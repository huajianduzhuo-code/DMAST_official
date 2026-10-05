import json
import random
import numpy as np
import wandb

from accelerate import Accelerator
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import (
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
)
from trl.trainer.utils import disable_dropout_in_model, first_true_indices

from configs.assign_credit_args import AssignCreditArgs
from data.dataloaders import TextDataset, credit_assignment_data_collactor

from tools.trainers import compute_log_prob, compute_kl_div, EPS

from utils.load_model import load_from_pretrained_model
from utils.utils import (
    IGNORE_INDEX,
    read_json_or_jsonl_data,
    setup_distributed_print,
    turn_off_deepspeed_info_logs,
)
from utils.wandb import resume_wandb_from_env_vars, wandb_log
from utils.two_hot import symexp, symlog, compute_softmax_over_buckets


def recover_episode_list(data_list):
    """
    Groups data indices by episode, ordered by transition index.

    Args:
        data_list: A list of dicts with keys: episode_idx, transition_idx, ....

    Returns:
        A list of lists, where the outer list index corresponds to the
        episode_idx, and each inner list contains the data for that episode
    """
    if not data_list:
        return []

    ep_dict = defaultdict(list)                             # episode_idx -> list of (transition_idx, data)
    max_ep_idx = -1
    for data in data_list:
        ep_idx = data['episode_idx']
        trans_idx = data['transition_idx']

        ep_dict[ep_idx].append((trans_idx, data))
        if ep_idx > max_ep_idx:
            max_ep_idx = ep_idx

    episodes = [[] for _ in range(max_ep_idx + 1)]

    for ep_idx, transitions in ep_dict.items():
        sorted_transitions = sorted(transitions, key=lambda item: item[0])
        episodes[ep_idx] = [data for _, data in sorted_transitions]

    return episodes


def get_episode_length(episode):
    terminated = episode[-1]['terminated']
    truncated = episode[-1]['truncated']
    
    # Give termination precedence when both outcome flags are set.
    if terminated and truncated:
        # Use the terminal outcome rather than the time-limit bootstrap path.
        terminated = True
        truncated = False
    
    # Use all transitions when no terminal or truncated observation is available.
    if not terminated and not truncated:
        # (use all transitions)
        return len(episode)
    
    if terminated:
        num_transitions = len(episode)
    elif truncated:
        # if truncated, episode[-1] is just used for calculating the value target,
        # so it won't be included as policy / value training data
        
        # Standard case: both episode[-2] and episode[-1] should have truncated=True
        # Edge case: if episode[-2] doesn't have truncated=True, it means the final 
        # observation with truncated=True was skipped during postprocessing
        if len(episode) >= 2 and episode[-2]['truncated']:
            # Standard truncated episode
            assert not episode[-1]['terminated']
            assert not episode[-2]['terminated']
            num_transitions = len(episode) - 1
        else:
            # Edge case: final observation was skipped
            # episode[-1] is actually the last real transition, not a final obs
            # So use all transitions
            num_transitions = len(episode)
    else:
        raise ValueError(f"Episode is not terminated or truncated")
    
    return num_transitions


def get_grpo_group_key(episode):
    task_name = episode[0]['task_name']
    use_attacker = episode[0]['use_attacker']
    seed = episode[0]['seed']
    return f"{task_name}_{use_attacker}_{seed}"


def compute_task_reward_stats(episodes):
    """
    episodes: List[List[Dict[str, Any]]], episode[transition[key -> value]]
    """
    task_reward = defaultdict(list)    # grpo_group_key -> reward_list
    batch_level_last_reward = []
    for episode in episodes:
        grpo_group_key = get_grpo_group_key(episode)

        num_transitions = get_episode_length(episode)
        last_reward = episode[num_transitions - 1]['raw_reward'] # reward of the last transition

        task_reward[grpo_group_key].append(last_reward)
        batch_level_last_reward.append(last_reward)

    batch_level_std = np.std(batch_level_last_reward)

    task_reward_stats = {
        grpo_group_key: (np.mean(reward_list), np.std(reward_list), batch_level_std)
        for grpo_group_key, reward_list in task_reward.items()
    }

    return task_reward_stats


def assign_credit(
    args: AssignCreditArgs,
):
    accelerator = Accelerator()
    setup_distributed_print(accelerator.is_main_process)
    turn_off_deepspeed_info_logs()

    # setup wandb
    #---------------------------------------------------------------------------------
    resume_wandb_from_env_vars(accelerator.is_main_process)
    accelerator.wait_for_everyone()

    # setup tokenizer
    #---------------------------------------------------------------------------------        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,   
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Try to load processor for multimodal models
    #---------------------------------------------------------------------------------
    processor = None
    if args.is_multimodal:
        from transformers import AutoProcessor
        import json
        
        # Determine the correct path for processor loading
        # If model_name_or_path is a LoRA adapter (has adapter_config.json but no config.json),
        # we need to load processor from the base model instead
        processor_path = args.model_name_or_path
        adapter_config_path = Path(args.model_name_or_path) / "adapter_config.json"
        model_config_path = Path(args.model_name_or_path) / "config.json"
        
        if adapter_config_path.exists() and not model_config_path.exists():
            # This is a LoRA adapter, read base_model_name_or_path
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get('base_model_name_or_path')
            if base_model_path:
                processor_path = base_model_path
                print(f"ℹ️  Detected LoRA adapter, loading processor from base model: {processor_path}")
        
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        print(f"✅ Loaded processor for multimodal model: {processor_path}")
    else:
        print(f"ℹ️  Skipping processor loading (text-only model, is_multimodal=False)")

    # load data
    #---------------------------------------------------------------------------------
    compute_value = args.rl_method == "ppo"
    data_list = read_json_or_jsonl_data(args.reward_data_path)
    
    # Filter out corrupted samples BEFORE creating dataset
    # This ensures all ranks have the same data and process the same number of batches
    # (Filtering during collation can cause NCCL desync and hang on exit)
    IMAGE_TOKEN_ID = 262144  # Gemma3 image soft token
    original_count = len(data_list)
    data_list = [
        item for item in data_list 
        if IMAGE_TOKEN_ID not in item.get('response_token_ids', [])
    ]
    filtered_count = original_count - len(data_list)
    if filtered_count > 0:
        print(f"⚠️ Pre-filtered {filtered_count} samples with image tokens in response_ids")
    
    dataset = TextDataset(data_list)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_batch_size, 
        shuffle=False,
        collate_fn=lambda batch: credit_assignment_data_collactor(
            batch,
            tokenizer,
            prepare_value_input_ids=compute_value,
            processor=processor,
        ),
        drop_last=False,
    )

    # setup model
    #---------------------------------------------------------------------------------
    print(f"Begin loading model from {args.model_name_or_path}")

    model = load_from_pretrained_model(
        PeftModelForCausalLM if args.use_lora else AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
    )
    ref_model = load_from_pretrained_model(
        PeftModelForCausalLM if args.ref_model_uses_lora else AutoModelForCausalLM,
        args.ref_model_name_or_path,
        tokenizer,
    )

    print(f"Finished loading model from {args.model_name_or_path}")
    print(f"Finished loading ref_model from {args.ref_model_name_or_path}")

    if compute_value:
        value_model = load_from_pretrained_model(
            PeftModelForSequenceClassification if args.use_lora else AutoModelForSequenceClassification,
            args.value_model_name_or_path,
            tokenizer,
            # if value_model_path == model_path, it means that the value model is not trained yet, so we reinitialize value head to 0
            zero_init_value_head=args.value_model_name_or_path == args.model_name_or_path,
            num_value_buckets=args.num_value_buckets,
        )
        print(f"Finished loading value_model from {args.value_model_name_or_path}")
    else:
        value_model = None

    # disable dropout in all models to avoid large discrepancy between policy model and ref model
    for module in [model, ref_model, value_model]:
        if module is not None:
            disable_dropout_in_model(module)

    # prepare distributed inference
    #---------------------------------------------------------------------------------
    # deepspeed can only prepare one model, so we use this workaround to prepare reference and value models
    model.ref_model = ref_model
    if compute_value:
        model.value_model = value_model

    model, dataloader = accelerator.prepare(model, dataloader)

    ref_model = accelerator.unwrap_model(model).ref_model
    if compute_value:
        value_model = accelerator.unwrap_model(model).value_model

    # compute credit
    #---------------------------------------------------------------------------------
    skipped_indices = set()  # Track indices that were skipped due to OOM
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Assigning credit", disable=not accelerator.is_main_process):
            # Skip empty batches (all items were corrupted/skipped)
            if batch is None:
                continue
            
            # Track OOM status across all GPUs
            local_oom = torch.tensor([0], device=accelerator.device)
            
            try:
                input_ids = batch['input_ids'].to(accelerator.device) # (batch_size, seq_len), includes both query and response token ids, padding with 0s in the front
                attention_mask = batch['attention_mask'].to(accelerator.device) # (batch_size, seq_len), 1 for non-padding tokens, 0 for padding tokens; non-padding tokens are the query and response tokens
                position_ids = batch['position_ids'].to(accelerator.device) # (batch_size, seq_len), start counting (from 0) from the first non-padding token (query token), padding with 0s in the front
                label = batch['labels'].to(accelerator.device) # (batch_size, seq_len), includes only response token ids, padding with -100s in the front

                # For multimodal inputs: extract pixel_values and image_grid_thw if present
                if args.is_multimodal:
                    pixel_values = batch.get('pixel_values')
                    pixel_values = pixel_values.to(accelerator.device) # (batch_size, num_images, C, H, W)
                    # For Qwen3-VL models, also extract image_grid_thw (required for image position encoding)
                    image_grid_thw = batch.get('image_grid_thw')
                    if image_grid_thw is not None:
                        image_grid_thw = image_grid_thw.to(accelerator.device)
                else:
                    pixel_values = None
                    image_grid_thw = None

                sampling_logprob = batch['sampling_logprob'].to(accelerator.device) # NOTE: Credit assignment uses batch size 1; larger batches require checking sampling-logprob alignment with padded response tokens.

                # compute generation logprob
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                # (batch_size, seq_len), (batch_size, seq_len)
                generation_logprob, token_valid = compute_log_prob(model_outputs.logits / args.temperature, label, valid_only=False)
                del model_outputs
                torch.cuda.empty_cache()

                # compute ref_model logprob
                ref_model_outputs = ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                # (batch_size, seq_len)
                ref_logprob, _ = compute_log_prob(ref_model_outputs.logits / args.temperature, label, valid_only=False)
                del ref_model_outputs
                torch.cuda.empty_cache()

                # NOTE: PPO value prediction is experimental; GRPO and SPAG do not use a value model.
                if compute_value:

                    value_input_ids = batch['value_input_ids'].to(accelerator.device)
                    value_attention_mask = batch['value_attention_mask'].to(accelerator.device)
                    value_position_ids = batch['value_position_ids'].to(accelerator.device)

                    # (batch_size, seq_len, num_value_buckets)
                    value_model_outputs = value_model(
                        input_ids=value_input_ids,
                        attention_mask=value_attention_mask,
                        position_ids=value_position_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    sequence_lengths = first_true_indices(value_input_ids == tokenizer.eos_token_id)
                    value = value_model_outputs[torch.arange(value_model_outputs.size(0)), sequence_lengths]        # (batch_size,)

                    num_value_buckets = value.size(-1)
                    if num_value_buckets == 1:
                        value = value.squeeze(-1)
                        if args.value_activation == "none":
                            value = torch.clip(value, args.value_min, args.value_max)
                        elif args.value_activation == "sigmoid":
                            value = torch.sigmoid(value) * (args.value_max - args.value_min) + args.value_min
                        else:
                            raise ValueError(f"Invalid value activation: {args.value_activation}")
                    else:
                        assert args.value_activation == "none"
                        value = symexp(compute_softmax_over_buckets(
                            value,
                            x_min=symlog(args.value_min),
                            x_max=symlog(args.value_max),
                        ))

                    del value_model_outputs
                    torch.cuda.empty_cache()
                else:
                    value = None

            except torch.cuda.OutOfMemoryError as e:
                # Mark that this GPU had OOM
                local_oom = torch.tensor([1], device=accelerator.device)
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    print(f"⚠️ CUDA OOM on rank {accelerator.process_index}, seq_len={batch['input_ids'].shape[-1]}")

            # Synchronize OOM status across all GPUs - if ANY GPU had OOM, all skip
            global_oom = accelerator.reduce(local_oom, reduction="sum")
            
            if global_oom.item() > 0:
                # At least one GPU had OOM, all GPUs skip this batch
                batch_indices = batch['index'].cpu().numpy().tolist()
                skipped_indices.update(batch_indices)
                if accelerator.is_main_process:
                    print(f"⚠️ Skipping batch (OOM on {global_oom.item()} GPU(s)), indices: {batch_indices}")
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                continue

            # compute reward = raw_reward - kl_regularization - length_penalty
            # kl regularization
            ref_kl_div = compute_kl_div(ref_logprob, generation_logprob)                                        # (batch_size, seq_len)
            ref_kl_div = ref_kl_div.sum(dim=-1) / token_valid.sum(dim=-1)                                       # (batch_size,)

            sampling_kl_div = compute_kl_div(sampling_logprob, generation_logprob)
            sampling_kl_div = sampling_kl_div.sum(dim=-1) / token_valid.sum(dim=-1)

            # length penalty
            num_total_tokens = (input_ids != IGNORE_INDEX).sum(dim=-1)
            response_length = (label != IGNORE_INDEX).sum(dim=-1)
            query_length = num_total_tokens - response_length

            new_tokens_ratio = response_length / args.max_new_tokens
            ratio = (new_tokens_ratio - args.length_penalty_start) / (args.length_penalty_end - args.length_penalty_start)
            ratio = torch.clip(ratio, 0.0, 1.0)

            if args.length_penalty_type == "none":
                length_penalty = 0.0
            elif args.length_penalty_type == "linear":
                length_penalty = ratio
            elif args.length_penalty_type == "cosine":
                length_penalty = 0.5 - 0.5 * torch.cos(ratio * np.pi)

            raw_reward = batch['reward'].to(accelerator.device)

            # gather across all GPUs and convert to numpy
            generation_logprob = accelerator.pad_across_processes(generation_logprob, dim=-1)
            ref_logprob = accelerator.pad_across_processes(ref_logprob, dim=-1)
            token_valid = accelerator.pad_across_processes(token_valid, dim=-1).bool()

            index = accelerator.gather(batch['index']).cpu().numpy()
            token_valid = accelerator.gather(token_valid).cpu().numpy()

            (
                generation_logprob,
                ref_logprob,
                generation_value_pred,
                raw_reward,
                ref_kl_div,
                sampling_kl_div,
                length_penalty,
                response_length,
                query_length,
            ) = map(
                lambda x: None if x is None else accelerator.gather(x.double()).cpu().numpy(),     # json expects float64
                (
                    generation_logprob,
                    ref_logprob,
                    value,
                    raw_reward,
                    ref_kl_div,
                    sampling_kl_div,
                    length_penalty,
                    response_length,
                    query_length,
                ),
            )

            # add ppo training data back to data_list
            if accelerator.is_main_process:
                for i, idx in enumerate(index):
                    transition = data_list[idx]
                    token_valid_i = token_valid[i] # token_valid is a boolean tensor, True for non-padding tokens, False for padding tokens in label
                    generation_logprob_i = generation_logprob[i]
                    ref_logprob_i = ref_logprob[i]
                    transition['generation_logprob'] = generation_logprob_i[token_valid_i].tolist()
                    transition['ref_logprob'] = ref_logprob_i[token_valid_i].tolist()
                    transition['raw_reward'] = raw_reward[i]
                    transition['ref_kl_div'] = ref_kl_div[i]
                    transition['sampling_kl_div'] = sampling_kl_div[i]
                    transition['length_penalty'] = length_penalty[i]
                    transition['response_length'] = response_length[i]
                    transition['query_length'] = query_length[i]

                    if compute_value:
                        transition['generation_value_pred'] = generation_value_pred[i]

            accelerator.wait_for_everyone()

    # Report skipped samples
    if accelerator.is_main_process and skipped_indices:
        print(f"⚠️ Total skipped samples due to OOM: {len(skipped_indices)}")

    if accelerator.is_main_process:
        
        # Filter out skipped samples (they won't have 'generation_logprob')
        original_count = len(data_list)
        if args.rl_method == "spag":
            # Keep role trajectories intact so an OOM cannot make an earlier
            # action or a truncated bootstrap observation become the outcome.
            incomplete_episode_ids = {
                data['episode_idx'] for data in data_list
                if 'generation_logprob' not in data
            }
            data_list = [
                data for data in data_list
                if data['episode_idx'] not in incomplete_episode_ids
            ]
        else:
            data_list = [data for data in data_list if 'generation_logprob' in data]
        filtered_count = len(data_list)
        if original_count != filtered_count:
            print(f"Filtered out {original_count - filtered_count} samples due to incomplete credit assignment")
            print(f"   Remaining samples: {filtered_count}")

        # compute advantage and return
        episodes = recover_episode_list(data_list)      # List[Dict[str, Any]] -> List[List[Dict[str, Any]]], is a shallow copy of data_list
        
        # Filter out empty episodes (can happen if all transitions were skipped due to OOM)
        original_episode_count = len(episodes)
        episodes = [ep for ep in episodes if len(ep) > 0]
        if len(episodes) != original_episode_count:
            print(f"✅ Filtered out {original_episode_count - len(episodes)} empty episodes")

        if args.rl_method == "grpo":
            # for each task, compute the mean and std of the reward, which is used for normalizing the advantage
            task_reward_stats = compute_task_reward_stats(episodes)

        for episode_idx, episode in enumerate(episodes):
            num_transitions = get_episode_length(episode)

            if args.rl_method == "ppo": # NOTE: PPO advantage calculation is experimental; DMAST uses GRPO.
                # compute Generalized Advantage Estimation
                lastgaelam = 0
                for t in reversed(range(num_transitions)):
                    if t == num_transitions - 1:
                        if episode[-1]['terminated']:
                            nextvalues = 0.0
                        elif episode[-1]['truncated']:
                            # nextvalues = episode[t + 1]['generation_value_pred']
                            nextvalues = 0.0
                        else:
                            raise ValueError(f"Episode {episode_idx} is not terminated or truncated")
                    else:
                        nextvalues = episode[t + 1]['generation_value_pred']

                    raw_reward = episode[t]['raw_reward']
                    length_penalty = episode[t]['length_penalty']
                    value = episode[t]['generation_value_pred']
                    reward = raw_reward - length_penalty * args.length_penalty_coef
                    delta = reward + args.gamma * nextvalues - value
                    advantage = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam

                    episode[t]['raw_advantage'] = advantage
                    episode[t]['return'] = np.clip(advantage + value, args.value_min, args.value_max)

            elif args.rl_method == "spag":
                # SPAG advantage = discounted final reward - length penalty
                # Paper: "Self-playing Adversarial Language Game Enhances LLM Reasoning"
                # Uses this role's episode outcome with temporal discounting.
                last_reward = episode[num_transitions - 1]['raw_reward']

                for t in reversed(range(num_transitions)):
                    length_penalty = episode[t]['length_penalty']
                    episode[t]['advantage'] = last_reward * (args.spag_gamma ** (num_transitions - 1 - t)) - length_penalty * args.length_penalty_coef
                    # Broadcast the unshaped outcome to every real action.
                    # Do not use a transition's immediate reward for filtering.
                    episode[t]['spag_episode_reward'] = last_reward
                    
                    # Sample weight (will be adjusted dynamically if episode filtering is enabled)
                    # Default: 0.5 for both attacker and defender
                    # With filtering: balance the retained roles' sample weights.
                    episode[t]['weight'] = 0.5

            elif args.rl_method == "grpo":
                grpo_group_key = get_grpo_group_key(episode)
                mean_reward, group_level_std, batch_level_std = task_reward_stats[grpo_group_key]
                last_reward = episode[num_transitions - 1]['raw_reward']

                for t in range(num_transitions):
                    # 'success' is an episode-level boolean (final_reward > 0)
                    # All transitions in the same episode share the same success value
                    if episode[t]['success']:
                        # For successful episodes: Apply length penalty to encourage efficiency
                        # Since the task is already completed, we want the model to learn to be more concise
                        length_penalty = episode[t]['length_penalty']
                        reward = last_reward - length_penalty * args.length_penalty_coef
                    else:
                        # For failed episodes: No length penalty
                        # The primary goal is to learn task completion, not brevity
                        # Applying length penalty here could mislead the model to generate shorter
                        # but still unsuccessful responses
                        reward = last_reward

                    episode[t]['reward'] = reward
                    if args.normalization_type == "batch":
                        episode[t]['advantage'] = (reward - mean_reward) / (batch_level_std+ EPS)
                    elif args.normalization_type == "group":
                        episode[t]['advantage'] = (reward - mean_reward) / (group_level_std + EPS)
                    else:
                        raise ValueError(f"Invalid normalization type: {args.normalization_type}")
                    episode[t]['reward_mean'] = mean_reward
                    episode[t]['reward_group_std'] = group_level_std
                    episode[t]['reward_batch_std'] = batch_level_std

            else:
                raise ValueError(f"Invalid RL method: {args.rl_method}")

        if args.rl_method == "ppo": # NOTE: PPO advantage calculation is experimental; DMAST uses GRPO.
            # normalize the advantage

            # remove the last item in each truncated episode, as it's only used for calculating the value target
            data_list = [data for data in data_list if 'raw_advantage' in data]

            advantages = [data['raw_advantage'] for data in data_list]
            mean, std = np.mean(advantages), np.std(advantages)
            for data in data_list:
                data['advantage'] = (data['raw_advantage'] - mean) / (std + EPS)
        elif args.rl_method == "grpo":
            # remove zero variance data
            data_list = [data for data in data_list if 'reward_group_std' in data and data['reward_group_std'] != 0]
        elif args.rl_method == "spag":
            # Step 1: Basic filtering (keep data with advantage)
            data_list = [data for data in data_list if 'advantage' in data]
            
            # Step 2: Keep winning role trajectories, including intermediate
            # zero-reward actions. Both roles store their own positive reward
            # for success; agent rewards are not attacker-perspective scores.
            if args.spag_use_episode_filtering:
                original_count = len(data_list)
                data_list = [
                    data for data in data_list
                    if data['spag_episode_reward'] > args.spag_reward_threshold
                ]
                filtered_count = len(data_list)

                # Preserve transition-count balancing: with both roles present,
                # each role has half the retained dataset's total sample weight.
                role_counts = {
                    role: sum(data['role'] == role for data in data_list)
                    for role in ('agent', 'attacker')
                }
                role_weights = {
                    role: filtered_count / (2 * count) if count else 0.0
                    for role, count in role_counts.items()
                }
                for data in data_list:
                    data['weight'] = role_weights[data['role']]

                if accelerator.is_main_process:
                    print(f"\n{'='*80}")
                    print("SPAG filtering by each role's episode outcome")
                    print(f"{'='*80}")
                    print(f"Original transitions: {original_count}")
                    print(f"Retained transitions: {filtered_count}")
                    keep_ratio = filtered_count / original_count if original_count else 0.0
                    print(f"Keep ratio: {keep_ratio:.1%}")
                    for role, count in role_counts.items():
                        episode_count = len({
                            data['episode_idx'] for data in data_list if data['role'] == role
                        })
                        print(f"{role}: {episode_count} winning episodes, {count} transitions "
                              f"(weight: {role_weights[role]:.4f})")
                    print(f"{'='*80}\n")

        # wandb logging
        if wandb.run is not None:
            logs = {}
            for role in args.roles:
                role_data = [data for data in data_list if data['role'] == role]
                log_keys = ["ref_kl_div", "sampling_kl_div", "query_length", "response_length", "length_penalty"]
                if args.rl_method == "ppo":
                    log_keys.append("return")
                logs.update({
                    f"{role}_{k}": np.mean([data[k] for data in role_data])
                    for k in log_keys if role_data
                })
                if args.rl_method == "grpo":
                    logs.update({
                        f"{role}_num_transitions": len([data for data in data_list if data['role'] == role]),
                        f"{role}_positive_advantage_ratio": np.mean([data['advantage'] > 0 for data in data_list if data['role'] == role]),
                    })
                if args.rl_method == "spag":
                    # Log zeros too; absence of a role must remain visible.
                    logs[f"{role}_num_transitions"] = len(role_data)
                    logs[f"{role}_num_episodes"] = len({data['episode_idx'] for data in role_data})
                    if role_data:
                        logs.update({
                            f"{role}_avg_weight": np.mean([data.get('weight', 0.5) for data in role_data]),
                            f"{role}_avg_reward": np.mean([data.get('raw_reward', 0) for data in role_data]),
                            f"{role}_positive_reward_ratio": np.mean([data.get('raw_reward', 0) > 0 for data in role_data]),
                        })
            wandb_log(logs, step_metric="iteration", step_value=args.iteration, prefix=args.wandb_prefix)

        if args.rl_method == "spag" and not data_list:
            raise ValueError(
                "SPAG credit assignment retained no training transitions. "
                "Check episode outcomes and collection errors; no policy dataset was written."
            )

        # save value training data
        if compute_value:
            Path(args.value_data_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.value_data_path, 'w') as f:
                json.dump(data_list, f, indent=4)

        # save policy training data
        Path(args.policy_data_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.policy_data_path, 'w') as f:
            json.dump(data_list, f, indent=4)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(AssignCreditArgs)
    args = parser.parse_args_into_dataclasses()[0]
    assign_credit(args)

    if wandb.run is not None:
        wandb.finish()
