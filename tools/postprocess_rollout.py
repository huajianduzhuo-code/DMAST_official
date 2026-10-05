import json
import tyro
import wandb
import logging

from pathlib import Path
from tqdm import tqdm
from typing import List, Literal, Dict, Any
from agentlab.llm.llm_utils import BaseMessage
from configs.postprocess_rollout_args import PostprocessRolloutArg
from pathlib import Path
from utils.wandb import wandb_log

logger = logging.getLogger(__name__)


def _extract_text_tokens_from_messages(messages: List[Dict[str, Any]], tokenizer=None) -> List[str]:
    """
    Extract text tokens from messages when vLLM doesn't return prompt_tokens (multimodal case).
    
    For multimodal inputs, vLLM doesn't return prompt_tokens because the prompt includes
    image embeddings. This function extracts tokens from the text portion only.
    
    Args:
        messages: List of message dicts (first 2 are prompt)
        tokenizer: Optional tokenizer instance (lazy loaded if None)
        
    Returns:
        List of token strings for the text portions of the prompt
    """
    if tokenizer is None:
        # Lazy load tokenizer only when needed
        from transformers import AutoTokenizer
        
        # Use local model path (same as used in configs/train_adversarial_args.py)
        local_model_path = Path.home() / "dmast_ckpt/pretrained/gemma-3-12b-it"
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_path),
            trust_remote_code=True,
        )
    
    tokens = []
    for msg in messages[:2]:  # First 2 messages are the prompt
        content = msg.get("content", "")
        
        if isinstance(content, str):
            # Simple text message
            msg_tokens = tokenizer.tokenize(content)
            tokens.extend(msg_tokens)
        elif isinstance(content, list):
            # Multimodal message - extract only text portions
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content = item.get("text", "")
                    msg_tokens = tokenizer.tokenize(text_content)
                    tokens.extend(msg_tokens)
                # Note: We skip image tokens as they're handled separately by the vision tower
    
    return tokens


def post_process_episodes(
    game_episodes,
    mode: Literal["imitation", "rl_on_policy", "rl_off_policy", "eval"],
    roles: List[Literal["agent", "attacker"]],
    use_attacker: bool,
    model_path: str = None,  # Optional model path for tokenizer.
):
    if not use_attacker:
        assert "attacker" not in roles, "attacker is not used"

    remove_failure_samples = mode == "imitation"
    remove_off_policy_samples = mode == "rl_on_policy"

    attacker_success_episode_num = attacker_failure_episode_num = 0
    agent_adversarial_success_episode_num = agent_adversarial_failure_episode_num = 0
    agent_normal_success_episode_num = agent_normal_failure_episode_num = 0

    attacker_success_transition_num = attacker_failure_transition_num = 0
    agent_adversarial_success_transition_num = agent_adversarial_failure_transition_num = 0
    agent_normal_success_transition_num = agent_normal_failure_transition_num = 0

    outputs = []

    # during post-processing, we break episodes into independent transitions,
    # so we need to keep track of the episode index to recover the episode
    episode_idx = 0

    # Load the tokenizer once for the full episode batch.
    # Model path can be specified explicitly or inferred from episode data
    from transformers import AutoTokenizer
    
    # Try to get model path from episodes if not provided
    if model_path is None:
        for episode in game_episodes[:5]:  # Check first few episodes
            if episode.get('agent_model_name'):
                model_path = episode.get('agent_model_name')
                break
    
    # Fall back to the default model path.
    if model_path is None:
        # Try common model paths
        possible_paths = [
            Path.home() / "dmast_ckpt/pretrained/gemma-3-12b-it",
            Path.home() / "dmast_ckpt/pretrained/gemma-3-27b-it",
            Path.home() / "dmast_ckpt/pretrained/qwen3-vl-8b-instruct",
        ]
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        if model_path is None:
            model_path = str(Path.home() / "dmast_ckpt/pretrained/gemma-3-12b-it")
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    logger.info(f"Loaded tokenizer from {model_path}")

    for episode in tqdm(game_episodes):
        history = episode['history']
        error = episode.get('error', None)
        epi_use_attacker = episode['use_attacker']
        terminated = episode['terminated']
        truncated = episode.get('truncated', not terminated)

        if error is not None:
            continue

        # due to agent / env error, some episodes may be empty
        if not history:
            continue

        # if we only want to keep non-adversarial data, we skip episodes where attacker is used
        if not use_attacker and epi_use_attacker:
            continue

        assert terminated or truncated, "Episode is not terminated or truncated"
        if terminated:
            assert history[-1]['terminated']
        else:
            assert not history[-2]['terminated']
            assert not history[-1]['terminated']
            assert history[-1]['truncated']

        for role in roles:
            assert role in ["attacker", "agent"], f"Invalid role: {role}"

            if terminated:
                final_reward = history[-1][f"{role}_reward"]
            else:
                # history[-1] is the next_obs for the last transition
                final_reward = history[-2][f"{role}_reward"]

            # when only collecting attacker data, we skip episodes where attacker is not used
            if role == "attacker" and not epi_use_attacker:
                continue

            # for imitation training, we remove failure samples
            if remove_failure_samples and final_reward <= 0:
                continue

            # the opponent is not using latest model, so we need to drop those off-policy samples
            if remove_off_policy_samples and not episode[f"{role}_on_policy"]:
                continue

            # count attacker and agent success and fail, for reweighting
            if terminated:
                num_transitions = len(history)
            else:
                num_transitions = len(history) - 1

            if role == "agent":
                if final_reward > 0:
                    if epi_use_attacker:
                        agent_adversarial_success_episode_num += 1
                        agent_adversarial_success_transition_num += num_transitions
                    else:
                        agent_normal_success_episode_num += 1
                        agent_normal_success_transition_num += num_transitions
                else:
                    if epi_use_attacker:
                        agent_adversarial_failure_episode_num += 1
                        agent_adversarial_failure_transition_num += num_transitions
                    else:
                        agent_normal_failure_episode_num += 1
                        agent_normal_failure_transition_num += num_transitions

            elif role == "attacker":
                if final_reward > 0:
                    attacker_success_episode_num += 1
                    attacker_success_transition_num += num_transitions
                else:
                    attacker_failure_episode_num += 1
                    attacker_failure_transition_num += num_transitions

            else:
                raise ValueError(f"Invalid role: {role}")

            episode_info = []
            for i, transition in enumerate(history):
                message_dicts = transition[f"{role}_messages"]
                
                # Skip transitions without messages for this role.
                if message_dicts is None:
                    print(f"Warning: Skipping transition {i} for role '{role}' in episode {episode_idx} - message_dicts is None")
                    continue
                    
                assert len(message_dicts) in [3, 4], f"message length is not 3 or 4: {len(message_dicts)}" # NOTE: for attacker, message length is 3 (system, user, assistant). For agent, message length is 3 or 4: most of the cases are 3 (system, user, assistant). Rarely, it is 4 (system, user, assistant, user), where the last user is Error while parsing action.

                training_info = transition[f"{role}_training_info"]
                
                # Check if this is a multimodal input (has images in messages)
                has_images = any(
                    isinstance(msg.get("content"), list) and 
                    any(item.get("type") == "image_url" for item in msg.get("content", []) if isinstance(item, dict))
                    for msg in message_dicts
                )
                
                # ============================================================================
                # STEP 1: Get query tokens
                # Priority: Use vLLM's prompt_token_ids if available (ensures consistency)
                # Fallback: Use prompt_tokens (token strings) or manual extraction
                # ============================================================================
                prompt_token_ids = training_info.get("prompt_token_ids", [])
                query_tokens = training_info["prompt_tokens"]
                
                # Keep the string representation of query tokens alongside token IDs.
                if not query_tokens:
                    # Decode token IDs to strings if needed
                    query_tokens = [tokenizer.convert_ids_to_tokens(tid) for tid in prompt_token_ids]
                # ============================================================================
                # STEP 2: Get response token IDs
                # Priority: Use vLLM's response_token_ids if available
                # Fallback: Use response_tokens (token strings)
                # ============================================================================
                response_token_ids = training_info.get("response_token_ids", [])
                target_tokens = training_info["response_tokens"]
                
                if response_token_ids:
                    logger.debug(f"Using {len(response_token_ids)} response_token_ids from vLLM")
                    # Retain response token strings alongside token IDs.
                    if not target_tokens:
                        target_tokens = [tokenizer.convert_ids_to_tokens(tid) for tid in response_token_ids]
                
                # ============================================================================
                # STEP 2.5: FILTER - Skip samples with image tokens in response
                # ============================================================================
                # Model-specific image token IDs that should NOT appear in response:
                # - 262144: Gemma3 image soft token
                # - 151655, 151656, 151657: Qwen3-VL vision tokens
                IMAGE_TOKEN_IDS = [262144, 151655, 151656, 151657]
                has_image_token_in_response = any(tok in response_token_ids for tok in IMAGE_TOKEN_IDS)
                if has_image_token_in_response:
                    num_image_tokens = sum(response_token_ids.count(tok) for tok in IMAGE_TOKEN_IDS)
                    logger.warning(f"⚠️ SKIPPING transition {i} for role '{role}' in episode {episode_idx} - "
                                   f"response contains {num_image_tokens} image tokens (corrupted data)")
                    continue  # Skip this corrupted transition
                
                transition_info = {
                    # Episode/transition metadata (used for grouping and ordering)
                    "episode_idx": episode_idx,
                    "task_name": episode['task_name'],
                    "transition_idx": i,
                    "role": role,
                    "on_policy": episode.get(f"{role}_on_policy", None),
                    "use_attacker": epi_use_attacker,
                    "seed": episode.get("seed", None),
                    "attacker_model_name": episode.get("attacker_model_name", None),  # For GRPO grouping by attacker
                    "success": final_reward > 0,        # for weight calculation
                    # "query": query,
                    # "query_tokens": query_tokens,
                    # "query_logprob": training_info["prompt_logprob"],
                    # "target": target.strip(),
                    # "target_tokens": target_tokens,
                    "target_logprob": training_info["response_logprob"],
                    
                    # Episode state
                    "reward": transition[f"{role}_reward"],
                    "terminated": transition["terminated"],
                    "truncated": transition["truncated"],
                    # Store messages, including image content, for multimodal training.
                    "messages": message_dicts if has_images else None,  # Only store for multimodal
                    "is_multimodal": has_images,
                    "prompt_token_ids": prompt_token_ids,
                    "response_token_ids": response_token_ids,
                }

                episode_info.append(transition_info)

            # Assign an episode index only when valid transitions are available.
            # This prevents creating gaps in episode_idx sequence when all transitions are skipped
            if episode_info:
                outputs.extend(episode_info)
                episode_idx += 1
            else:
                # Log when an episode/role combination produces no valid transitions
                print(f"Warning: Episode role '{role}' (use_attacker={epi_use_attacker}) produced no valid transitions. Skipping episode_idx increment.")

    total_transition_num = (
        attacker_success_transition_num +
        attacker_failure_transition_num +
        agent_adversarial_success_transition_num +
        agent_adversarial_failure_transition_num +
        agent_normal_success_transition_num +
        agent_normal_failure_transition_num
    )

    if roles == ["agent"]:
        divider = 2
    elif roles == ["attacker"]:
        divider = 1
    elif roles == ["agent", "attacker"]:
        divider = 3
    else:
        raise ValueError(f"Invalid roles: {roles}")

    if not remove_failure_samples:
        divider *= 2

    attacker_success_weight = (
        (total_transition_num / divider / attacker_success_transition_num)
        if attacker_success_transition_num > 0
        else 0
    )
    attacker_failure_weight = (
        (total_transition_num / divider / attacker_failure_transition_num)
        if attacker_failure_transition_num > 0
        else 0
    )
    agent_adversarial_success_weight = (
        (total_transition_num / divider / agent_adversarial_success_transition_num)
        if agent_adversarial_success_transition_num > 0
        else 0
    )
    agent_adversarial_failure_weight = (
        (total_transition_num / divider / agent_adversarial_failure_transition_num)
        if agent_adversarial_failure_transition_num > 0
        else 0
    )
    agent_normal_success_weight = (
        (total_transition_num / divider / agent_normal_success_transition_num)
        if agent_normal_success_transition_num > 0
        else 0
    )
    agent_normal_failure_weight = (
        (total_transition_num / divider / agent_normal_failure_transition_num)
        if agent_normal_failure_transition_num > 0
        else 0
    )

    for item in outputs:
        if item['role'] == "attacker":
            if item['success']:
                item['weight'] = attacker_success_weight
            else:
                item['weight'] = attacker_failure_weight
        elif item['role'] == "agent":
            if item['success']:
                if item['use_attacker']:
                    item['weight'] = agent_adversarial_success_weight
                else:
                    item['weight'] = agent_normal_success_weight
            else:
                if item['use_attacker']:
                    item['weight'] = agent_adversarial_failure_weight
                else:
                    item['weight'] = agent_normal_failure_weight
        else:
            raise ValueError(f"Invalid role: {item['role']}")

    stats = {}
    if "agent" in roles:
        stats.update({
            "agent_adversarial_success_episode_num": agent_adversarial_success_episode_num,
            "agent_adversarial_failure_episode_num": agent_adversarial_failure_episode_num,
            "agent_normal_success_episode_num": agent_normal_success_episode_num,
            "agent_normal_failure_episode_num": agent_normal_failure_episode_num,
            "agent_adversarial_success_transition_num": agent_adversarial_success_transition_num,
            "agent_adversarial_failure_transition_num": agent_adversarial_failure_transition_num,
            "agent_normal_success_transition_num": agent_normal_success_transition_num,
            "agent_normal_failure_transition_num": agent_normal_failure_transition_num,
        })

        agent_adversarial_episode_num = agent_adversarial_success_episode_num + agent_adversarial_failure_episode_num
        agent_normal_episode_num = agent_normal_success_episode_num + agent_normal_failure_episode_num

        if agent_adversarial_episode_num:
            stats["agent_adversarial_success_rate"] = (
                100 * agent_adversarial_success_episode_num / agent_adversarial_episode_num
            )
            stats["agent_adversarial_failure_rate"] = (
                100 * agent_adversarial_failure_episode_num / agent_adversarial_episode_num
            )
        if agent_normal_episode_num:
            stats["agent_normal_success_rate"] = (
                100 * agent_normal_success_episode_num / agent_normal_episode_num
            )
            stats["agent_normal_failure_rate"] = (
                100 * agent_normal_failure_episode_num / agent_normal_episode_num
            )

    if "attacker" in roles:
        stats.update({
            "attacker_success_episode_num": attacker_success_episode_num,
            "attacker_failure_episode_num": attacker_failure_episode_num,
            "attacker_success_transition_num": attacker_success_transition_num,
            "attacker_failure_transition_num": attacker_failure_transition_num,
        })

        attacker_episode_num = attacker_success_episode_num + attacker_failure_episode_num
        if attacker_episode_num:
            stats["attacker_success_rate"] = (
                100 * attacker_success_episode_num / attacker_episode_num
            )
            stats["attacker_failure_rate"] = (
                100 * attacker_failure_episode_num / attacker_episode_num
            )

    return outputs, stats


def postprocess_rollout(
    args: PostprocessRolloutArg,
    wandb_prefix: str = "episode_stats",
    iteration: int = 1,
):
    with open(args.input_data_path, 'r') as f:
        game_episodes = json.load(f)

    rl_data, stats = post_process_episodes(
        game_episodes,
        args.mode,
        args.roles,
        args.use_attacker,
    )

    # NOTE: length of game_episodes is the number of real episodes collected. For example, if num_train_envs_per_iter=8 and num_train_responses_per_env=16, then length of game_episodes is 8 * 16 = 128.
    # length of rl_data, the output of post_process_episodes, is the number of transitions * len(args.roles). number of transitions is the sum of steps in all game_episodes. If args.roles = ["agent", "attacker"], then length of rl_data is the number of transitions * 2.
    # in each item of rl_data, there is an episode_idx, the range of episode_idx is [0, length of game_episodes * len(args.roles) - 1]. So when args.roles = ["agent", "attacker"], each episode is split into 2 episode_idx: one idx for agent and one idx for attacker.
    # save rollout data to file
    output_data_path = Path(args.output_data_path)
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_data_path, 'w') as f:
        json.dump(rl_data, f, ensure_ascii=False, indent=2)

    # iteration 0 is for imitation training, so we don't log and save success / failure stats
    if iteration == 0:
        return

    # log stats to wandb
    if wandb.run is not None:
        logs = {k: v for k, v in stats.items() if isinstance(v, float)}
        wandb_log(logs, step_metric="iteration", step_value=iteration, prefix=wandb_prefix)

    # save stats to file
    for k, v in stats.items():
        if isinstance(v, float):
            stats[k] = f"{v:.1f}"

    input_data_path = Path(args.input_data_path)
    stats_path = input_data_path.parent / f"{input_data_path.stem}_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            existing_stats = json.load(f)
        stats = {**existing_stats, **stats}
    with open(stats_path, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = tyro.cli(PostprocessRolloutArg)
    postprocess_rollout(args)
