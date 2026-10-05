import json
import logging
import os
import wandb
import time
import numpy as np
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Optional

from browsergym.experiments.loop import EnvArgs
import bgym

from agents.llm.vllm_model import VllmModelArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp
from configs.postprocess_rollout_args import PostprocessRolloutArg
from configs.collect_episode_args import CollectEpisodeArgs, MultimodalCollectEpisodeArgs, VllmModelArgsList, AttackerPrivilegeObsFlags
from configs.train_adversarial_args import TrainAdversarialConfig

# Import unified collection system (supports both text-only and multimodal)
from tools.multimodal_collect_episode import collect_episodes, prepare_collect_kwargs

from tools.postprocess_rollout import postprocess_rollout

from utils.vllm import (
    is_multimodal_model,
    get_population_model_paths_for_multimodal,
    start_vllm_for_single_pair,
)

# HTML injection is used instead of image editing - no GPU-based image editor needed

from utils.accelerate_config import update_accelerate_config
from utils.bgym import setup_bgym_env_variables
from utils.path import (
    get_model_path_by_role_and_iteration,
    get_current_policy_path_by_role,
    get_model_path_by_iteration,
)
from utils.subprocess import run_commands
from utils.utils import assert_equal, print_separator
from utils.vllm import (
    start_vllm,
    sleep_vllm,
    unload_lora_modules,
)
from utils.wandb import retrieve_existing_run_id


def collect_episodes_with_population(
    config,
    current_model_paths: dict,
    on_policy_role: str,
    cur_iter: int,
    data_split: str,
    rollout_data_path: Path,
    collect_episode_args,
    multimodal_args,
    is_multimodal_collection: bool,
):
    """
    Collect episodes with population-based sampling for multimodal models.
    
    For multimodal models, vLLM can only load one full model at a time (no LoRA).
    This function implements sequential collection:
    1. Pre-allocate episodes to different opponent models (uniform distribution)
    2. For each opponent model:
       - Start vLLM server with current on-policy model + that opponent
       - Collect all episodes allocated to that opponent
       - Unload the server
    3. Combine all collected episodes into the final output file
    
    Asymmetric strategy:
    - When training agent: agent interacts with ALL historical attackers uniformly
    - When training attacker: attacker only interacts with the LATEST agent
    """
    from copy import deepcopy
    import random
    import shutil
    
    # Get population model paths
    agent_model_paths, attacker_model_paths = get_population_model_paths_for_multimodal(
        config, current_model_paths, on_policy_role, cur_iter
    )
    
    # Determine which side has population
    if on_policy_role == "agent":
        # Agent is on-policy, attacker uses population
        on_policy_model_path = agent_model_paths[0]  # Should be single element
        opponent_model_paths = attacker_model_paths
        on_policy_role_name = "agent"
        opponent_role_name = "attacker"
    elif on_policy_role == "attacker":
        # Attacker is on-policy, agent is latest only (no population for attacker training)
        on_policy_model_path = attacker_model_paths[0]
        opponent_model_paths = agent_model_paths  # Should be single element
        on_policy_role_name = "attacker"
        opponent_role_name = "agent"
    else:
        # on_policy_role == "both" - no population
        opponent_model_paths = [None]  # Placeholder
        on_policy_model_path = None
        on_policy_role_name = "both"
        opponent_role_name = "both"
    
    num_opponents = len(opponent_model_paths)
    print_separator(f"🎯 Population-based collection: {on_policy_role_name} vs {num_opponents} {opponent_role_name}(s)")
    
    # Check if all episodes already collected
    if rollout_data_path.is_file():
        with open(rollout_data_path, 'r') as f:
            existing_data = json.load(f)
        
        # Get expected number of episodes
        args = multimodal_args if is_multimodal_collection else collect_episode_args
        expected_kwargs = prepare_collect_kwargs(
            benchmark=args.benchmark,
            env_ids=args.env_ids,
            use_env_ids=args.use_env_ids,
            mode=args.mode,
            train_prop=args.train_prop,
            valid_prop=args.valid_prop,
            num_responses_per_env=args.num_responses_per_env,
            num_envs_per_iter=args.num_envs_per_iter,
            iteration=cur_iter,
            on_policy_role=on_policy_role,
            use_attacker=args.use_attacker,
            fix_seed=args.fix_seed,
            seed_offset=args.seed,
        )
        
        if len(existing_data) == len(expected_kwargs):
            print(f"✅ All episodes already collected at {rollout_data_path}. Skipping.")
            return
        else:
            print(f"⚠️ Partial data exists ({len(existing_data)}/{len(expected_kwargs)}). Re-collecting...")
    
    # For "both" mode or single opponent, use standard collection
    if num_opponents <= 1 or on_policy_role == "both":
        print("📦 Single opponent mode - using standard collection")
        
        # Start vLLM with standard approach
        if on_policy_role == "both":
            agent_path = agent_model_paths[0] if agent_model_paths else config.agent.model_path
            attacker_path = attacker_model_paths[0] if attacker_model_paths else config.attacker.model_path
        elif on_policy_role == "agent":
            agent_path = on_policy_model_path
            attacker_path = opponent_model_paths[0] if opponent_model_paths else config.attacker.model_path
        else:
            agent_path = opponent_model_paths[0] if opponent_model_paths else config.agent.model_path
            attacker_path = on_policy_model_path
        
        vllm_model_arguments, vllm_server_args_list = start_vllm_for_single_pair(
            config, agent_path, attacker_path, on_policy_role
        )
        
        # Update collect args with vLLM info
        args = multimodal_args if is_multimodal_collection else collect_episode_args
        args = deepcopy(args)
        args.agent_args_list.model_names = vllm_model_arguments["agent"]["model_names"]
        args.agent_args_list.base_urls = vllm_model_arguments["agent"]["urls"]
        if config.use_attacker and "attacker" in vllm_model_arguments:
            args.attacker_args_list.model_names = vllm_model_arguments["attacker"]["model_names"]
            args.attacker_args_list.base_urls = vllm_model_arguments["attacker"]["urls"]
        
        # IMPORTANT: Regenerate agent_args and attacker_args from updated lists
        args.agent_args = args.create_agent_args(args.agent_args_list)
        if config.use_attacker:
            args.attacker_args = args.create_agent_args(args.attacker_args_list)
        
        # Run collection
        collect_episodes(args, iteration=cur_iter)
        
        # Sleep vLLM servers
        for vllm_server_args in vllm_server_args_list:
            sleep_vllm(vllm_server_args["url"], vllm_server_args["api_key"])
        
        return
    
    # Multiple opponents - need sequential collection
    print(f"📊 Sequential population collection with {num_opponents} opponents")
    for i, opponent_path in enumerate(opponent_model_paths):
        print(f"   [{i+1}/{num_opponents}] {opponent_path}")
    
    # Prepare collection kwargs and allocate to opponents uniformly
    args = multimodal_args if is_multimodal_collection else collect_episode_args
    all_collect_kwargs = prepare_collect_kwargs(
        benchmark=args.benchmark,
        env_ids=args.env_ids,
        use_env_ids=args.use_env_ids,
        mode=args.mode,
        train_prop=args.train_prop,
        valid_prop=args.valid_prop,
        num_responses_per_env=args.num_responses_per_env,
        num_envs_per_iter=args.num_envs_per_iter,
        iteration=cur_iter,
        on_policy_role=on_policy_role,
        use_attacker=args.use_attacker,
        fix_seed=args.fix_seed,
        seed_offset=args.seed,
    )
    
    total_episodes = len(all_collect_kwargs)
    
    # ========================================================================
    # GRPO-aware allocation: ensure episodes in the same GRPO group are 
    # assigned to the same attacker. This prevents different attackers from
    # affecting variance within the same group.
    # ========================================================================
    from collections import defaultdict
    
    def get_grpo_group_key_from_kwargs(kwargs):
        """Get GRPO group key from collect kwargs (matches get_grpo_group_key in assign_credit.py)"""
        return f"{kwargs['env_id']}_{kwargs['use_attacker']}_{kwargs['seed']}"
    
    # Group all episodes by their GRPO group key
    grpo_groups = defaultdict(list)
    for idx, kwargs in enumerate(all_collect_kwargs):
        group_key = get_grpo_group_key_from_kwargs(kwargs)
        grpo_groups[group_key].append(kwargs)
    
    # Assign GRPO groups to opponents using RECENCY-WEIGHTED sampling
    # More recent opponents (later in the list) get higher probability of being selected
    # This ensures the agent primarily learns to counter newer, stronger attacker strategies
    # while still maintaining some diversity from historical attackers
    random.seed(cur_iter)  # Deterministic for reproducibility
    np.random.seed(cur_iter)
    
    # Configuration parameters
    min_latest_ratio = getattr(config, 'min_latest_opponent_ratio', 0.5)
    # Recency weight decay: higher = more weight to recent opponents
    # Default 0.7 means each older opponent has exp(-0.7) ≈ 0.5x weight of the next newer one
    recency_weight_decay = getattr(config, 'recency_weight_decay', 0.7)
    
    opponent_allocations = {opponent_path: [] for opponent_path in opponent_model_paths}
    group_assignments = {}  # For logging: group_key -> opponent_path
    
    all_group_keys = list(grpo_groups.keys())
    random.shuffle(all_group_keys)  # Shuffle for random selection
    
    # The latest opponent is at the end of the list (current_attacker_policy_path)
    latest_opponent = opponent_model_paths[-1]
    
    # Calculate how many groups to guarantee for the latest opponent
    num_groups_for_latest = max(1, int(len(all_group_keys) * min_latest_ratio))
    
    # First, assign guaranteed groups to the latest opponent
    guaranteed_groups = all_group_keys[:num_groups_for_latest]
    remaining_groups = all_group_keys[num_groups_for_latest:]
    
    for group_key in guaranteed_groups:
        opponent_allocations[latest_opponent].extend(grpo_groups[group_key])
        group_assignments[group_key] = latest_opponent
    
    # Then, distribute remaining groups using RECENCY-WEIGHTED sampling
    # Compute exponential weights: weight[i] = exp(recency_weight_decay * i)
    # where i=0 is oldest, i=num_opponents-1 is newest (latest)
    num_opponents = len(opponent_model_paths)
    recency_weights = np.array([np.exp(recency_weight_decay * i) for i in range(num_opponents)])
    recency_weights = recency_weights / recency_weights.sum()  # Normalize to probabilities
    
    print(f"   🎯 Recency-weighted sampling (decay={recency_weight_decay}):")
    for i, (opponent_path, weight) in enumerate(zip(opponent_model_paths, recency_weights)):
        is_latest = " (LATEST)" if opponent_path == latest_opponent else ""
        print(f"      Opponent {i+1}{is_latest}: weight={weight:.3f}")
    
    for group_key in remaining_groups:
        selected_idx = np.random.choice(num_opponents, p=recency_weights)
        selected_opponent = opponent_model_paths[selected_idx]
        opponent_allocations[selected_opponent].extend(grpo_groups[group_key])
        group_assignments[group_key] = selected_opponent
    
    # Print allocation summary
    print(f"   📊 Latest opponent guaranteed {num_groups_for_latest} groups ({min_latest_ratio*100:.0f}% of {len(all_group_keys)})")
    for i, opponent_path in enumerate(opponent_model_paths):
        num_groups = sum(1 for g in group_assignments.values() if g == opponent_path)
        num_episodes = len(opponent_allocations[opponent_path])
        is_latest = " (LATEST)" if opponent_path == latest_opponent else ""
        print(f"   Allocated {num_episodes} episodes ({num_groups} GRPO groups) to opponent {i+1}{is_latest}")
    
    # Create temp directory for partial results
    temp_dir = rollout_data_path.parent / f".temp_population_{on_policy_role}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    all_episodes = []
    
    # Collect episodes for each opponent sequentially
    for opponent_idx, (opponent_path, allocated_kwargs) in enumerate(opponent_allocations.items()):
        print_separator(f"[Opponent {opponent_idx + 1}/{num_opponents}] Collecting {len(allocated_kwargs)} episodes")
        print(f"   On-policy ({on_policy_role_name}): {on_policy_model_path}")
        print(f"   Opponent ({opponent_role_name}): {opponent_path}")
        
        # Check for partial results from previous run
        temp_file = temp_dir / f"opponent_{opponent_idx}.json"
        if temp_file.is_file():
            with open(temp_file, 'r') as f:
                partial_episodes = json.load(f)
            if len(partial_episodes) == len(allocated_kwargs):
                print(f"   ✅ Using cached results from {temp_file}")
                all_episodes.extend(partial_episodes)
                continue
        
        # Determine agent and attacker paths based on on_policy_role
        if on_policy_role == "agent":
            agent_path = on_policy_model_path
            attacker_path = opponent_path
        else:
            agent_path = opponent_path
            attacker_path = on_policy_model_path
        
        # Start vLLM server for this opponent
        vllm_model_arguments, vllm_server_args_list = start_vllm_for_single_pair(
            config, agent_path, attacker_path, on_policy_role
        )
        
        # Create modified args for this opponent
        opponent_args = deepcopy(args)
        opponent_args.output_path = str(temp_file)
        opponent_args.agent_args_list.model_names = vllm_model_arguments["agent"]["model_names"]
        opponent_args.agent_args_list.base_urls = vllm_model_arguments["agent"]["urls"]
        if config.use_attacker and "attacker" in vllm_model_arguments:
            opponent_args.attacker_args_list.model_names = vllm_model_arguments["attacker"]["model_names"]
            opponent_args.attacker_args_list.base_urls = vllm_model_arguments["attacker"]["urls"]
        
        # IMPORTANT: Regenerate agent_args and attacker_args from updated lists
        # (the original args have "placeholder" values from __post_init__)
        opponent_args.agent_args = opponent_args.create_agent_args(opponent_args.agent_args_list)
        if config.use_attacker:
            opponent_args.attacker_args = opponent_args.create_agent_args(opponent_args.attacker_args_list)
        
        # Collect episodes for this opponent using pebble ProcessPool
        # (same as non-population training - can actually kill stuck processes)
        from tools.multimodal_collect_episode import collect_episode, multimodal_collect_episode
        from pebble import ProcessPool
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        import signal
        
        def _worker_initializer():
            """
            Initialize worker process to use a new process group.
            This ensures that when pebble kills the worker, we can also kill
            any child processes (like Chrome/Playwright) by killing the process group.
            """
            os.setpgrp()  # Create new process group
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        collection_func = multimodal_collect_episode if is_multimodal_collection else collect_episode
        
        # Get timeout from config or use default 5 minutes
        job_timeout = getattr(config.data_collection, 'job_timeout', 300)
        if job_timeout <= 0:
            job_timeout = None
        
        # Adjust workers based on number of vLLM servers
        # When 2 servers (different models): each server has half the GPUs, so reduce workers
        num_vllm_servers = len(vllm_server_args_list)
        base_workers = config.data_collection.num_jobs
        
        if num_vllm_servers > 1:
            # Reduce workers when using multiple servers (each has fewer GPUs)
            # Each server has ~half throughput, so we reduce to prevent overload
            adjusted_workers = max(1, base_workers // num_vllm_servers)
            print(f"   ⚠️ {num_vllm_servers} vLLM servers detected (different models) - reducing workers: {base_workers} → {adjusted_workers}")
        else:
            adjusted_workers = base_workers
            print(f"   ✅ 1 vLLM server (same model for agent/attacker) - using full workers: {adjusted_workers}")
        
        n_workers = min(adjusted_workers, len(allocated_kwargs))
        batch_size = len(allocated_kwargs)
        
        print(f"   📦 Using pebble ProcessPool with {n_workers} workers, timeout={job_timeout}s per job")
        
        opponent_episodes = []
        
        with ProcessPool(max_workers=n_workers, initializer=_worker_initializer) as pool:
            # Submit all tasks with individual timeouts
            futures = []
            for kwargs in allocated_kwargs:
                future = pool.schedule(
                    collection_func,
                    args=(opponent_args,),
                    kwargs=kwargs,
                    timeout=job_timeout,  # pebble will TERMINATE the process after this timeout
                )
                futures.append((future, kwargs))
            
            # Process results
            for idx, (future, kwargs_dict) in enumerate(futures):
                env_id = kwargs_dict.get("env_id", "unknown")
                completed = idx + 1
                
                try:
                    result = future.result()  # This blocks until result or timeout
                    opponent_episodes.append(result)
                    if completed % 50 == 0 or completed == batch_size:
                        print(f"   ✅ Progress: {completed}/{batch_size} episodes completed")
                    
                except FuturesTimeoutError:
                    print(f"   ⏰ Episode {completed}/{batch_size}: {env_id} TIMEOUT after {job_timeout}s - PROCESS KILLED")
                    timeout_episode = {
                        "task_name": env_id,
                        "error": f"Job timeout after {job_timeout} seconds (process terminated)",
                        "terminated": False,
                        "truncated": True,
                        "history": [],
                        "use_attacker": kwargs_dict.get("use_attacker", False),
                        "seed": kwargs_dict.get("seed"),
                    }
                    opponent_episodes.append(timeout_episode)
                    
                except Exception as e:
                    print(f"   ❌ Episode {completed}/{batch_size}: {env_id} failed - {e}")
                    error_episode = {
                        "task_name": env_id,
                        "error": str(e),
                        "terminated": False,
                        "truncated": False,
                        "history": [],
                        "use_attacker": kwargs_dict.get("use_attacker", False),
                        "seed": kwargs_dict.get("seed"),
                    }
                    opponent_episodes.append(error_episode)
        
        # Save temp results
        with open(temp_file, 'w') as f:
            json.dump(opponent_episodes, f, ensure_ascii=False, indent=2)
        
        all_episodes.extend(opponent_episodes)
        print(f"   ✅ Collected {len(opponent_episodes)} episodes for opponent {opponent_idx + 1}")
        
        # Sleep vLLM servers
        for vllm_server_args in vllm_server_args_list:
            sleep_vllm(vllm_server_args["url"], vllm_server_args["api_key"])
    
    # Combine all episodes and save to final output
    print_separator(f"💾 Saving {len(all_episodes)} total episodes to {rollout_data_path}")
    rollout_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rollout_data_path, 'w') as f:
        json.dump(all_episodes, f, ensure_ascii=False, indent=2)
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
        print(f"   🧹 Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        print(f"   ⚠️ Failed to clean up temp directory: {e}")
    
    print(f"✅ Population-based collection complete: {len(all_episodes)} episodes")


def main(config: Optional[TrainAdversarialConfig] = None):
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if config is None:
        config = TrainAdversarialConfig()
    setup_bgym_env_variables()

    # ----------------------------- Initialize WandB -----------------------------
    wandb_disabled = "test" in config.exp_name
    wandb.init(
        project=config.wandb_project,
        group=config.group_name,
        name=config.exp_name,
        id=retrieve_existing_run_id(config.wandb_project, config.group_name, config.exp_name),
        config=asdict(config),
        mode="disabled" if wandb_disabled else "online",
    )
    wandb_env_vars = {
        "WANDB_PROJECT": config.wandb_project,
        "WANDB_GROUP": config.group_name,
        "WANDB_NAME": config.exp_name,
        "WANDB_RESUME": "must",
        "WANDB_RUN_ID": wandb.run.id,
        "WANDB_DISABLED": wandb_disabled,
    }

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------- Check if resuming training ------------------------------- #
    # ------------------------------------------------------------------------------------------ #
    exp_ckpt_path = Path(config.exp_cloud_ckpt_path).resolve() # NOTE: config file saved in bucket ckpt folder, not local ckpt folder
    local_ckpt_path = Path(config.exp_local_ckpt_path).resolve() # NOTE: config file saved in local ckpt folder
    config_file = exp_ckpt_path / "config.json"
    local_config_file = local_ckpt_path / "config.json"

    if exp_ckpt_path.exists() and exp_ckpt_path.is_dir() and config_file.exists():
        print(f"Checkpoint directory {exp_ckpt_path} already exists.")

        with open(config_file, 'r') as f:
            existing_config = json.load(f)

        assert_equal(existing_config, asdict(config), ignore_keys=["wandb_project", "group_name", "data_collection.job_timeout"])

        if local_ckpt_path.exists() and local_ckpt_path.is_dir() and local_config_file.exists():
            print(f"Local checkpoint directory {local_ckpt_path} already exists.")

            with open(local_config_file, 'r') as f:
                existing_local_config = json.load(f)

            assert_equal(existing_local_config, asdict(config), ignore_keys=["wandb_project", "group_name", "data_collection.job_timeout"])

        else:
            print(f"Creating local checkpoint directory {local_ckpt_path}.")

            local_ckpt_path.mkdir(parents=True, exist_ok=True)
            with open(local_config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)

    else:
        print(f"Creating checkpoint directory {exp_ckpt_path}.")

        exp_ckpt_path.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        print(f"Creating local checkpoint directory {local_ckpt_path}.")

        local_ckpt_path.mkdir(parents=True, exist_ok=True)
        with open(local_config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)

    rl_train_method = config.rl.train_method
    if rl_train_method == "ppo":
        current_model_paths = {
            "agent_policy": config.agent.model_path,
            "agent_value": config.agent.model_path,
            "attacker_policy": config.attacker.model_path,
            "attacker_value": config.attacker.model_path,
        }
    elif rl_train_method in ["spag", "grpo", "online_sft"]:
        current_model_paths = {
            "agent": config.agent.model_path,
            "attacker": config.attacker.model_path,
        }
    else:
        raise ValueError(f"Unknown training method: {rl_train_method}")

    # ------------------------------- helper variables and functions -------------------------------
    roles = config.roles
    agent_configs = config.agent_configs

    # ------------------------------------------------------------------------------------------ #
    # ----------------------------- Warmup with imitation training ----------------------------- #
    # ------------------------------------------------------------------------------------------ #
    if config.agent.train_imitation or config.attacker.train_imitation:
        print_separator("Warmup with imitation training with the following models:", current_model_paths)

        for role in roles:
            agent_config = agent_configs[role]

            if not agent_config.train_imitation:
                continue

            current_model_paths.update(get_model_path_by_role_and_iteration(config, role, "imitation")) # This is a LOCAL PATH
            imitation_ckpt_path = Path(get_current_policy_path_by_role(config, current_model_paths, role)) # This is a LOCAL PATH
            print(f"Imitation checkpoint path: {imitation_ckpt_path}")
            print(f"Current model paths: {current_model_paths}")

            # if imitation checkpoint path already exists, skip imitation training
            if (
                imitation_ckpt_path.exists() and 
                imitation_ckpt_path.is_dir() and
                any(imitation_ckpt_path.iterdir())
            ):
                print(f"Imitation checkpoint path already exists at {imitation_ckpt_path}, skipping imitation training.")
                continue

            imitation_data_base_path = Path(config.imitation.data_base_path) # This is a LOCAL PATH
            
            # ============================================================================
            # Enhanced SFT Support: Check if data is already preprocessed
            # ============================================================================
            if config.imitation.is_preprocessed:
                # Data is already in transitions format (NOTE: from enhanced SFT pipeline)
                # Use role-specific paths if provided, otherwise use raw_data_path
                if role == "agent" and config.imitation.agent_data_path:
                    imitation_data_path = Path(config.imitation.agent_data_path)
                    print(f"[Enhanced SFT] Using preprocessed agent data: {imitation_data_path}")
                elif role == "attacker" and config.imitation.attacker_data_path:
                    imitation_data_path = Path(config.imitation.attacker_data_path)
                    print(f"[Enhanced SFT] Using preprocessed attacker data: {imitation_data_path}")
                else:
                    raise ValueError(f"No preprocessed data path provided for role: {role}")
                
                if not imitation_data_path.exists():
                    raise FileNotFoundError(
                        f"Preprocessed imitation data not found at {imitation_data_path}. "
                        f"Please run the enhanced SFT pipeline first or set is_preprocessed=False."
                    )
                print(f"[Enhanced SFT] Skipping postprocess_rollout (data already in transitions format)")
            else:
                # Convert episode data to training transitions with postprocess_rollout.
                imitation_data_path = imitation_data_base_path / f"{'both' if config.self_play else role}_data.json" # This is a LOCAL PATH
                print(f"Imitation data output path: {imitation_data_path}")
                # --- step 1: postprocess rollout ---
                print(f"[Warmup] Postprocessing rollout for imitation data for {role}...")

                postprocess_rollout_args = PostprocessRolloutArg(
                    mode="imitation",
                    roles=roles if config.self_play else [role],
                    use_attacker=config.use_attacker,
                    input_data_path=config.imitation.raw_data_path, # This is the raw imitation data path in the bucket
                    output_data_path=str(imitation_data_path),
                )
                postprocess_rollout(postprocess_rollout_args, iteration=0)

                print("[Warmup] Imitation rollout postprocessing finished.")

            # --- step 2: imitation training ---
            print("[Warmup] Starting imitation training...")

            common_training_commands = config.get_common_training_commands(role, is_imitation=True, wandb_disabled=wandb_disabled)
            specific_training_commands = [
                "--model_name_or_path", agent_config.model_path,
                "--train_data_path", str(imitation_data_path),
                "--output_dir", str(imitation_ckpt_path),
                "--local_model_save_path", str(imitation_ckpt_path),
                "--cloud_model_save_path", str(imitation_ckpt_path), # NOTE: imitation training only saves the local checkpoints
            ]
            imitation_commands = \
                common_training_commands + \
                specific_training_commands + \
                config.imitation.to_str_list()

            run_commands(imitation_commands, wandb_env_vars, f"{role} Imitation Training")
            print("[Warmup] Imitation training finished.")
    else:
        print("[Warmup] Skipping imitation training as it is not enabled.")

    # ------------------------------------------------------------------------------------------ #
    # ----------------------------- Enhanced SFT Training (Stage 2) ---------------------------- #
    # ------------------------------------------------------------------------------------------ #
    if config.enhanced_sft.enabled and (config.agent.train_imitation or config.attacker.train_imitation):
        print_separator("Enhanced SFT Training (Stage 2) - Fine-tuning with defense reasoning")
        
        for role in roles:
            agent_config = agent_configs[role]
            
            if not agent_config.train_imitation:
                continue
            
            # Update current_model_paths with enhanced SFT checkpoint path (same pattern as imitation)
            current_model_paths.update(get_model_path_by_role_and_iteration(config, role, "enhanced_sft"))
            enhanced_sft_ckpt_path = Path(get_current_policy_path_by_role(config, current_model_paths, role))
            print(f"[Enhanced SFT] Enhanced SFT checkpoint path: {enhanced_sft_ckpt_path}")
            print(f"[Enhanced SFT] Current model paths: {current_model_paths}")
            
            # Check if enhanced SFT checkpoint already exists
            if (
                enhanced_sft_ckpt_path.exists() and 
                enhanced_sft_ckpt_path.is_dir() and
                any(enhanced_sft_ckpt_path.iterdir())
            ):
                print(f"[Enhanced SFT] Checkpoint already exists at {enhanced_sft_ckpt_path}, skipping training.")
                continue
            
            # Get the starting model path (from imitation stage, using merged model)
            # Enhanced SFT uses SFTTrainer which requires a full merged model, not a LoRA adapter
            imitation_model_paths = get_model_path_by_role_and_iteration(config, role, "imitation", merged=True)
            starting_model_path = imitation_model_paths[role] if rl_train_method in ["spag", "grpo", "online_sft"] else imitation_model_paths[f"{role}_policy"]
            
            # Check if merged model exists, auto-merge if only LoRA adapter is found
            starting_model_path_obj = Path(starting_model_path)
            if not (starting_model_path_obj.exists() and (starting_model_path_obj / "config.json").exists()):
                # Try to find the non-merged LoRA adapter
                non_merged_path = starting_model_path.replace("_merged", "")
                if Path(non_merged_path).exists() and (Path(non_merged_path) / "adapter_config.json").exists():
                    # Auto-merge the LoRA adapter (same as RL process)
                    print(f"[Enhanced SFT] Found LoRA adapter at {non_merged_path}, auto-merging...")
                    from utils.vllm import merge_lora_weights_for_vllm
                    merged_path = merge_lora_weights_for_vllm(non_merged_path)
                    print(f"[Enhanced SFT] Merged model saved to: {merged_path}")
                    # Update starting_model_path to use the merged path
                    starting_model_path = str(merged_path)
                else:
                    raise FileNotFoundError(
                        f"[Enhanced SFT] Merged model not found at {starting_model_path}. "
                        f"Please complete imitation training first."
                    )
            
            print(f"[Enhanced SFT] Role: {role}")
            print(f"[Enhanced SFT] Starting model (from imitation): {starting_model_path}")
            print(f"[Enhanced SFT] Output checkpoint: {enhanced_sft_ckpt_path}")
            
            # Get data path for this role
            if role == "agent" and config.enhanced_sft.agent_data_path:
                enhanced_sft_data_path = Path(config.enhanced_sft.agent_data_path)
            elif role == "attacker" and config.enhanced_sft.attacker_data_path:
                enhanced_sft_data_path = Path(config.enhanced_sft.attacker_data_path)
            else:
                raise ValueError(f"[Enhanced SFT] No data path provided for role: {role}")
            
            if not enhanced_sft_data_path.exists():
                raise FileNotFoundError(
                    f"[Enhanced SFT] Data not found at {enhanced_sft_data_path}. "
                    f"Please prepare enhanced SFT data first."
                )
            
            print(f"[Enhanced SFT] Training data: {enhanced_sft_data_path}")
            
            # Build training commands
            common_training_commands = config.get_common_training_commands(role, is_imitation=True, wandb_disabled=wandb_disabled)
            specific_training_commands = [
                "--model_name_or_path", str(starting_model_path),
                "--train_data_path", str(enhanced_sft_data_path),
                "--output_dir", str(enhanced_sft_ckpt_path),
                "--local_model_save_path", str(enhanced_sft_ckpt_path),
                "--cloud_model_save_path", str(enhanced_sft_ckpt_path),
            ]
            
            enhanced_sft_commands = \
                common_training_commands + \
                specific_training_commands + \
                config.enhanced_sft.to_str_list()
            
            run_commands(enhanced_sft_commands, wandb_env_vars, f"{role} Enhanced SFT Training")
            print(f"[Enhanced SFT] {role} training finished.")
            
            # Merge LoRA weights immediately after training (vLLM doesn't support LoRA on vision towers)
            from utils.vllm import merge_lora_weights_for_vllm
            print(f"[Enhanced SFT] Merging LoRA weights for {role}...")
            merged_path = merge_lora_weights_for_vllm(enhanced_sft_ckpt_path)
            print(f"[Enhanced SFT] {role} merged model saved to: {merged_path}")
        
        print("[Enhanced SFT] Stage 2 complete. Current model paths:", current_model_paths)
    else:
        print("[Enhanced SFT] Skipping Enhanced SFT training (not enabled or imitation not trained).")

    # ------------------------------------------------------------------------------------------ #
    # -------------------------------- Set up resuming training -------------------------------- #
    # ------------------------------------------------------------------------------------------ #
    start_iteration = 1
    
    rl_progress_file = Path(config.rl.progress_file_path)

    if rl_progress_file.is_file():
        with open(rl_progress_file, 'r') as f:
            last_completed_iteration = int(f.read().strip())

        start_iteration = last_completed_iteration + 1

        for role in roles:
            agent_config = agent_configs[role]
            if agent_config.train_rl:
                # here we assume the training is resumed on the same local machine, so use local model paths
                current_model_paths.update(
                    get_model_path_by_role_and_iteration(config, role, last_completed_iteration, cloud=False)
                )

        for model_path in current_model_paths.values():
            model_path = Path(model_path)
            if not (model_path.exists() and model_path.is_dir()):
                raise ValueError(f"Progress file found, but RL checkpoint directory {model_path} does not exist.")

        print_separator(f"Resuming from iteration {start_iteration} with models from:", current_model_paths)


    # ------------------------------------------------------------------------------------------ #
    # --------------------------------- Initialize Ray Server ---------------------------------- #
    # ------------------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------------------ #
    # -------------------------------------- Main RL Loop -------------------------------------- #
    # ------------------------------------------------------------------------------------------ #
    for cur_iter in range(start_iteration, config.rl.num_iterations + 1):
        print_separator(f"Starting RL Iteration {cur_iter} / {config.rl.num_iterations}", current_model_paths)

        # --- initialize paths for current iteration ---
        iter_data_path = Path(config.exp_data_path) / f"iter_{cur_iter}"

        # some keys may not be used depending on the current iteration and attacker training flags
        data_paths = defaultdict(dict)
        for role in roles:
            if config.self_play and not config.agent.train_use_population and not config.attacker.train_use_population:
                role_desc = "both"
            else:
                role_desc = role

            for data_split in ["train", "valid"]:
                if rl_train_method in ["spag", "grpo"]:
                    data_paths[f"{role}_reward"][data_split] = iter_data_path / data_split / f"{role_desc}_reward_data.json"
                    data_paths[f"{role}_policy"][data_split] = iter_data_path / f"{role_desc}_{rl_train_method}_data.json"
                elif rl_train_method == "online_sft":
                    # For online SFT: reward_data is the postprocessed data (filtered to successful episodes)
                    # This filtered data is directly used for SFT training (no credit assignment needed)
                    data_paths[f"{role}_reward"][data_split] = iter_data_path / data_split / f"{role_desc}_reward_data.json"
                elif rl_train_method == "ppo":
                    data_paths[f"{role}_reward"][data_split] = iter_data_path / data_split / f"{role_desc}_reward_data.json"
                    data_paths[f"{role}_policy"][data_split] = iter_data_path / f"{role_desc}_ppo_policy_data.json"
                    data_paths[f"{role}_value"][data_split] = iter_data_path / f"{role_desc}_ppo_value_data.json"
                else:
                    raise ValueError(f"Unknown training method: {rl_train_method}")

        # ----------------------------- Start vLLM Server and Collect Rollouts ----------------------------- 
        print_separator(f"[Iteration {cur_iter}] Data Collection")

        # decide number of data collection turns
        # Determine which roles need on-policy data collection
        agent_training = config.agent.train_rl
        attacker_training = config.attacker.train_rl
        
        if agent_training and attacker_training:
            # Both are training
            if (config.agent.train_use_population or config.attacker.train_use_population) and cur_iter > 1:
                # Population training: collect in separate turns
                on_policy_roles = ["agent", "attacker"]
                num_turns = 2
            else:
                # Co-training: collect together
                on_policy_roles = ["both"]
                num_turns = 1
        elif agent_training and not attacker_training:
            # Only agent training (attacker fixed): collect agent-only data
            on_policy_roles = ["agent"]
            num_turns = 1
        elif attacker_training and not agent_training:
            # Only attacker training (agent fixed): collect attacker data with fixed agent
            on_policy_roles = ["attacker"]
            num_turns = 1
        else:
            raise ValueError("At least one of agent or attacker must be training")

        # decide split to collect data on
        data_splits = ["train", "valid"] if config.data_collection.use_valid else ["train"]

        # prepare rollout data paths
        rollout_data_paths = defaultdict(dict)
        for role in roles:
            for data_split in data_splits:
                if num_turns == 1:
                    # When only one turn, use the on_policy_role name (could be "both", "agent", or "attacker")
                    rollout_data_paths[role][data_split] = iter_data_path / data_split / f"data_{on_policy_roles[0]}.json"
                elif num_turns == 2:
                    rollout_data_paths[role][data_split] = iter_data_path / data_split / f"data_{role}.json"
                else:
                    raise ValueError(f"Unknown number of turns: {num_turns}")

        collect_episode_start_time = time.time()
        
        # Check if we need population-based collection for multimodal models
        is_multimodal = is_multimodal_model(config.agent.model_path)
        use_population_collection = (
            is_multimodal and 
            config.agent.lora.use_lora and
            cur_iter > 1 and
            (config.agent.train_use_population or config.attacker.train_use_population)
        )
        
        if use_population_collection:
            print_separator(f"🎯 [Iteration {cur_iter}] Using POPULATION-BASED collection for multimodal model")
        
        for turn_idx, on_policy_role in enumerate(on_policy_roles):
            for data_split in data_splits:

                # ------ step 1: start vLLM server(s) or use population-based collection ------ 
                print_separator(f"[Iteration {cur_iter} turn {turn_idx + 1}] Starting vLLM server for {on_policy_role} - {data_split}:", current_model_paths)

                # For population training with multimodal models, we'll handle vLLM inside collect_episodes_with_population
                if not use_population_collection:
                    vllm_model_arguments, vllm_server_args_list = start_vllm(
                        config,
                        current_model_paths,
                        on_policy_role,
                        cur_iter,
                    )
                else:
                    # Population-based collection will handle vLLM internally
                    vllm_model_arguments = None
                    vllm_server_args_list = []

                # ------ step 2: check if data already exists ------ 
                # prepare data collection args, so we can check if the existing data matches the args
                if data_split == "train":
                    mode = "train"
                    env_ids = config.data_collection.train_env_ids
                    num_envs_per_iter = config.data_collection.num_train_envs_per_iter
                    num_responses_per_env = config.data_collection.num_train_responses_per_env
                elif data_split == "valid":
                    mode = "valid"
                    env_ids = config.data_collection.valid_env_ids
                    num_envs_per_iter = 0     # use all env_ids for valid data collection
                    num_responses_per_env = config.data_collection.num_valid_responses_per_env
                else:
                    raise ValueError(f"Unknown data split: {data_split}")
                fix_seed = (rl_train_method == "grpo" and data_split == "train")

                rollout_data_path = iter_data_path / data_split / f"data_{on_policy_role}.json"
                if rollout_data_path.is_file():
                    with open(rollout_data_path, 'r') as f:
                        rollout_data = json.load(f)

                    data_collection_kwargs = prepare_collect_kwargs(
                        benchmark=config.data_collection.benchmark,
                        env_ids=env_ids,
                        use_env_ids=config.data_collection.use_env_ids,
                        mode=mode,
                        train_prop=config.data_collection.train_prop,
                        valid_prop=config.data_collection.valid_prop,
                        num_responses_per_env=num_responses_per_env,
                        num_envs_per_iter=num_envs_per_iter,
                        iteration=cur_iter,
                        on_policy_role=on_policy_role,
                        use_attacker=config.use_attacker,
                        fix_seed=fix_seed,
                        seed_offset=config.seed,
                    )
                    if len(rollout_data) == len(data_collection_kwargs):
                        print(f"Rollout data already exists at {rollout_data_path}. Skipping collection.")
                        continue

                # ------ step 3: collect episodes ------ 
                print_separator(f"[Iteration {cur_iter}] Collecting rollout episodes for {on_policy_role} - {data_split}...")

                extra_token_length = 16
                max_input_tokens = config.agent.vllm.max_model_len - config.data_collection.max_new_tokens - extra_token_length

                # HTML injection is used instead of image editing
                # No GPU-based image editor server needed
                if config.use_attacker:
                    print_separator("🚀 HTML Injection Attack Mode")
                    logging.info(f"   Attacker will generate HTML/CSS injections")
                    logging.info(f"   Injections modify DOM directly via JavaScript")
                    logging.info(f"   Both screenshot and AXTree are automatically updated")
                else:
                    print_separator("⏭️  Running in agent-only mode (no attacker)")
                    logging.info(f"   use_attacker: {config.use_attacker}")

                
                # 🎯 Configure prompt flags for multimodal collection
                prompt_flags = GenericPromptFlags(
                    obs=dp.ObsFlags(
                        use_html=False,
                        use_ax_tree=True,
                        use_focused_element=True,
                        use_error_logs=False,
                        use_history=True,
                        use_past_error_logs=False,
                        use_action_history=True,
                        use_think_history=True,
                        use_diff=False,
                        html_type="pruned_html",
                        use_screenshot=config.data_collection.enable_agent_vision if config.data_collection.enable_multimodal else False,
                        use_som=False,  # Disable SOM for training data collection
                        extract_visible_tag=True,
                        extract_clickable_tag=False,
                        extract_coords="False",
                        filter_visible_elements_only=False,
                    ),
                    action=dp.ActionFlags(
                        action_set=bgym.HighLevelActionSetArgs(
                            subsets=("bid",),
                            multiaction=False,
                        ),
                        long_description=False,
                        individual_examples=True,
                        multi_actions=False,
                        is_strict=False,
                    ),
                    use_plan=False,
                    use_criticise=False,
                    use_thinking=True,
                    use_memory=False,
                    use_concrete_example=True,
                    use_abstract_example=True,
                    use_hints=True,
                    enable_chat=False,
                    be_cautious=True,
                    extra_instructions=None,
                    add_missparsed_messages=True,
                    max_trunc_itr=50,
                )
                
                # For population-based collection, use placeholder URLs that will be filled later
                if use_population_collection:
                    agent_model_names = ["placeholder"]
                    agent_urls = "http://placeholder"
                    attacker_model_names = ["placeholder"]
                    attacker_urls = "http://placeholder"
                else:
                    agent_model_names = vllm_model_arguments["agent"]["model_names"]
                    agent_urls = vllm_model_arguments["agent"]["urls"]
                    if config.use_attacker:
                        attacker_model_names = vllm_model_arguments["attacker"]["model_names"]
                        attacker_urls = vllm_model_arguments["attacker"]["urls"]
                    else:
                        attacker_model_names = ["placeholder"]
                        attacker_urls = "http://placeholder"
                
                collect_episode_args = CollectEpisodeArgs(
                    seed=config.seed,
                    # data collection params
                    mode=mode,
                    benchmark=config.data_collection.benchmark,
                    env_ids=env_ids,
                    use_env_ids=config.data_collection.use_env_ids,
                    train_prop=config.data_collection.train_prop,
                    valid_prop=config.data_collection.valid_prop,
                    eval_prop=config.data_collection.eval_prop,
                    num_envs_per_iter=num_envs_per_iter,
                    num_responses_per_env=num_responses_per_env,
                    num_jobs=config.data_collection.num_jobs,
                    # for each task, whether to fix the init state, required for GRPO training
                    fix_seed=fix_seed,
                    output_path=str(rollout_data_path),
                    # agent and attacker params
                    on_policy_role=on_policy_role,
                    use_attacker=config.use_attacker,
                    attacker_use_art=config.attacker_use_art,
                    collect_responses_with_same_opponent=config.data_collection.collect_responses_with_same_opponent,
                    # Skip model validation when using population-based collection (models set later)
                    skip_model_validation=use_population_collection,
                    default_model_args=VllmModelArgs(
                        model_name="placeholder",
                        temperature=config.data_collection.temperature,
                        top_p=config.data_collection.top_p,
                        top_k=config.data_collection.top_k,
                        min_p=config.data_collection.min_p,
                        repetition_penalty=config.data_collection.repetition_penalty,
                        max_new_tokens=config.data_collection.max_new_tokens,
                        # 0 means return the logprobs of sampled tokens
                        log_probs=0,
                        # Enable vision support for multimodal models
                        vision_support=config.data_collection.enable_multimodal and config.data_collection.enable_agent_vision,
                    ),
                    agent_args_list=VllmModelArgsList(
                        model_names=agent_model_names,
                        base_urls=agent_urls,
                        max_input_tokens=max_input_tokens,
                    ),
                    attacker_args_list=VllmModelArgsList(
                        model_names=attacker_model_names,
                        base_urls=attacker_urls,
                        max_input_tokens=max_input_tokens,
                    ) if config.use_attacker else None,
                    env_args=EnvArgs(
                        task_name="placeholder",
                        max_steps=config.data_collection.max_episode_steps,
                    ),
                    attacker_privilege=AttackerPrivilegeObsFlags(
                        attacker_uses_web_agent_history=config.data_collection.attacker_uses_web_agent_history,
                        attacker_uses_web_agent_action_history=config.data_collection.attacker_uses_web_agent_action_history,
                        attacker_uses_web_agent_think_history=config.data_collection.attacker_uses_web_agent_think_history,
                        attacker_uses_web_agent_memory=config.data_collection.attacker_uses_web_agent_memory,
                    ),
                    prompt_flags=prompt_flags,  # Add prompt flags for multimodal
                )

                # 🎨 Choose between multimodal and text-only collection based on config
                is_multimodal_collection = config.data_collection.enable_multimodal
                
                if is_multimodal_collection:
                    print("🎨 Using multimodal episode collection (vision + language)")
                    
                    # Convert CollectEpisodeArgs to MultimodalCollectEpisodeArgs (with HTML injection)
                    multimodal_args = MultimodalCollectEpisodeArgs(
                        # Copy all base args
                        seed=collect_episode_args.seed,
                        mode=collect_episode_args.mode,
                        benchmark=collect_episode_args.benchmark,
                        env_ids=collect_episode_args.env_ids,
                        use_env_ids=collect_episode_args.use_env_ids,
                        train_prop=collect_episode_args.train_prop,
                        valid_prop=collect_episode_args.valid_prop,
                        eval_prop=collect_episode_args.eval_prop,
                        num_envs_per_iter=collect_episode_args.num_envs_per_iter,
                        num_responses_per_env=collect_episode_args.num_responses_per_env,
                        num_jobs=collect_episode_args.num_jobs,
                        save_every_n_episodes=collect_episode_args.save_every_n_episodes,
                        env_step_timeout=collect_episode_args.env_step_timeout,
                        fix_seed=collect_episode_args.fix_seed,
                        collect_responses_with_same_opponent=collect_episode_args.collect_responses_with_same_opponent,
                        on_policy_role=collect_episode_args.on_policy_role,
                        use_attacker=collect_episode_args.use_attacker,
                        attacker_use_art=collect_episode_args.attacker_use_art,
                        skip_model_validation=collect_episode_args.skip_model_validation,
                        default_model_args=collect_episode_args.default_model_args,
                        agent_args_list=collect_episode_args.agent_args_list,
                        attacker_args_list=collect_episode_args.attacker_args_list,
                        env_args=collect_episode_args.env_args,
                        attacker_privilege=collect_episode_args.attacker_privilege,
                        prompt_flags=collect_episode_args.prompt_flags,
                        output_path=collect_episode_args.output_path,
                        
                        # Add multimodal-specific args from config (HTML injection)
                        enable_agent_vision=config.data_collection.enable_agent_vision,
                        execute_text_injection=config.data_collection.execute_text_injection,
                        execute_image_attacks=config.data_collection.execute_image_attacks,
                        max_image_size=config.data_collection.max_image_size,
                    )
                else:
                    multimodal_args = None
                
                # 🎯 Use population-based collection if enabled, otherwise standard collection
                if use_population_collection:
                    print("🎯 Using POPULATION-based episode collection")
                    collect_episodes_with_population(
                        config=config,
                        current_model_paths=current_model_paths,
                        on_policy_role=on_policy_role,
                        cur_iter=cur_iter,
                        data_split=data_split,
                        rollout_data_path=rollout_data_path,
                        collect_episode_args=collect_episode_args,
                        multimodal_args=multimodal_args,
                        is_multimodal_collection=is_multimodal_collection,
                    )
                elif is_multimodal_collection:
                    collect_episodes(multimodal_args, iteration=cur_iter)
                else:
                    # Fall back to text-only collection
                    if config.data_collection.enable_multimodal:
                        print("⚠️ Multimodal collection requested but not available, using text-only collection")
                    else:
                        print("📝 Using text-only episode collection")
                    
                    collect_episodes(collect_episode_args, iteration=cur_iter)


            # unload LoRA modules and sleep vLLM servers
            # Skip if using population-based collection (it handles vLLM servers internally)
            if not use_population_collection:
                for vllm_server_args in vllm_server_args_list:
                    url = vllm_server_args["url"]
                    api_key = vllm_server_args["api_key"]
                    if config.agent.lora.use_lora:
                        lora_module_names = vllm_server_args["lora_module_names"]
                        unload_lora_modules(url, api_key, lora_module_names)
                    sleep_vllm(url, api_key)

        collect_episode_end_time = time.time()
        episode_collection_time = collect_episode_end_time - collect_episode_start_time
        print(f"🎨 Multimodal episode collection finished in {episode_collection_time:.2f} seconds")

        print_separator(f"[Iteration {cur_iter}] Rollout collection finished successfully.")

        # ------ step 3: post-process rollout data ------ 
        # NOTE: this step generates transition-level data, where each transition is a single step in an episode, and each transition is a data point for training.
        print_separator(f"[Iteration {cur_iter}] Post-processing rollout data ...")
        postprocess_rollout_start_time = time.time()

        # For online_sft, use "imitation" mode to filter to successful episodes only
        postprocess_mode = "imitation" if rl_train_method == "online_sft" else "rl_on_policy"

        for role in roles:

            for data_split in data_splits:

                agent_config = agent_configs[role]
                rollout_data_path = rollout_data_paths[role][data_split]
                reward_data_path = data_paths[f"{role}_reward"][data_split]

                if role == "attacker" and not config.use_attacker:
                    print(f"Skipping reward assignment for attacker as it is not used.")
                    continue

                # Check if postprocessing is already done
                if reward_data_path.is_file():
                    print(f"Reward data already exists at {reward_data_path}. Skipping postprocessing for {role} {data_split}.")
                    continue

                # data preprocessing
                if config.self_play and not config.agent.train_use_population and not config.attacker.train_use_population: # NOTE: question: what is "self_play"? answer: self_play is the same model shared by both attacker and agent.
                    roles_to_postprocess = roles
                else:
                    roles_to_postprocess = [role]

                postprocess_rollout_args = PostprocessRolloutArg(
                    mode=postprocess_mode,
                    roles=roles_to_postprocess,
                    use_attacker=config.use_attacker,
                    input_data_path=str(rollout_data_path),
                    output_data_path=str(reward_data_path),
                )
                postprocess_rollout(
                    postprocess_rollout_args,
                    wandb_prefix=f"episode_stats/{data_split}",
                    iteration=cur_iter,
                )
        postprocess_rollout_end_time = time.time()
        postprocess_rollout_time = postprocess_rollout_end_time - postprocess_rollout_start_time
        print(f"🎨 Post-processing rollout data finished in {postprocess_rollout_time:.2f} seconds")

        # ==================================================================================
        # Online SFT: Skip credit assignment and use SFT training instead of RL
        # ==================================================================================
        if rl_train_method == "online_sft":
            # ------ step 4 (online_sft): SKIP credit assignment ------
            print_separator(f"[Iteration {cur_iter}] [Online SFT] Skipping credit assignment (not needed for SFT).")
            credit_assignment_time = 0.0

            # ------ step 5 (online_sft): run SFT training ------
            print_separator(f"[Iteration {cur_iter}] [Online SFT] Starting SFT training on successful episodes...")
            rl_training_start_time = time.time()

            role = "agent"  # Only train the agent in online_sft
            agent_config = agent_configs[role]

            # --- Prepare data path ---
            # Use the postprocessed reward data (filtered to successful episodes)
            sft_data_path = data_paths[f"{role}_reward"]["train"]

            if not sft_data_path.is_file():
                raise FileNotFoundError(
                    f"[Online SFT] SFT data not found at {sft_data_path}. "
                    f"Postprocessing may have failed or produced no successful episodes."
                )

            # Check if there are any training samples
            with open(sft_data_path, 'r') as f:
                sft_data = json.load(f)
            num_sft_samples = len(sft_data)
            print(f"[Online SFT] Found {num_sft_samples} training samples (from successful episodes).")

            if num_sft_samples == 0:
                print(f"[Online SFT] ⚠️ No successful episodes in iteration {cur_iter}. Skipping SFT training.")
            else:
                # --- Prepare model paths ---
                module_name = role
                current_model_path = current_model_paths[module_name]

                # For SFT training, we need a full model (not LoRA adapter).
                # If the current model is a LoRA adapter, merge it first.
                current_model_path_obj = Path(current_model_path)
                if (current_model_path_obj / "adapter_config.json").exists():
                    print(f"[Online SFT] Current model is a LoRA adapter, merging for SFT training...")
                    from utils.vllm import merge_lora_weights_for_vllm
                    current_model_path_for_sft = merge_lora_weights_for_vllm(current_model_path)
                else:
                    current_model_path_for_sft = current_model_path

                optimizer_save_path = agent_config.rl.module_to_optimizer_save_path[module_name]
                local_model_save_paths = get_model_path_by_iteration(config, cur_iter, cloud=False)
                loca_model_save_path = local_model_save_paths[module_name]
                cloud_model_save_paths = get_model_path_by_iteration(config, cur_iter, cloud=True)
                cloud_model_save_path = cloud_model_save_paths[module_name]
                loca_model_save_path = Path(loca_model_save_path)

                # Update current model paths to the new model path
                current_model_paths[module_name] = str(loca_model_save_path)

                if (
                    loca_model_save_path.exists() and
                    loca_model_save_path.is_dir() and
                    any(loca_model_save_path.iterdir())
                ):
                    print_separator(f"[Online SFT] Model checkpoint already exists at {loca_model_save_path}. Skipping training.")
                else:
                    # --- Build training commands ---
                    # Use is_imitation=False so that ref_model_name_or_path = agent_config.ref_model_path
                    # (the initial SFT/imitation checkpoint for KL regularization)
                    common_training_commands = config.get_common_training_commands(
                        role, is_imitation=False, wandb_disabled=wandb_disabled
                    )

                    data_save_load_commands = [
                        "--model_name_or_path", current_model_path_for_sft,
                        "--train_data_path", str(sft_data_path),
                        "--output_dir", optimizer_save_path,
                        "--local_model_save_path", str(loca_model_save_path),
                        "--cloud_model_save_path", cloud_model_save_path,
                    ]

                    # Online SFT specific training commands
                    online_sft_config = agent_config.rl.online_sft
                    num_repeats_per_episode = online_sft_config.num_agent_repeats_per_episode

                    # Compute training steps for persistent LR scheduler
                    num_total_epochs = num_epochs_this_iter = -1

                    num_batches_per_iteration = (
                        config.data_collection.num_train_envs_per_iter *
                        config.data_collection.num_train_responses_per_env *
                        num_repeats_per_episode
                    ) // agent_config.batch_size
                    num_total_steps = config.rl.num_iterations * num_batches_per_iteration
                    num_steps_this_iter = cur_iter * num_batches_per_iteration

                    training_step_commands = [
                        "--use_persistent_lr_scheduler", config.rl.use_persistent_lr_scheduler,
                        "--num_total_train_epochs", num_total_epochs,
                        "--num_total_train_steps", num_total_steps,
                        "--num_train_epochs", num_epochs_this_iter,
                        "--max_steps", num_steps_this_iter,
                    ]

                    # Set up resume_from_checkpoint commands
                    if config.rl.use_persistent_lr_scheduler and cur_iter > 1:
                        ckpt_glob = list(Path(optimizer_save_path).glob("checkpoint-*"))
                        if ckpt_glob:
                            prev_iter_ckpt_path = max(ckpt_glob)
                            training_step_commands.extend(["--resume_from_checkpoint", str(prev_iter_ckpt_path)])
                            print_separator(f"Resuming from checkpoint: {prev_iter_ckpt_path}")

                    # Get online SFT method commands (returns --train_method sft, --learning_rate, etc.)
                    sft_method_commands = online_sft_config.to_str_list()

                    sft_training_commands = (
                        common_training_commands +
                        data_save_load_commands +
                        training_step_commands +
                        sft_method_commands
                    )

                    print_separator(f"[Online SFT] Training agent, model will be saved to: {loca_model_save_path}")
                    run_commands(sft_training_commands, wandb_env_vars, f"Iteration {cur_iter} Online SFT training for {role}")

                    # After SFT training, merge LoRA weights immediately
                    # This ensures the merged model is ready for vLLM in the next iteration
                    if agent_config.lora.use_lora and (loca_model_save_path / "adapter_config.json").exists():
                        from utils.vllm import merge_lora_weights_for_vllm
                        print(f"[Online SFT] Merging LoRA weights for next iteration's vLLM serving...")
                        merged_path = merge_lora_weights_for_vllm(str(loca_model_save_path))
                        print(f"[Online SFT] Merged model saved to: {merged_path}")

            print_separator(f"[Iteration {cur_iter}] [Online SFT] Training finished.")
            rl_training_end_time = time.time()
            rl_training_time = rl_training_end_time - rl_training_start_time
            print(f"🎨 Online SFT training finished in {rl_training_time:.2f} seconds")

        else:
            # ==================================================================================
            # Standard RL methods: SPAG, GRPO, PPO
            # ==================================================================================
            # ------ step 4: assign credits ------ 
            print_separator(f"[Iteration {cur_iter}] Assigning credits ...")
            credit_assignment_start_time = time.time()

            for role in roles:

                agent_config = agent_configs[role]

                # prepare the data and model paths
                reward_data_path = data_paths[f"{role}_reward"]["train"]
                policy_data_path = data_paths[f"{role}_policy"]["train"]

                if rl_train_method in ["spag", "grpo"]:
                    # value_data_path and value_model_name_or_path are not used for SPAG and GRPO
                    value_data_path = policy_data_path
                    model_name_or_path = value_model_name_or_path = current_model_paths[role]

                elif rl_train_method == "ppo":
                    value_data_path = data_paths[f"{role}_value"]["train"]
                    model_name_or_path = current_model_paths[f"{role}_policy"]
                    value_model_name_or_path = current_model_paths[f"{role}_value"]

                else:
                    raise ValueError(f"Unknown training method: {rl_train_method}")

                # if resume training, check if credit assignment is already done
                if not agent_config.train_rl:
                    print(f"Skipping credit assignment for {role} as it is not trained.")
                    continue

                if policy_data_path.is_file() and value_data_path.is_file():
                    if rl_train_method == "spag":
                        with open(policy_data_path, 'r') as f:
                            cached_spag_data = json.load(f)
                        if not cached_spag_data or any(
                            data.get('spag_episode_reward', 0) <= 0 for data in cached_spag_data
                        ):
                            raise ValueError(
                                f"SPAG cache {policy_data_path} is empty or lacks positive episode "
                                "outcomes from the corrected winner filter. Regenerate credit-assignment data in a fresh output "
                                "directory before resuming; the existing file was not changed."
                            )
                    print(f"Policy and value data already exists at {policy_data_path} and {value_data_path}. Skipping credit assignment.")
                    continue

                # update accelerate config and commands
                update_accelerate_config(
                    "configs/assign_credit_accelerate_config.yaml",
                    agent_config.gradient_accumulation_steps,
                    zero_stage=(
                        3
                        if rl_train_method == "ppo" or agent_config.zero_stage == 3
                        else 0
                    ),   # hack to avoid init an optimizer
                )

                if config.self_play and not config.agent.train_use_population and not config.attacker.train_use_population:
                    roles_to_assign_credit = roles
                else:
                    roles_to_assign_credit = [role]

                credit_assignment_commands = [ # NOTE: question: what are they? answer: they are the commands to assign credit to the agent and attacker.
                    "accelerate", "launch",
                    "--config_file", "configs/assign_credit_accelerate_config.yaml",
                    "tools/assign_credit.py",
                    # if self-play, assign credit for both agent and attacker,
                    # as the model is going to be trained on both data
                    "--roles", *roles_to_assign_credit,
                    "--rl_method", rl_train_method,
                    # model
                    "--model_name_or_path", model_name_or_path,
                    "--ref_model_name_or_path", agent_config.ref_model_path,
                    "--value_model_name_or_path", value_model_name_or_path,
                    # data load and save
                    "--reward_data_path", str(reward_data_path),
                    "--policy_data_path", str(policy_data_path),
                    "--value_data_path", str(value_data_path),
                    # wandb logging
                    "--iteration", str(cur_iter),
                    # sampling params
                    "--temperature", str(config.data_collection.temperature),
                    # ppo params
                    "--per_device_batch_size", str(agent_config.per_device_train_batch_size),
                    "--gamma", str(agent_config.rl.ppo.gamma),
                    "--gae_lambda", str(agent_config.rl.ppo.gae_lambda),
                    "--num_value_buckets", str(agent_config.rl.ppo.num_value_buckets),
                    "--value_activation", agent_config.rl.ppo.value_activation,
                    "--value_min", str(agent_config.rl.ppo.value_min),
                    "--value_max", str(agent_config.rl.ppo.value_max),
                    # spag params
                    "--spag_gamma", str(agent_config.rl.spag.gamma),
                    "--spag_use_episode_filtering", "True",  # Keep each role's successful trajectories
                    "--spag_reward_threshold", "0.0",  # Each role's own episode reward must be positive
                    # grpo params
                    "--normalization_type", agent_config.rl.grpo.normalization_type,
                    # reward params
                    "--length_penalty_type", agent_config.rl.length_penalty_type,
                    "--length_penalty_coef", str(agent_config.rl.length_penalty_coef),
                    "--length_penalty_start", str(agent_config.rl.length_penalty_start),
                    "--length_penalty_end", str(agent_config.rl.length_penalty_end),
                    "--max_new_tokens", str(config.data_collection.max_new_tokens),
                    # lora params
                    "--use_lora", str(agent_config.lora.use_lora),
                    "--ref_model_uses_lora", str(agent_config.lora.use_lora and agent_config.train_imitation),
                ]

                run_commands(credit_assignment_commands, wandb_env_vars, f"Iteration {cur_iter} credit assignment for {role}")

            print_separator(f"[Iteration {cur_iter}] Reward / credit assignment finished.")
            credit_assignment_end_time = time.time()
            credit_assignment_time = credit_assignment_end_time - credit_assignment_start_time
            print(f"🎨 Reward / credit assignment finished in {credit_assignment_time:.2f} seconds")
            # ------ step 5: run RL training ------  # NOTE: question: how are they trained?
            print_separator(f"[Iteration {cur_iter}] Starting RL training...")
            rl_training_start_time = time.time()

            for role in roles:

                agent_config = agent_configs[role]

                if not agent_config.train_rl:
                    print(f"Skipping RL training for {role} as it is not trained.")
                    continue

                if rl_train_method == "spag":
                    if config.self_play:
                        data_keys = [["agent_policy", "attacker_policy"]]
                    else:
                        data_keys = [[f"{role}_policy"]]

                    module_names = [role]
                    num_repeats_per_episodes = [getattr(agent_config.rl.spag, f"num_{role}_repeats_per_episode")]
                    rl_method_commands = [agent_config.rl.to_str_list()]

                elif rl_train_method == "grpo":
                    if config.self_play:
                        data_keys = [["agent_policy", "attacker_policy"]]
                    else:
                        data_keys = [[f"{role}_policy"]]

                    module_names = [role]
                    num_repeats_per_episodes = [getattr(agent_config.rl.grpo, f"num_{role}_repeats_per_episode")]
                    rl_method_commands = [agent_config.rl.to_str_list()]

                elif rl_train_method == "ppo": # Experimental PPO branch; GRPO is the paper configuration.
                    data_keys, module_names, num_repeats_per_episodes, rl_method_commands = [], [], [], []
                    for module_name in ["value", "policy"]:
                        if config.self_play:
                            data_keys.append([f"agent_{module_name}", f"attacker_{module_name}"])
                        else:
                            data_keys.append([f"{role}_{module_name}"])

                        module_names.append(f"{role}_{module_name}")
                        num_repeats_per_episodes.append(getattr(agent_config.rl.ppo, f"num_{role}_{module_name}_repeats_per_episode"))
                        rl_method_commands.append(agent_config.rl.to_str_list() + ["--train_method", f"ppo_{module_name}"])

                else:
                    raise ValueError(f"Unknown training method: {rl_train_method}")

                for (
                    data_key,
                    module_name,
                    num_repeats_per_episode,
                    rl_method_command,
                ) in zip(
                    data_keys,
                    module_names,
                    num_repeats_per_episodes,
                    rl_method_commands,
                ): # NOTE: for spag and grpo, all those lists in zip have length 1. for ppo, all those lists in zip have length 2 (one for policy, one for value).

                    # prepare data, save, and load commands
                    data_path = [
                        str(data_paths[data_key_i]["train"])
                        for data_key_i in data_key
                    ]

                    current_model_path = current_model_paths[module_name]

                    optimizer_save_path = agent_config.rl.module_to_optimizer_save_path[module_name]
                    local_model_save_paths = get_model_path_by_iteration(config, cur_iter, cloud=False)
                    loca_model_save_path = local_model_save_paths[module_name]
                    cloud_model_save_paths = get_model_path_by_iteration(config, cur_iter, cloud=True)
                    cloud_model_save_path = cloud_model_save_paths[module_name]
                    loca_model_save_path = Path(loca_model_save_path)

                    # update current model paths to the new model path
                    current_model_paths[module_name] = str(loca_model_save_path)

                    if (
                        loca_model_save_path.exists() and
                        loca_model_save_path.is_dir() and
                        any(loca_model_save_path.iterdir())
                    ):
                        print_separator(f"Model checkpoint already exists at {loca_model_save_path}. Skipping {module_name} training.")
                        continue

                    data_save_load_commands = [
                        "--model_name_or_path", current_model_path,
                        "--train_data_path", *data_path,
                        "--output_dir", optimizer_save_path,
                        "--local_model_save_path", loca_model_save_path,
                        "--cloud_model_save_path", cloud_model_save_path,
                    ]

                    print_separator(f"Training {module_name} for {role}, model will be saved to: {loca_model_save_path}")

                    # prepare training commands
                    common_training_commands = config.get_common_training_commands(role, is_imitation=False, wandb_disabled=wandb_disabled)

                    # compute training step commands and parameters for persistent LR scheduler
                    num_total_epochs = num_epochs_this_iter = -1

                    num_batches_per_iteration = (
                        config.data_collection.num_train_envs_per_iter * 
                        config.data_collection.num_train_responses_per_env *
                        num_repeats_per_episode
                    ) // agent_config.batch_size
                    num_total_steps = config.rl.num_iterations * num_batches_per_iteration
                    num_steps_this_iter = cur_iter * num_batches_per_iteration

                    training_step_commands = [
                        "--use_persistent_lr_scheduler", config.rl.use_persistent_lr_scheduler,
                        "--num_total_train_epochs", num_total_epochs,
                        "--num_total_train_steps", num_total_steps,
                        "--num_train_epochs", num_epochs_this_iter,
                        "--max_steps", num_steps_this_iter,
                    ]

                    # set up resume_from_checkpoint commands
                    if config.rl.use_persistent_lr_scheduler and cur_iter > 1:
                        prev_iter_ckpt_path = max(Path(optimizer_save_path).glob("checkpoint-*"))
                        training_step_commands.extend(["--resume_from_checkpoint", str(prev_iter_ckpt_path)])
                        print_separator(f"Resuming from checkpoint: {prev_iter_ckpt_path}")

                    rl_training_commands = common_training_commands + \
                        data_save_load_commands + \
                        training_step_commands + \
                        rl_method_command

                    # breakpoint()
                    run_commands(rl_training_commands, wandb_env_vars, f"Iteration {cur_iter} RL {module_name} training for {role}")

            print_separator(f"[Iteration {cur_iter}] RL training finished.")
            rl_training_end_time = time.time()
            rl_training_time = rl_training_end_time - rl_training_start_time
            print(f"🎨 RL training finished in {rl_training_time:.2f} seconds")

        # --- Save Progress ---
        with open(rl_progress_file, 'w') as f:
            f.write(str(cur_iter))

        print_separator(f"Completed Iteration {cur_iter}", current_model_paths)
        print_separator(f"Time breakdown:")
        print_separator(f"   Training: {rl_training_time:.2f} seconds")
        print_separator(f"   Credit assignment: {credit_assignment_time:.2f} seconds")
        print_separator(f"   Post-processing rollout: {postprocess_rollout_time:.2f} seconds")
        print_separator(f"   Episode collection: {episode_collection_time:.2f} seconds")

    print_separator(f"All {config.rl.num_iterations} iterations completed successfully.")


if __name__ == "__main__":
    main()
