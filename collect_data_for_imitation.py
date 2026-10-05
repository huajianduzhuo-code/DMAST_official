#!/usr/bin/env python3
"""
Collect high-quality imitation data with the 27B teacher model.

This script rolls out a strong teacher in the environment and saves
demonstration episodes for student imitation learning.

Usage:
    python collect_data_for_imitation.py --num-episodes 2000 --all-tasks --output data/imitation_27b.json
"""

import argparse
import sys
import logging
from pathlib import Path

# Suppress verbose logs from httpx and openai (HTTP Request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from configs.task_splits import MINIWOB_TRAIN_TASKS, select_sft_tasks, select_attack_study_tasks
from utils.training_data import validate_training_data_file

from configs.train_adversarial_args import TrainAdversarialConfig, DataCollectionConfig, AgentConfig, LoraConfig, VLLMConfig
from configs.collect_episode_args import CollectEpisodeArgs, MultimodalCollectEpisodeArgs, VllmModelArgsList, AttackerPrivilegeObsFlags
from agents.llm.vllm_model import VllmModelArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp
import bgym
from browsergym.experiments.loop import EnvArgs
from utils.bgym import setup_bgym_env_variables
from tools.multimodal_collect_episode import collect_episodes


def create_data_collection_config(
    model_path: str,
    output_path: str,
    num_episodes: int = 2000,
    num_tasks: int = len(MINIWOB_TRAIN_TASKS),
    all_tasks: bool = False,  # Use all registered MiniWoB++ tasks except the 28 held-out tasks
    enable_multimodal: bool = True,
    enable_agent_vision: bool = True,  # Independently control agent vision
    enable_attacker: bool = True,
    execute_text_injection: bool = True,
    execute_image_attacks: bool = True,
    on_policy_role: str = "both",
    num_jobs: int = 128,
    save_every_n_episodes: int = 128,
    job_timeout: int = 300,  # Timeout per job in seconds (0 = no timeout)
    attacker_model_path: str = None,  # Separate model path for attacker
    evaluation: bool = False,
):
    """Build the data-collection config.

    Args:
        attacker_model_path: If set, use a different model as the attacker
            (not self-play). If None, reuse the agent model (self-play).
        evaluation: Use the attack-mode study task pool and eval collection mode.
            Evaluation outputs are not training data.
    """

    setup_bgym_env_variables()
    task_selector = select_attack_study_tasks if evaluation else select_sft_tasks
    task_ids = task_selector(num_tasks, all_tasks=all_tasks)
    num_tasks = len(task_ids)
    if num_episodes < num_tasks:
        raise ValueError("num_episodes must provide at least one episode per selected task")
    if not evaluation and Path(output_path).exists():
        validate_training_data_file(output_path)

    # Self-play iff agent and attacker share the same model
    effective_attacker_model_path = attacker_model_path if attacker_model_path else model_path
    is_self_play = enable_attacker and (effective_attacker_model_path == model_path)

    # Full base model for collection; LoRA is not used
    agent_config = AgentConfig(
        model_path=model_path,
        train_imitation=False, # not actually used in data collection
        train_rl=True, # Required for config validation
        train_use_population=False, # not actually used in data collection
        lora=LoraConfig(
            use_lora=False,  # data collection uses the full base model
        ),
        vllm=VLLMConfig(
            max_model_len=32768,
            gpu_memory_utilization=0.8,
        ),
    )
    
    # Attacker config
    if not enable_attacker:
        # Disable RL training on the unused attacker to satisfy config asserts
        attacker_config = AgentConfig(
            model_path=model_path,
            train_imitation=False,
            train_rl=False,  # no attacker: do not train this role
            train_use_population=False,
            lora=LoraConfig(use_lora=False),
            vllm=VLLMConfig(max_model_len=32768, gpu_memory_utilization=0.8),
        )
    elif is_self_play:
        # Self-play: share the agent model and config
        attacker_config = agent_config
    else:
        # Separate attacker model
        attacker_config = AgentConfig(
            model_path=effective_attacker_model_path,
            train_imitation=False,
            train_rl=True,  # Required for config validation
            train_use_population=False,
            lora=LoraConfig(use_lora=False),
            vllm=VLLMConfig(
                max_model_len=32768,
                gpu_memory_utilization=0.8,
            ),
        )
    
    # Full training config, reused here for data collection
    base_config = TrainAdversarialConfig(
        seed=42, # unused in data collection because fix_seed is False
        group_name="imitation_data", # used in imitationg training to locate the checkpoint saving directory, not used in data collection
        exp_name="collect_27b_demos_for_imitation", # used in imitationg training to locate the checkpoint saving file, not used in data collection
        
        agent=agent_config,
        attacker=attacker_config,
        self_play=is_self_play,  # self-play only when both roles share a model
        use_attacker=enable_attacker,
        
        data_collection=DataCollectionConfig(
            # Multimodal settings (HTML injection mode)
            enable_multimodal=enable_multimodal,
            enable_agent_vision=enable_agent_vision,  # Independent from enable_multimodal
            # HTML injection flags - independently control what agent sees after injection
            # execute_text_injection: If True, agent sees modified AXTree
            # execute_image_attacks: If True, agent sees modified screenshot
            execute_text_injection=execute_text_injection and enable_attacker,
            execute_image_attacks=execute_image_attacks and enable_attacker,
            
            # Task set
            use_env_ids=True,  # Use resolved IDs; evaluation may include held-out tasks
            # These fields configure the shared collector, not training in evaluation mode.
            train_prop=1.0,
            valid_prop=0.0,
            eval_prop=0.0,
            train_env_ids=task_ids,
            
            # Collection size
            num_train_envs_per_iter=num_tasks,  # number of tasks
            num_train_responses_per_env=num_episodes // num_tasks,  # episodes per task
            max_episode_steps=5,
            
            # Sampling
            num_jobs=num_jobs,  # parallel workers
            temperature=1.2,
            max_new_tokens=4096,
        ),
    )
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from utils.vllm import start_vllm

    print("=" * 80)
    print("Imitation data collection (HTML injection mode)")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"   Agent model: {model_path}")
    if enable_attacker and not is_self_play:
        print(f"   Attacker model: {effective_attacker_model_path}")
        print(f"   Mode: not self-play (agent and attacker use different models)")
    elif enable_attacker:
        print(f"   Mode: self-play (agent and attacker share a model)")
    print(f"   Tasks: {num_tasks} ({'evaluation' if evaluation else 'training; held-out tasks excluded'})")
    print(f"   Episodes per task: {num_episodes // num_tasks}")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Multimodal: {enable_multimodal}")
    print(f"   Agent vision: {enable_agent_vision}")
    print(f"   Attacker (HTML injection): {enable_attacker}")
    if enable_attacker:
        print(f"   └─ Agent sees modified AXTree: {execute_text_injection}")
        print(f"   └─ Agent sees modified screenshot: {execute_image_attacks}")
    print(f"   On-policy role: {on_policy_role}")
    if on_policy_role == "both" and enable_attacker:
        print(f"   └─ Note: episode count is doubled (half with attacker, half without)")
    elif on_policy_role == "attacker" and enable_attacker:
        print(f"   └─ Collect only episodes that include an attacker")
    print(f"   Parallel workers: {num_jobs}")
    print(f"   Save every: {save_every_n_episodes} episodes")
    print(f"   └─ Effective parallelism: min({num_jobs}, {save_every_n_episodes}) = {min(num_jobs, save_every_n_episodes)}")
    print(f"   Job timeout: {job_timeout}s (timed-out jobs are skipped)" if job_timeout > 0 else "   Job timeout: none")
    print(f"   Output: {output_path}")
    print()

    current_model_paths = {
        "agent": model_path,
        "attacker": effective_attacker_model_path,
    }

    print("Starting vLLM server...")
    vllm_model_arguments, vllm_server_args_list = start_vllm(
        base_config,
        current_model_paths,
        on_policy_role=on_policy_role,
        current_iter=1,  # iteration must start at 1
    )

    from configs.collect_episode_args import AttackerPrivilegeObsFlags

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
            use_screenshot=enable_agent_vision,  # Use agent vision flag
            use_som=False,
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
    
    collect_args = CollectEpisodeArgs(
        seed=base_config.seed,
        mode="eval" if evaluation else "train",
        benchmark=base_config.data_collection.benchmark,
        env_ids=base_config.data_collection.train_env_ids,
        use_env_ids=base_config.data_collection.use_env_ids,
        train_prop=base_config.data_collection.train_prop,
        valid_prop=base_config.data_collection.valid_prop,
        eval_prop=base_config.data_collection.eval_prop,
        num_envs_per_iter=num_tasks,
        num_responses_per_env=num_episodes // num_tasks,
        num_jobs=base_config.data_collection.num_jobs,
        save_every_n_episodes=save_every_n_episodes,
        job_timeout=job_timeout,  # Timeout per job in seconds
        fix_seed=False,  # unfixed seed for diversity
        output_path=str(output_file),
        
        on_policy_role=on_policy_role,
        use_attacker=enable_attacker,
        attacker_use_art=False,
        collect_responses_with_same_opponent=False,
        
        default_model_args=VllmModelArgs(
            model_name="placeholder",
            temperature=base_config.data_collection.temperature,
            top_p=base_config.data_collection.top_p,
            top_k=base_config.data_collection.top_k,
            min_p=base_config.data_collection.min_p,
            repetition_penalty=base_config.data_collection.repetition_penalty,
            max_new_tokens=base_config.data_collection.max_new_tokens,
            log_probs=0,
            # Attacker always needs vision to generate better CSS selectors
            # Agent vision is controlled separately via enable_agent_vision
            vision_support=True,  # Always enable for attacker
        ),
        
        agent_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["agent"]["model_names"],
            base_urls=vllm_model_arguments["agent"]["urls"],
            max_input_tokens=base_config.agent.vllm.max_model_len - base_config.data_collection.max_new_tokens - 16,
        ),
        
        attacker_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["attacker"]["model_names"],
            base_urls=vllm_model_arguments["attacker"]["urls"],
            max_input_tokens=base_config.attacker.vllm.max_model_len - base_config.data_collection.max_new_tokens - 16,
        ) if enable_attacker else None,
        
        env_args=EnvArgs(
            task_name="placeholder",
            max_steps=base_config.data_collection.max_episode_steps,
        ),
        
        attacker_privilege=AttackerPrivilegeObsFlags(
            attacker_uses_web_agent_history=base_config.data_collection.attacker_uses_web_agent_history,
            attacker_uses_web_agent_action_history=base_config.data_collection.attacker_uses_web_agent_action_history,
            attacker_uses_web_agent_think_history=base_config.data_collection.attacker_uses_web_agent_think_history,
            attacker_uses_web_agent_memory=base_config.data_collection.attacker_uses_web_agent_memory,
        ),
        
        prompt_flags=prompt_flags,
    )
    
    return base_config, collect_args


def main():
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Collect demonstration data for imitation learning")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "dmast_ckpt/pretrained/gemma-3-27b-it"),
        help="Teacher model path (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "bucket/data/imitation_data/gemma_3_27b.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2000,
        help="Total episodes to collect (default: 2000). Doubled when --on-policy-role=both",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=len(MINIWOB_TRAIN_TASKS),
        help="Number of tasks (default: 42); training pool: 42, --evaluation study pool: 56",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Use all registered tasks, excluding 28 held-out tasks unless --evaluation is set",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Collect attack-mode evaluation episodes, allowing held-out tasks; outputs are not for training",
    )
    parser.add_argument(
        "--agent-no-vision",
        action="store_true",
        help="Agent sees only the AXTree (no screenshot); attacker can still inject into the AXTree",
    )
    parser.add_argument(
        "--attacker-model-path",
        type=str,
        default=None,
        help="Attacker model path (default: same as agent). A different path disables self-play",
    )
    parser.add_argument(
        "--no-attacker",
        action="store_true",
        help="Collect agent-only demonstrations (no attacker)",
    )
    parser.add_argument(
        "--no-text-injection",
        action="store_true",
        help="Do not deliver the injected AXTree (agent sees the original AXTree)",
    )
    parser.add_argument(
        "--no-image-attack",
        action="store_true",
        help="Do not deliver the injected screenshot (agent sees the original screenshot)",
    )
    parser.add_argument(
        "--on-policy-role",
        type=str,
        choices=["agent", "attacker", "both"],
        default="both",
        help="Collection mode: 'both' splits with/without attacker (doubles count), "
             "'agent' collects the agent view only, 'attacker' collects attacked episodes only",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=128,
        help="Max parallel workers (default: 128)",
    )
    parser.add_argument(
        "--save-every-n-episodes",
        type=int,
        default=128,
        help="Episodes per save batch; effective parallelism is min(num_jobs, this value)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=300,
        help="Per-episode timeout in seconds (default: 300). Timed-out jobs are skipped. 0 disables the timeout",
    )
    
    args = parser.parse_args()
    
    try:
        config, collect_args = create_data_collection_config(
            model_path=args.model_path,
            output_path=args.output,
            num_episodes=args.num_episodes,
            num_tasks=args.num_tasks,
        all_tasks=args.all_tasks,  # Use all registered MiniWoB++ tasks except the 28 held-out tasks
            enable_multimodal=True,  # Always use multimodal collection for HTML injection support
            enable_agent_vision=not args.agent_no_vision,
            enable_attacker=not args.no_attacker,
            execute_text_injection=not args.no_text_injection,
            execute_image_attacks=not args.no_image_attack,
            on_policy_role=args.on_policy_role,
            num_jobs=args.num_jobs,
            save_every_n_episodes=args.save_every_n_episodes,
            job_timeout=args.job_timeout,
            attacker_model_path=args.attacker_model_path,
            evaluation=args.evaluation,
        )
        
        # HTML Injection mode - no GPU image editor needed
        if config.use_attacker:
            print("\n🎨 HTML Injection Attack Mode")
            print("   Attacker will generate HTML/CSS injections")
            print("   Injections modify DOM directly via JavaScript")
            print("   Both screenshot and AXTree are automatically updated")
        
        print("\nCollecting demonstration episodes...")
        print("🎨 Using multimodal episode collection (HTML injection mode)")
        
        # Convert CollectEpisodeArgs to MultimodalCollectEpisodeArgs
        multimodal_args = MultimodalCollectEpisodeArgs(
            # Copy all base args
            seed=collect_args.seed,
            mode=collect_args.mode,
            benchmark=collect_args.benchmark,
            env_ids=collect_args.env_ids,
            use_env_ids=collect_args.use_env_ids,
            train_prop=collect_args.train_prop,
            valid_prop=collect_args.valid_prop,
            eval_prop=collect_args.eval_prop,
            num_envs_per_iter=collect_args.num_envs_per_iter,
            num_responses_per_env=collect_args.num_responses_per_env,
            num_jobs=collect_args.num_jobs,
            save_every_n_episodes=collect_args.save_every_n_episodes,
            job_timeout=collect_args.job_timeout,
            env_step_timeout=collect_args.env_step_timeout,
            fix_seed=collect_args.fix_seed,
            collect_responses_with_same_opponent=collect_args.collect_responses_with_same_opponent,
            on_policy_role=collect_args.on_policy_role,
            use_attacker=collect_args.use_attacker,
            attacker_use_art=collect_args.attacker_use_art,
            default_model_args=collect_args.default_model_args,
            agent_args_list=collect_args.agent_args_list,
            attacker_args_list=collect_args.attacker_args_list,
            env_args=collect_args.env_args,
            attacker_privilege=collect_args.attacker_privilege,
            prompt_flags=collect_args.prompt_flags,
            output_path=collect_args.output_path,
            
            # Add multimodal-specific args (HTML injection mode)
            enable_agent_vision=config.data_collection.enable_agent_vision,
            execute_text_injection=config.data_collection.execute_text_injection,
            execute_image_attacks=config.data_collection.execute_image_attacks,
            max_image_size=config.data_collection.max_image_size,
        )
        
        collect_episodes(multimodal_args, iteration=1)
        
        print("\nData collection finished.")
        print(f"   Saved to: {args.output}")
        if args.evaluation:
            print("Evaluation episodes saved for analysis.")
        else:
            print(f"\nNext: set imitation.raw_data_path to {args.output} and run python train.py")
        
    except Exception as e:
        print(f"\nData collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
