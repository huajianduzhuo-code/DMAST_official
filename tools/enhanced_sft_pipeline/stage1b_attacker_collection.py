#!/usr/bin/env python3
"""
Stage 1b: Attacker Imitation Data Collection
=============================================

Collects imitation training data for the attacker model:
- Uses attacker with full attack execution (text + image attacks)
- Agent interacts with attacked observations
- Filters for attacker-successful episodes
- Output format matches existing imitation training pipeline

This stage runs in parallel with Stage 1 (agent golden trajectories).
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))


from configs.task_splits import MINIWOB_TRAIN_TASKS, select_sft_tasks
from utils.training_data import validate_training_data_file


@dataclass
class AttackerStep:
    """
    Represents a single step of attacker data for imitation training.
    
    Mirrors the format used in collect_data_for_imitation.py
    """
    episode_idx: int
    task_name: str
    transition_idx: int
    role: str  # "attacker"
    on_policy: bool
    use_attacker: bool
    seed: Optional[int]
    success: bool  # Attacker success (agent failed task)
    reward: float  # Attacker reward (1 if successful)
    terminated: bool
    truncated: bool
    messages: List[Dict[str, Any]]  # Attacker conversation
    is_multimodal: bool
    prompt_token_ids: List[int]
    response_token_ids: List[int]
    weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_attacker_data(
    model_path: str,
    output_path: str,
    num_episodes: int = 4000,
    num_tasks: int = len(MINIWOB_TRAIN_TASKS),
    num_jobs: int = 128,
    max_episode_steps: int = 5,
    all_tasks: bool = True,
    save_every_n_episodes: int = 128,
    job_timeout: int = 300,
):
    """
    Collect attacker imitation training data.
    
    This is essentially the same as collect_data_for_imitation.py with:
    - on_policy_role="attacker" (only collect attacker-focused episodes)
    - use_attacker=True
    - execute_text_injection=True
    - execute_image_attacks=True
    - Filter for attacker-successful episodes
    
    Args:
        model_path: Path to the 27B model for attacker
        output_path: Where to save the attacker data JSON
        num_episodes: Nominal episode budget, rounded down per task
        num_tasks: Number of shared training tasks when all_tasks=False
        num_jobs: Parallel workers
        max_episode_steps: Max steps per episode
        all_tasks: Use all registered MiniWoB++ tasks except the 28 held-out tasks
        save_every_n_episodes: Batch size for saving
        job_timeout: Timeout per job in seconds
    """
    print("=" * 80)
    print("🎭 Stage 1b: Attacker Imitation Data Collection")
    print("=" * 80)
    
    # Import dependencies
    from configs.train_adversarial_args import (
        TrainAdversarialConfig, DataCollectionConfig, AgentConfig, LoraConfig, VLLMConfig
    )
    from configs.collect_episode_args import (
        MultimodalCollectEpisodeArgs, VllmModelArgsList, AttackerPrivilegeObsFlags
    )
    from agents.llm.vllm_model import VllmModelArgs
    from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
    from agentlab.agents import dynamic_prompting as dp
    import bgym
    from browsergym.experiments.loop import EnvArgs
    from utils.bgym import setup_bgym_env_variables
    from tools.multimodal_collect_episode import collect_episodes
    from utils.vllm import start_vllm
    
    # Resolve the training split before starting inference or collection.
    setup_bgym_env_variables()
    task_ids = select_sft_tasks(num_tasks, all_tasks=all_tasks)
    num_tasks = len(task_ids)
    if num_episodes < num_tasks:
        raise ValueError("num_episodes must provide at least one episode per selected task")
    if Path(output_path).exists():
        validate_training_data_file(output_path)
    
    # Configure agent and attacker
    agent_config = AgentConfig(
        model_path=model_path,
        train_imitation=False,
        train_rl=True,  # Required for config validation
        train_use_population=False,
        lora=LoraConfig(use_lora=False),
        vllm=VLLMConfig(max_model_len=32768, gpu_memory_utilization=0.8),
    )
    
    # Create training config
    base_config = TrainAdversarialConfig(
        seed=42,
        group_name="enhanced_sft_attacker",
        exp_name="collect_attacker_demos",
        
        agent=agent_config,
        attacker=agent_config,  # Self-play: same config
        self_play=True,
        use_attacker=True,
        
        data_collection=DataCollectionConfig(
            enable_multimodal=True,
            enable_agent_vision=True,
            # Full attack execution - agent sees attacked observations
            execute_text_injection=True,
            execute_image_attacks=True,
            
            use_env_ids=True,  # Resolved training IDs always exclude held-out tasks
            train_prop=1.0,
            valid_prop=0.0,
            eval_prop=0.0,
            train_env_ids=task_ids,
            
            num_train_envs_per_iter=num_tasks,
            num_train_responses_per_env=num_episodes // num_tasks,
            max_episode_steps=max_episode_steps,
            
            num_jobs=num_jobs,
            temperature=1.2,
            max_new_tokens=4096,
        ),
    )
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Tasks: {num_tasks}" + (" (all non-held-out MiniWoB++)" if all_tasks else ""))
    print(f"   Nominal episode budget: {num_episodes}")
    print(f"   Planned episodes: {num_tasks * (num_episodes // num_tasks)}")
    print(f"   Max steps: {max_episode_steps}")
    print(f"   Parallel jobs: {num_jobs}")
    print(f"   Attack mode: Full (text + image)")
    print(f"   On-policy role: attacker")
    print(f"   Output: {output_path}")
    print()
    
    # Prepare model paths
    current_model_paths = {
        "agent": model_path,
        "attacker": model_path,
    }
    
    # Start vLLM servers
    print("🚀 Starting vLLM servers...")
    vllm_model_arguments, vllm_server_args_list = start_vllm(
        base_config,
        current_model_paths,
        on_policy_role="attacker",  # Focus on attacker data
        current_iter=1,
    )
    
    # Setup prompt flags
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
            use_screenshot=True,
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
    
    # Create collection args
    collect_args = MultimodalCollectEpisodeArgs(
        seed=base_config.seed,
        mode="train",
        benchmark=base_config.data_collection.benchmark,
        env_ids=base_config.data_collection.train_env_ids,
        use_env_ids=base_config.data_collection.use_env_ids,
        train_prop=base_config.data_collection.train_prop,
        valid_prop=base_config.data_collection.valid_prop,
        eval_prop=base_config.data_collection.eval_prop,
        num_envs_per_iter=num_tasks,
        num_responses_per_env=num_episodes // num_tasks,
        num_jobs=num_jobs,
        save_every_n_episodes=save_every_n_episodes,
        job_timeout=job_timeout,
        fix_seed=False,
        output_path=str(output_file),
        
        # Attacker-focused collection
        on_policy_role="attacker",
        use_attacker=True,
        attacker_use_art=False,
        collect_responses_with_same_opponent=False,
        
        # Full attack execution
        enable_agent_vision=True,
        execute_text_injection=True,
        execute_image_attacks=True,
        
        default_model_args=VllmModelArgs(
            model_name="placeholder",
            temperature=base_config.data_collection.temperature,
            top_p=base_config.data_collection.top_p,
            top_k=base_config.data_collection.top_k,
            min_p=base_config.data_collection.min_p,
            repetition_penalty=base_config.data_collection.repetition_penalty,
            max_new_tokens=base_config.data_collection.max_new_tokens,
            log_probs=0,
            vision_support=True,
        ),
        
        agent_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["agent"]["model_names"],
            base_urls=vllm_model_arguments["agent"]["urls"],
            max_input_tokens=base_config.agent.vllm.max_model_len - base_config.data_collection.max_new_tokens - 16,
        ),
        
        attacker_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["attacker"]["model_names"],
            base_urls=vllm_model_arguments["attacker"]["urls"],
            max_input_tokens=base_config.agent.vllm.max_model_len - base_config.data_collection.max_new_tokens - 16,
        ),
        
        env_args=EnvArgs(
            task_name="placeholder",
            max_steps=max_episode_steps,
        ),
        
        attacker_privilege=AttackerPrivilegeObsFlags(
            attacker_uses_web_agent_history=base_config.data_collection.attacker_uses_web_agent_history,
            attacker_uses_web_agent_action_history=base_config.data_collection.attacker_uses_web_agent_action_history,
            attacker_uses_web_agent_think_history=base_config.data_collection.attacker_uses_web_agent_think_history,
            attacker_uses_web_agent_memory=base_config.data_collection.attacker_uses_web_agent_memory,
        ),
        
        prompt_flags=prompt_flags,
    )
    
    # Collect episodes
    print("\n📊 Collecting attacker imitation data...")
    collect_episodes(collect_args, iteration=1)
    
    print(f"\n✅ Attacker data collection complete!")
    print(f"   Raw data saved to: {output_path}")
    
    return output_path


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stage 1b: Collect attacker imitation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use a nominal budget of 4000 episodes across all non-held-out tasks
    python stage1b_attacker_collection.py --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it --all-tasks
    
    # Collect with specific number of tasks
    python stage1b_attacker_collection.py --model-path ~/model --no-all-tasks --num-tasks 42 --num-episodes 2000
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "dmast_ckpt/pretrained/gemma-3-27b-it"),
        help="Path to 27B model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/stage1b_attacker_data.json"),
        help="Output path for attacker data",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=4000,
        help="Nominal episodes, rounded down per task (default: 4000)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=len(MINIWOB_TRAIN_TASKS),
        help="Number of shared training tasks (1-42; default: 42); requires --no-all-tasks",
    )
    parser.add_argument(
        "--all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all registered non-held-out MiniWoB++ tasks (default); --no-all-tasks enables --num-tasks",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=128,
        help="Parallel workers",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Max steps per episode",
    )
    
    args = parser.parse_args()
    
    collect_attacker_data(
        model_path=args.model_path,
        output_path=args.output,
        num_episodes=args.num_episodes,
        num_tasks=args.num_tasks,
        num_jobs=args.num_jobs,
        max_episode_steps=args.max_steps,
        all_tasks=args.all_tasks,
    )


if __name__ == "__main__":
    main()
