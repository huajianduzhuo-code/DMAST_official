#!/usr/bin/env python3
"""
Stage 1: Golden Trajectory Collection
=====================================

Collect successful agent episodes using clean observations and actions.
When enabled, the attacker supplies paired observations for the same states;
the agent acts on the clean view.
This provides the "ground truth" of how a well-performing agent should behave.

Output per successful step:
- clean_screenshot: PIL Image of the clean webpage
- clean_axtree: Accessibility tree text without any injection
- task_instruction: The user's task goal
- golden_action: The correct browser action taken by the agent

Only successful episodes are retained.
"""

import argparse
import json
import logging
import os
import re
import sys
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from copy import deepcopy
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from configs.task_splits import MINIWOB_TRAIN_TASKS, select_sft_tasks
from utils.training_data import validate_training_data_file

from configs.train_adversarial_args import TrainAdversarialConfig, DataCollectionConfig, AgentConfig, LoraConfig, VLLMConfig
from configs.collect_episode_args import MultimodalCollectEpisodeArgs, VllmModelArgsList, AttackerPrivilegeObsFlags
from agents.llm.vllm_model import VllmModelArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp
import bgym
from browsergym.experiments.loop import EnvArgs
from utils.bgym import setup_bgym_env_variables

# Import model family detection and think extraction utilities
from utils.vllm import get_model_family, extract_think_content

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class GoldenStep:
    """
    Represents a single step in a golden trajectory.
    
    When collected with attacker enabled, includes both clean and attacked observations.
    When collected in clean-only mode, attacked fields will be None/empty.
    
    IMPORTANT: agent_messages contains the complete LLM conversation including:
    - System prompt
    - User prompt (with observation)
    - Assistant response (with think + action)
    
    Stage 4 can directly use agent_messages and only replace specific parts
    (observation → attacked, think → defense_cot) to ensure format consistency.
    
    Token IDs from vLLM are preserved for training consistency:
    - prompt_token_ids: Token IDs for the prompt (system + user messages)
    - response_token_ids: Token IDs for the response (assistant message)
    """
    episode_idx: int
    step_idx: int
    task_name: str
    task_instruction: str
    clean_screenshot_b64: str  # Base64 encoded PNG of clean webpage
    clean_axtree: str          # Accessibility tree without injection
    golden_action: str         # Correct action taken by agent
    agent_think: Optional[str] = None
    
    # 🔑 Complete agent messages for format consistency in Stage 4
    agent_messages: Optional[List[Dict[str, Any]]] = None
    
    # 🔑 Token IDs from vLLM for training consistency
    prompt_token_ids: Optional[List[int]] = None
    response_token_ids: Optional[List[int]] = None
    
    # Attacked observation data (populated when use_attacker=True)
    attacked_screenshot_b64: Optional[str] = None  # Base64 PNG after HTML injection
    attacked_axtree: Optional[str] = None          # AXTree after HTML injection
    injection_dict: Optional[Dict[str, Any]] = None  # Details of the HTML injection
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def image_to_base64(image) -> Optional[str]:
    """Convert PIL Image or numpy array to base64 string"""
    if image is None:
        return None
    try:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, str):
            # Already base64
            return image
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
    except Exception as e:
        logging.warning(f"Failed to convert image to base64: {e}")
    return None


def extract_task_instruction(obs: Dict[str, Any]) -> str:
    """Extract task instruction from observation"""
    # Try goal_object first
    goal_object = obs.get('goal_object', [])
    if isinstance(goal_object, list):
        for item in goal_object:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
            elif isinstance(item, str):
                return item
    
    # Try chat_messages
    chat_messages = obs.get('chat_messages', [])
    for msg in chat_messages:
        if msg.get('role') == 'user':
            return msg.get('content', '')
    
    # Fallback
    return obs.get('goal', '')


def collect_golden_trajectories(
    model_path: str,
    output_path: str,
    num_episodes: int = 4000,
    num_tasks: int = len(MINIWOB_TRAIN_TASKS),
    num_jobs: int = 64,
    save_every_n_episodes: int = 64,
    job_timeout: int = 300,
    max_episode_steps: int = 5,
    attacker_model_path: str = None,
    use_attacker: bool = True,
    all_tasks: bool = True,  # Use all registered MiniWoB++ tasks except the 28 held-out tasks
):
    """
    Collect golden trajectories with optional attacker.
    
    When use_attacker=True (default), this collects:
    - clean_obs: Original observation before any attack
    - attacked_obs: Observation after HTML injection attack
    - golden_action: The correct action taken by the agent
    
    This enables the enhanced SFT pipeline to have both clean and attacked
    versions of each step without needing a separate attack phase.
    The same call also collects independent clean-only episodes. Each mode
    uses the nominal num_episodes budget, rounded down to equal counts per task.
    
    Args:
        model_path: Path to the 27B teacher model (agent)
        output_path: Where to save the golden trajectories JSON
        num_episodes: Nominal episode budget per mode, rounded down per task
        num_tasks: Number of shared training tasks when all_tasks=False
        num_jobs: Parallel workers
        save_every_n_episodes: Batch size for saving
        job_timeout: Timeout per job in seconds
        max_episode_steps: Maximum steps per episode
        attacker_model_path: Path to attacker model (defaults to same as agent)
        use_attacker: Whether to enable attacker (default True for enhanced SFT)
    """
    # Default attacker model to same as agent
    if attacker_model_path is None:
        attacker_model_path = model_path
    setup_bgym_env_variables()
    
    task_ids = select_sft_tasks(num_tasks, all_tasks=all_tasks)
    num_tasks = len(task_ids)
    if num_episodes < num_tasks:
        raise ValueError("num_episodes must provide at least one episode per selected task")
    raw_output_filename = "raw_dual_episodes.json" if use_attacker else "raw_clean_episodes.json"
    raw_path = Path(output_path).parent / raw_output_filename
    if raw_path.exists():
        validate_training_data_file(raw_path)
    
    mode_str = "with Attacker (Clean + Attacked)" if use_attacker else "Clean Only"
    tasks_str = f"{num_tasks} training tasks (28 held-out tasks excluded)"
    print("=" * 80)
    print(f"🏆 Stage 1: Golden Trajectory Collection ({mode_str})")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   Teacher Model: {model_path}")
    if use_attacker:
        print(f"   Attacker Model: {attacker_model_path}")
    print(f"   Tasks: {tasks_str}")
    print(f"   Episodes per task: {num_episodes // num_tasks}")
    print(f"   Nominal episode budget per mode: {num_episodes}")
    episodes_per_mode = num_tasks * (num_episodes // num_tasks)
    print(f"   Planned episodes per mode: {episodes_per_mode}")
    print(f"   Total planned episodes: {episodes_per_mode * (2 if use_attacker else 1)}")
    print(f"   Max steps per episode: {max_episode_steps}")
    print(f"   Parallel workers: {num_jobs}")
    print(f"   Use Attacker: {use_attacker}")
    print(f"   Output: {output_path}")
    print()
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Agent config (no LoRA for data collection)
    agent_config = AgentConfig(
        model_path=model_path,
        train_imitation=False,
        train_rl=True,  # Required for config validation (not actually used in data collection)
        train_use_population=False,
        lora=LoraConfig(use_lora=False),
        vllm=VLLMConfig(
            max_model_len=32768,
            gpu_memory_utilization=0.8,
        ),
    )
    
    # Attacker config (same as agent, just different role)
    attacker_agent_config = AgentConfig(
        model_path=attacker_model_path,
        train_imitation=False,
        train_rl=False,  # Only one needs train_rl=True
        train_use_population=False,
        lora=LoraConfig(use_lora=False),
        vllm=VLLMConfig(
            max_model_len=32768,
            gpu_memory_utilization=0.8,
        ),
    )
    
    # Base config - optionally with attacker for dual data collection
    base_config = TrainAdversarialConfig(
        seed=42,
        group_name="golden_trajectory",
        exp_name="stage1_dual_collection" if use_attacker else "stage1_clean_collection",
        agent=agent_config,
        attacker=attacker_agent_config,
        self_play=False,
        use_attacker=use_attacker,  # 🔑 KEY: Enable attacker for dual data
        data_collection=DataCollectionConfig(
            enable_multimodal=True,
            enable_agent_vision=True,
            execute_text_injection=use_attacker,  # Inject into DOM when attacker enabled
            execute_image_attacks=use_attacker,   # Inject into screenshot when attacker enabled
            use_env_ids=True,  # Resolved training IDs always exclude held-out tasks
            train_prop=1.0,
            valid_prop=0.0,
            eval_prop=0.0,
            train_env_ids=task_ids,
            num_train_envs_per_iter=num_tasks,
            num_train_responses_per_env=num_episodes // num_tasks,
            max_episode_steps=max_episode_steps,
            num_jobs=num_jobs,
            temperature=1.2,  # Match collect_data_for_imitation.py
            max_new_tokens=4096,
        ),
    )
    
    # Start vLLM server
    from utils.vllm import start_vllm
    
    print("🚀 Starting vLLM server...")
    current_model_paths = {
        "agent": model_path,
        "attacker": attacker_model_path if use_attacker else model_path,
    }
    
    vllm_model_arguments, vllm_server_args_list = start_vllm(
        base_config,
        current_model_paths,
        on_policy_role="agent",  # Only agent is on-policy
        current_iter=1,
    )
    
    # Prompt flags
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
            use_screenshot=True,  # Need screenshots
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
    
    # Build attacker args list if using attacker
    attacker_args_list = None
    if use_attacker and "attacker" in vllm_model_arguments:
        attacker_args_list = VllmModelArgsList(
            model_names=vllm_model_arguments["attacker"]["model_names"],
            base_urls=vllm_model_arguments["attacker"]["urls"],
            max_input_tokens=base_config.attacker.vllm.max_model_len - base_config.data_collection.max_new_tokens - 16,
        )
    
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
        output_path=str(output_file.parent / raw_output_filename),
        on_policy_role="agent",
        use_attacker=use_attacker,  # 🔑 Enable attacker for dual data collection
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
            vision_support=True,
        ),
        agent_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["agent"]["model_names"],
            base_urls=vllm_model_arguments["agent"]["urls"],
            max_input_tokens=base_config.agent.vllm.max_model_len - base_config.data_collection.max_new_tokens - 16,
        ),
        attacker_args_list=attacker_args_list,  # Enable attacker if configured
        env_args=EnvArgs(
            task_name="placeholder",
            max_steps=max_episode_steps,
        ),
        attacker_privilege=AttackerPrivilegeObsFlags(
            attacker_uses_web_agent_history=False,
            attacker_uses_web_agent_action_history=False,
            attacker_uses_web_agent_think_history=False,
            attacker_uses_web_agent_memory=False,
        ),
        prompt_flags=prompt_flags,
        enable_agent_vision=True,
        # 🔑 KEY: Agent always sees CLEAN data to generate correct golden actions
        # Attacker still generates attacks (saved in attacked_obs), but agent is not affected
        execute_text_injection=False,  # Agent sees CLEAN AXTree
        execute_image_attacks=False,   # Agent sees CLEAN screenshot
        max_image_size=base_config.data_collection.max_image_size,
        save_raw_observations=True,  # 🔑 Enable for enhanced SFT pipeline
    )
    
    # Collect episodes
    collect_mode = "dual (clean + attacked)" if use_attacker else "clean"
    print(f"\n📊 Collecting {collect_mode} episodes...")
    from tools.multimodal_collect_episode import collect_episodes
    collect_episodes(collect_args, iteration=1)
    
    # Load raw episodes and filter for successful ones
    raw_path = output_file.parent / raw_output_filename
    print(f"\n📂 Loading raw episodes from {raw_path}...")
    
    raw_episodes = validate_training_data_file(raw_path)
    
    print(f"   Total raw episodes: {len(raw_episodes)}")
    
    # Detect model family for format-aware processing
    model_family = get_model_family(model_path)
    print(f"   🔍 Detected model family: {model_family}")
    
    # Extract golden steps from successful episodes
    golden_steps = []
    successful_episodes = 0
    failed_episodes = 0
    
    for ep_idx, episode in enumerate(raw_episodes):
        # Skip episodes with errors
        if episode.get('error'):
            failed_episodes += 1
            continue
        
        # Check if episode was successful (agent got positive reward)
        history = episode.get('history', [])
        if not history:
            failed_episodes += 1
            continue
        
        # Find final reward
        final_reward = 0
        for step in reversed(history):
            if step.get('terminated') or step.get('truncated'):
                final_reward = step.get('agent_reward', 0)
                break
        
        if final_reward <= 0:
            failed_episodes += 1
            continue
        
        # This is a successful episode - extract steps
        successful_episodes += 1
        task_name = episode.get('task_name', 'unknown')
        
        # Get task instruction from episode metadata (set at the start of collection)
        episode_task_instruction = episode.get('goal', '')
        
        for step_idx, step in enumerate(history):
            # =========================================================================
            # Read observation data from step_data
            # multimodal_collect_episode.py records these paired observations.
            # =========================================================================
            
            # Extract clean observation data
            clean_obs_data = step.get('clean_obs', {})
            clean_screenshot_b64 = clean_obs_data.get('screenshot_b64', '') if clean_obs_data else ''
            clean_axtree = clean_obs_data.get('axtree_txt', '') if clean_obs_data else ''
            
            # Extract attacked observation data (if available)
            attacked_obs_data = step.get('attacked_obs', {})
            attacked_screenshot_b64 = None
            attacked_axtree = None
            injection_dict = None
            
            if attacked_obs_data:
                attacked_screenshot_b64 = attacked_obs_data.get('screenshot_b64', '')
                attacked_axtree = attacked_obs_data.get('axtree_txt', '')
                injection_dict = step.get('injection_dict')
            
            # Get golden action
            golden_action = step.get('agent_action', '')
            
            # Try to extract think from agent response
            # Use model-aware extraction to handle different formats (Gemma vs Qwen)
            agent_think = None
            agent_messages = step.get('agent_messages', [])
            response_msg = agent_messages[-1] if agent_messages else {}
            response_content = response_msg.get('content', '')
            if isinstance(response_content, str):
                # Use model-aware think extraction
                agent_think = extract_think_content(response_content, model_family=model_family)
            
            # 🔑 Extract token IDs from agent_training_info
            agent_training_info = step.get('agent_training_info', {})
            prompt_token_ids = agent_training_info.get('prompt_token_ids', [])
            response_token_ids = agent_training_info.get('response_token_ids', [])
            
            # 🔑 FILTER: Skip samples with image tokens in response
            IMAGE_TOKEN_IDS = [262144, 151655, 151656, 151657]
            if any(tok in response_token_ids for tok in IMAGE_TOKEN_IDS):
                num_image_tokens = sum(response_token_ids.count(tok) for tok in IMAGE_TOKEN_IDS)
                logging.warning(f"⚠️ SKIPPING golden step (episode={ep_idx}, step={step_idx}) - "
                               f"response contains {num_image_tokens} image tokens (corrupted data)")
                continue
            
            # Skip if we don't have essential data
            if not golden_action:
                continue
            
            # Skip steps without clean observation data (fallback warning)
            if not clean_screenshot_b64 and not clean_axtree:
                logging.warning(f"Episode {ep_idx} Step {step_idx}: No clean observation data found, skipping")
                continue
            
            golden_step = GoldenStep(
                episode_idx=ep_idx,
                step_idx=step_idx,
                task_name=task_name,
                task_instruction=episode_task_instruction,
                clean_screenshot_b64=clean_screenshot_b64 or "",
                clean_axtree=clean_axtree,
                golden_action=golden_action,
                agent_think=agent_think,
                # 🔑 Complete agent messages for Stage 4 format consistency
                agent_messages=agent_messages if agent_messages else None,
                # 🔑 Token IDs from vLLM for training consistency
                prompt_token_ids=prompt_token_ids if prompt_token_ids else None,
                response_token_ids=response_token_ids if response_token_ids else None,
                # Attacked data (may be None if use_attacker=False)
                attacked_screenshot_b64=attacked_screenshot_b64,
                attacked_axtree=attacked_axtree,
                injection_dict=injection_dict,
            )
            golden_steps.append(golden_step.to_dict())
    
    # Count steps with attacked data
    steps_with_attacks = sum(1 for step in golden_steps if step.get('attacked_screenshot_b64'))
    
    print(f"\n📊 Extraction Results:")
    print(f"   Successful episodes: {successful_episodes}")
    print(f"   Failed episodes: {failed_episodes}")
    print(f"   Total golden steps: {len(golden_steps)}")
    if use_attacker:
        print(f"   Steps with attacked data: {steps_with_attacks}")
    
    # Save golden trajectories
    output_data = {
        "metadata": {
            "stage": "golden_trajectory_collection",
            "model_path": model_path,
            "attacker_model_path": attacker_model_path if use_attacker else None,
            "use_attacker": use_attacker,
            "timestamp": datetime.now().isoformat(),
            "num_successful_episodes": successful_episodes,
            "num_failed_episodes": failed_episodes,
            "num_golden_steps": len(golden_steps),
            "num_steps_with_attacks": steps_with_attacks,
        },
        "golden_steps": golden_steps,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Golden trajectories saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(
        description="Stage 1: Collect Golden Trajectories with optional attacker for dual data"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "dmast_ckpt/pretrained/gemma-3-27b-it"),
        help="Path to teacher model (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--attacker-model-path",
        type=str,
        default=None,
        help="Path to attacker model (default: same as teacher model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/stage1_golden_trajectories.json"),
        help="Output path for golden trajectories",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=4000,
        help="Nominal episodes per mode, rounded down per task (default: 4000)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=len(MINIWOB_TRAIN_TASKS),
        help="Number of shared training tasks (1-42; default: 42); requires --no-all-tasks",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=64,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=64,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=300,
        help="Timeout per job in seconds",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--use-attacker",
        action="store_true",
        default=True,
        help="Enable attacker for dual data collection (clean + attacked). Default: True",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        default=False,
        help="Disable attacker, collect clean data only",
    )
    parser.add_argument(
        "--all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all registered non-held-out MiniWoB++ tasks (default); --no-all-tasks enables --num-tasks",
    )
    
    args = parser.parse_args()
    
    # Determine whether to use attacker (--clean-only takes precedence)
    use_attacker = not args.clean_only
    
    collect_golden_trajectories(
        model_path=args.model_path,
        output_path=args.output,
        num_episodes=args.num_episodes,
        num_tasks=args.num_tasks,
        num_jobs=args.num_jobs,
        save_every_n_episodes=args.save_every,
        job_timeout=args.job_timeout,
        max_episode_steps=args.max_steps,
        attacker_model_path=args.attacker_model_path,
        use_attacker=use_attacker,
        all_tasks=args.all_tasks,
    )


if __name__ == "__main__":
    main()
