#!/usr/bin/env python3
"""
Multimodal Adversarial Evaluation Script

This script evaluates different combinations of agent and attacker models
by collecting episodes with HTML injection attacks.

Key features:
- Reuses training's vLLM server logic (automatic 1 or 2 servers)
- Uses HTML injection for attacks (no GPU image editor needed)
- Tests multiple (agent, attacker) combinations
- Collects episodes with HTML injection modifying both screenshot and AXTree
- Pure 1v1 evaluation (population training disabled)

Important design decision:
- train_use_population is set to False for evaluation
- This ensures we test specific agent vs attacker combinations (1v1)
- Not on-policy vs population (which is used in training data collection)
"""

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import bgym
from browsergym.experiments.loop import EnvArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp

from agents.llm.vllm_model import VllmModelArgs
from configs.train_adversarial_args import TrainAdversarialConfig, AgentConfig
from configs.collect_episode_args import CollectEpisodeArgs, MultimodalCollectEpisodeArgs, VllmModelArgsList, AttackerPrivilegeObsFlags
from utils.bgym import setup_bgym_env_variables
from utils.vllm import start_vllm, sleep_vllm, unload_lora_modules
from tools.multimodal_collect_episode import collect_episodes


# ================================================================================================
# Configuration
# ================================================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    # Model paths to evaluate
    agent_model_paths: List[str]
    attacker_model_paths: List[str]
    
    # Base configuration (will be loaded from training config)
    base_config_path: str
    
    # Evaluation parameters
    eval_env_ids: List[str] = None  # None means use eval_env_ids from base_config, will be overwritten by loaded config if not None
    num_eval_episodes_per_env: int = 5
    num_jobs: int = 8
    
    # Output
    output_dir: str = "./eval_results"
    exp_name: str = "multimodal_adversarial_eval"
    
    # Data collection settings
    seed: int = 42
    max_episode_steps: int = 15
    temperature: float = 0.0  # Greedy for evaluation
    
    # Enable both text and image attacks
    enable_multimodal: bool = True
    enable_agent_vision: bool = True
    enable_attacker_image_attacks: bool = True
    
    # Evaluation mode: collect with attacker only, or both with/without attacker
    eval_attacker_only: bool = True  # If True, only collect with use_attacker=True (half the episodes)
    
    # Timeout settings
    job_timeout: int = 0  # 0 = no job-level timeout (only env_step_timeout applies)

    # Agent prompt-defense controls:
    #   defense_level: 0 = none, 1 = enhanced, 2 = maximum (paper's "Prompt Defense" baseline)
    #   system_prompt_addendum_override: arbitrary text appended to the system prompt instead
    #     of the hardcoded `security_rules_addendum` for custom prompt comparisons.
    defense_level: int = 0
    system_prompt_addendum_override: str | None = None
    # Zero-shot transfer to a different attack objective.
    #   "leakage"        — paper default (extract sensitive info)
    #   "misinformation" — attacker tries to plant a per-task wrong value
    attack_goal: str = "leakage"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multimodal adversarial models")
    
    # Required: Base configuration from training
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Path to training config (e.g., ckpt/.../config.json)"
    )
    
    # Model selection
    parser.add_argument(
        "--agent-models",
        type=str,
        nargs="+",
        required=True,
        help="Agent model paths to evaluate (e.g., base_model_path ckpt/.../agent/iter_5)"
    )
    
    parser.add_argument(
        "--attacker-models",
        type=str,
        nargs="+",
        default=None,
        help="Attacker model paths to evaluate (optional, omit for agent-only evaluation)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--eval-env-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific env IDs to evaluate (default: use eval_env_ids from base_config)"
    )
    
    parser.add_argument(
        "--num-episodes-per-env",
        type=int,
        default=4,
        help="Number of episodes per environment"
    )
    
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs for episode collection"
    )
    
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=5,
        help="Maximum steps per episode"
    )
    
    # NOTE: For adversarial evaluation, pass this flag to report ASR over attacked
    # episodes only (as in the paper's main evaluation commands). Without it, collection
    # includes both clean and attacked episodes, and the summary uses their combined
    # count as the ASR denominator. Agent-only evaluation is unaffected by this flag.
    parser.add_argument(
        "--eval-attacker-only",
        action="store_true",
        help="Only collect episodes with attacker (use_attacker=True). If not set, collects both with and without attacker (2x episodes)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generate with timestamp)"
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=0,
        help="Timeout per episode job in seconds. 0 = no timeout (only env_step_timeout applies). "
             "Set to 0 to avoid timeouts from vLLM queue waiting. Default: 0"
    )

    # Prompt-defense flags
    parser.add_argument(
        "--defense-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Agent defense instruction level: "
             "0=basic (default, no security hardening), "
             "1=enhanced (detailed attack-pattern warnings), "
             "2=maximum (full THREAT MODEL + DEFENSE PROTOCOL — paper Prompt Defense row)"
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Path to a text file appended to the agent system prompt instead of "
             "the hardcoded security_rules_addendum. Takes precedence over "
             "--defense-level for the addendum (the user-info defense block is "
             "still controlled by --defense-level)."
    )
    parser.add_argument(
        "--attack-goal",
        type=str,
        default="leakage",
        choices=["leakage", "misinformation"],
        help="Attacker objective: 'leakage' (default) exfiltrates user PII; "
             "'misinformation' plants a per-task wrong value from "
             "configs/misinformation_wrong_values.py (Table 4)."
    )

    return parser.parse_args()


def _filter_dataclass_fields(cls, d: dict) -> dict:
    """Drop dict keys that aren't init-able fields of a dataclass.

    Saved configs can include fields outside the selected dataclass schema.
    Filter those keys before construction to avoid unexpected-keyword errors;
    the config loader logs any excluded fields.
    """
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
    extra = set(d.keys()) - valid_keys
    if extra:
        import logging
        logging.getLogger(__name__).debug(
            f"[load_base_config] dropping unknown {cls.__name__} keys: {sorted(extra)}"
        )
    return {k: v for k, v in d.items() if k in valid_keys}


def load_base_config(config_path: str) -> TrainAdversarialConfig:
    """Load base configuration from training"""
    from configs.train_adversarial_args import (
        AgentConfig, LoraConfig, VLLMConfig, RLConfig,
        SPAGConfig, PPOConfig, GRPOConfig,
        DataCollectionConfig, ImitationConfig, EnhancedSFTConfig
    )
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # ============================================================================
    # Save ALL original values that will be overwritten by __post_init__
    # This ensures we restore the exact configuration from the saved JSON
    # ============================================================================
    
    # Top-level TrainAdversarialConfig fields
    original_values = {
        'exp_name': config_dict.get('exp_name'),
        'local_ckpt_base_path': config_dict.get('local_ckpt_base_path'),
        'cloud_ckpt_base_path': config_dict.get('cloud_ckpt_base_path'),
        'data_base_path': config_dict.get('data_base_path'),
        'exp_local_ckpt_path': config_dict.get('exp_local_ckpt_path'),
        'exp_cloud_ckpt_path': config_dict.get('exp_cloud_ckpt_path'),
        'exp_data_path': config_dict.get('exp_data_path'),
    }
    
    # Imitation config fields
    imitation_dict = config_dict.get('imitation', {})
    if isinstance(imitation_dict, dict):
        original_values['imitation'] = {
            'ckpt_base_path': imitation_dict.get('ckpt_base_path'),
            'data_base_path': imitation_dict.get('data_base_path'),
        }
    
    # Agent and attacker RL paths and module_to_optimizer_save_path
    original_values['agent_configs'] = {}
    for agent_key in ['agent', 'attacker']:
        if agent_key in config_dict and isinstance(config_dict[agent_key], dict):
            rl_dict = config_dict[agent_key].get('rl', {})
            if isinstance(rl_dict, dict):
                original_values['agent_configs'][agent_key] = {
                    'local_ckpt_base_path': rl_dict.get('local_ckpt_base_path'),
                    'cloud_ckpt_base_path': rl_dict.get('cloud_ckpt_base_path'),
                    'progress_file_path': rl_dict.get('progress_file_path'),
                    'module_to_optimizer_save_path': rl_dict.get('module_to_optimizer_save_path', {}),
                }
    
    # Filter out fields with init=False (these are computed in __post_init__)
    # Top-level TrainAdversarialConfig init=False fields
    config_dict.pop('exp_local_ckpt_path', None)
    config_dict.pop('exp_cloud_ckpt_path', None)
    config_dict.pop('exp_data_path', None)
    config_dict.pop('rl', None)  # RLConfig at top level, computed from agent.rl
    
    # Convert nested dicts to dataclass objects
    # AgentConfig conversion (agent and attacker)
    for agent_key in ['agent', 'attacker']:
        if agent_key in config_dict and isinstance(config_dict[agent_key], dict):
            agent_dict = config_dict[agent_key]
            
            # Remove init=False fields
            agent_dict.pop('gradient_accumulation_steps', None)
            agent_dict.pop('model_name', None)
            
            # Convert LoraConfig
            if 'lora' in agent_dict and isinstance(agent_dict['lora'], dict):
                agent_dict['lora'] = LoraConfig(**_filter_dataclass_fields(LoraConfig, agent_dict['lora']))

            # Convert VLLMConfig
            if 'vllm' in agent_dict and isinstance(agent_dict['vllm'], dict):
                agent_dict['vllm'] = VLLMConfig(**_filter_dataclass_fields(VLLMConfig, agent_dict['vllm']))

            # Convert RLConfig
            if 'rl' in agent_dict and isinstance(agent_dict['rl'], dict):
                rl_dict = agent_dict['rl']

                # Remove init=False fields
                rl_dict.pop('local_ckpt_base_path', None)
                rl_dict.pop('cloud_ckpt_base_path', None)
                rl_dict.pop('progress_file_path', None)

                # Convert nested training configs
                if 'spag' in rl_dict and isinstance(rl_dict['spag'], dict):
                    rl_dict['spag'] = SPAGConfig(**_filter_dataclass_fields(SPAGConfig, rl_dict['spag']))
                if 'ppo' in rl_dict and isinstance(rl_dict['ppo'], dict):
                    rl_dict['ppo'] = PPOConfig(**_filter_dataclass_fields(PPOConfig, rl_dict['ppo']))
                if 'grpo' in rl_dict and isinstance(rl_dict['grpo'], dict):
                    rl_dict['grpo'] = GRPOConfig(**_filter_dataclass_fields(GRPOConfig, rl_dict['grpo']))

                agent_dict['rl'] = RLConfig(**_filter_dataclass_fields(RLConfig, rl_dict))

            # Convert AgentConfig
            config_dict[agent_key] = AgentConfig(**_filter_dataclass_fields(AgentConfig, agent_dict))

    # Convert DataCollectionConfig
    if 'data_collection' in config_dict and isinstance(config_dict['data_collection'], dict):
        config_dict['data_collection'] = DataCollectionConfig(
            **_filter_dataclass_fields(DataCollectionConfig, config_dict['data_collection'])
        )

    # Convert ImitationConfig
    if 'imitation' in config_dict and isinstance(config_dict['imitation'], dict):
        imitation_dict = config_dict['imitation']
        # Remove init=False fields
        imitation_dict.pop('data_base_path', None)
        imitation_dict.pop('ckpt_base_path', None)
        config_dict['imitation'] = ImitationConfig(
            **_filter_dataclass_fields(ImitationConfig, imitation_dict)
        )

    # Convert EnhancedSFTConfig
    if 'enhanced_sft' in config_dict and isinstance(config_dict['enhanced_sft'], dict):
        enhanced_sft_dict = config_dict['enhanced_sft']
        # Remove init=False fields
        enhanced_sft_dict.pop('data_base_path', None)
        enhanced_sft_dict.pop('ckpt_base_path', None)
        config_dict['enhanced_sft'] = EnhancedSFTConfig(
            **_filter_dataclass_fields(EnhancedSFTConfig, enhanced_sft_dict)
        )

    config = TrainAdversarialConfig(
        **_filter_dataclass_fields(TrainAdversarialConfig, config_dict)
    )
    
    # ============================================================================
    # CRITICAL: Restore ALL original values that were overwritten by __post_init__
    # This is the same pattern used in test_multimodal_training.py (line 127-146)
    # ============================================================================
    
    # Restore top-level paths
    if original_values['exp_name'] is not None:
        config.exp_name = original_values['exp_name']
    if original_values['local_ckpt_base_path'] is not None:
        config.local_ckpt_base_path = original_values['local_ckpt_base_path']
    if original_values['cloud_ckpt_base_path'] is not None:
        config.cloud_ckpt_base_path = original_values['cloud_ckpt_base_path']
    if original_values['data_base_path'] is not None:
        config.data_base_path = original_values['data_base_path']
    if original_values['exp_local_ckpt_path'] is not None:
        config.exp_local_ckpt_path = original_values['exp_local_ckpt_path']
    if original_values['exp_cloud_ckpt_path'] is not None:
        config.exp_cloud_ckpt_path = original_values['exp_cloud_ckpt_path']
    if original_values['exp_data_path'] is not None:
        config.exp_data_path = original_values['exp_data_path']
    
    # Restore imitation paths
    if 'imitation' in original_values:
        imitation_paths = original_values['imitation']
        if imitation_paths['ckpt_base_path'] is not None:
            config.imitation.ckpt_base_path = imitation_paths['ckpt_base_path']
        if imitation_paths['data_base_path'] is not None:
            config.imitation.data_base_path = imitation_paths['data_base_path']
    
    # Restore agent and attacker RL paths
    for agent_key in ['agent', 'attacker']:
        if agent_key in original_values['agent_configs']:
            paths = original_values['agent_configs'][agent_key]
            agent_config = getattr(config, agent_key)
            
            if paths['local_ckpt_base_path'] is not None:
                agent_config.rl.local_ckpt_base_path = paths['local_ckpt_base_path']
            if paths['cloud_ckpt_base_path'] is not None:
                agent_config.rl.cloud_ckpt_base_path = paths['cloud_ckpt_base_path']
            if paths['progress_file_path'] is not None:
                agent_config.rl.progress_file_path = paths['progress_file_path']
            if paths['module_to_optimizer_save_path']:
                agent_config.rl.module_to_optimizer_save_path = paths['module_to_optimizer_save_path']
    
    print(f"✅ Loaded base config from {config_path}")
    print(f"   📁 Checkpoint directory: {config.exp_local_ckpt_path}")
    return config


def create_eval_combinations(
    agent_models: List[str],
    attacker_models: List[str]
) -> List[Tuple[str, str, str]]:
    """
    Create all combinations of (agent_model, attacker_model, combo_name)
    
    Returns:
        List of tuples: (agent_path, attacker_path, combo_name)
    """
    combinations = []
    
    for agent_path in agent_models:
        for attacker_path in attacker_models:
            # Create readable names
            agent_name = Path(agent_path).name
            if "iter_" in agent_name:
                agent_label = agent_name
            elif "enhanced_sft" in agent_path:
                agent_label = "enhanced_sft"
            elif "imitation" in agent_path:
                agent_label = "imitation"
            else:
                agent_label = "base"
            
            attacker_name = Path(attacker_path).name
            if "27b" in attacker_name:
                attacker_label = "27b"
            elif "iter_" in attacker_name:
                attacker_label = attacker_name
            elif "enhanced_sft" in attacker_path:
                attacker_label = "enhanced_sft"
            elif "imitation" in attacker_path:
                attacker_label = "imitation"
            else:
                attacker_label = "base"
            
            combo_name = f"agent_{agent_label}_vs_attacker_{attacker_label}"
            combinations.append((agent_path, attacker_path, combo_name))
    
    print(f"\n📊 Created {len(combinations)} evaluation combinations:")
    for agent_p, attacker_p, name in combinations:
        print(f"   • {name}")
        print(f"     Agent:    {agent_p}")
        print(f"     Attacker: {attacker_p}")
    
    return combinations


def print_separator(message: str = ""):
    """Print a separator line"""
    if message:
        print(f"\n{'='*80}")
        print(f"{message}")
        print(f"{'='*80}\n")
    else:
        print(f"{'='*80}")


# ================================================================================================
# Main Evaluation Logic
# ================================================================================================

def evaluate_model_combination(
    base_config: TrainAdversarialConfig,
    agent_model_path: str,
    attacker_model_path: str,
    combo_name: str,
    output_dir: Path,
    eval_config: EvalConfig,
    is_agent_only: bool = False,
) -> Dict:
    """
    Evaluate a single agent (agent-only mode) or (agent, attacker) combination (adversarial mode)
    
    This function handles two modes:
    
    1. Agent-only mode (is_agent_only=True):
       - Sets use_attacker=False
       - Evaluates agent on clean environments (no attacker)
       - Uses eval_env_ids (eval + valid combined)
       - No Ray image server needed
    
    2. Adversarial mode (is_agent_only=False):
       - Sets use_attacker=True
       - Evaluates agent vs attacker with text and image attacks
       - Uses eval_env_ids (eval + valid combined)
       - Starts Ray image server for image attacks
    
    Note: Population training is disabled to ensure pure 1v1 testing.
    In both modes, eval_env_ids = eval_env_ids + valid_env_ids from base config.
    """
    print_separator(f"🎯 Evaluating: {combo_name}")
    
    # ================================================================================================
    # Step 0: Check if this combination is already completed
    # ================================================================================================
    
    rollout_data_path = output_dir / f"{combo_name}_episodes.json"
    
    if rollout_data_path.exists():
        print(f"✅ Found existing results at {rollout_data_path}")
        print(f"   Skipping evaluation for {combo_name}")
        
        # Load and return existing results
        with open(rollout_data_path, 'r') as f:
            episodes = json.load(f)
        
        # Calculate statistics
        total_episodes = len(episodes)
        agent_successes = sum(1 for ep in episodes if ep.get("reward_summary", {}).get("agent_success", False))
        attacker_successes = sum(1 for ep in episodes if ep.get("reward_summary", {}).get("attacker_success", False))
        agent_success_rate = agent_successes / total_episodes if total_episodes > 0 else 0.0
        attacker_success_rate = attacker_successes / total_episodes if total_episodes > 0 else 0.0
        avg_steps = sum(len(ep.get("history", [])) for ep in episodes) / total_episodes if total_episodes > 0 else 0.0
        avg_agent_reward = sum(ep.get("reward_summary", {}).get("agent_total_reward", 0) for ep in episodes) / total_episodes if total_episodes > 0 else 0.0
        
        return {
            "combo_name": combo_name,
            "agent_model": agent_model_path,
            "attacker_model": attacker_model_path,
            "total_episodes": total_episodes,
            "agent_successes": agent_successes,
            "agent_success_rate": agent_success_rate,
            "attacker_successes": attacker_successes,
            "attacker_success_rate": attacker_success_rate,
            "avg_steps": avg_steps,
            "avg_agent_reward": avg_agent_reward,
            "collection_time": 0.0,
            "data_path": str(rollout_data_path),
            "skipped": True,
        }
    
    # ================================================================================================
    # Step 1: Prepare configuration
    # ================================================================================================
    
    # Create a modified config for this evaluation
    # Use deepcopy to properly copy all nested objects
    eval_run_config = copy.deepcopy(base_config)

    # Architecture/config queries (e.g. `get_tensor_parallel_size` in start_vllm)
    # use `config.agent.model_path` / `config.attacker.model_path` to call
    # `AutoConfig.from_pretrained(...)`. If the saved config.json was produced
    # on a different machine, those paths may not exist locally and the load
    # fails. Since the eval-time agent (`agent_model_path`) shares the same
    # base architecture as the original training base model, it is safe to
    # point these architecture-query paths at the locally-available eval ckpt.
    import os as _os
    if not _os.path.isdir(eval_run_config.agent.model_path):
        print(f"⚠️ config.agent.model_path '{eval_run_config.agent.model_path}' not found locally; "
              f"overriding with --agent-models path '{agent_model_path}' for architecture queries.")
        eval_run_config.agent.model_path = agent_model_path
    if not is_agent_only and not _os.path.isdir(eval_run_config.attacker.model_path):
        print(f"⚠️ config.attacker.model_path '{eval_run_config.attacker.model_path}' not found locally; "
              f"overriding with --attacker-models path '{attacker_model_path}' for architecture queries.")
        eval_run_config.attacker.model_path = attacker_model_path
    
    # Handle agent-only vs adversarial mode
    if is_agent_only:
        # 🎯 Agent-only mode: disable attacker
        eval_run_config.use_attacker = False
        eval_run_config.data_collection.enable_multimodal = eval_config.enable_agent_vision  # Only enable if agent needs vision
        eval_run_config.data_collection.enable_agent_vision = eval_config.enable_agent_vision
        eval_run_config.data_collection.enable_attacker_image_attacks = False
        eval_run_config.data_collection.execute_image_attacks = False
        
        # Prepare current_model_paths dict
        # Note: start_vllm() expects both keys, but will only use agent when use_attacker=False
        current_model_paths = {
            "agent": agent_model_path,
            "attacker": agent_model_path,  # Dummy path, won't be used since use_attacker=False
        }
    else:
        # 🎯 Adversarial mode: enable attacker and multimodal attacks
        eval_run_config.use_attacker = True
        eval_run_config.data_collection.enable_multimodal = True
        eval_run_config.data_collection.enable_agent_vision = True
        eval_run_config.data_collection.enable_attacker_image_attacks = True
        
        # Prepare current_model_paths dict (agent and attacker)
        # The actual model paths are passed here, NOT in config.*model_path
        current_model_paths = {
            "agent": agent_model_path,
            "attacker": attacker_model_path,
        }
    
    # 🎯 Both modes use the same eval_env_ids (eval + valid combined)
    eval_env_ids = eval_config.eval_env_ids
    mode = "eval"
    
    # 🎯 CRITICAL: Disable population training for evaluation
    # We want pure 1v1 testing, not on-policy vs population
    eval_run_config.agent.train_use_population = False
    eval_run_config.attacker.train_use_population = False
    
    # Set evaluation parameters
    eval_run_config.data_collection.temperature = eval_config.temperature
    eval_run_config.data_collection.max_episode_steps = eval_config.max_episode_steps
    
    # ================================================================================================
    # Step 2: Start vLLM Server(s)
    # ================================================================================================
    
    print_separator(f"🚀 Starting vLLM server(s) for {combo_name}")
    print(f"   Agent:    {agent_model_path}")
    if not is_agent_only:
        print(f"   Attacker: {attacker_model_path}")
    
    # Use the same start_vllm() function as in training
    # It will automatically decide 1 or 2 servers based on model paths and on_policy_role
    on_policy_role = "agent" if is_agent_only else "both"
    
    vllm_model_arguments, vllm_server_args_list = start_vllm(
        config=eval_run_config,
        current_model_paths=current_model_paths,
        on_policy_role=on_policy_role,
        current_iter=0,  # Dummy iteration number for eval
    )
    
    print(f"✅ Started {len(vllm_server_args_list)} vLLM server(s)")
    
    # ================================================================================================
    # Step 3: HTML Injection Attack Mode (no GPU image editor needed)
    # ================================================================================================
    
    if not is_agent_only:
        print_separator("🎨 HTML Injection Attack Mode")
        logging.info(f"   Attacker will generate HTML/CSS injections")
        logging.info(f"   Injections modify DOM directly via JavaScript")
        logging.info(f"   Both screenshot and AXTree are automatically updated")
    else:
        print("⏭️  Running in agent-only mode (no attacker)")
    
    # ================================================================================================
    # Step 4: Collect Episodes
    # ================================================================================================
    
    print_separator(f"📊 Collecting evaluation episodes for {combo_name}")
    
    # Note: rollout_data_path already defined in Step 0
    
    # Prepare prompt flags (same as training)
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
            use_screenshot=eval_run_config.data_collection.enable_agent_vision,  # Based on config
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
    
    # Calculate max_input_tokens
    extra_token_length = 16
    max_input_tokens = eval_run_config.agent.vllm.max_model_len - eval_run_config.data_collection.max_new_tokens - extra_token_length
    
    # Prepare collection args
    collect_episode_args = CollectEpisodeArgs(
        seed=eval_config.seed,
        # Evaluation mode
        mode=mode,
        benchmark=eval_run_config.data_collection.benchmark,
        env_ids=eval_env_ids,
        use_env_ids=True,
        train_prop=0.8,
        valid_prop=0.1,
        eval_prop=0.1,
        num_envs_per_iter=0,  # Use all eval envs
        num_responses_per_env=eval_config.num_eval_episodes_per_env,
        num_jobs=eval_config.num_jobs,
        job_timeout=eval_config.job_timeout,  # 0 = no job-level timeout, only env_step_timeout applies
        fix_seed=False,  # Don't fix seed for evaluation
        output_path=str(rollout_data_path),
        # Agent and attacker params
        # 🎯 For agent-only: use_attacker=False, on_policy_role="agent"
        # 🎯 For adversarial: use_attacker=True, on_policy_role based on eval_attacker_only flag
        on_policy_role=on_policy_role if is_agent_only else ("attacker" if eval_config.eval_attacker_only else "both"), # this affects use_attacker_modes in prepare_collect_kwargs
        use_attacker=not is_agent_only,
        attacker_use_art=eval_run_config.attacker_use_art if not is_agent_only else False,
        collect_responses_with_same_opponent=False,  # Not needed for eval
        default_model_args=VllmModelArgs(
            model_name="placeholder",
            temperature=eval_config.temperature,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=eval_run_config.data_collection.max_new_tokens,
            log_probs=None,  # Disable logprobs for eval - avoids "Out of range float values" vLLM error
            vision_support=eval_run_config.data_collection.enable_agent_vision,
        ),
        agent_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["agent"]["model_names"],
            base_urls=vllm_model_arguments["agent"]["urls"],
            max_input_tokens=max_input_tokens,
        ),
        attacker_args_list=VllmModelArgsList(
            model_names=vllm_model_arguments["attacker"]["model_names"],
            base_urls=vllm_model_arguments["attacker"]["urls"],
            max_input_tokens=max_input_tokens,
        ) if not is_agent_only else None,
        env_args=EnvArgs(
            task_name="placeholder",
            max_steps=eval_config.max_episode_steps,
        ),
        attacker_privilege=AttackerPrivilegeObsFlags(
            attacker_uses_web_agent_history=eval_run_config.data_collection.attacker_uses_web_agent_history if not is_agent_only else False,
            attacker_uses_web_agent_action_history=eval_run_config.data_collection.attacker_uses_web_agent_action_history if not is_agent_only else False,
            attacker_uses_web_agent_think_history=eval_run_config.data_collection.attacker_uses_web_agent_think_history if not is_agent_only else False,
            attacker_uses_web_agent_memory=eval_run_config.data_collection.attacker_uses_web_agent_memory if not is_agent_only else False,
        ),
        prompt_flags=prompt_flags,
    )
    
    # Collect episodes
    start_time = time.time()
    
    # 🎨 Choose between multimodal and text-only collection based on config
    if eval_run_config.data_collection.enable_multimodal:
        print("🎨 Using multimodal episode collection (vision + language)")
        
        # Convert CollectEpisodeArgs to MultimodalCollectEpisodeArgs
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
            job_timeout=collect_episode_args.job_timeout,
            save_every_n_episodes=collect_episode_args.save_every_n_episodes,
            env_step_timeout=collect_episode_args.env_step_timeout,
            fix_seed=collect_episode_args.fix_seed,
            collect_responses_with_same_opponent=collect_episode_args.collect_responses_with_same_opponent,
            on_policy_role=collect_episode_args.on_policy_role,
            use_attacker=collect_episode_args.use_attacker,
            attacker_use_art=collect_episode_args.attacker_use_art,
            default_model_args=collect_episode_args.default_model_args,
            agent_args_list=collect_episode_args.agent_args_list,
            attacker_args_list=collect_episode_args.attacker_args_list,
            env_args=collect_episode_args.env_args,
            attacker_privilege=collect_episode_args.attacker_privilege,
            prompt_flags=collect_episode_args.prompt_flags,
            output_path=collect_episode_args.output_path,
            
            # Add multimodal-specific args from config (HTML injection)
            enable_agent_vision=eval_run_config.data_collection.enable_agent_vision,
            execute_text_injection=eval_run_config.data_collection.execute_text_injection,
            execute_image_attacks=eval_run_config.data_collection.execute_image_attacks,
            max_image_size=eval_run_config.data_collection.max_image_size,

            # Prompt-based defense
            defense_level=eval_config.defense_level,
            system_prompt_addendum_override=eval_config.system_prompt_addendum_override,

            # Attack-goal transfer
            attack_goal=eval_config.attack_goal,
        )
        
        collect_episodes(multimodal_args, iteration=999)  # Dummy iteration
    else:
        # Fall back to text-only collection
        # NOTE: This path is used for agent-only evaluation with vision disabled.
        # Its episodes lack reward_summary, so this script's success-rate aggregation
        # (including cached results) reports zero even when tasks succeed.
        # The paper's multimodal evaluations, including clean runs with vision enabled,
        # use the collector above, which provides reward_summary, and are unaffected.
        if eval_run_config.data_collection.enable_multimodal:
            print("⚠️ Multimodal collection requested but not available, using text-only collection")
        else:
            print("📝 Using text-only episode collection")
        
        collect_episodes(collect_episode_args, iteration=999)  # Dummy iteration
    collection_time = time.time() - start_time
    
    print(f"✅ Episode collection finished in {collection_time:.2f}s")
    print(f"   Saved to: {rollout_data_path}")
    
    # ================================================================================================
    # Step 5: Cleanup
    # ================================================================================================
    
    print_separator("🧹 Cleaning up servers")
    
    # Sleep vLLM servers (but don't kill them for next evaluation)
    for vllm_server_args in vllm_server_args_list:
        url = vllm_server_args["url"]
        api_key = vllm_server_args["api_key"]
        if eval_run_config.agent.lora.use_lora:
            lora_module_names = vllm_server_args["lora_module_names"]
            unload_lora_modules(url, api_key, lora_module_names)
        sleep_vllm(url, api_key)
    
    print("✅ vLLM servers put to sleep")
    
    # ================================================================================================
    # Step 6: Load and analyze results
    # ================================================================================================
    
    print_separator(f"📈 Analyzing results for {combo_name}")
    
    with open(rollout_data_path, 'r') as f:
        episodes = json.load(f)
    
    # Calculate basic statistics
    total_episodes = len(episodes)
    
    # Use the correct field structure: reward_summary.agent_success and reward_summary.attacker_success
    agent_successes = sum(1 for ep in episodes if ep.get("reward_summary", {}).get("agent_success", False))
    attacker_successes = sum(1 for ep in episodes if ep.get("reward_summary", {}).get("attacker_success", False))
    
    agent_success_rate = agent_successes / total_episodes if total_episodes > 0 else 0.0
    attacker_success_rate = attacker_successes / total_episodes if total_episodes > 0 else 0.0
    
    # Calculate average steps from history length
    avg_steps = sum(len(ep.get("history", [])) for ep in episodes) / total_episodes if total_episodes > 0 else 0.0
    
    # Calculate average rewards
    avg_agent_reward = sum(ep.get("reward_summary", {}).get("agent_total_reward", 0) for ep in episodes) / total_episodes if total_episodes > 0 else 0.0
    
    results = {
        "combo_name": combo_name,
        "agent_model": agent_model_path,
        "attacker_model": attacker_model_path,
        "total_episodes": total_episodes,
        "agent_successes": agent_successes,
        "agent_success_rate": agent_success_rate,
        "attacker_successes": attacker_successes,
        "attacker_success_rate": attacker_success_rate,
        "avg_steps": avg_steps,
        "avg_agent_reward": avg_agent_reward,
        "collection_time": collection_time,
        "data_path": str(rollout_data_path),
    }
    
    print(f"   Total episodes: {total_episodes}")
    print(f"   🤖 Agent success rate: {agent_success_rate:.2%} ({agent_successes}/{total_episodes})")
    print(f"   🎯 Attacker success rate: {attacker_success_rate:.2%} ({attacker_successes}/{total_episodes})")
    print(f"   📊 Avg steps: {avg_steps:.1f}")
    print(f"   💰 Avg agent reward: {avg_agent_reward:.2f}")
    
    return results


# ================================================================================================
# Main Function
# ================================================================================================

def main():
    args = parse_args()
    
    # Setup environment variables (MINIWOB_URL, etc.)
    setup_bgym_env_variables()
    
    # Set seed
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ================================================================================================
    # Determine evaluation mode: Agent-only or Adversarial
    # ================================================================================================
    
    is_agent_only = args.attacker_models is None or len(args.attacker_models) == 0
    
    if is_agent_only:
        print_separator("🎯 Agent-Only Evaluation (No Attacker)")
        print("   Mode: Evaluating agent performance on clean environments")
    else:
        print_separator("🎯 Adversarial Evaluation (Agent vs Attacker)")
        print("   Mode: Evaluating agent vs attacker combinations")
    
    # ================================================================================================
    # Step 1: Load base configuration
    # ================================================================================================
    
    base_config = load_base_config(args.base_config)
    
    # ================================================================================================
    # Step 2: Determine evaluation combinations
    # ================================================================================================
    
    if is_agent_only:
        # Agent-only mode: evaluate each agent separately
        print(f"📊 Evaluating {len(args.agent_models)} agent model(s)")
        
        # Create "combinations" - one per agent
        combinations = []
        for agent_path in args.agent_models:
            agent_name = Path(agent_path).name
            if "imitation" in agent_path:
                agent_label = "imitation"
            elif "iter_" in agent_name:
                agent_label = agent_name
            else:
                agent_label = "base"
            
            combo_name = f"{agent_label}"
            combinations.append((agent_path, None, combo_name))
        
        print(f"\n📋 Created {len(combinations)} evaluation tasks:")
        for agent_p, _, name in combinations:
            print(f"   • {name}: {agent_p}")
    else:
        # Adversarial mode: evaluate all agent × attacker combinations
        combinations = create_eval_combinations(
            agent_models=args.agent_models,
            attacker_models=args.attacker_models,
        )
    
    # ================================================================================================
    # Step 3: Setup output directory
    # ================================================================================================
    
    # Use checkpoint directory if output_dir is default, otherwise use provided path
    if args.output_dir == "./eval_results":
        # Default: use checkpoint directory (already restored from JSON)
        output_dir = Path(base_config.exp_local_ckpt_path) / "eval_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir} (using checkpoint directory)")
    else:
        # User specified custom path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir} (custom path)")
    
    # Create eval config
    # Always combine eval + valid environments for both agent-only and adversarial modes
    eval_env_ids = args.eval_env_ids
    if eval_env_ids is None:
        # Use combined eval and valid environments from base config
        eval_env_ids = base_config.data_collection.eval_env_ids + base_config.data_collection.valid_env_ids
        print(f"📋 Using eval_env_ids from base config: {len(eval_env_ids)} environments")
        print(f"   Eval envs: {len(base_config.data_collection.eval_env_ids)}")
        print(f"   Valid envs: {len(base_config.data_collection.valid_env_ids)}")
    
    # Resolve --system-prompt-path → string contents for the agent factory.
    system_prompt_addendum_override = None
    if args.system_prompt_path:
        with open(args.system_prompt_path, "r") as _f:
            system_prompt_addendum_override = _f.read()
        print(f"🛡️ Loaded prompt addendum from {args.system_prompt_path} "
              f"({len(system_prompt_addendum_override)} chars)")

    eval_config = EvalConfig(
        agent_model_paths=args.agent_models,
        attacker_model_paths=args.attacker_models if not is_agent_only else [],
        base_config_path=args.base_config,
        eval_env_ids=eval_env_ids,
        num_eval_episodes_per_env=args.num_episodes_per_env,
        num_jobs=args.num_jobs,
        output_dir=str(output_dir),
        exp_name="eval_results",  # Fixed name, no timestamp
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        eval_attacker_only=args.eval_attacker_only if not is_agent_only else True,  # Not relevant for agent-only
        enable_agent_vision=base_config.data_collection.enable_agent_vision if is_agent_only else True,
        job_timeout=args.job_timeout,
        defense_level=args.defense_level,
        system_prompt_addendum_override=system_prompt_addendum_override,
        attack_goal=args.attack_goal,
    )

    # Save eval config
    eval_config_path = output_dir / "eval_config.json"
    with open(eval_config_path, 'w') as f:
        json.dump({
            "agent_models": args.agent_models,
            "attacker_models": args.attacker_models,
            "base_config": args.base_config,
            "eval_env_ids": args.eval_env_ids,
            "num_episodes_per_env": args.num_episodes_per_env,
            "num_jobs": args.num_jobs,
            "max_episode_steps": args.max_episode_steps,
            "seed": args.seed,
            "eval_attacker_only": args.eval_attacker_only,
            "defense_level": args.defense_level,
            "system_prompt_path": args.system_prompt_path,
            "attack_goal": args.attack_goal,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"💾 Saved eval config to {eval_config_path}")
    
    # ================================================================================================
    # Step 4: Evaluate all combinations
    # ================================================================================================
    
    all_results = []
    
    for idx, combination in enumerate(combinations, 1):
        if is_agent_only:
            agent_path, attacker_path, combo_name = combination
        else:
            agent_path, attacker_path, combo_name = combination
        
        print_separator(f"[{idx}/{len(combinations)}] Evaluating {combo_name}")
        
        try:
            results = evaluate_model_combination(
                base_config=base_config,
                agent_model_path=agent_path,
                attacker_model_path=attacker_path,
                combo_name=combo_name,
                output_dir=output_dir,
                eval_config=eval_config,
                is_agent_only=is_agent_only,
            )
            all_results.append(results)
            
        except Exception as e:
            logging.error(f"❌ Error evaluating {combo_name}: {e}")
            print(f"❌ Error evaluating {combo_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Still record the failure
            all_results.append({
                "combo_name": combo_name,
                "agent_model": agent_path,
                "attacker_model": attacker_path,
                "error": str(e),
            })
    
    # ================================================================================================
    # Step 5: Save and print summary
    # ================================================================================================
    
    print_separator("📊 Evaluation Summary")
    
    # Save detailed results
    results_path = output_dir / "results_summary.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"💾 Saved results to {results_path}")
    
    # Print summary table
    print("\n" + "="*130)
    print(f"{'Combination':<45} {'Agent Success':<18} {'Attacker Success':<18} {'Avg Steps':<12} {'Avg Reward':<12} {'Episodes':<10}")
    print("="*130)
    
    for result in all_results:
        if "error" in result:
            print(f"{result['combo_name']:<45} {'ERROR':<18} {'ERROR':<18} {'-':<12} {'-':<12} {'-':<10}")
        else:
            agent_succ = f"{result['agent_success_rate']:.1%} ({result['agent_successes']}/{result['total_episodes']})"
            attacker_succ = f"{result['attacker_success_rate']:.1%} ({result['attacker_successes']}/{result['total_episodes']})"
            avg_steps = f"{result['avg_steps']:.1f}"
            avg_reward = f"{result['avg_agent_reward']:.2f}"
            total = f"{result['total_episodes']}"
            skipped_marker = " [CACHED]" if result.get('skipped', False) else ""
            print(f"{result['combo_name']:<45} {agent_succ:<18} {attacker_succ:<18} {avg_steps:<12} {avg_reward:<12} {total:<10}{skipped_marker}")
    
    print("="*130)
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
