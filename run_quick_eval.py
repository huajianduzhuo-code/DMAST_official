#!/usr/bin/env python3
"""
Quick evaluation launcher

This script provides a convenient way to launch evaluations with predefined settings.
Modify the paths and settings below to match your experiment.

Supported iterations:
- "base": Uses BASE_MODEL
- "imitation": Uses IMITATION_MODEL (if set)
- "iter_X": Uses <EXP_DIR>/rl/iter_X_merged (or role-specific path for single-role training)

Training modes supported by this script:
1. Agent-only: Set USE_ATTACKER=False, IS_AGENT_ONLY_TRAINING=True
   - Only agent is trained, no attacker present during evaluation
   - Evaluates agent performance on clean environments
   
2. Attacker-only: Set USE_ATTACKER=True, IS_ATTACKER_ONLY_TRAINING=True
   - Only attacker is trained, agent is fixed at imitation checkpoint
   - Evaluates how attacker performs against fixed agent
   
3. Co-training/Self-play: Set USE_ATTACKER=True, both training flags=False
   - Both agent and attacker are trained together
   - Evaluates different (agent, attacker) combinations
"""

import subprocess
import sys
from pathlib import Path


# ================================================================================================
# Configuration - MODIFY THESE TO MATCH YOUR SETUP
# ================================================================================================

# Your experiment directory
# UPDATE THIS to your experiment directory
CKPT_ROOT = Path.home() / "dmast_ckpt"
EXP_DIR = CKPT_ROOT / "multimodal_adversarial_html_enhanced_sft_task_focused_qwen8b/grpo_qwen3_vl_8b_instruct_self_play_agent_iterative_attacker_iterative_seed_0"

# Base model path (the original model you started training from)
BASE_MODEL = str(CKPT_ROOT / "pretrained/qwen3-vl-8b-instruct")
# Teacher / eval attacker used when an iteration is labeled "27b"
MODEL_27B = str(CKPT_ROOT / "pretrained/gemma-3-27b-it")

# Imitation model paths (for single-role training, only the trained role needs imitation checkpoint)
# UPDATE THESE to your actual imitation checkpoint paths
AGENT_IMITATION_MODEL = str(CKPT_ROOT / "imitation_qwen8b/qwen3-vl-8b-instruct-both_merged")
ATTACKER_IMITATION_MODEL = str(CKPT_ROOT / "imitation_qwen8b/qwen3-vl-8b-instruct-both_merged")

AGENT_ENHANCED_SFT_TASK_FOCUSED_MODEL = str(CKPT_ROOT / "enhanced_sft_task_focused_qwen8b/qwen3-vl-8b-instruct-both_merged")
AGENT_ENHANCED_SFT_DEFENSE_FOCUSED_MODEL = str(CKPT_ROOT / "enhanced_sft_defense_focused_qwen8b/qwen3-vl-8b-instruct-both_merged")
ATTACKER_ENHANCED_SFT_TASK_FOCUSED_MODEL = str(CKPT_ROOT / "enhanced_sft_task_focused_qwen8b/qwen3-vl-8b-instruct-both_merged")
ATTACKER_ENHANCED_SFT_DEFENSE_FOCUSED_MODEL = str(CKPT_ROOT / "enhanced_sft_defense_focused_qwen8b/qwen3-vl-8b-instruct-both_merged")

# ================================================================================================
# Training mode configuration - SET ONE OF THESE SCENARIOS:
# ================================================================================================

# # Scenario 1: Agent-only training (no attacker)
# USE_ATTACKER = False  # No attacker during evaluation
# IS_AGENT_ONLY_TRAINING = True  # Agent was trained alone
# IS_ATTACKER_ONLY_TRAINING = False

# # Scenario 2: Attacker-only training (agent fixed at imitation)
# USE_ATTACKER = True  # Attacker is present
# IS_AGENT_ONLY_TRAINING = False
# IS_ATTACKER_ONLY_TRAINING = True  # Only attacker was trained

# # Scenario 3: Co-training or self-play (both trained)
USE_ATTACKER = True  # Attacker is present
IS_AGENT_ONLY_TRAINING = False
IS_ATTACKER_ONLY_TRAINING = False  # Both were trained

# ================================================================================================
# Model selection - ADJUST BASED ON YOUR SCENARIO:
# ================================================================================================

# Which iterations to evaluate
# Options: "base", "imitation", "iter_1", "iter_2", etc.

# For AGENT-ONLY training:
AGENT_ITERATIONS = ["base"]
ATTACKER_ITERATIONS = ["base", "enhanced_sft_task_focused", "iter_4", "iter_8"]

# For ATTACKER-ONLY training:
# AGENT_ITERATIONS = ["imitation"]  # Fixed agent at imitation
# ATTACKER_ITERATIONS = ["imitation", "iter_1", "iter_2", "iter_3", "iter_4", "iter_5"]

# For CO-TRAINING (both trained):
# AGENT_ITERATIONS = ["imitation", "iter_1", "iter_2", "iter_3"]
# ATTACKER_ITERATIONS = ["imitation", "iter_1", "iter_2", "iter_3"]

# ================================================================================================
# Evaluation settings
# ================================================================================================

NUM_EPISODES_PER_ENV = 32  # Episodes per environment per model combination
NUM_JOBS = 32              # Number of parallel workers (reduced for 2-server stability)
MAX_EPISODE_STEPS = 5     # Maximum steps per episode
SEED = 42                 # Random seed for reproducibility
JOB_TIMEOUT = 600           # Timeout per episode in seconds. 0 = no job-level timeout (only env_step_timeout applies)
                          # Set to 0 to prevent timeouts from vLLM queue waiting time

# NOTE: All evaluations use eval_env_ids + valid_env_ids from base config
# This is the same for both agent-only and adversarial modes

# Evaluation mode (for adversarial modes only)
EVAL_ATTACKER_ONLY = True  # If True, only collect with attacker (15 tasks × 4 episodes = 60 total)
                            # If False, collect both with/without attacker (15 tasks × 4 episodes × 2 modes = 120 total)

# Output directory (optional - set to None to use default)
# Default: <EXP_DIR>/eval_results
# Custom: Any path you specify
OUTPUT_DIR = None  # Set to a custom path if you want results elsewhere
# Example: OUTPUT_DIR = str(Path.home() / "eval_results/my_evaluation")

# Note: For agent-only training, results will show agent performance across iterations
# For adversarial training, results will include agent × attacker comparison matrix

# ================================================================================================
# Build command
# ================================================================================================

def get_model_path(iteration: str, role: str) -> str:
    """Get model path for a given iteration and role
    
    Args:
        iteration: One of "base", "imitation", or "iter_X"
        role: Either "agent" or "attacker"
    
    Returns:
        Full path to the model
    
    Path structure examples:
    - Agent-only: {EXP_DIR}/rl/agent/iter_X_merged
    - Attacker-only: {EXP_DIR}/rl/attacker/iter_X_merged (attacker), imitation checkpoint (agent)
    - Co-training: {EXP_DIR}/rl/iter_X_merged (shared) or {EXP_DIR}/rl/{role}/iter_X_merged (separate)
    """
    if iteration == "base":
        return BASE_MODEL
    elif iteration == "27b":
        return MODEL_27B
    elif iteration == "enhanced_sft_task_focused":
        if role == "agent":
            return AGENT_ENHANCED_SFT_TASK_FOCUSED_MODEL
        else:
            return ATTACKER_ENHANCED_SFT_TASK_FOCUSED_MODEL
    elif iteration == "enhanced_sft_defense_focused":
        if role == "agent":
            return AGENT_ENHANCED_SFT_DEFENSE_FOCUSED_MODEL
        else:
            return ATTACKER_ENHANCED_SFT_DEFENSE_FOCUSED_MODEL
    elif iteration == "imitation":
        # Use role-specific imitation models
        if role == "agent":
            imitation_model = AGENT_IMITATION_MODEL
        else:  # attacker
            imitation_model = ATTACKER_IMITATION_MODEL
        
        if imitation_model is None:
            print(f"❌ Error: 'imitation' specified but {role.upper()}_IMITATION_MODEL is None")
            print(f"   Please set {role.upper()}_IMITATION_MODEL in the script or remove 'imitation' from iterations")
            sys.exit(1)
        if not Path(imitation_model).exists():
            print(f"❌ Error: {role} imitation model not found at {imitation_model}")
            print(f"   Have you completed imitation training?")
            sys.exit(1)
        return imitation_model
    else:
        # iteration already includes "iter_" prefix (e.g., "iter_5")
        # Path structure depends on training mode
        if IS_ATTACKER_ONLY_TRAINING or IS_AGENT_ONLY_TRAINING:
            # Single-role training: models are in {EXP_DIR}/rl/{role}/iter_X_merged
            return f"{EXP_DIR}/rl/{role}/{iteration}_merged"
        else:
            # Co-training/self-play: models could be in either location
            # Try role-specific first (typical for co-training without self-play)
            role_specific_path = Path(f"{EXP_DIR}/rl/{role}/{iteration}_merged")
            if role_specific_path.exists():
                return str(role_specific_path)
            # Otherwise use shared path (typical for self-play)
            return f"{EXP_DIR}/rl/{iteration}_merged"


def main():
    # Build agent and attacker model lists with role-specific paths
    agent_models = [get_model_path(it, "agent") for it in AGENT_ITERATIONS]
    
    # Only build attacker models if using attacker
    if USE_ATTACKER and ATTACKER_ITERATIONS:
        attacker_models = [get_model_path(it, "attacker") for it in ATTACKER_ITERATIONS]
    else:
        attacker_models = []
    
    # Check if base config exists
    config_path = EXP_DIR / "config.json"
    if not config_path.exists():
        print(f"❌ Error: Config not found at {config_path}")
        print(f"   Please update EXP_DIR in this script to point to your experiment directory")
        sys.exit(1)
    
    # Determine training mode for display
    if not USE_ATTACKER:
        mode_name = "Agent-Only Mode"
        mode_desc = "Agent trained alone, no attacker"
    elif IS_ATTACKER_ONLY_TRAINING:
        mode_name = "Attacker-Only Mode"
        mode_desc = "Attacker trained, agent fixed"
    else:
        mode_name = "Adversarial Mode (Co-training)"
        mode_desc = "Agent and attacker trained together"
    
    # Print summary
    print("=" * 80)
    print(f"🎯 Quick Evaluation Launch - {mode_name}")
    print("=" * 80)
    print(f"\nTraining Mode: {mode_desc}")
    print(f"Experiment: {EXP_DIR}")
    print(f"Config: {config_path}")
    print(f"\n🤖 Agent models ({len(agent_models)}):")
    for iteration, model in zip(AGENT_ITERATIONS, agent_models):
        if iteration == "base":
            label = "Base Model"
        elif iteration == "imitation":
            label = "After Imitation"
        elif iteration == "enhanced_sft":
            label = "After Enhanced SFT"
        else:
            label = f"After {iteration.upper()}"
        print(f"  • {label}: {model}")
    
    if USE_ATTACKER:
        print(f"\n🎯 Attacker models ({len(attacker_models)}):")
        for iteration, model in zip(ATTACKER_ITERATIONS, attacker_models):
            if iteration == "base":
                label = "Base Model"
            elif iteration == "27b":
                label = "27b Model"
            elif iteration == "imitation":
                label = "After Imitation"
            elif iteration == "enhanced_sft":
                label = "After Enhanced SFT"
            else:
                label = f"After {iteration.upper()}"
            print(f"  • {label}: {model}")
    
    print(f"\n📊 Evaluation Settings:")
    if USE_ATTACKER:
        print(f"  Total combinations: {len(agent_models)} × {len(attacker_models)} = {len(agent_models) * len(attacker_models)}")
        print(f"  Mode: Adversarial evaluation (agent vs attacker)")
        if EVAL_ATTACKER_ONLY:
            print(f"  Episodes per combination: ~{NUM_EPISODES_PER_ENV * 10} (10 eval tasks × {NUM_EPISODES_PER_ENV} episodes)")
        else:
            print(f"  Episodes per combination: ~{NUM_EPISODES_PER_ENV * 10 * 2} (10 eval tasks × {NUM_EPISODES_PER_ENV} episodes × 2 modes)")
    else:
        print(f"  Models to evaluate: {len(agent_models)}")
        print(f"  Mode: Clean environment (no attacker)")
        print(f"  Environments: eval + valid (from base config)")
    print(f"  Episodes per env: {NUM_EPISODES_PER_ENV}")
    print(f"  Max steps: {MAX_EPISODE_STEPS}")
    print(f"  Parallel jobs: {NUM_JOBS}")
    
    # Determine output directory
    if OUTPUT_DIR is None:
        output_path = EXP_DIR / ("eval_results_agent_only" if not USE_ATTACKER else "eval_results")
        print(f"\n📁 Results will be saved to: {output_path} (default)")
    else:
        output_path = Path(OUTPUT_DIR)
        print(f"\n📁 Results will be saved to: {output_path} (custom)")
    
    print("=" * 80)
    
    # Build command - use unified evaluation script for both modes
    cmd = [
        "python", "eval_multimodal_adversarial.py",
        "--base-config", str(config_path),
        "--agent-models", *agent_models,
        "--num-episodes-per-env", str(NUM_EPISODES_PER_ENV),
        "--num-jobs", str(NUM_JOBS),
        "--max-episode-steps", str(MAX_EPISODE_STEPS),
        "--seed", str(SEED),
        "--job-timeout", str(JOB_TIMEOUT),
    ]
    
    if USE_ATTACKER:
        # Adversarial mode: add attacker models
        cmd.extend(["--attacker-models", *attacker_models])
        
        # Add eval-attacker-only flag if enabled
        if EVAL_ATTACKER_ONLY:
            cmd.append("--eval-attacker-only")
    
    # Add output-dir - always specify to ensure results go to bucket path
    if OUTPUT_DIR is not None:
        cmd.extend(["--output-dir", str(OUTPUT_DIR)])
    else:
        # Default: save to EXP_DIR (bucket path), not to local dmast_ckpt path
        default_output = EXP_DIR / ("eval_results_agent_only" if not USE_ATTACKER else "eval_results")
        cmd.extend(["--output-dir", str(default_output)])
    
    print("\n▶️  Running command:")
    print(" ".join(cmd))
    print()
    
    # Ask for confirmation
    try:
        input("Press Enter to start evaluation or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        sys.exit(0)
    
    # Run evaluation
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

