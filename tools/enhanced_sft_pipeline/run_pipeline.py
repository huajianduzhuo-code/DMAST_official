#!/usr/bin/env python3
"""
Enhanced SFT Data Pipeline
==========================

A unified pipeline that orchestrates all 4 stages of enhanced SFT data generation:

Stage 1: Golden Trajectory Collection
    - Collect paired golden and independent clean-only trajectories
    - Use a nominal budget of 4000 episodes per mode
    - Keep only successful episodes
    - Store: clean_screenshot, clean_axtree, task_instruction, golden_action

Stage 1b: Attacker Trajectory Collection
    - Collect a separate nominal budget of 4000 adversarial episodes
    - Extract attacker steps from successful attacks during formatting

Stage 2: Attack Factory  
    - Reuse paired attacked observations from Stage 1
    - An offline synthesis fallback is available but not recommended for this pipeline
    - Keep golden_action as the correct label

Stage 3: Oracle Hindsight Labeling
    - Generate task-focused Chain-of-Thought (CoT) reasoning by default
    - Oracle receives the attacked view, task goal, and golden action
    - Optional defense-focused mode also provides the clean view
    - Heuristic validation and fallback templates do not guarantee prompt compliance

Stage 4: Data Formatting & Mixing
    - Format data for SFT training
    - Combine augmented agent, clean agent, and attacker samples
    - Count the retained transitions in each category after formatting
    - Output format compatible with existing train.py

Usage:
    # Run full pipeline
    python run_pipeline.py --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it
    
    # Run specific stages
    python run_pipeline.py --stages 2,3,4 --skip-collection
    
    # Test with small dataset
    python run_pipeline.py --test-mode
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from configs.task_splits import MINIWOB_TRAIN_TASKS
from utils.training_data import validate_training_data_file


def run_stage1_golden_collection(
    model_path: str,
    output_dir: str,
    num_episodes: int = 4000,
    num_tasks: int = len(MINIWOB_TRAIN_TASKS),
    num_jobs: int = 64,
    max_steps: int = 5,
    all_tasks: bool = True,
) -> str:
    """Run Stage 1: Golden Trajectory Collection"""
    print("\n" + "=" * 80)
    print("🏆 STAGE 1: Golden Trajectory Collection")
    print("=" * 80)
    
    output_path = f"{output_dir}/stage1_golden_trajectories.json"
    
    from tools.enhanced_sft_pipeline.stage1_golden_trajectory import collect_golden_trajectories
    
    collect_golden_trajectories(
        model_path=model_path,
        output_path=output_path,
        num_episodes=num_episodes,
        num_tasks=num_tasks,
        num_jobs=num_jobs,
        max_episode_steps=max_steps,
        all_tasks=all_tasks,
    )
    
    return output_path


def run_stage2_attack_factory(
    golden_path: str,
    output_dir: str,
    attacker_model: str,
    attacker_url: str,
    max_samples: int = -1,
) -> str:
    """Run Stage 2: Attack Factory"""
    print("\n" + "=" * 80)
    print("🎭 STAGE 2: Attack Factory")
    print("=" * 80)
    
    output_path = f"{output_dir}/stage2_attacked_data.json"
    
    from tools.enhanced_sft_pipeline.stage2_attack_factory import synthesize_attacks
    
    synthesize_attacks(
        golden_trajectories_path=golden_path,
        output_path=output_path,
        attacker_model_path=attacker_model,
        attacker_base_url=attacker_url,
        max_samples=max_samples,
    )
    
    return output_path


def run_stage3_oracle_cot(
    attacked_path: str,
    output_dir: str,
    oracle_model: str,
    oracle_url: str,
    use_images: bool = True,
    max_samples: int = -1,
    task_focused: bool = True,  # Use task-focused CoT by default
) -> str:
    """Run Stage 3: Oracle CoT Generation
    
    Args:
        task_focused: If True (default), generate task-focused CoT that ignores attacks.
                     If False, generate defense-analysis CoT that identifies attacks.
    """
    cot_mode = "Task-Focused" if task_focused else "Defense-Analysis"
    print("\n" + "=" * 80)
    print(f"🔮 STAGE 3: Oracle Hindsight Labeling ({cot_mode})")
    print("=" * 80)
    
    output_path = f"{output_dir}/stage3_cot_data.json"
    
    from tools.enhanced_sft_pipeline.stage3_oracle_cot import generate_cot_labels
    
    generate_cot_labels(
        attacked_data_path=attacked_path,
        output_path=output_path,
        oracle_model_path=oracle_model,
        oracle_base_url=oracle_url,
        use_images=use_images,
        max_samples=max_samples,
        task_focused=task_focused,  # Select the CoT reasoning mode
    )
    
    return output_path


def run_stage1b_attacker_collection(
    model_path: str,
    output_dir: str,
    num_episodes: int = 4000,
    num_tasks: int = len(MINIWOB_TRAIN_TASKS),
    num_jobs: int = 128,
    max_steps: int = 5,
    all_tasks: bool = True,
) -> str:
    """Run Stage 1b: Attacker Imitation Data Collection"""
    print("\n" + "=" * 80)
    print("🎭 STAGE 1b: Attacker Imitation Data Collection")
    print("=" * 80)
    
    output_path = f"{output_dir}/stage1b_attacker_data.json"
    
    from tools.enhanced_sft_pipeline.stage1b_attacker_collection import collect_attacker_data
    
    collect_attacker_data(
        model_path=model_path,
        output_path=output_path,
        num_episodes=num_episodes,
        num_tasks=num_tasks,
        num_jobs=num_jobs,
        max_episode_steps=max_steps,
        all_tasks=all_tasks,
    )
    
    return output_path


def run_stage4_format_mix(
    golden_path: str,
    cot_path: str,
    output_dir: str,
    model_path: str = None,
    attacker_path: str = None,
    max_samples: int = -1,
) -> str:
    """Run Stage 4: Data Formatting & Mixing"""
    print("\n" + "=" * 80)
    print("📦 STAGE 4: Data Formatting & Mixing")
    print("=" * 80)
    
    output_path = f"{output_dir}/final_sft_dataset.json"
    
    from tools.enhanced_sft_pipeline.stage4_format_mix import format_and_mix_data
    
    format_and_mix_data(
        golden_trajectories_path=golden_path,
        cot_data_path=cot_path,
        output_path=output_path,
        model_path=model_path,  # Pass model_path for tokenization
        attacker_data_path=attacker_path,
        max_samples=max_samples,
    )
    
    return output_path


def start_vllm_server(model_path: str, port: int = 8000) -> Optional[subprocess.Popen]:
    """Start a vLLM server for inference"""
    print(f"\n🚀 Starting vLLM server for {model_path} on port {port}...")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", "4",  # Adjust based on your GPU setup
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.8",
        "--trust-remote-code",  # Required for Qwen-VL/Gemma3 custom vision processors
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for server to start
        import time
        for _ in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/v1/models")
                if response.status_code == 200:
                    print(f"   ✅ vLLM server started successfully")
                    return process
            except:
                continue
        
        print(f"   ⚠️ Server may not be ready, continuing anyway...")
        return process
        
    except Exception as e:
        print(f"   ❌ Failed to start vLLM server: {e}")
        return None


def run_full_pipeline(
    model_path: str,
    output_dir: str,
    vllm_url: str,
    stages: List[int] = [1, 2, 3, 4],
    num_episodes: int = 4000,
    num_tasks: int = len(MINIWOB_TRAIN_TASKS),
    num_jobs: int = 64,
    max_steps: int = 5,
    use_images: bool = True,
    max_samples_per_stage: int = -1,
    skip_vllm_start: bool = True,
    all_tasks: bool = True,  # Use all registered MiniWoB++ tasks except the 28 held-out tasks
    task_focused: bool = True,  # Use task-focused CoT (recommended)
    collect_attacker_data: bool = True,  # Also collect attacker imitation data
    attacker_episodes: int = 4000,  # Episodes for attacker data collection
):
    """
    Run the full enhanced SFT data pipeline.
    
    Args:
        model_path: Path to the 27B model (used for agent, attacker, and oracle)
        output_dir: Directory to save all outputs
        vllm_url: vLLM server URL
        stages: List of stages to run (1, 2, 3, 4)
        num_episodes: Nominal episode budget per golden mode (paired and clean-only)
        num_tasks: Number of shared training tasks when all_tasks=False
        num_jobs: Parallel workers for Stage 1
        max_steps: Max steps per episode
        use_images: Whether to use images in Oracle prompt
        max_samples_per_stage: Limit samples per stage (-1 = all)
        skip_vllm_start: Skip starting vLLM server (assume it's already running)
        collect_attacker_data: Whether to also collect attacker imitation data (Stage 1b)
        attacker_episodes: Nominal episode budget for independent attacker collection
    """
    # Each collector resolves --all-tasks from the installed registry, excluding holdout.
    tasks_str = ("all registered training tasks (28 held-out tasks excluded)"
                 if all_tasks else f"{num_tasks} tasks from the shared training split")
    
    cot_mode = "Task-Focused (recommended)" if task_focused else "Defense-Analysis"
    
    print("=" * 80)
    print("🚀 Enhanced SFT Data Pipeline")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_dir}")
    print(f"   vLLM URL: {vllm_url}")
    print(f"   Stages to run: {stages}")
    print(f"   Golden episode budget per mode: {num_episodes}")
    print(f"   Tasks: {tasks_str}")
    print(f"   Max steps: {max_steps}")
    print(f"   Use images: {use_images}")
    print(f"   CoT Mode: {cot_mode}")
    print(f"   Collect attacker data: {collect_attacker_data}")
    if collect_attacker_data:
        print(f"   Attacker episode budget: {attacker_episodes}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Track paths
    golden_path = f"{output_dir}/stage1_golden_trajectories.json"
    attacker_path = f"{output_dir}/stage1b_attacker_data.json"
    attacked_path = f"{output_dir}/stage2_attacked_data.json"
    cot_path = f"{output_dir}/stage3_cot_data.json"
    final_path = f"{output_dir}/final_sft_dataset.json"
    
    # Check reused inputs before optionally launching inference. Individual stages
    # also validate their inputs, including when called outside this entry point.
    reused_inputs = set()
    if 1 not in stages and (2 in stages or 4 in stages):
        reused_inputs.add(golden_path)
    if 2 not in stages and 3 in stages:
        reused_inputs.add(attacked_path)
    if 3 not in stages and 4 in stages:
        reused_inputs.add(cot_path)
    if collect_attacker_data and 1 not in stages and 4 in stages and Path(attacker_path).exists():
        reused_inputs.add(attacker_path)
    if 4 not in stages and Path(final_path).exists():
        reused_inputs.add(final_path)
        reused_inputs.add(final_path.replace('.json', '_simplified.json'))
    for input_path in sorted(reused_inputs):
        validate_training_data_file(input_path)

    vllm_process = None
    
    try:
        # Start vLLM server if needed
        if not skip_vllm_start:
            vllm_process = start_vllm_server(model_path)
        
        # Run each stage
        if 1 in stages:
            golden_path = run_stage1_golden_collection(
                model_path=model_path,
                output_dir=output_dir,
                num_episodes=num_episodes,
                num_tasks=num_tasks,
                num_jobs=num_jobs,
                max_steps=max_steps,
                all_tasks=all_tasks,
            )
        else:
            print(f"\n⏭️ Skipping Stage 1 (using existing: {golden_path})")
        
        # Stage 1b: Attacker data collection (optional)
        if collect_attacker_data:
            if 1 in stages:  # Only run Stage 1b if Stage 1 is being run
                attacker_path = run_stage1b_attacker_collection(
                    model_path=model_path,
                    output_dir=output_dir,
                    num_episodes=attacker_episodes,
                    num_tasks=num_tasks,
                    num_jobs=num_jobs,
                    max_steps=max_steps,
                    all_tasks=all_tasks,
                )
            elif not Path(attacker_path).exists():
                print(f"\n⚠️ Attacker data collection skipped but file not found: {attacker_path}")
                attacker_path = None
            else:
                print(f"\n⏭️ Using existing attacker data: {attacker_path}")
        else:
            attacker_path = None
        
        if 2 in stages:
            attacked_path = run_stage2_attack_factory(
                golden_path=golden_path,
                output_dir=output_dir,
                attacker_model=model_path,  # Pass full path for tokenizer loading
                attacker_url=vllm_url,
                max_samples=max_samples_per_stage,
            )
        else:
            print(f"\n⏭️ Skipping Stage 2 (using existing: {attacked_path})")
        
        if 3 in stages:
            cot_path = run_stage3_oracle_cot(
                attacked_path=attacked_path,
                output_dir=output_dir,
                oracle_model=model_path,  # Pass full path for tokenizer loading
                oracle_url=vllm_url,
                use_images=use_images,
                max_samples=max_samples_per_stage,
                task_focused=task_focused,  # Select the CoT reasoning mode
            )
        else:
            print(f"\n⏭️ Skipping Stage 3 (using existing: {cot_path})")
        
        if 4 in stages:
            final_path = run_stage4_format_mix(
                golden_path=golden_path,
                cot_path=cot_path,
                output_dir=output_dir,
                model_path=model_path,  # Pass model_path for tokenization
                attacker_path=attacker_path,  # Include attacker data if available
                max_samples=max_samples_per_stage * 2 if max_samples_per_stage > 0 else -1,
            )
        else:
            print(f"\n⏭️ Skipping Stage 4 (using existing: {final_path})")
        
        # Print final summary
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\n📁 Output files:")
        print(f"   Stage 1 (Agent Golden): {golden_path}")
        if attacker_path:
            print(f"   Stage 1b (Attacker): {attacker_path}")
        print(f"   Stage 2 (Attacked): {attacked_path}")
        print(f"   Stage 3 (CoT): {cot_path}")
        print(f"   Stage 4 (Final): {final_path}")
        
        # Load and show final stats
        if Path(final_path).exists():
            final_data = validate_training_data_file(final_path)
            validate_training_data_file(final_path.replace('.json', '_simplified.json'))
            
            metadata = final_data.get('metadata', {})
            print(f"\n📊 Final Dataset:")
            print(f"   Total samples: {metadata.get('total_samples', 'N/A')}")
            print(f"   By role:")
            print(f"   └─ Agent: {metadata.get('num_agent_samples', 'N/A')}")
            print(f"   └─ Attacker: {metadata.get('num_attacker_samples', 'N/A')}")
            print(f"   By sample type:")
            print(f"   └─ Agent attacked (with CoT): {metadata.get('num_attacked_with_cot_samples', 'N/A')}")
            print(f"   └─ Agent clean: {metadata.get('num_clean_samples', 'N/A')}")
            print(f"   └─ Attacker: {metadata.get('num_attacker_role_samples', 'N/A')}")
            print(f"   Multimodal: {metadata.get('num_multimodal_samples', 'N/A')}")
        
        print(f"\n🎓 Next step: Use this dataset for SFT training:")
        print(f"   python train.py --imitation-data {final_path.replace('.json', '_simplified.json')}")
        
    finally:
        # Cleanup vLLM server if we started it
        if vllm_process:
            print("\n🧹 Shutting down vLLM server...")
            vllm_process.terminate()
            vllm_process.wait()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(
        description="Enhanced SFT Data Pipeline with CoT Defense Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python run_pipeline.py --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it
    
    # Run specific stages (2, 3, 4) assuming Stage 1 data exists
    python run_pipeline.py --stages 2,3,4
    
    # Test mode with small dataset
    python run_pipeline.py --test-mode
    
    # Custom configuration
    python run_pipeline.py \\
        --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it \\
        --num-episodes 1000 \\
        --output-dir ~/bucket/data/my_sft_data
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "dmast_ckpt/pretrained/gemma-3-27b-it"),
        help="Path to 27B model (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft"),
        help="Output directory for all stages",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://127.0.0.1:1234/v1",
        help="vLLM server URL (default matches VllmModelArgs default)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of stages to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=4000,
        help="Nominal episodes per golden mode, rounded down per task (default: 4000)",
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
        default=64,
        help="Parallel workers for Stage 1 (default: 64)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Max steps per episode (default: 5)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't use images in Oracle prompt",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Max samples per stage (-1 = all)",
    )
    parser.add_argument(
        "--start-vllm",
        action="store_true",
        help="Start vLLM server automatically",
    )
    parser.add_argument(
        "--no-attacker-data",
        action="store_true",
        help="Skip attacker imitation data collection (Stage 1b)",
    )
    parser.add_argument(
        "--attacker-episodes",
        type=int,
        default=4000,
        help="Nominal attacker episodes, rounded down per task (default: 4000)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with small dataset (10 episodes, stages 2-4 only)",
    )
    parser.add_argument(
        "--task-focused",
        action="store_true",
        default=True,
        help="Generate task-focused CoT (default, recommended). Agent learns to focus purely on task.",
    )
    parser.add_argument(
        "--defense-focused",
        action="store_true",
        help="Generate defense-analysis CoT. Agent learns to identify and reason about attacks.",
    )
    
    args = parser.parse_args()
    
    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(',')]
    
    # Determine CoT mode
    task_focused = not args.defense_focused  # Default is task-focused unless --defense-focused
    
    # Test mode overrides
    if args.test_mode:
        print("🧪 TEST MODE: Using small dataset")
        args.num_episodes = 10
        args.max_samples = 5
        stages = [2, 3, 4]  # Skip Stage 1, assume clean data exists
    
    run_full_pipeline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        stages=stages,
        num_episodes=args.num_episodes,
        num_tasks=args.num_tasks,
        num_jobs=args.num_jobs,
        max_steps=args.max_steps,
        use_images=not args.no_images,
        max_samples_per_stage=args.max_samples,
        skip_vllm_start=not args.start_vllm,
        all_tasks=args.all_tasks,
        task_focused=task_focused,
        collect_attacker_data=not args.no_attacker_data,
        attacker_episodes=args.attacker_episodes,
    )


if __name__ == "__main__":
    main()
