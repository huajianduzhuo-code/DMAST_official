#!/usr/bin/env python3
"""
Study different attack modes and their effects on agent and attacker performance.

This script runs multiple data collection experiments with different configurations
and summarizes the statistics in a JSON file.

Configurations tested:
1. dual_attack: Agent sees both modified AXTree and screenshot (full attack)
2. text_only_attack: Agent sees modified AXTree only, original screenshot
3. image_only_attack: Agent sees original AXTree, modified screenshot only
4. no_attack: Multimodal agent without attacker (pure baseline)
5. text_agent_no_attacker: Text-only agent without attacker (pure baseline)
6. text_agent_with_attack: Text-only agent with attacker (AXTree attack only, no vision)

Usage:
    python study_attack_modes.py --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it
    
    # Run specific modes only
    python study_attack_modes.py --modes dual_attack text_only_attack
    
    # Skip data collection and only analyze existing files
    python study_attack_modes.py --analyze-only
"""

import argparse
import subprocess
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Suppress verbose logs from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Define experiment configurations
EXPERIMENT_CONFIGS = {
    "dual_attack": {
        "name": "Dual Attack (AXTree + Screenshot)",
        "description": "Agent sees both modified AXTree and screenshot",
        "flags": ["--on-policy-role", "attacker"],
        "output_file": "study_dual_attack.json",
    },
    "text_only_attack": {
        "name": "Text-Only Attack (AXTree only)",
        "description": "Agent sees modified AXTree, original screenshot",
        "flags": ["--no-image-attack", "--on-policy-role", "attacker"],
        "output_file": "study_text_only_attack.json",
    },
    "image_only_attack": {
        "name": "Image-Only Attack (Screenshot only)",
        "description": "Agent sees original AXTree, modified screenshot",
        "flags": ["--no-text-injection", "--on-policy-role", "attacker"],
        "output_file": "study_image_only_attack.json",
    },
    "no_attack": {
        "name": "No Attack (Baseline)",
        "description": "Multimodal agent without attacker - pure baseline",
        "flags": ["--no-attacker"],
        "output_file": "study_no_attack.json",
    },
    "text_agent_no_attacker": {
        "name": "Text-Only Agent (No Attacker)",
        "description": "Pure text-based agent without attacker, no screenshots",
        "flags": ["--agent-no-vision", "--no-attacker"],
        "output_file": "study_text_agent_no_attacker.json",
    },
    "text_agent_with_attack": {
        "name": "Text-Only Agent with Attack",
        "description": "Text-based agent (no screenshots) with AXTree attack",
        "flags": ["--agent-no-vision", "--no-image-attack", "--on-policy-role", "attacker"],
        "output_file": "study_text_agent_with_attack.json",
    },
}


def run_data_collection(
    model_path: str,
    output_dir: Path,
    config_name: str,
    num_episodes: int,
    num_tasks: int,
    num_jobs: int,
    save_every_n_episodes: int,
    job_timeout: int = 300,
    all_tasks: bool = False,
) -> Dict[str, Any]:
    """Run data collection for a specific configuration."""
    
    config = EXPERIMENT_CONFIGS[config_name]
    output_file = output_dir / config["output_file"]
    
    print("\n" + "=" * 80)
    print(f"🎯 Running: {config['name']}")
    print(f"   {config['description']}")
    print("=" * 80)
    
    # Build command
    cmd = [
        "python", "collect_data_for_imitation.py",
        "--evaluation",
        "--model-path", model_path,
        "--output", str(output_file),
        "--num-episodes", str(num_episodes),
        "--num-tasks", str(num_tasks),
        "--num-jobs", str(num_jobs),
        "--save-every-n-episodes", str(save_every_n_episodes),
        "--job-timeout", str(job_timeout),
    ]
    
    # Add --all-tasks flag if needed
    if all_tasks:
        cmd.append("--all-tasks")
    
    # Add config-specific flags
    cmd.extend(config["flags"])
    
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=False,  # Show output in real-time
        )
        
        if result.returncode != 0:
            return {
                "config": config_name,
                "name": config["name"],
                "success": False,
                "error": f"Command exited with code {result.returncode}",
                "output_file": str(output_file),
            }
        
        return {
            "config": config_name,
            "name": config["name"],
            "success": True,
            "error": None,
            "output_file": str(output_file),
        }
        
    except Exception as e:
        return {
            "config": config_name,
            "name": config["name"],
            "success": False,
            "error": str(e),
            "output_file": str(output_file),
        }


def analyze_episodes(json_path: Path) -> Dict[str, Any]:
    """Analyze episodes from a JSON file and compute statistics."""
    
    if not json_path.exists():
        return {
            "error": f"File not found: {json_path}",
            "total_episodes": 0,
        }
    
    with open(json_path, 'r') as f:
        episodes = json.load(f)
    
    total_episodes = len(episodes)
    if total_episodes == 0:
        return {
            "error": "No episodes found",
            "total_episodes": 0,
        }
    
    # Statistics
    agent_success_count = 0
    attacker_success_count = 0
    episodes_with_attacker = 0
    
    task_stats = {}
    
    for ep in episodes:
        task_name = ep.get('task_name', 'unknown')
        use_attacker = ep.get('use_attacker', False)
        
        # Initialize task stats
        if task_name not in task_stats:
            task_stats[task_name] = {
                'total': 0,
                'agent_success': 0,
                'attacker_success': 0,
                'with_attacker': 0,
            }
        
        task_stats[task_name]['total'] += 1
        
        # Check agent success (cumulative reward > 0 or last step reward > 0)
        history = ep.get('history', [])
        agent_reward = ep.get('reward', 0)
        
        # If no top-level reward, sum from history
        if agent_reward == 0 and history:
            agent_reward = sum(step.get('agent_reward', 0) for step in history)
        
        if agent_reward > 0:
            agent_success_count += 1
            task_stats[task_name]['agent_success'] += 1
        
        # Check attacker success
        if use_attacker:
            episodes_with_attacker += 1
            task_stats[task_name]['with_attacker'] += 1
            
            # Attacker success: any step has attacker_reward > 0
            attacker_reward = sum(step.get('attacker_reward', 0) for step in history)
            if attacker_reward > 0:
                attacker_success_count += 1
                task_stats[task_name]['attacker_success'] += 1
    
    # Compute rates
    agent_success_rate = agent_success_count / total_episodes if total_episodes > 0 else 0
    attacker_success_rate = attacker_success_count / episodes_with_attacker if episodes_with_attacker > 0 else 0
    
    return {
        "total_episodes": total_episodes,
        "episodes_with_attacker": episodes_with_attacker,
        "agent_success_count": agent_success_count,
        "agent_success_rate": agent_success_rate,
        "attacker_success_count": attacker_success_count,
        "attacker_success_rate": attacker_success_rate,
        "task_stats": task_stats,
    }


def print_summary_table(results: Dict[str, Dict[str, Any]]):
    """Print a summary table of all results."""
    
    print("\n" + "=" * 100)
    print("📊 SUMMARY OF ALL EXPERIMENTS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Configuration':<35} {'Episodes':>10} {'Agent ✓':>12} {'Agent %':>10} {'Attacker ✓':>12} {'Attacker %':>12}")
    print("-" * 95)
    
    for config_name, stats in results.items():
        config = EXPERIMENT_CONFIGS.get(config_name, {"name": config_name})
        name = config.get("name", config_name)[:33]
        
        if "error" in stats and stats.get("total_episodes", 0) == 0:
            print(f"{name:<35} {'ERROR':>10} {'-':>12} {'-':>10} {'-':>12} {'-':>12}")
            continue
        
        total = stats.get("total_episodes", 0)
        agent_count = stats.get("agent_success_count", 0)
        agent_rate = stats.get("agent_success_rate", 0) * 100
        attacker_count = stats.get("attacker_success_count", 0)
        attacker_rate = stats.get("attacker_success_rate", 0) * 100
        with_attacker = stats.get("episodes_with_attacker", 0)
        
        attacker_str = f"{attacker_count}/{with_attacker}" if with_attacker > 0 else "N/A"
        attacker_rate_str = f"{attacker_rate:.1f}%" if with_attacker > 0 else "N/A"
        
        print(f"{name:<35} {total:>10} {agent_count:>12} {agent_rate:>9.1f}% {attacker_str:>12} {attacker_rate_str:>12}")
    
    print("-" * 95)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Study different attack modes and their effects on agent/attacker performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "dmast_ckpt/pretrained/gemma-3-27b-it"),
        help="Model path for agent and attacker",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment modes to run (default: all)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Total episodes per experiment. If not set, uses --num-tasks * --episodes-per-task",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=56,
        help="Number of tasks to use (default: 56, ignored if --all-tasks is set)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Use all 125 MiniWoB++ tasks instead of the custom 56-task subset",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=6,
        help="Episodes per task (default: 6). Total episodes = num_tasks * episodes_per_task",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=128,
        help="Number of parallel jobs (default: 128)",
    )
    parser.add_argument(
        "--save-every-n-episodes",
        type=int,
        default=128,
        help="Save every N episodes (default: 128)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=300,
        help="Timeout per job in seconds (default: 300 = 5 minutes, 0 = no timeout)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip data collection and only analyze existing files",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="attack_modes_summary.json",
        help="Output file for summary statistics (default: attack_modes_summary.json)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate num_tasks and num_episodes
    num_tasks = 125 if args.all_tasks else args.num_tasks
    if args.num_episodes is not None:
        num_episodes = args.num_episodes
    else:
        num_episodes = num_tasks * args.episodes_per_task
    
    print("=" * 80)
    print("🔬 Attack Modes Study")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Experiments: {', '.join(args.modes)}")
    print(f"   Tasks: {num_tasks}" + (" (all MiniWoB++)" if args.all_tasks else " (custom subset)"))
    print(f"   Episodes per task: {args.episodes_per_task}")
    print(f"   Total episodes per experiment: {num_episodes}")
    if not args.analyze_only:
        print(f"   Parallel jobs: {args.num_jobs}")
        print(f"   Job timeout: {args.job_timeout}s" + (" (auto-skip on timeout)" if args.job_timeout > 0 else " (no limit)"))
    print()
    
    # Run data collection (if not analyze-only)
    collection_results = []
    if not args.analyze_only:
        for mode in args.modes:
            result = run_data_collection(
                model_path=args.model_path,
                output_dir=output_dir,
                config_name=mode,
                num_episodes=num_episodes,
                num_tasks=num_tasks,
                num_jobs=args.num_jobs,
                save_every_n_episodes=args.save_every_n_episodes,
                job_timeout=args.job_timeout,
                all_tasks=args.all_tasks,
            )
            collection_results.append(result)
            
            if not result["success"]:
                print(f"⚠️ Warning: {mode} failed: {result['error']}")
    
    # Analyze results
    print("\n📊 Analyzing results...")
    all_stats = {}
    
    for mode in args.modes:
        config = EXPERIMENT_CONFIGS[mode]
        json_path = output_dir / config["output_file"]
        stats = analyze_episodes(json_path)
        stats["config_name"] = mode
        stats["config_description"] = config["description"]
        all_stats[mode] = stats
    
    # Print summary table
    print_summary_table(all_stats)
    
    # Save summary to JSON
    summary_path = output_dir / args.summary_file
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "num_episodes": args.num_episodes,
        "num_tasks": args.num_tasks,
        "experiments": args.modes,
        "results": all_stats,
        "collection_results": collection_results if not args.analyze_only else None,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_path}")
    
    # Print key findings
    print("\n📈 Key Findings:")
    
    # Find baseline (no attack or text_agent_no_attacker)
    baseline_mode = None
    baseline_rate = None
    for mode in ["no_attack", "text_agent_no_attacker"]:
        if mode in all_stats and all_stats[mode].get("total_episodes", 0) > 0:
            baseline_mode = mode
            baseline_rate = all_stats[mode].get("agent_success_rate", 0)
            break
    
    if baseline_mode and baseline_rate is not None:
        print(f"   • Baseline ({EXPERIMENT_CONFIGS[baseline_mode]['name']}): {baseline_rate*100:.1f}% agent success")
        
        for mode, stats in all_stats.items():
            if mode != baseline_mode and stats.get("total_episodes", 0) > 0:
                agent_rate = stats.get("agent_success_rate", 0)
                diff = (agent_rate - baseline_rate) * 100
                direction = "↓" if diff < 0 else "↑" if diff > 0 else "→"
                
                attacker_rate = stats.get("attacker_success_rate", 0)
                attacker_str = f", attacker: {attacker_rate*100:.1f}%" if stats.get("episodes_with_attacker", 0) > 0 else ""
                
                print(f"   • {EXPERIMENT_CONFIGS[mode]['name']}: agent {agent_rate*100:.1f}% ({direction} {abs(diff):.1f}%){attacker_str}")
    
    print("\n✅ Study complete!")


if __name__ == "__main__":
    main()
