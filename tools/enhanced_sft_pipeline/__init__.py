"""
Enhanced SFT Data Pipeline
==========================

A multi-stage pipeline for generating enhanced SFT training data with 
Chain-of-Thought (CoT) defense reasoning against adversarial attacks.

Stages:
    1. Golden Trajectory Collection: Collect clean agent trajectories
    1b. Attacker Data Collection: Collect attacker imitation data (optional)
    2. Attack Factory: Synthesize attacks on clean data
    3. Oracle Hindsight Labeling: Generate CoT defense reasoning
    4. Data Formatting & Mixing: Create final SFT dataset (agent + attacker)

Usage:
    # Run full pipeline (includes attacker data collection)
    python -m tools.enhanced_sft_pipeline.run_pipeline --model-path ~/dmast_ckpt/pretrained/gemma-3-27b-it
    
    # Run without attacker data collection
    python -m tools.enhanced_sft_pipeline.run_pipeline --no-attacker-data
    
    # Run individual stages
    python -m tools.enhanced_sft_pipeline.stage1_golden_trajectory --help
    python -m tools.enhanced_sft_pipeline.stage1b_attacker_collection --help
    python -m tools.enhanced_sft_pipeline.stage2_attack_factory --help
    python -m tools.enhanced_sft_pipeline.stage3_oracle_cot --help
    python -m tools.enhanced_sft_pipeline.stage4_format_mix --help
"""

from .stage1_golden_trajectory import collect_golden_trajectories, GoldenStep
from .stage1b_attacker_collection import collect_attacker_data, AttackerStep
from .stage2_attack_factory import synthesize_attacks, AttackedStep
from .stage3_oracle_cot import generate_cot_labels, CoTEnhancedStep
from .stage4_format_mix import format_and_mix_data, SFTSample
from .run_pipeline import run_full_pipeline

__all__ = [
    # Stage 1
    'collect_golden_trajectories',
    'GoldenStep',
    
    # Stage 1b
    'collect_attacker_data',
    'AttackerStep',
    
    # Stage 2
    'synthesize_attacks',
    'AttackedStep',
    
    # Stage 3
    'generate_cot_labels',
    'CoTEnhancedStep',
    
    # Stage 4
    'format_and_mix_data',
    'SFTSample',
    
    # Full pipeline
    'run_full_pipeline',
]
