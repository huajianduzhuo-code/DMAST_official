#!/usr/bin/env python3
"""
Stage 2: Attack Factory (Malicious Attack Synthesis)
====================================================

This stage processes golden trajectories from Stage 1:

1. If Stage 1 was run with use_attacker=True:
   - Data already contains both clean and attacked observations
   - This stage just validates and formats the data

2. If no paired attacked observations are available, or regeneration is requested:
   - An offline fallback synthesizes attack text and screenshot overlays
   - This does not reproduce browser-based HTML injection
   - Not recommended for the pipeline: Stage 4 cannot match synthesized samples
     whose Stage 1 records lack attacked screenshots

Input: Stage 1 golden trajectories
Output: Pairs of (clean_data, attacked_data) with same golden_action
"""

import argparse
import json
import logging
import os
import sys
import base64
import io
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from copy import deepcopy
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from agents.html_injection_attacker import (
    HTMLInjectionAttacker,
    HTMLInjectionAttackerArgs,
    generate_injection_script,
)
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp
from agents.llm.vllm_model import VllmModelArgs
from utils.training_data import validate_training_data_file
import bgym

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass 
class AttackedStep:
    """Represents a step with both clean and attacked versions"""
    # Original clean data
    episode_idx: int
    step_idx: int
    task_name: str
    task_instruction: str
    clean_screenshot_b64: str
    clean_axtree: str
    golden_action: str
    agent_think: Optional[str]
    
    # Attack data
    attacked_screenshot_b64: str
    attacked_axtree: str
    attack_html: str
    attack_css: str
    attack_target: str
    attack_position: str
    attacker_think: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def base64_to_pil(b64_string: str) -> Optional[Image.Image]:
    """Convert base64 string to PIL Image"""
    if not b64_string:
        return None
    try:
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logging.warning(f"Failed to decode base64 image: {e}")
        return None


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def inject_html_into_axtree(clean_axtree: str, html_content: str) -> str:
    """
    Simulate HTML injection into the accessibility tree.
    
    Since we can't actually render the page, we approximate the effect
    by inserting relevant text from the HTML into the axtree.
    """
    if not html_content:
        return clean_axtree
    
    # Extract visible text from HTML
    # Remove tags but keep text content
    text_content = re.sub(r'<[^>]+>', ' ', html_content)
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    
    if not text_content:
        return clean_axtree
    
    # Find a suitable injection point in the axtree
    lines = clean_axtree.split('\n')
    
    # Find lines with [bid] markers to inject near
    bid_lines = [i for i, line in enumerate(lines) if re.search(r'\[\d+\]', line)]
    
    if bid_lines:
        # Inject after a random bid line (simulating DOM injection)
        import random
        inject_idx = random.choice(bid_lines)
        
        # Create fake axtree entry for the injected content
        # Use a high bid number that wouldn't normally exist
        fake_bid = 9999
        injected_entry = f"\t[{fake_bid}] StaticText '{text_content}'"
        
        lines.insert(inject_idx + 1, injected_entry)
        return '\n'.join(lines)
    else:
        # Append to end if no bid markers found
        return clean_axtree + f"\n[9999] StaticText '{text_content}'"


def create_attacked_screenshot(
    clean_screenshot: Image.Image, 
    html_content: str,
    css_content: str,
) -> Image.Image:
    """
    Create a simulated attacked screenshot.
    
    Since we can't render actual HTML, we overlay text on the image
    to simulate the visual effect of the injection.
    """
    from PIL import ImageDraw, ImageFont
    
    if not html_content:
        return clean_screenshot
    
    # Create a copy
    attacked = clean_screenshot.copy()
    draw = ImageDraw.Draw(attacked)
    
    # Extract key text from HTML (look for common attack patterns)
    text_to_show = ""
    
    # Look for common attack UI elements
    patterns = [
        r'Security[^<]*',
        r'Verification[^<]*',
        r'Password[^<]*',
        r'Alert[^<]*',
        r'Warning[^<]*',
        r'Required[^<]*',
        r'Enter your[^<]*',
        r'Please[^<]*verify[^<]*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            text_to_show = match.group(0).strip()
            break
    
    if not text_to_show:
        # Fallback: extract first text content
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        text_to_show = text_content[:50] if len(text_content) > 50 else text_content
    
    if text_to_show:
        # Draw a semi-transparent overlay
        width, height = attacked.size
        
        # Create overlay box
        box_height = 60
        box_y = 20
        
        # Draw background rectangle
        overlay = Image.new('RGBA', attacked.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [10, box_y, width - 10, box_y + box_height],
            fill=(255, 243, 205, 220),  # Light yellow with transparency
            outline=(255, 193, 7, 255),  # Orange border
            width=2
        )
        
        # Try to use a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()
        
        # Draw the attack text
        overlay_draw.text(
            (15, box_y + 10),
            f"⚠️ {text_to_show}",
            fill=(133, 100, 4, 255),  # Dark yellow/brown
            font=font
        )
        
        # Composite the overlay
        attacked = Image.alpha_composite(attacked.convert('RGBA'), overlay)
        attacked = attacked.convert('RGB')
    
    return attacked


def synthesize_attacks(
    golden_trajectories_path: str,
    output_path: str,
    attacker_model_path: str = None,
    attacker_base_url: str = None,
    batch_size: int = 32,
    max_samples: int = -1,
    force_regenerate: bool = False,
):
    """
    Process golden trajectory data and ensure attacked versions exist.
    
    If Stage 1 was run with use_attacker=True, the data already contains
    attacked observations and this stage just validates and formats it.
    
    If Stage 1 was run with use_attacker=False (or force_regenerate=True),
    this stage synthesizes attacks using the attacker model.
    
    Args:
        golden_trajectories_path: Path to Stage 1 output
        output_path: Where to save attacked data
        attacker_model_path: Path/name of the attacker model (required if generating)
        attacker_base_url: vLLM server URL for attacker (required if generating)
        batch_size: Process this many samples at a time
        max_samples: Maximum samples to process (-1 = all)
        force_regenerate: Force regeneration even if attacked data exists
    """
    print("=" * 80)
    print("🎭 Stage 2: Attack Factory")
    print("=" * 80)
    
    # Load golden trajectories
    print(f"\n📂 Loading golden trajectories from {golden_trajectories_path}...")
    golden_data = validate_training_data_file(golden_trajectories_path)
    if not isinstance(golden_data, dict) or 'golden_steps' not in golden_data:
        raise ValueError(f"{golden_trajectories_path}: Stage 2 requires a golden_steps wrapper")

    golden_steps = golden_data['golden_steps']
    metadata = golden_data.get('metadata', {})
    
    print(f"   Loaded {len(golden_steps)} golden steps")
    print(f"   Stage 1 used attacker: {metadata.get('use_attacker', 'unknown')}")
    
    if max_samples > 0:
        golden_steps = golden_steps[:max_samples]
        print(f"   Limited to {max_samples} samples")
    
    # Check if attacked data already exists
    steps_with_attacks = sum(1 for step in golden_steps if step.get('attacked_screenshot_b64'))
    
    if steps_with_attacks > 0 and not force_regenerate:
        print(f"\n✅ Found {steps_with_attacks}/{len(golden_steps)} steps with existing attacked data")
        print("   Using existing attacked data from Stage 1 (use --force-regenerate to override)")
        
        # Only process steps with attacked screenshots for CoT generation.
        # Stage 4 treats steps without attacked screenshots as clean samples.
        steps_to_process = [step for step in golden_steps if step.get('attacked_screenshot_b64')]
        print(f"   Filtering: {len(steps_to_process)} steps with attacked data (skipping {len(golden_steps) - len(steps_to_process)} pure clean steps)")
        
        # Just format existing data (only steps with attacks)
        attacked_steps = []
        for step in tqdm(steps_to_process, desc="Formatting existing data"):
            # Get attack details from injection_dict if available
            injection_dict = step.get('injection_dict', {}) or {}
            
            attacked_step = AttackedStep(
                # Clean data (preserved)
                episode_idx=step.get('episode_idx', 0),
                step_idx=step.get('step_idx', 0),
                task_name=step.get('task_name', ''),
                task_instruction=step.get('task_instruction', ''),
                clean_screenshot_b64=step.get('clean_screenshot_b64', ''),
                clean_axtree=step.get('clean_axtree', ''),
                golden_action=step.get('golden_action', ''),
                agent_think=step.get('agent_think'),
                
                # Attack data from Stage 1
                attacked_screenshot_b64=step.get('attacked_screenshot_b64', ''),
                attacked_axtree=step.get('attacked_axtree', ''),
                attack_html=injection_dict.get('html', ''),
                attack_css=injection_dict.get('css', ''),
                attack_target=injection_dict.get('target', '#area'),
                attack_position=injection_dict.get('position', 'prepend'),
                attacker_think=None,  # Not available from Stage 1
            )
            attacked_steps.append(attacked_step.to_dict())
        
        # Save and return
        return _save_attacked_steps(attacked_steps, output_path, golden_trajectories_path)
    
    # NOTE: This offline fallback is not recommended for the paper pipeline.
    # Synthesized views are saved separately, without updating the original golden data.
    # Stage 4 only indexes golden steps that already have attacked_screenshot_b64,
    # so augmented samples from originally clean-only steps are dropped there.
    # Prefer real browser-paired collection in Stage 1 and reuse its attacked views.
    logging.warning(
        "NOTE: Entering the offline attack-generation fallback; this path is not "
        "recommended for the paper pipeline. Stage 4 drops augmented samples whose "
        "original golden steps lack attacked_screenshot_b64, even if attacks and CoT "
        "are generated here and in Stage 3. Prefer rerunning Stage 1 with browser-paired "
        "collection, ensuring paired data survives filtering/sample limits, and "
        "reusing those views without --force-regenerate."
    )

    # Need to generate attacks - require attacker model
    if not attacker_model_path or not attacker_base_url:
        raise ValueError(
            "Stage 1 data does not contain attacked observations. "
            "Please provide --attacker-model-path and --attacker-base-url to generate attacks, "
            "or re-run Stage 1 with --use-attacker."
        )
    
    print(f"\n🤖 Initializing attacker model for attack generation...")
    print(f"   Model: {attacker_model_path}")
    print(f"   Server: {attacker_base_url}")
    
    # Prompt flags for attacker
    prompt_flags = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_error_logs=False,
            use_history=False,  # No history for attack generation
            use_past_error_logs=False,
            use_action_history=False,
            use_think_history=False,
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
        be_cautious=False,
        extra_instructions=None,
        add_missparsed_messages=True,
        max_trunc_itr=50,
    )
    
    attacker_model_args = VllmModelArgs(
        model_name=attacker_model_path,
        base_url=attacker_base_url,
        temperature=1.0,  # Higher temperature for diverse attacks
        max_new_tokens=2048,
        vision_support=True,
    )
    
    attacker_args = HTMLInjectionAttackerArgs(
        chat_model_args=attacker_model_args,
        flags=prompt_flags,
        max_retry=2,
        in_data_collection=True,
        use_art=True,  # Use ART mutations for diversity
    )
    
    attacker = attacker_args.make_agent()
    
    # Process golden steps and generate attacks
    attacked_steps = []
    failed_attacks = 0
    
    print(f"\n🎯 Generating attacks for {len(golden_steps)} steps...")
    
    for i, step in enumerate(tqdm(golden_steps, desc="Generating attacks")):
        try:
            # Reset attacker for fresh context
            attacker.reset()
            
            # Reconstruct observation for attacker
            clean_screenshot = base64_to_pil(step.get('clean_screenshot_b64', ''))
            clean_axtree = step.get('clean_axtree', '')
            task_instruction = step.get('task_instruction', '')
            
            if not clean_axtree and not clean_screenshot:
                logging.warning(f"Step {i}: Missing both screenshot and axtree, skipping")
                failed_attacks += 1
                continue
            
            # Create observation dict for attacker
            obs = {
                'axtree_txt': clean_axtree,
                'screenshot': clean_screenshot if clean_screenshot else None,
                'goal_object': [{'type': 'text', 'text': task_instruction}],
                'chat_messages': [{'role': 'user', 'content': task_instruction}],
                'focused_element': '',
            }
            
            # Get attack from attacker
            injection_dict, attacker_info = attacker.get_action(obs)
            
            if not injection_dict or not injection_dict.get('html'):
                logging.warning(f"Step {i}: Attacker failed to generate valid HTML")
                failed_attacks += 1
                continue
            
            # Extract attacker's think
            attacker_think = attacker_info.extra_info.get('think') if attacker_info else None
            if not attacker_think:
                # Try to extract from chat messages
                for msg in attacker_info.chat_messages.messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        if think_match:
                            attacker_think = think_match.group(1).strip()
                            break
            
            # Apply attack to create attacked versions
            attacked_axtree = inject_html_into_axtree(
                clean_axtree, 
                injection_dict.get('html', '')
            )
            
            attacked_screenshot = None
            attacked_screenshot_b64 = ""
            if clean_screenshot:
                attacked_screenshot = create_attacked_screenshot(
                    clean_screenshot,
                    injection_dict.get('html', ''),
                    injection_dict.get('css', ''),
                )
                attacked_screenshot_b64 = pil_to_base64(attacked_screenshot)
            
            # Create attacked step
            attacked_step = AttackedStep(
                # Clean data (preserved)
                episode_idx=step.get('episode_idx', 0),
                step_idx=step.get('step_idx', 0),
                task_name=step.get('task_name', ''),
                task_instruction=task_instruction,
                clean_screenshot_b64=step.get('clean_screenshot_b64', ''),
                clean_axtree=clean_axtree,
                golden_action=step.get('golden_action', ''),
                agent_think=step.get('agent_think'),
                
                # Attack data
                attacked_screenshot_b64=attacked_screenshot_b64,
                attacked_axtree=attacked_axtree,
                attack_html=injection_dict.get('html', ''),
                attack_css=injection_dict.get('css', ''),
                attack_target=injection_dict.get('target', '#area'),
                attack_position=injection_dict.get('position', 'prepend'),
                attacker_think=attacker_think,
            )
            
            attacked_steps.append(attacked_step.to_dict())
            
            # Save periodically
            if (i + 1) % batch_size == 0:
                print(f"\n   Progress: {i + 1}/{len(golden_steps)} steps processed, {failed_attacks} failed")
                _save_intermediate(output_path, attacked_steps, golden_data, failed_attacks)
                
        except Exception as e:
            logging.error(f"Step {i}: Error generating attack: {e}")
            failed_attacks += 1
            continue
    
    # Final save
    print(f"\n📊 Attack Synthesis Results:")
    print(f"   Successful attacks: {len(attacked_steps)}")
    print(f"   Failed attacks: {failed_attacks}")
    print(f"   Success rate: {len(attacked_steps) / len(golden_steps) * 100:.1f}%")
    
    output_data = {
        "metadata": {
            "stage": "attack_factory",
            "source": golden_trajectories_path,
            "attacker_model": attacker_model_path,
            "timestamp": datetime.now().isoformat(),
            "num_attacked_steps": len(attacked_steps),
            "num_failed_attacks": failed_attacks,
        },
        "attacked_steps": attacked_steps,
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Attacked data saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def _save_attacked_steps(attacked_steps: List[Dict], output_path: str, source_path: str) -> str:
    """
    Save processed attacked steps to output file.
    
    Used when Stage 1 already has attacked data and we just need to format it.
    """
    output_data = {
        "metadata": {
            "stage": "attack_factory",
            "source": source_path,
            "attacker_model": "from_stage1",  # Attacks were generated during Stage 1
            "timestamp": datetime.now().isoformat(),
            "num_attacked_steps": len(attacked_steps),
            "num_failed_attacks": 0,
        },
        "attacked_steps": attacked_steps,
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Attacked data saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def _save_intermediate(output_path: str, attacked_steps: List[Dict], golden_data: Dict, failed_attacks: int):
    """Save intermediate results"""
    output_data = {
        "metadata": {
            "stage": "attack_factory",
            "status": "in_progress",
            "timestamp": datetime.now().isoformat(),
            "num_attacked_steps": len(attacked_steps),
            "num_failed_attacks": failed_attacks,
        },
        "attacked_steps": attacked_steps,
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Stage 2: Attack Factory")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/stage1_golden_trajectories.json"),
        help="Path to Stage 1 golden trajectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/stage2_attacked_data.json"),
        help="Output path for attacked data",
    )
    parser.add_argument(
        "--attacker-model",
        type=str,
        default="gemma-3-27b-it",
        help="Attacker model name/path",
    )
    parser.add_argument(
        "--attacker-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL for attacker",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Save checkpoint every N samples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum samples to process (-1 = all)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        default=False,
        help="Force regeneration of attacks even if Stage 1 already has attacked data",
    )
    
    args = parser.parse_args()
    
    synthesize_attacks(
        golden_trajectories_path=args.input,
        output_path=args.output,
        attacker_model_path=args.attacker_model,
        attacker_base_url=args.attacker_url,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        force_regenerate=args.force_regenerate,
    )


if __name__ == "__main__":
    main()
