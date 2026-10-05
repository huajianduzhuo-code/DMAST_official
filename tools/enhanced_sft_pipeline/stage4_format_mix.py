#!/usr/bin/env python3
"""
Stage 4: Data Formatting & Mixing
=================================

Create the final SFT training dataset by:
1. Formatting CoT-enhanced attacked data
2. Formatting clean data (standard trajectories)
3. Combining available agent samples with successful attacker samples
   (subject to max_samples for agent data)

KEY DESIGN: Instead of rebuilding messages from templates, we directly reuse
the original agent_messages from Stage 1 and only replace specific parts:
- For attacked samples: replace observation (clean → attacked) and think (→ defense_cot)
- For clean samples: use agent_messages as-is

This ensures perfect format consistency with the original imitation training data.

Input: 
- Stage 1 golden trajectories (with agent_messages)
- Stage 3 CoT-enhanced data (with defense_cot)

Output: Final SFT training dataset in the format expected by train.py
"""

import argparse
import json
import logging
import os
import sys
import re
import random
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from copy import deepcopy
from tqdm import tqdm

from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

# Import model family detection utility
from utils.vllm import get_model_family
from utils.training_data import validate_training_data_file
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# TOKENIZATION HELPERS
# ============================================================================

def _decode_base64_image(image_url: str) -> Image.Image:
    """Decode base64 image from data URL"""
    if image_url.startswith('data:image'):
        base64_data = image_url.split(',', 1)[1]
        image_data = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image URL format: {image_url[:50]}...")


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    """Extract PIL Images from multimodal messages."""
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    if url:
                        try:
                            image = _decode_base64_image(url)
                            images.append(image)
                        except Exception as e:
                            logging.error(f"Failed to decode image: {e}")
    return images


def _convert_messages_to_processor_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert saved message format to processor-expected format."""
    converted = []
    for msg in messages:
        new_msg = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        if isinstance(new_msg["content"], list):
            new_content = []
            for item in new_msg["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(item)
            new_msg["content"] = new_content
        converted.append(new_msg)
    return converted


def tokenize_multimodal_messages(
    messages: List[Dict[str, Any]],
    processor,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize multimodal messages into prompt and response token IDs.
    
    Args:
        messages: List of message dicts (system, user, assistant)
        processor: Transformers processor with apply_chat_template and tokenizer
        
    Returns:
        Tuple of (prompt_token_ids, response_token_ids)
    """
    # Convert to processor format
    messages_converted = _convert_messages_to_processor_format(messages)
    
    # Extract images
    images = _extract_images_from_messages(messages)
    
    # Split into prompt (system + user) and response (assistant)
    prompt_messages = messages_converted[:2]  # system + user
    response_message = messages_converted[2] if len(messages_converted) > 2 else None
    
    # Tokenize prompt with images
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    if images:
        # NOTE: add_special_tokens=False is critical because apply_chat_template
        # already adds <bos> token. Without this, we get duplicate BOS tokens.
        prompt_inputs = processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,  # Avoid duplicate BOS token
        )
        prompt_token_ids = prompt_inputs['input_ids'][0].tolist()
    else:
        # Text-only: add_special_tokens=False for the same reason
        prompt_token_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Tokenize response
    response_token_ids = []
    if response_message:
        response_content = response_message.get('content', '')
        if isinstance(response_content, list):
            # Extract text from multimodal content
            response_text = ' '.join(
                part.get('text', '') for part in response_content
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        else:
            response_text = str(response_content)
        
        response_token_ids = processor.tokenizer.encode(response_text, add_special_tokens=False)
    
    return prompt_token_ids, response_token_ids


@dataclass
class SFTSample:
    """
    A single SFT training sample.
    
    Format matches the existing imitation training infrastructure.
    """
    # Metadata
    episode_idx: int
    transition_idx: int
    task_name: str
    role: str  # "agent"
    
    # Training data
    messages: List[Dict[str, Any]]  # Full conversation including images
    is_multimodal: bool
    
    # Token IDs (will be populated during preprocessing)
    prompt_token_ids: List[int]
    response_token_ids: List[int]
    
    # Labels
    target_logprob: List[float]
    reward: float
    terminated: bool
    truncated: bool
    
    # Sample info
    on_policy: bool
    use_attacker: bool
    success: bool
    weight: float
    
    # Additional metadata
    sample_type: str  # "attacked_with_cot" or "clean"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def replace_observation_in_message(
    message: Dict[str, Any],
    new_axtree: str,
    new_screenshot_b64: Optional[str],
    old_axtree: str = None,
) -> Dict[str, Any]:
    """
    Replace observation (axtree and screenshot) in a user message.
    
    This preserves the original message structure and only swaps out the observation.
    """
    message = deepcopy(message)
    content = message.get('content', '')
    
    if isinstance(content, list):
        # Multimodal message - list of content items
        new_content = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    # Replace axtree in text content
                    text = item.get('text', '')
                    if old_axtree and old_axtree in text:
                        text = text.replace(old_axtree, new_axtree)
                    new_content.append({'type': 'text', 'text': text})
                elif item.get('type') == 'image_url' and new_screenshot_b64:
                    # Replace screenshot (use JPEG to match agent system format)
                    new_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{new_screenshot_b64}",
                            'detail': item.get('image_url', {}).get('detail', 'auto')
                        }
                    })
                else:
                    new_content.append(item)
            else:
                new_content.append(item)
        message['content'] = new_content
    elif isinstance(content, str):
        # Text-only message
        if old_axtree and old_axtree in content:
            message['content'] = content.replace(old_axtree, new_axtree)
    
    return message


def replace_think_in_response(
    message: Dict[str, Any],
    new_think: str,
    model_family: str = "gemma",
) -> Dict[str, Any]:
    """
    Replace or insert reasoning content in an assistant response.
    
    Handles different model formats:
    - Gemma: Uses explicit <think>...</think> tags - will replace content inside tags
    - Qwen: Does NOT use <think> tags - will prepend reasoning before action WITHOUT tags
    
    This ensures format consistency: if clean samples don't have <think> tags,
    attacked samples with CoT won't have them either.
    
    Preserves the action part of the response.
    
    Args:
        message: The assistant message dict
        new_think: The new thinking/reasoning content to insert
        model_family: "gemma", "qwen", or "unknown"
    """
    message = deepcopy(message)
    content = message.get('content', '')
    
    if isinstance(content, str):
        # Check if content has explicit <think> tags
        has_think_tags = bool(re.search(r'<think>.*?</think>', content, re.DOTALL))
        
        if has_think_tags:
            # Replace existing <think> content (for Gemma-style models)
            new_content = re.sub(
                r'<think>(.*?)</think>',
                f'<think>\n{new_think}\n</think>',
                content,
                flags=re.DOTALL
            )
            message['content'] = new_content
        else:
            # No <think> tags found - this is for models like Qwen
            # that output reasoning without explicit tags
            
            # Find the action part (if any)
            # IMPORTANT: Check <action>...</action> FIRST to preserve the tags if present
            action_patterns = [
                r'(<action>.*?</action>)',  # Must be first to preserve action tags
                r'(click\([^)]+\))',
                r'(fill\([^)]+\))',
                r'(scroll\([^)]+\))',
                r'(goto\([^)]+\))',
                r'(select_option\([^)]+\))',
                r'(send_msg_to_user\([^)]+\))',
                r'(drag_and_drop\([^)]+\))',
            ]
            
            action_match = None
            action_text = ""
            for pattern in action_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    action_match = match
                    action_text = match.group(0)
                    break
            
            # For Qwen and other models without native <think> tags:
            # Do NOT wrap in <think> tags to maintain format consistency with clean samples
            # Just prepend the reasoning before the action
            if action_match:
                # Found action - combine reasoning with action (no <think> tags)
                new_content = f"{new_think}\n\n{action_text}"
            else:
                # No action found - prepend reasoning to the content (no <think> tags)
                new_content = f"{new_think}\n\n{content.strip()}"
            
            message['content'] = new_content
    
    return message


def format_attacked_sample(
    golden_step: Dict[str, Any],
    cot_step: Dict[str, Any],
    episode_idx: int,
    transition_idx: int,
    processor=None,
    model_family: str = "gemma",
) -> Optional[Dict[str, Any]]:
    """
    Format an attacked sample by modifying the original agent_messages.
    
    Instead of rebuilding from templates, we:
    1. Copy the original agent_messages from golden_step
    2. Replace observation (clean → attacked) in user message
    3. Replace <think>...</think> content with defense_cot in assistant message
    4. Tokenize the modified messages to get token IDs
    
    This ensures perfect format consistency with original imitation training.
    
    Args:
        golden_step: Original step data from Stage 1
        cot_step: CoT-enhanced step from Stage 3
        episode_idx: Episode index for the sample
        transition_idx: Transition index within the episode
        processor: Transformers processor for tokenization (required for multimodal)
        model_family: "gemma", "qwen", or "unknown" - affects how think tags are handled
    """
    agent_messages = golden_step.get('agent_messages')
    if not agent_messages:
        logging.warning(f"No agent_messages in golden_step, skipping")
        return None
    
    # Get data for replacement
    clean_axtree = golden_step.get('clean_axtree', '')
    attacked_axtree = cot_step.get('attacked_axtree', '')
    attacked_screenshot_b64 = cot_step.get('attacked_screenshot_b64', '')
    defense_cot = cot_step.get('defense_cot', '')
    golden_action = cot_step.get('golden_action', '')
    task_name = cot_step.get('task_name', '')
    
    if not golden_action or not defense_cot:
        return None
    
    # Deep copy and modify messages
    messages = deepcopy(agent_messages)
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        
        if role == 'user':
            # Replace observation (clean → attacked)
            messages[i] = replace_observation_in_message(
                msg,
                new_axtree=attacked_axtree,
                new_screenshot_b64=attacked_screenshot_b64,
                old_axtree=clean_axtree,
            )
        elif role == 'assistant':
            # Replace think content with defense_cot
            # Pass model_family to handle different model formats
            messages[i] = replace_think_in_response(msg, defense_cot, model_family=model_family)
    
    # Check if multimodal
    is_multimodal = bool(attacked_screenshot_b64)
    
    # 🔑 Tokenize modified messages to get token IDs
    prompt_token_ids = []
    response_token_ids = []
    
    if processor is not None:
        try:
            prompt_token_ids, response_token_ids = tokenize_multimodal_messages(messages, processor)
            logging.debug(f"Tokenized attacked sample: prompt={len(prompt_token_ids)}, response={len(response_token_ids)}")
            
            # 🔑 FILTER: Skip samples with image tokens in response
            # Model-specific image token IDs that should NOT appear in response:
            IMAGE_TOKEN_IDS = [262144, 151655, 151656, 151657]
            if any(tok in response_token_ids for tok in IMAGE_TOKEN_IDS):
                num_image_tokens = sum(response_token_ids.count(tok) for tok in IMAGE_TOKEN_IDS)
                logging.warning(f"⚠️ SKIPPING attacked sample (episode={episode_idx}, transition={transition_idx}) - "
                               f"response contains {num_image_tokens} image tokens (corrupted data)")
                return None
        except Exception as e:
            logging.warning(f"Failed to tokenize attacked sample: {e}")
            # NOTE: This sample retains empty token IDs after tokenization failure;
            # it requires successful tokenization before training.
    else:
        logging.warning("No processor provided, attacked sample will have empty token IDs")
    
    sample = SFTSample(
        episode_idx=episode_idx,
        transition_idx=transition_idx,
        task_name=task_name,
        role="agent",
        messages=messages,
        is_multimodal=is_multimodal,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        target_logprob=[],  # Not used in SFT
        reward=1.0,  # Positive reward for correct action
        terminated=True,
        truncated=False,
        on_policy=True,
        use_attacker=True,  # This is attacked data
        success=True,  # Golden action is correct by definition
        weight=1.0,
        sample_type="attacked_with_cot",
    )
    
    return sample.to_dict()


def format_clean_sample(
    golden_step: Dict[str, Any],
    episode_idx: int,
    transition_idx: int,
    processor=None,
) -> Optional[Dict[str, Any]]:
    """
    Format a clean sample (no attack, standard trajectory).
    
    Directly uses the original agent_messages from Stage 1.
    No modification needed - this preserves perfect format consistency.
    
    🔑 Token IDs from Stage 1 are passed through directly for training consistency.
    
    Args:
        golden_step: Step data from Stage 1
        episode_idx: Episode index for the sample
        transition_idx: Transition index within the episode
        processor: Transformers processor for tokenization (only needed for fallback case)
    """
    agent_messages = golden_step.get('agent_messages')
    golden_action = golden_step.get('golden_action', '')
    task_name = golden_step.get('task_name', '')
    clean_screenshot_b64 = golden_step.get('clean_screenshot_b64', '')
    
    # 🔑 Get original token IDs from Stage 1
    prompt_token_ids = golden_step.get('prompt_token_ids', []) or []
    response_token_ids = golden_step.get('response_token_ids', []) or []
    
    if not golden_action:
        return None
    
    # Use original agent_messages directly
    if agent_messages:
        messages = deepcopy(agent_messages)
    else:
        # Fallback: build messages from available data (shouldn't happen normally)
        logging.warning(f"No agent_messages in golden_step, using fallback")
        task_instruction = golden_step.get('task_instruction', '')
        clean_axtree = golden_step.get('clean_axtree', '')
        agent_think = golden_step.get('agent_think', '')
        
        user_content = []
        user_content.append({
            "type": "text",
            "text": f"# Task\n{task_instruction}\n\n# Webpage State\n{clean_axtree[:4000]}"
        })
        if clean_screenshot_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{clean_screenshot_b64}", "detail": "auto"}
            })
        
        if agent_think:
            response = f"<think>\n{agent_think}\n</think>\n\n{golden_action}"
        else:
            response = golden_action
        
        messages = [
            {"role": "system", "content": "You are a web browsing agent."},
            {"role": "user", "content": user_content if clean_screenshot_b64 else user_content[0]["text"]},
            {"role": "assistant", "content": response}
        ]
        
        # 🔑 Tokenize fallback messages to get token IDs
        prompt_token_ids = []
        response_token_ids = []
        
        if processor is not None:
            try:
                prompt_token_ids, response_token_ids = tokenize_multimodal_messages(messages, processor)
                logging.debug(f"Tokenized fallback clean sample: prompt={len(prompt_token_ids)}, response={len(response_token_ids)}")
            except Exception as e:
                logging.warning(f"Failed to tokenize fallback clean sample: {e}")
        else:
            logging.warning("No processor provided for fallback clean sample, will have empty token IDs")
    
    # 🔑 FILTER: Skip samples with image tokens in response
    IMAGE_TOKEN_IDS = [262144, 151655, 151656, 151657]
    if any(tok in response_token_ids for tok in IMAGE_TOKEN_IDS):
        num_image_tokens = sum(response_token_ids.count(tok) for tok in IMAGE_TOKEN_IDS)
        logging.warning(f"⚠️ SKIPPING clean sample (episode={episode_idx}, transition={transition_idx}) - "
                       f"response contains {num_image_tokens} image tokens (corrupted data)")
        return None
    
    # Check if multimodal
    is_multimodal = bool(clean_screenshot_b64)
    
    sample = SFTSample(
        episode_idx=episode_idx,
        transition_idx=transition_idx,
        task_name=task_name,
        role="agent",
        messages=messages,
        is_multimodal=bool(clean_screenshot_b64),
        # 🔑 Use original token IDs from Stage 1 for clean samples
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        target_logprob=[],
        reward=1.0,
        terminated=True,
        truncated=False,
        on_policy=True,
        use_attacker=False,  # This is clean data
        success=True,
        weight=1.0,
        sample_type="clean",
    )
    
    return sample.to_dict()


def _load_and_filter_attacker_data(attacker_data_path: str) -> List[Dict[str, Any]]:
    """
    Load attacker data and filter for successful attacks.
    
    The attacker data format (from collect_episodes) is a list of episodes,
    where each episode has a 'history' containing steps. We need to extract
    attacker-specific data from each step.
    
    Args:
        attacker_data_path: Path to Stage 1b attacker data JSON
        
    Returns:
        List of attacker samples ready for SFT training
    """
    raw_data = validate_training_data_file(attacker_data_path)
    if isinstance(raw_data, dict) and not any(
        key in raw_data for key in ('samples', 'attacker_steps')
    ):
        raise ValueError(
            f"{attacker_data_path}: attacker data requires an episode list, samples, or attacker_steps"
        )
    
    attacker_samples = []
    
    # Handle different data formats
    if isinstance(raw_data, list):
        # Format from collect_episodes: list of episodes
        for ep_idx, episode in enumerate(raw_data):
            # Skip non-attacker episodes
            if not episode.get('use_attacker', False):
                continue
            
            # Check if episode is attacker-successful
            # Read the attacker's independently recorded success flag.
            reward_summary = episode.get('reward_summary', {})
            attacker_success = reward_summary.get('attacker_success', False)
            
            # For records without reward_summary, infer attacker success from
            # agent failure; this fallback assumes mutually exclusive outcomes.
            if not reward_summary:
                final_reward = episode.get('reward', episode.get('final_reward', 0))
                agent_success = episode.get('success', final_reward > 0)
                attacker_success = not agent_success
            
            if not attacker_success:
                continue  # Skip episodes where attacker failed
            
            history = episode.get('history', [])
            task_name = episode.get('task_name', '')
            
            for step_idx, step in enumerate(history):
                # Extract attacker data from step
                attacker_messages = step.get('attacker_messages')
                if not attacker_messages:
                    continue
                
                # Get attacker response token IDs from attacker_training_info
                # The token IDs are stored inside attacker_training_info, not at step level
                attacker_training_info = step.get('attacker_training_info', {}) or {}
                attacker_prompt_ids = attacker_training_info.get('prompt_token_ids', []) or []
                attacker_response_ids = attacker_training_info.get('response_token_ids', []) or []
                
                # 🔑 FILTER: Skip samples with image tokens in response
                IMAGE_TOKEN_IDS = [262144, 151655, 151656, 151657]
                if any(tok in attacker_response_ids for tok in IMAGE_TOKEN_IDS):
                    num_image_tokens = sum(attacker_response_ids.count(tok) for tok in IMAGE_TOKEN_IDS)
                    logging.warning(f"⚠️ SKIPPING attacker sample (episode={ep_idx}, step={step_idx}) - "
                                   f"response contains {num_image_tokens} image tokens (corrupted data)")
                    continue
                
                # Check if multimodal
                is_multimodal = False
                for msg in attacker_messages:
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                is_multimodal = True
                                break
                    if is_multimodal:
                        break
                
                sample = {
                    'episode_idx': ep_idx,
                    'transition_idx': step_idx,
                    'task_name': task_name,
                    'role': 'attacker',
                    'on_policy': True,
                    'use_attacker': True,
                    'seed': episode.get('seed'),
                    'success': True,  # Attacker success (we filtered for this)
                    'reward': 1.0,  # Attacker reward
                    'terminated': step.get('terminated', False),
                    'truncated': step.get('truncated', False),
                    'messages': attacker_messages,
                    'is_multimodal': is_multimodal,
                    'prompt_token_ids': attacker_prompt_ids,
                    'response_token_ids': attacker_response_ids,
                    'weight': 1.0,
                    'sample_type': 'attacker',
                }
                
                attacker_samples.append(sample)
    
    elif isinstance(raw_data, dict):
        # Format: dict with 'samples' key (already processed)
        samples = raw_data.get('samples', raw_data.get('attacker_steps', []))
        for sample in samples:
            if sample.get('role') == 'attacker' and sample.get('success', False):
                sample['sample_type'] = 'attacker'
                attacker_samples.append(sample)
    
    return attacker_samples


def format_and_mix_data(
    golden_trajectories_path: str,
    cot_data_path: str,
    output_path: str,
    model_path: str = None,
    attacker_data_path: str = None,
    shuffle: bool = True,
    seed: int = 42,
    max_samples: int = -1,
):
    """
    Format and mix clean and attacked data for SFT training.
    
    Uses ALL available data from sources:
    - Agent clean samples: from use_attacker=False episodes (no attacked data)
    - Agent attacked samples: from Stage 3 CoT data (with defense reasoning)
    - Attacker samples (optional): from Stage 1b attacker collection
    
    Args:
        golden_trajectories_path: Path to Stage 1 clean data
        cot_data_path: Path to Stage 3 CoT-enhanced data
        output_path: Where to save the final SFT dataset
        model_path: Path to the model for tokenization (required for attacked samples)
        attacker_data_path: Optional path to Stage 1b attacker data
        shuffle: Whether to shuffle the final dataset
        seed: Random seed for reproducibility
        max_samples: Maximum total samples (-1 = all)
    """
    print("=" * 80)
    print("📦 Stage 4: Data Formatting & Mixing")
    print("=" * 80)
    
    random.seed(seed)

    # Validate every source before sampling, success filtering, or model loading.
    print(f"\n📂 Loading data...")
    print(f"   Loading golden data from {golden_trajectories_path}...")
    golden_data = validate_training_data_file(golden_trajectories_path)
    if not isinstance(golden_data, dict) or 'golden_steps' not in golden_data:
        raise ValueError(f"{golden_trajectories_path}: Stage 4 requires a golden_steps wrapper")
    all_golden_steps = golden_data['golden_steps']
    print(f"   → {len(all_golden_steps)} total golden steps")

    print(f"   Loading CoT data from {cot_data_path}...")
    cot_data = validate_training_data_file(cot_data_path)
    if not isinstance(cot_data, dict) or 'cot_steps' not in cot_data:
        raise ValueError(f"{cot_data_path}: Stage 4 requires a cot_steps wrapper")
    cot_steps = cot_data['cot_steps']
    print(f"   → {len(cot_steps)} attacked steps with CoT")

    attacker_samples = []
    if attacker_data_path:
        print(f"\n🎭 Loading attacker data from {attacker_data_path}...")
        attacker_samples = _load_and_filter_attacker_data(attacker_data_path)
        print(f"   → {len(attacker_samples)} attacker samples loaded (successful attacks only)")
    
    # Load processor for tokenization and detect model family
    processor = None
    model_family = "unknown"
    if model_path:
        print(f"\n🔧 Loading processor from {model_path}...")
        
        # Detect model family for format handling
        model_family = get_model_family(model_path)
        print(f"   🔍 Detected model family: {model_family}")
        
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print(f"   ✅ Processor loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load processor: {e}")
            print(f"   ⚠️ Failed to load processor: {e}")
            print(f"   ⚠️ Attacked samples will have empty token IDs")
    else:
        print(f"\n⚠️ No model_path provided, attacked samples will have empty token IDs")
    
    # Partition samples by the presence of an attacked screenshot.
    # NOTE: Screenshot presence is a proxy for sample type, not an episode-mode check.
    # Steps without attacked screenshots are treated as clean samples.
    # Attacked samples use the matching Stage 3 CoT data.
    clean_only_steps = [step for step in all_golden_steps if not step.get('attacked_screenshot_b64')]
    attacked_golden_steps = [step for step in all_golden_steps if step.get('attacked_screenshot_b64')]
    
    print(f"   → {len(clean_only_steps)} pure clean steps (from use_attacker=False episodes)")
    print(f"   → {len(attacked_golden_steps)} steps with attacked data (from use_attacker=True episodes)")
    
    # Build index for quick golden_step lookup by (episode_idx, step_idx)
    # Only index attacked steps (for matching with CoT data)
    # NOTE: This assumes attacked views already exist in the original golden data.
    # Stage 2's offline fallback does not update that data, so CoT samples synthesized
    # from originally clean-only steps will miss this index and be skipped below.
    # Use browser-paired Stage 1 data; the offline fallback is not recommended.
    golden_steps_index = {}
    for step in attacked_golden_steps:
        key = (step.get('episode_idx'), step.get('step_idx'))
        golden_steps_index[key] = step
    
    # Use ALL available data (no ratio-based sampling)
    # - Attacked samples: all cot_steps
    # - Clean samples: all clean_only_steps (without attacked screenshots)
    num_attacked = len(cot_steps)
    num_clean = len(clean_only_steps)
    total_available = num_attacked + num_clean
    
    # Apply max_samples limit if specified (proportionally)
    if max_samples > 0 and total_available > max_samples:
        ratio = max_samples / total_available
        num_attacked = int(num_attacked * ratio)
        num_clean = int(num_clean * ratio)
        # Ensure we don't exceed max_samples
        if num_attacked + num_clean > max_samples:
            num_clean = max_samples - num_attacked
    
    print(f"\n📊 Dataset Composition:")
    print(f"   Attacked samples (with CoT): {num_attacked}")
    print(f"   Clean samples: {num_clean}")
    print(f"   Total samples: {num_attacked + num_clean}")
    if num_attacked + num_clean > 0:
        actual_ratio = num_attacked / (num_attacked + num_clean)
        print(f"   Actual ratio: {actual_ratio * 100:.1f}% attacked / {(1 - actual_ratio) * 100:.1f}% clean")
    
    # Format attacked data (use all or limited by max_samples)
    print(f"\n🎭 Formatting attacked samples...")
    selected_cot_steps = cot_steps[:num_attacked]
    
    attacked_samples = []
    skipped_no_golden = 0
    for i, cot_step in enumerate(tqdm(selected_cot_steps, desc="Formatting attacked")):
        # Find corresponding golden_step for agent_messages
        key = (cot_step.get('episode_idx'), cot_step.get('step_idx'))
        golden_step = golden_steps_index.get(key)
        
        if not golden_step:
            skipped_no_golden += 1
            continue
        
        sample = format_attacked_sample(
            golden_step=golden_step,
            cot_step=cot_step,
            episode_idx=i,
            transition_idx=cot_step.get('step_idx', 0),
            processor=processor,  # Pass processor for tokenization
            model_family=model_family,  # Pass model family for format handling
        )
        if sample:
            attacked_samples.append(sample)
    
    if skipped_no_golden > 0:
        logging.warning(f"Skipped {skipped_no_golden} CoT steps without matching golden_step")
    print(f"   → {len(attacked_samples)} attacked samples formatted")
    
    # Format clean data (use all or limited by max_samples)
    # Clean samples are steps without attacked screenshots.
    print(f"\n✨ Formatting clean samples...")
    selected_golden_steps = clean_only_steps[:num_clean]
    
    clean_samples = []
    for i, step in enumerate(tqdm(selected_golden_steps, desc="Formatting clean")):
        sample = format_clean_sample(
            golden_step=step,
            episode_idx=len(attacked_samples) + i,  # Continue episode numbering
            transition_idx=step.get('step_idx', 0),
            processor=processor,  # Pass processor for tokenization (only needed for fallback)
        )
        if sample:
            clean_samples.append(sample)
    
    print(f"   → {len(clean_samples)} clean samples formatted")
    
    # Add role field to agent samples
    for sample in attacked_samples:
        sample['role'] = 'agent'
    for sample in clean_samples:
        sample['role'] = 'agent'
    
    # Combine and shuffle
    all_samples = attacked_samples + clean_samples + attacker_samples

    if not all_samples:
        raise ValueError(
            "No usable SFT samples remain after formatting and filtering. "
            "Check the input data and preceding warnings."
        )
    
    if shuffle:
        print(f"\n🔀 Shuffling {len(all_samples)} samples...")
        random.shuffle(all_samples)
        
        # Re-index episode_idx after shuffle
        for i, sample in enumerate(all_samples):
            sample['episode_idx'] = i
    
    # Calculate final statistics
    final_attacked = sum(1 for s in all_samples if s.get('sample_type') == 'attacked_with_cot')
    final_clean = sum(1 for s in all_samples if s.get('sample_type') == 'clean')
    final_attacker = sum(1 for s in all_samples if s.get('sample_type') == 'attacker')
    final_multimodal = sum(1 for s in all_samples if s.get('is_multimodal'))
    final_agent = sum(1 for s in all_samples if s.get('role') == 'agent')
    final_attacker_role = sum(1 for s in all_samples if s.get('role') == 'attacker')
    
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   Total samples: {len(all_samples)}")
    print(f"\n   By role:")
    print(f"   └─ Agent: {final_agent} ({final_agent / len(all_samples) * 100:.1f}%)")
    print(f"   └─ Attacker: {final_attacker_role} ({final_attacker_role / len(all_samples) * 100:.1f}%)")
    print(f"\n   By sample type:")
    print(f"   └─ Agent attacked (with CoT): {final_attacked}")
    print(f"   └─ Agent clean: {final_clean}")
    print(f"   └─ Attacker: {final_attacker}")
    print(f"\n   Multimodal: {final_multimodal} ({final_multimodal / len(all_samples) * 100:.1f}%)")
    
    # Save the dataset
    output_data = {
        "metadata": {
            "stage": "final_sft_dataset",
            "golden_source": golden_trajectories_path,
            "cot_source": cot_data_path,
            "attacker_source": attacker_data_path if attacker_data_path else None,
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            # By role
            "num_agent_samples": final_agent,
            "num_attacker_samples": final_attacker_role,
            # By sample type
            "num_attacked_with_cot_samples": final_attacked,
            "num_clean_samples": final_clean,
            "num_attacker_role_samples": final_attacker,
            # Other stats
            "num_multimodal_samples": final_multimodal,
            "total_samples": len(all_samples),
        },
        "samples": all_samples,
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Final SFT dataset saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Also save in simplified format for direct use with train.py
    simplified_path = output_path.replace('.json', '_simplified.json')
    with open(simplified_path, 'w') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"   Simplified format: {simplified_path}")
    
    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Stage 4: Format and Mix Data for SFT")
    parser.add_argument(
        "--golden-input",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/stage1_golden_trajectories.json"),
        help="Path to Stage 1 golden trajectories",
    )
    parser.add_argument(
        "--cot-input",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/stage3_cot_data.json"),
        help="Path to Stage 3 CoT-enhanced data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "bucket/data/enhanced_sft/final_sft_dataset.json"),
        help="Output path for final SFT dataset",
    )
    parser.add_argument(
        "--attacker-input",
        type=str,
        default=None,
        help="Optional path to Stage 1b attacker data (if not provided, only agent data is used)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the final dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum total samples (-1 = all)",
    )
    
    args = parser.parse_args()
    
    format_and_mix_data(
        golden_trajectories_path=args.golden_input,
        cot_data_path=args.cot_input,
        output_path=args.output,
        attacker_data_path=args.attacker_input,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
