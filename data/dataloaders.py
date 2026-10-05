from copy import deepcopy
import logging
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.utils import IGNORE_INDEX

logger = logging.getLogger(__name__)


def _decode_base64_image(image_url: str) -> Image.Image:
    """Decode base64 image from data URL"""
    if image_url.startswith('data:image'):
        # Extract base64 data after the comma
        base64_data = image_url.split(',', 1)[1]
        image_data = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image URL format: {image_url[:50]}...")


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    """
    Extract PIL Images from multimodal messages.
    
    Args:
        messages: List of message dicts with potential image content
        
    Returns:
        List of PIL Image objects found in the messages
    """
    images = []
    for msg in messages:
        content = msg.get("content", "")
        
        if isinstance(content, list):
            # Multimodal message - look for images
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
                            logger.error(f"Failed to decode image: {e}")
    
    return images


def _convert_messages_to_processor_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert saved message format to processor-expected format.
    
    Saved format:     {"type": "image_url", "image_url": {"url": "..."}}
    Processor format: {"type": "image"}
    
    This is critical for the processor's chat template to recognize images and insert
    proper image tokens (like <image> or <start_of_image>).
    """
    converted = []
    for msg in messages:
        new_msg = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        
        # Convert content format if it's a list (multimodal)
        if isinstance(new_msg["content"], list):
            new_content = []
            for item in new_msg["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # Convert image_url to simple image marker
                        new_content.append({"type": "image"})
                    else:
                        # Keep text and other types as-is
                        new_content.append(item)
            new_msg["content"] = new_content
        
        converted.append(new_msg)
    
    return converted


def _format_messages_for_processor(messages: List[Dict[str, Any]]) -> str:
    """
    Format messages into text for processor.
    For multimodal messages, this extracts only the text portions.
    """
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if isinstance(content, str):
            # Simple text message
            text_parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            # Multimodal message - extract text portions
            msg_text = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    msg_text.append(item.get("text", ""))
            if msg_text:
                text_parts.append(f"{role}: {' '.join(msg_text)}")
    
    return "\n".join(text_parts)


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __getitem__(self, index):
        data = self.data[index]
        data["index"] = index
        return data

    def __len__(self,):
        return len(self.data)


def sft_data_collactor(batch, tokenizer, prepare_value_input_ids=False, overwrite_labels=False, processor=None):
    """
    Data collator that handles both text-only and multimodal inputs.
    
    Args:
        batch: List of data items
        tokenizer: Tokenizer for text-only processing
        prepare_value_input_ids: Whether to prepare value input IDs for value models
        overwrite_labels: Whether to overwrite labels for value training
        processor: Optional processor for multimodal inputs (handles text + images)
    
    Returns:
        Dict with input_ids, labels, and optionally pixel_values for multimodal
    """
    # Check if any items are multimodal
    has_multimodal = any(item.get('is_multimodal', False) for item in batch)
    
    
    # Decide processing mode
    use_multimodal_processing = has_multimodal and processor is not None
    
    # If multimodal but no processor, raise error
    if has_multimodal and not use_multimodal_processing:
        logging.error("❌ Processor is required for multimodal inputs, but not provided")
        raise ValueError("❌ Processor is required for multimodal inputs, but not provided")
    
    input_ids, label = [], []
    value_input_ids = []
    pixel_values_list = []  # For multimodal
    image_grid_thw_list = []  # For Qwen3-VL models (stores image grid info)
    has_images = False

    query_token_lens, target_token_lens = [], []

    for item in batch:
        is_multimodal_item = item.get('is_multimodal', False)
        messages = item.get('messages') if is_multimodal_item else None
        
        # MULTIMODAL PATH: Use processor to handle text + images
        if use_multimodal_processing and is_multimodal_item and messages:
            logger.debug(f"🖼️ Processing multimodal item (episode {item.get('episode_idx')}, transition {item.get('transition_idx')})")
            
            try:
                # ============================================================================
                # CRITICAL IMAGE TOKEN HANDLING FOR VLM MODELS
                # ============================================================================
                # 1. Saved format has: {"type": "image_url", "image_url": {"url": "data:..."}}
                # 2. Convert to processor format: {"type": "image"}
                # 3. Chat template converts: {"type": "image"} → <image> tokens (or model-specific)
                # 4. Processor replaces <image> tokens with actual image embeddings
                # 5. Model receives: [text_tokens, IMAGE_EMBEDDING, more_text_tokens]
                # ============================================================================
                
                # Step 1: Convert to processor format
                # NOTE: this only modifies the image url part. It converts the content of:
                #     {'type': 'image_url',
                #      'image_url': {'url': 'data:...'}}
                # to:
                #     {'type': 'image'}
                # and leaves the rest of the content unchanged.
                messages_converted = _convert_messages_to_processor_format(messages)
                
                # Step 2: Extract actual image data (PIL Images)
                # NOTE: Samples are expected to contain one screenshot per item.
                images = _extract_images_from_messages(messages)
                
                if images:
                    has_images = True
                    
                    # Get prompt messages (first 2) and response (3rd)
                    prompt_messages = messages_converted[:2]
                    response_message = messages_converted[2]
                    
                    # CRITICAL: Apply chat template to messages (preserves image placeholders)
                    # The chat template sees {"type": "image"} and converts to proper token <start_of_image>. For texts, it concates them together with proper combining tokens, e,g., <bos>, <end_of_turn>, <start_of_turn>, etc.
                    # Example prompt text with image and turn-boundary tokens:
                    #       "Your visual analysis should be as detailed as your textual analysis:<start_of_image><end_of_turn>\n<start_of_turn>model\n"
                    prompt_text = processor.apply_chat_template(
                        prompt_messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    logger.debug(f"   📝 Generated prompt with image tokens: {prompt_text[:200]}...")
                    
                    # Process prompt with images
                    # The processor will replace image tokens with actual image embeddings
                    # NOTE: prompt_inputs has attributes: pixel_values, input_ids, attention_mask, token_type_ids.
                    #       There will be 256 <image_soft_token>s in the input_ids.
                    #       pixel_values is a tensor of shape (1, 3, 896, 896), regardless of the original image shape.
                    #       token_type_ids is a tensor indicating token type of input_ids, which is 0 for others and 1 for <image_soft_token>.
                    
                    # Extract pixel_values using processor (pixel_values are independent of text)
                    # We only need pixel_values, not input_ids
                    prompt_inputs = processor(
                        text=prompt_text,  # Still needed for processor to know where images go
                        images=images,
                        return_tensors="pt",
                        padding=False
                    )
                    # ============================================================================
                    # CRITICAL: Use token IDs from JSON directly for training consistency
                    # ============================================================================
                    # Token IDs must be pre-computed during data generation (Stage 1 for clean,
                    # Stage 4 for attacked samples) and stored in the JSON file.
                    # This ensures training data is self-contained and training process is independent.
                    prompt_ids = item.get('prompt_token_ids')
                    response_ids = item.get('response_token_ids')
                    
                    if not prompt_ids or not response_ids:
                        raise ValueError(
                            f"Missing token IDs in sample (episode={item.get('episode_idx')}, "
                            f"transition={item.get('transition_idx')}, type={item.get('sample_type', 'unknown')}). "
                            f"prompt_token_ids={len(prompt_ids) if prompt_ids else 0}, "
                            f"response_token_ids={len(response_ids) if response_ids else 0}. "
                            f"Please regenerate the dataset with token IDs included."
                        )
                    
                    # ============================================================================
                    # CRITICAL: Skip samples with image tokens in response_ids
                    # ============================================================================
                    # Sometimes the model may incorrectly generate image tokens in the response.
                    # This causes "Image features and image tokens do not match" error because:
                    # - prompt has N image tokens -> N image features
                    # - response has M image tokens -> M extra tokens but no features
                    # - total: N+M tokens vs N features -> MISMATCH!
                    # Drop samples whose response tokens require unavailable image features.
                    # Model-specific image token IDs:
                    IMAGE_TOKEN_IDS = [
                        262144,  # Gemma3 image soft token
                        151655,  # Qwen3-VL vision start token <|vision_start|>
                        151656,  # Qwen3-VL vision end token <|vision_end|>
                        151657,  # Qwen3-VL vision pad token <|vision_pad|>
                    ]
                    has_image_token_in_response = any(tok in response_ids for tok in IMAGE_TOKEN_IDS)
                    if has_image_token_in_response:
                        num_image_tokens = sum(response_ids.count(tok) for tok in IMAGE_TOKEN_IDS)
                        logger.warning(f"⚠️ SKIPPING sample with {num_image_tokens} image tokens in response_ids (episode {item.get('episode_idx')}, transition {item.get('transition_idx')})")
                        continue  # Skip this corrupted sample
                    
                    query_token_len = len(prompt_ids)
                    target_token_len = len(response_ids)
                    
                    query_token_lens.append(query_token_len)
                    target_token_lens.append(target_token_len)
                    
                    # Store input IDs and pixel values
                    input_ids.append(prompt_ids + response_ids)
                    label.append([IGNORE_INDEX] * query_token_len + response_ids)
                    
                    # Handle pixel_values based on model type:
                    # - Gemma3: processor may return different shapes:
                    #   * Single image: (C, H, W) or (1, C, H, W)
                    #   * Multiple images: (num_images, C, H, W) or (1, num_images, C, H, W)
                    # - Qwen3-VL: processor returns (num_patches, hidden_dim)
                    pixel_values = prompt_inputs['pixel_values']
                    if 'image_grid_thw' in prompt_inputs:
                        # Qwen3-VL: pixel_values is (num_patches, hidden_dim)
                        pixel_values_list.append(pixel_values)
                        image_grid_thw_list.append(prompt_inputs['image_grid_thw'])
                    else:
                        # Gemma3: normalize to (num_images, C, H, W)
                        if pixel_values.dim() == 5:
                            # (1, num_images, C, H, W) -> (num_images, C, H, W)
                            pixel_values = pixel_values[0]
                        elif pixel_values.dim() == 3:
                            # (C, H, W) -> (1, C, H, W)
                            pixel_values = pixel_values.unsqueeze(0)
                    # pixel_values has shape (num_images, C, H, W).
                        pixel_values_list.append(pixel_values)
                    
                    if prepare_value_input_ids:
                        value_input_ids.append(prompt_ids + [tokenizer.eos_token_id])
                    
                    logger.debug(f"   ✅ Processed {len(images)} images, prompt_tokens={query_token_len}, response_tokens={target_token_len}")
                    continue  # Skip text-only path
                    
            except Exception as e:
                logger.error(f"❌ Failed to process multimodal item: {e}")
                logger.info("   Falling back to text-only tokens for this item")
                # Fall through to text-only path
        
        # TEXT-ONLY PATH: Use token-based processing.
        # query_tokens = item['query_tokens']
        # target_tokens = item['target_tokens']

        # query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        # query_token_len = len(query_token_ids)
        # query_token_lens.append(query_token_len)

        # target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        # target_token_len = len(target_token_ids)
        # target_token_lens.append(target_token_len)

        # assert tokenizer.unk_token_id not in target_token_ids, \
        #     f"target_token_ids contains unk token: {target_tokens[target_token_ids.index(tokenizer.unk_token_id)]}"

        # input_ids.append(
        #     deepcopy(query_token_ids) + deepcopy(target_token_ids)
        # )
        # label.append(
        #     [IGNORE_INDEX] * query_token_len + deepcopy(target_token_ids)
        # )
        
        # # For multimodal batches with mixed content, add placeholder for text-only items
        # if has_images:
        #     # Add zero-filled pixel_values as placeholder (will be masked out)
        #     pixel_values_list.append(None)

        # if prepare_value_input_ids:
        #     value_input_ids.append(
        #         deepcopy(query_token_ids) + [tokenizer.eos_token_id]
        #     )

    # Handle case where all items in batch were skipped (e.g., due to corrupted data)
    # CRITICAL: In distributed training, we CANNOT return None because all ranks must
    # participate in the same number of forward/backward passes for gradient all-reduce.
    # Instead, return a dummy batch with labels=IGNORE_INDEX so num_items_in_batch=0.
    if len(input_ids) == 0:
        logger.warning(f"⚠️ All {len(batch)} items in batch were skipped - returning empty batch (num_items=0)")
        
        # Create a minimal dummy batch that the model can process as TEXT-ONLY
        # The labels are all IGNORE_INDEX, so no loss will be computed
        # IMPORTANT: Do NOT include pixel_values here - the input_ids don't have image tokens,
        # so providing pixel_values would cause "Image features and image tokens do not match" error.
        dummy_ids = [tokenizer.pad_token_id or tokenizer.eos_token_id]
        dummy_batch = {
            "input_ids": torch.tensor([dummy_ids]).long(),
            "attention_mask": torch.tensor([[1.0]]).float(),
            "position_ids": torch.tensor([[0]]).long(),
            "labels": torch.tensor([[IGNORE_INDEX]]).long(),  # All ignored -> num_items_in_batch = 0
            "index": torch.tensor([batch[0].get('index', 0)]).long(),
            "weight": torch.tensor([1.0]).float(),
            "query_token_lens": [1],
            "target_token_lens": [0],
        }
        
        return dummy_batch

    input_batch = batch_padding(input_ids, tokenizer)
    label_batch = batch_padding(label, tokenizer, pad_token_id=IGNORE_INDEX)

    outputs = {
        "input_ids": torch.Tensor(input_batch['input_ids']).long(),
        "attention_mask": torch.Tensor(input_batch['attention_mask']).float(),
        "position_ids": torch.Tensor(input_batch['position_ids']).long(),
        "labels": torch.Tensor(label_batch['input_ids']).long(),
        "index": torch.Tensor([item['index'] for item in batch]).long(),
        "weight": torch.Tensor([item.get("weight", 1.) for item in batch]).float(),
        "query_token_lens": query_token_lens,
        "target_token_lens":target_token_lens,
    }
    
    # Add pixel_values for multimodal inputs
    if has_images and pixel_values_list:
        # Filter out None placeholders and stack valid pixel_values
        assert None not in pixel_values_list, "❌ pixel_values_list contains None"
        try:
            # Concatenate pixel_values (different models have different shapes)
            # Gemma3: Each item is (num_images, C, H, W), stack to (batch, num_images, C, H, W)
            # Qwen3-VL: Each item is (num_patches, hidden_dim), concatenate to (total_patches, hidden_dim)
            if pixel_values_list[0].dim() == 2:
                # Qwen3-VL: (num_patches, hidden_dim) - concatenate patches from all samples
                outputs['pixel_values'] = torch.cat(pixel_values_list, dim=0)
                logger.info(f"✅ Added Qwen pixel_values: shape={outputs['pixel_values'].shape}")
            elif pixel_values_list[0].dim() == 4:
                # Gemma3: (num_images, C, H, W) - stack then flatten to (total_images, C, H, W)
                # SigLIP vision encoder expects (batch_of_images, C, H, W), not (batch, num_images, C, H, W)
                stacked = torch.stack(pixel_values_list, dim=0)  # (batch, num_images, C, H, W)
                outputs['pixel_values'] = stacked.view(-1, *stacked.shape[2:])  # (batch*num_images, C, H, W)
                logger.info(f"✅ Added Gemma pixel_values: shape={outputs['pixel_values'].shape}")
            else:
                # Unexpected shape - log warning and try stacking
                logger.warning(f"⚠️ Unexpected pixel_values shape: {pixel_values_list[0].shape}")
                outputs['pixel_values'] = torch.stack(pixel_values_list, dim=0)
            logger.debug(f"pixel_values shape: {outputs['pixel_values'].shape}")
        except Exception as e:
            logger.error(f"❌ Could not process pixel_values: {e}")
            # Store as list for models that can handle it
            outputs['pixel_values'] = pixel_values_list
            raise e
    
    # Add image_grid_thw for Qwen3-VL models
    if image_grid_thw_list:
        try:
            outputs['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)
            logger.info(f"✅ Added image_grid_thw to batch: shape={outputs['image_grid_thw'].shape}")
        except Exception as e:
            logger.error(f"❌ Could not process image_grid_thw: {e}")
            raise e

    if prepare_value_input_ids:
        value_input_batch = batch_padding(value_input_ids, tokenizer)
        outputs['value_input_ids'] = torch.Tensor(value_input_batch['input_ids']).long()
        outputs['value_attention_mask'] = torch.Tensor(value_input_batch['attention_mask']).float()
        outputs['value_position_ids'] = torch.Tensor(value_input_batch['position_ids']).long()

        if overwrite_labels:
            # overwrite labels, as trainer will use labels to compute num_items_in_batch
            labels = torch.ones_like(outputs['value_input_ids']) * IGNORE_INDEX
            value_input_ids = outputs['value_input_ids']
            labels[value_input_ids == tokenizer.eos_token_id] = tokenizer.eos_token_id
            outputs['labels'] = labels

    return outputs


def policy_data_collactor(batch, tokenizer, processor=None):
    results = sft_data_collactor(batch, tokenizer, prepare_value_input_ids=False, processor=processor)
    
    # Handle case where all items were skipped (empty batch with dummy data)
    # Check if this is a dummy batch (target_token_lens[0] == 0)
    if results['target_token_lens'][0] == 0:
        # Return results with empty logprob arrays for consistency
        results.update({
            'ref_logprob': torch.zeros((1, 1)).float(),
            'generation_logprob': torch.zeros((1, 1)).float(),
            'sampling_logprob': torch.zeros((1, 1)).float(),
            'advantage': torch.zeros((1,)).float(),
            'sft_mask': torch.zeros((1, 1)).float(),
        })
        return results

    batch_size, seq_len = results['labels'].shape
    
    # Determine logprob array size based on whether we have query tokens
    # Text-only: seq_len includes prompt, so logprob_size = seq_len - 1 (excludes 1st prompt token)
    # Multimodal: seq_len is only response, so logprob_size = seq_len (includes all response logprobs)
    # NOTE: Query lengths exclude leading padding; GRPO uses a per-device batch size of 1.
    has_query_tokens = any(results['query_token_lens'])  # True if any sample has query tokens.
    logprob_size = seq_len - 1 if has_query_tokens else seq_len
    
    ref_logprob = np.zeros((batch_size, logprob_size))
    generation_logprob = np.zeros((batch_size, logprob_size))
    sampling_logprob = np.zeros((batch_size, logprob_size))
    advantage = []
    
    # Create SFT mask for SPAG regularization
    # sft_mask marks response tokens (not query tokens) for SFT loss computation
    # Shape: (batch_size, logprob_size)
    sft_mask = np.zeros((batch_size, logprob_size))

    for i, item in enumerate(batch):
        # For multimodal models, query_token_lens may be 0 (no prompt tokens extracted)
        query_token_len_raw = results['query_token_lens'][i]
        target_token_len = results['target_token_lens'][i]
        
        if query_token_len_raw > 0:
            # Text-only case: query_token_len-1 points to last prompt position
            # that predicts the first response token
            query_token_len = query_token_len_raw - 1
        else:
            # Multimodal case: no query tokens extracted, but response_logprobs[0]
            # IS the logprob for generating the first response token (from the prompt).
            # We include all response logprobs starting from position 0.
            query_token_len = 0
        
        # Available space in the logprob array
        available_space = logprob_size - query_token_len

        if 'ref_logprob' in item:
            ref_logprobs = item['ref_logprob']
            assert len(ref_logprobs)<=available_space, f"❌ ref_logprob length {len(ref_logprobs)} is greater than available_space {available_space}"
            ref_logprob[i, query_token_len: query_token_len + len(ref_logprobs)] = ref_logprobs
            
        if 'generation_logprob' in item:
            gen_logprobs = item['generation_logprob']
            assert len(gen_logprobs)<=available_space, f"❌ generation_logprob length {len(gen_logprobs)} is greater than available_space {available_space}"
            generation_logprob[i, query_token_len: query_token_len + len(gen_logprobs)] = gen_logprobs
            
        if 'target_logprob' in item:
            target_logprobs = item['target_logprob']
            assert len(target_logprobs)<=available_space, f"❌ target_logprob length {len(target_logprobs)} is greater than available_space {available_space}"
            sampling_logprob[i, query_token_len: query_token_len + len(target_logprobs)] = target_logprobs

        advantage.append(item['advantage'])
        
        # Fill SFT mask for response tokens
        # Mark response tokens (from query_token_len onwards) as 1
        if target_token_len > 0:
            sft_mask[i, query_token_len: query_token_len + target_token_len] = 1.0

    results.update({
        'ref_logprob': torch.Tensor(ref_logprob).float(),
        'generation_logprob': torch.Tensor(generation_logprob).float(),
        'sampling_logprob': torch.Tensor(sampling_logprob).float(),
        'advantage': torch.Tensor(advantage).float(),
        'sft_mask': torch.Tensor(sft_mask).float(),  # SFT mask for SPAG regularization
    })

    return results


def ppo_value_data_collactor(batch, tokenizer, processor=None):
    results = sft_data_collactor(batch, tokenizer, prepare_value_input_ids=True, overwrite_labels=True, processor=processor)
    
    # Handle case where all items were skipped (empty batch with dummy data)
    # Check if this is a dummy batch (target_token_lens[0] == 0)
    if results['target_token_lens'][0] == 0:
        # Return results with empty value arrays for consistency
        results.update({
            'generation_value_pred': torch.zeros((1,)).float(),
            'return': torch.zeros((1,)).float(),
        })
        return results

    generation_value_pred, returns = [], []

    for item in batch:
        generation_value_pred.append(item['generation_value_pred'])
        returns.append(item['return'])

    results.update({
        'generation_value_pred': torch.Tensor(generation_value_pred).float(),
        'return': torch.Tensor(returns).float(),
    })

    return results


def credit_assignment_data_collactor(batch, tokenizer, prepare_value_input_ids, processor=None):
    results = sft_data_collactor(batch, tokenizer, prepare_value_input_ids=prepare_value_input_ids, processor=processor)
    
    # Handle case where all items were skipped (e.g., corrupted samples)
    if results is None:
        return None

    results['reward'] = torch.Tensor([item['reward'] for item in batch]).float()

    batch_size, seq_len = results['labels'].shape
    
    # Determine logprob array size based on whether we have query tokens
    # if has_query_tokens: seq_len includes prompt, so logprob_size = seq_len - 1 (excludes 1st prompt token)
    # if no query tokens, seq_len is only response, so logprob_size = seq_len (includes all response logprobs)
    # Query tokens are available for both text-only and multimodal inputs.
    # NOTE: Query lengths exclude leading padding; credit assignment uses a per-device batch size of 1.
    has_query_tokens = any(results['query_token_lens'])  # True if any sample has query tokens.
    logprob_size = seq_len - 1 if has_query_tokens else seq_len
    
    sampling_logprob = np.zeros((batch_size, logprob_size))

    for i, item in enumerate(batch):
        query_token_len_raw = results['query_token_lens'][i]
        target_token_len = results['target_token_lens'][i]
        
        if query_token_len_raw > 0:
            query_token_len = query_token_len_raw - 1
        else:
            # no query tokens, so query_token_len = 0
            query_token_len = 0
        
        available_space = logprob_size - query_token_len

        if 'target_logprob' in item:
            target_logprobs = item['target_logprob']
            # Use all available logprobs up to available_space
            assert len(target_logprobs)<=available_space, f"❌ target_logprob length {len(target_logprobs)} is greater than available_space {available_space}"
            sampling_logprob[i, query_token_len: query_token_len + len(target_logprobs)] = target_logprobs

    # Target logprobs occupy [query_token_len:query_token_len + len(target_logprobs)]; remaining positions are zero.
    results['sampling_logprob'] = torch.Tensor(sampling_logprob).float()

    return results


def batch_padding(input_ids, tokenizer, padding='longest', max_length=None, pad_token_id=None):
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_size = len(input_ids)
    max_length = tokenizer.model_max_length if max_length is None else max_length
    
    if padding == 'longest':
        max_input_length = max([len(inp_ids) for inp_ids in input_ids])
        max_length = min(tokenizer.model_max_length, max_input_length)                

    pad_input_ids = np.full((batch_size, max_length), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)

    for i, inp_ids in enumerate(input_ids):
        inp_ids_len = len(inp_ids)
        if inp_ids_len >= max_length:
            if tokenizer.truncation_side == 'left':
                inp_ids = inp_ids[-max_length :]
            else:
                inp_ids = inp_ids[:max_length]
        inp_ids = np.array(inp_ids, dtype=np.int64)

        if tokenizer.padding_side == 'left':
            pad_input_ids[i, -inp_ids_len:] = inp_ids
        else:
            pad_input_ids[i, :inp_ids_len] = inp_ids

    attention_mask = pad_input_ids != pad_token_id
    position_ids = attention_mask.cumsum(-1) - attention_mask

    return {
        'input_ids': pad_input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }
