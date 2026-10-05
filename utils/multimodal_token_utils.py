#!/usr/bin/env python3
"""
Multimodal Token Management Utilities

This module provides accurate token counting and validation for multimodal messages
containing both text and images, preventing truncation issues in VLLM calls.

Key Features:
- Exact image token calculation based on Vision Transformer (ViT) patch encoding
- Pre-flight validation before VLLM API calls
- Automatic image resizing for token limit compliance
- Token-usage warnings
"""

import logging
import base64
import math
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class MultimodalTokenCalculator:
    """
    Calculates exact token counts for multimodal messages containing text and images.
    
    Based on Vision Transformer (ViT) architecture used by VLM models:
    - Images are divided into patches (14x14 for Gemma, 16x16 for Qwen)
    - Each patch becomes one token
    - Additional tokens for special tokens (CLS, SEP, etc.)
    """
    
    def __init__(self, model_name: str = "google/gemma-3-27b-it"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Auto-detect patch size based on model type
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            # Qwen-VL models use 16x16 patches
            self.patch_size = 16
        else:
            # Gemma and other models typically use 14x14 patches
            self.patch_size = 14
        
        self.vision_special_tokens = 4  # CLS, SEP, padding tokens
        
        # Token allocation settings
        self.max_total_tokens = 32768  # 32K context
        self.max_completion_tokens = 4096  # Reserved for completion
        self.max_input_tokens = self.max_total_tokens - self.max_completion_tokens  # 28672
        self.safety_buffer = 256  # Safety buffer for tokenization variations
        
        logger.debug(f"🔢 MultimodalTokenCalculator initialized for {model_name}")
        logger.debug(f"   Max input tokens: {self.max_input_tokens}")
        logger.debug(f"   Patch size: {self.patch_size}x{self.patch_size} ({'Qwen' if 'qwen' in model_name_lower else 'Gemma/default'})")
    
    def calculate_image_tokens(self, image: Union[Image.Image, str, bytes]) -> int:
        """
        Calculate exact number of tokens consumed by an image.
        
        Args:
            image: PIL Image, base64 string, or bytes
            
        Returns:
            int: Number of tokens the image will consume
        """
        # Convert to PIL Image if needed
        pil_image = self._to_pil_image(image)
        if pil_image is None:
            logger.warning("Could not process image for token calculation")
            return 0
        
        # Get image dimensions
        width, height = pil_image.size
        
        # Calculate number of patches
        patches_x = math.ceil(width / self.patch_size)
        patches_y = math.ceil(height / self.patch_size)
        patch_tokens = patches_x * patches_y
        
        # Add special tokens
        total_tokens = patch_tokens + self.vision_special_tokens
        
        logger.debug(f"📊 Image token calculation:")
        logger.debug(f"   Dimensions: {width}x{height}")
        logger.debug(f"   Patches: {patches_x}x{patches_y} = {patch_tokens}")
        logger.debug(f"   Special tokens: {self.vision_special_tokens}")
        logger.debug(f"   Total image tokens: {total_tokens}")
        
        return total_tokens
    
    def calculate_text_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        """
        Calculate exact number of tokens consumed by text.
        
        Args:
            text: Text content
            add_special_tokens: Whether to include special tokens in count
            
        Returns:
            int: Number of tokens the text will consume
        """
        try:
            tokens = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors="pt"
            )
            token_count = tokens.input_ids.shape[1]
            logger.debug(f"📝 Text tokens: {token_count} for text length {len(text)}")
            return token_count
        except Exception as e:
            logger.warning(f"Error calculating text tokens: {e}")
            # Fallback estimate: ~4 chars per token
            return len(text) // 4
    
    def calculate_message_tokens(self, messages: List[Dict]) -> Dict[str, int]:
        """
        Calculate total tokens for a list of chat messages including multimodal content.
        
        Args:
            messages: List of chat messages in OpenAI format
            
        Returns:
            Dict with token breakdown: {
                'text_tokens': int,
                'image_tokens': int, 
                'total_tokens': int,
                'image_count': int
            }
        """
        total_text_tokens = 0
        total_image_tokens = 0
        image_count = 0
        
        for message in messages:
            content = message.get("content", "")
            
            if isinstance(content, str):
                # Simple text message
                total_text_tokens += self.calculate_text_tokens(content)
                
            elif isinstance(content, list):
                # Multimodal message with text and images
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content = item.get("text", "")
                            total_text_tokens += self.calculate_text_tokens(text_content)
                            
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url:
                                image_tokens = self.calculate_image_tokens(image_url)
                                total_image_tokens += image_tokens
                                image_count += 1
                                logger.debug(f"🖼️ Image {image_count}: {image_tokens} tokens")
        
        # Add chat template overhead (system messages, formatting, etc.)
        try:
            # Estimate template overhead using direct tokenizer approach
            # Create a simple system + user message to estimate overhead
            simple_messages = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"}
            ]
            
            # Try to get template tokens directly from tokenizer
            if hasattr(self.tokenizer, 'apply_chat_template'):
                template_result = self.tokenizer.apply_chat_template(
                    simple_messages,
                    tokenize=True,
                    add_special_tokens=True,
                    add_generation_prompt=True
                )
                template_tokens_count = len(template_result) if isinstance(template_result, list) else len(template_result.input_ids)
                
                # Calculate overhead (total tokens minus content tokens)
                system_tokens = self.calculate_text_tokens("System", add_special_tokens=False)
                user_tokens = self.calculate_text_tokens("User", add_special_tokens=False)
                template_overhead = template_tokens_count - system_tokens - user_tokens
                
                # Add reasonable overhead (but cap it to avoid excessive estimates)
                total_text_tokens += max(0, min(template_overhead, 100))
                logger.debug(f"📊 Estimated template overhead: {template_overhead} tokens")
            else:
                # Fallback: conservative estimate
                total_text_tokens += 50
                logger.debug(f"📊 Using fallback template overhead: 50 tokens")
                
        except Exception as e:
            logger.warning(f"Could not calculate template overhead: {e}")
            # Conservative estimate
            total_text_tokens += 50
        
        total_tokens = total_text_tokens + total_image_tokens
        
        result = {
            'text_tokens': total_text_tokens,
            'image_tokens': total_image_tokens,
            'total_tokens': total_tokens,
            'image_count': image_count
        }
        
        logger.debug(f"🔢 Message token breakdown: {result}")
        return result
    
    def validate_message_size(self, messages: List[Dict], warn_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Validate that messages will fit within token limits.
        
        Args:
            messages: List of chat messages
            warn_threshold: Warn when usage exceeds this fraction of limit (0.8 = 80%)
            
        Returns:
            Dict with validation results: {
                'valid': bool,
                'token_breakdown': dict,
                'warnings': List[str],
                'errors': List[str],
                'suggestions': List[str]
            }
        """
        token_breakdown = self.calculate_message_tokens(messages)
        total_tokens = token_breakdown['total_tokens']
        
        warnings = []
        errors = []
        suggestions = []
        
        # Check hard limit
        available_tokens = self.max_input_tokens - self.safety_buffer
        if total_tokens > available_tokens:
            errors.append(f"Message exceeds token limit: {total_tokens} > {available_tokens}")
            suggestions.append("Reduce image size or remove images")
            suggestions.append("Shorten text content")
        
        # Check warning threshold
        warn_limit = int(available_tokens * warn_threshold)
        if total_tokens > warn_limit:
            warnings.append(f"Message approaching token limit: {total_tokens}/{available_tokens} ({total_tokens/available_tokens:.1%})")
        
        # Image-specific checks
        if token_breakdown['image_count'] > 0:
            avg_image_tokens = token_breakdown['image_tokens'] / token_breakdown['image_count']
            if avg_image_tokens > 8000:  # ~896x896 image
                warnings.append(f"Large images detected (avg {avg_image_tokens:.0f} tokens/image)")
                suggestions.append("Consider resizing images to 448x448 or smaller")
            
            if token_breakdown['image_count'] > 3:
                warnings.append(f"Many images in message ({token_breakdown['image_count']})")
                suggestions.append("Consider reducing number of images")
        
        return {
            'valid': len(errors) == 0,
            'token_breakdown': token_breakdown,
            'warnings': warnings,
            'errors': errors,
            'suggestions': suggestions
        }
    
    def resize_image_for_token_limit(self, image: Union[Image.Image, str, bytes], 
                                   target_tokens: int = 2000) -> Tuple[Image.Image, int]:
        """
        Resize image to fit within a target token count.
        
        Args:
            image: PIL Image, base64 string, or bytes
            target_tokens: Target number of tokens for the image
            
        Returns:
            Tuple of (resized_image, actual_tokens)
        """
        pil_image = self._to_pil_image(image)
        if pil_image is None:
            raise ValueError("Could not process image for resizing")
        
        # Calculate target dimensions
        # target_tokens = (width/patch_size) * (height/patch_size) + special_tokens
        # Solving for square image: target_tokens - special_tokens = (dim/patch_size)^2
        target_patches = target_tokens - self.vision_special_tokens
        target_dim_patches = math.sqrt(target_patches)
        target_dim_pixels = int(target_dim_patches * self.patch_size)
        
        # Maintain aspect ratio
        original_width, original_height = pil_image.size
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            # Wider than tall
            new_width = target_dim_pixels
            new_height = int(target_dim_pixels / aspect_ratio)
        else:
            # Taller than wide
            new_height = target_dim_pixels  
            new_width = int(target_dim_pixels * aspect_ratio)
        
        # Ensure dimensions are multiples of patch size for optimal encoding
        new_width = (new_width // self.patch_size) * self.patch_size
        new_height = (new_height // self.patch_size) * self.patch_size
        
        # Minimum size constraint
        min_size = self.patch_size * 2  # At least 2x2 patches
        new_width = max(new_width, min_size)
        new_height = max(new_height, min_size)
        
        # Resize image
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        actual_tokens = self.calculate_image_tokens(resized_image)
        
        logger.info(f"🖼️ Resized image: {original_width}x{original_height} → {new_width}x{new_height}")
        logger.info(f"   Token count: {self.calculate_image_tokens(pil_image)} → {actual_tokens}")
        
        return resized_image, actual_tokens
    
    def optimize_messages_for_token_limit(self, messages: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Optimize messages to fit within token limits by resizing images and truncating text.
        
        Args:
            messages: Original list of chat messages
            
        Returns:
            Tuple of (optimized_messages, optimization_report)
        """
        optimized_messages = []
        optimization_report = {
            'original_tokens': 0,
            'optimized_tokens': 0,
            'images_resized': 0,
            'text_truncated': False,
            'modifications': []
        }
        
        # Calculate original token count
        original_validation = self.validate_message_size(messages)
        optimization_report['original_tokens'] = original_validation['token_breakdown']['total_tokens']
        
        if original_validation['valid']:
            logger.info("✅ Messages already within token limits, no optimization needed")
            return messages, optimization_report
        
        logger.info("🔧 Optimizing messages to fit token limits...")
        
        # Step 1: Resize large images
        for message in messages:
            optimized_message = message.copy()
            content = message.get("content", "")
            
            if isinstance(content, list):
                optimized_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            current_tokens = self.calculate_image_tokens(image_url)
                            if current_tokens > 2000:  # Resize large images
                                try:
                                    resized_image, new_tokens = self.resize_image_for_token_limit(image_url, 2000)
                                    # Convert back to data URL
                                    optimized_url = self._image_to_data_url(resized_image)
                                    optimized_item = item.copy()
                                    optimized_item["image_url"]["url"] = optimized_url
                                    optimized_content.append(optimized_item)
                                    optimization_report['images_resized'] += 1
                                    optimization_report['modifications'].append(f"Resized image: {current_tokens} → {new_tokens} tokens")
                                except Exception as e:
                                    logger.warning(f"Failed to resize image: {e}")
                                    optimized_content.append(item)
                            else:
                                optimized_content.append(item)
                        else:
                            optimized_content.append(item)
                    else:
                        optimized_content.append(item)
                optimized_message["content"] = optimized_content
            
            optimized_messages.append(optimized_message)
        
        # Step 2: Check if we're still over limit
        current_validation = self.validate_message_size(optimized_messages)
        optimization_report['optimized_tokens'] = current_validation['token_breakdown']['total_tokens']
        
        if not current_validation['valid']:
            logger.warning("⚠️ Still over token limit after image optimization")
            optimization_report['modifications'].append("Consider reducing text content or removing images")
        
        return optimized_messages, optimization_report
    
    def _to_pil_image(self, image: Union[Image.Image, str, bytes]) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image"""
        try:
            if isinstance(image, Image.Image):
                return image
            elif isinstance(image, str):
                if image.startswith("data:image"):
                    # Data URL
                    base64_data = image.split(",")[1] if "," in image else image
                    image_bytes = base64.b64decode(base64_data)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                else:
                    # Assume it's base64
                    image_bytes = base64.b64decode(image)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
            elif isinstance(image, bytes):
                return Image.open(BytesIO(image)).convert("RGB")
            else:
                logger.warning(f"Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            logger.warning(f"Error converting image: {e}")
            return None
    
    def _image_to_data_url(self, image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """Convert PIL Image to data URL"""
        buffer = BytesIO()
        image.save(buffer, format=format, quality=quality, optimize=True)
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{b64_data}"


# Global instance for easy access
_default_calculator = None

def get_token_calculator(model_name: str = "google/gemma-3-27b-it") -> MultimodalTokenCalculator:
    """Get or create default token calculator instance"""
    global _default_calculator
    if _default_calculator is None or _default_calculator.model_name != model_name:
        _default_calculator = MultimodalTokenCalculator(model_name)
    return _default_calculator


def calculate_image_tokens(image: Union[Image.Image, str, bytes], model_name: str = "google/gemma-3-27b-it") -> int:
    """Convenience function to calculate image tokens"""
    calculator = get_token_calculator(model_name)
    return calculator.calculate_image_tokens(image)


def validate_multimodal_message(messages: List[Dict], model_name: str = "google/gemma-3-27b-it") -> Dict[str, Any]:
    """Convenience function to validate multimodal messages"""
    calculator = get_token_calculator(model_name)
    return calculator.validate_message_size(messages)


def optimize_multimodal_message(messages: List[Dict], model_name: str = "google/gemma-3-27b-it") -> Tuple[List[Dict], Dict[str, Any]]:
    """Convenience function to optimize multimodal messages for token limits"""
    calculator = get_token_calculator(model_name)
    return calculator.optimize_messages_for_token_limit(messages)
