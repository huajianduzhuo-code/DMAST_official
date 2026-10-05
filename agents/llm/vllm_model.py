from dataclasses import dataclass
from typing import Optional, Tuple
import logging

import openai
from openai import OpenAI
from transformers import AutoTokenizer

import agentlab.llm.tracking as tracking
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import AIMessage
from agentlab.llm.chat_api import ChatModel, RetryError, handle_error

from agents.llm.base_model_args import BaseModelArgs

# Import model family detection utility
from utils.vllm import get_model_family
# Import multimodal token utilities
try:
    from utils.multimodal_token_utils import get_token_calculator, validate_multimodal_message
    MULTIMODAL_VALIDATION_AVAILABLE = True
except ImportError:
    MULTIMODAL_VALIDATION_AVAILABLE = False
    import logging
    logging.warning("Multimodal token validation not available. Install PIL: pip install Pillow")


@dataclass
class VllmModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""
    base_url: str = "http://127.0.0.1:1234"
    api_key: str = "api_key"

    # sampling params
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    log_probs: Optional[int] = None
    
    # Retry settings for vLLM API calls
    # With 2 servers (4 GPU each), transient errors are more common
    max_retry: int = 1
    min_retry_wait_time: int = 5  # Wait between retries

    def make_model(self):
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        return VllmChatModel(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
            max_retry=self.max_retry,
            min_retry_wait_time=self.min_retry_wait_time,
        )


class VllmChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        base_url=None,
        temperature=0.5,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs: Optional[int] = None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args={
                "base_url": base_url
            }
        )
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.log_probs = log_probs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Detect model family for model-specific handling
        self.model_family = get_model_family(model_name)
        logging.info(f"🔍 Detected model family: {self.model_family} for {model_name}")

    def _extract_text_only_logprobs(
        self,
        completion,
        prompt_tokens: list,
        prompt_logprob: list,
        response_tokens: list,
        response_logprob: list,
        prompt_token_ids: list,  # Output token IDs.
        response_token_ids: list,  # Output token IDs.
    ) -> str:
        """
        Extract logprobs and token IDs from completions endpoint (text-only).
        Returns the decoded response text.
        """
        # Extract prompt logprobs
        prompt_vllm_token_ids = []
        prompt_transformers_token_ids = []

        prompt_logprob_content = completion.choices[0].prompt_logprobs
        for logprob_dicts in prompt_logprob_content:
            if logprob_dicts is None:
                continue
            
            # vLLM may return multiple tokens with top_logprobs, take the first one (the actual token)
            if len(logprob_dicts) == 0: # NOTE: length should be self.log_probs+1 (because self.logprobs is the number of "extra" top logprobs to return)
                logging.warning("Empty logprob dict encountered in prompt_logprobs")
                continue
            # Get the first token (the actual token that was generated/in the prompt)
            logprob_dict = list(logprob_dicts.values())[0]
            prompt_token = logprob_dict["decoded_token"]
            prompt_tokens.append(prompt_token)
            prompt_logprob.append(logprob_dict["logprob"])

            prompt_vllm_token_id = int(list(logprob_dicts.keys())[0])
            prompt_vllm_token_ids.append(prompt_vllm_token_id)
            prompt_token_ids.append(prompt_vllm_token_id)  # Include token IDs in the output list.
            prompt_transformers_token_ids.append(self.tokenizer.convert_tokens_to_ids(prompt_token))

        # Verify prompt token consistency
        prompt_token_ids_match = [id1 == id2 for id1, id2 in zip(prompt_vllm_token_ids, prompt_transformers_token_ids)]
        if not all(prompt_token_ids_match):
            unmatched_idx = [i for i, t in enumerate(prompt_token_ids_match) if not t]
            unmatched_tokens = [prompt_tokens[i] for i in unmatched_idx]
            unmatched_vllm_token_ids = [prompt_vllm_token_ids[i] for i in unmatched_idx]
            unmatched_transformers_token_ids = [prompt_transformers_token_ids[i] for i in unmatched_idx]

            raise ValueError(
                "prompt token ids are not consistent between vllm and transformers\n"
                f"\tunmatched_tokens: {unmatched_tokens}\n"
                f"\tunmatched_vllm_token_ids: {unmatched_vllm_token_ids}\n"
                f"\tunmatched_transformers_token_ids: {unmatched_transformers_token_ids}\n"
            )

        # Extract response logprobs
        # token_id_str is in the format of "token:id"
        response_token_contents = completion.choices[0].logprobs.tokens
        response_logprob_values = completion.choices[0].logprobs.token_logprobs

        response_vllm_token_ids = [int(token_id_str.split(":")[-1]) for token_id_str in response_token_contents]
        response_token_ids.extend(response_vllm_token_ids)  # Include token IDs in the output list.
        response_tokens_list = [":".join(token_id_str.split(":")[:-1]) for token_id_str in response_token_contents]

        # Verify response token consistency
        response_transformers_token_ids = self.tokenizer.convert_tokens_to_ids(response_tokens_list)
        response_token_ids_match = [id1 == id2 for id1, id2 in zip(response_vllm_token_ids, response_transformers_token_ids)]

        if not all(response_token_ids_match):
            unmatched_idx = [i for i, t in enumerate(response_token_ids_match) if not t]
            unmatched_tokens = [response_tokens_list[i] for i in unmatched_idx]
            unmatched_response_vllm_token_ids = [response_vllm_token_ids[i] for i in unmatched_idx]
            unmatched_response_transformers_token_ids = self.tokenizer.convert_tokens_to_ids(unmatched_tokens)

            raise ValueError(
                "response token ids are not consistent between vllm and transformers\n"
                f"\tunmatched_tokens: {unmatched_tokens}\n"
                f"\tunmatched_response_vllm_token_ids: {unmatched_response_vllm_token_ids}\n"
                f"\tunmatched_response_transformers_token_ids: {unmatched_response_transformers_token_ids}\n"
            )

        # Update the lists passed by reference
        response_tokens.extend(response_tokens_list)
        response_logprob.extend(response_logprob_values)

        # Decode response text from token IDs for consistency
        response_text = self.tokenizer.decode(response_vllm_token_ids) # NOTE: Decoding converts token markers, e.g. '▁bids' becomes ' bids'.
        logging.debug(f"✅ Extracted {len(response_tokens)} response tokens with logprobs (text-only)")
        
        return response_text

    def _extract_multimodal_logprobs(
        self,
        completion,
        response_tokens: list,
        response_logprob: list,
        prompt_tokens: list = None,
        prompt_logprob: list = None,
    ) -> None:
        """
        Extract logprobs from chat.completions endpoint (multimodal).
        Modifies response_tokens and response_logprob lists in place.
        If prompt_tokens/prompt_logprob lists are provided, also extracts prompt tokens.
        
        Enable prompt_logprobs in extra_body to extract prompt tokens and check
        token alignment between vLLM and the dataloader.
        """
        logging.debug("📊 Extracting logprobs from chat.completions format (multimodal)")
        
        logprobs_content = completion.choices[0].logprobs
        if not logprobs_content or not logprobs_content.content:
            logging.warning("⚠️ No logprobs content found in chat.completions response")
            return
        
        response_vllm_token_ids = []
        
        # ============================================================================
        # 🔍 VERIFICATION: Log first few tokens to understand logprob alignment
        # ============================================================================
        logging.debug("=" * 80)
        logging.debug("🔍 LOGPROB VERIFICATION - First 5 tokens:")
        logging.debug("=" * 80)
        
        for idx, token_obj in enumerate(logprobs_content.content):
            # Each token_obj has: token, logprob, bytes, top_logprobs
            token = token_obj.token
            logprob = token_obj.logprob
            
            # Log first 5 tokens for verification
            if idx < 5:
                # Build context string to show what model "saw" before this token
                previous_tokens = response_tokens[-3:] if len(response_tokens) > 0 else []
                context = "".join(previous_tokens) if previous_tokens else "[START]"
                
                logging.debug(f"Token {idx}:")
                logging.debug(f"  Context before: '{context}'")
                logging.debug(f"  Current token: '{token}'")
                logging.debug(f"  Logprob value: {logprob:.4f}")
                logging.debug(f"  Interpretation: P(token='{token}' | context='{context}')")
                logging.debug("-" * 60)
            
            response_tokens.append(token)
            response_logprob.append(logprob)
            
            # Convert token to token ID using tokenizer
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                response_vllm_token_ids.append(token_ids[0])
            else:
                # Token encodes to multiple IDs - this can happen with special tokens
                logging.debug(f"Token '{token}' encoded to {len(token_ids)} IDs, using first")
                response_vllm_token_ids.append(token_ids[0] if token_ids else 0)
        
        # Verify token consistency (more lenient for multimodal)
        response_transformers_token_ids = self.tokenizer.convert_tokens_to_ids(response_tokens)
        mismatches = sum(1 for id1, id2 in zip(response_vllm_token_ids, response_transformers_token_ids) if id1 != id2)
        
        if mismatches > 0:
            logging.debug(f"⚠️ {mismatches}/{len(response_tokens)} token ID mismatches (expected for multimodal)")
        
        # ============================================================================
        # 🔍 VERIFICATION SUMMARY
        # ============================================================================
        logging.debug("=" * 80)
        logging.debug("🔍 LOGPROB VERIFICATION SUMMARY:")
        logging.debug("=" * 80)
        logging.debug(f"Total tokens extracted: {len(response_tokens)}")
        logging.debug(f"Total logprobs extracted: {len(response_logprob)}")
        logging.debug(f"First 10 tokens: {response_tokens[:10]}")
        logging.debug(f"Response text preview: {''.join(response_tokens[:20])}")
        logging.debug("")
        logging.debug("🔑 KEY INSIGHT:")
        logging.debug("   If logprob[i] is for predicting token[i] (given tokens[0:i]),")
        logging.debug("   then logprob[0] is for predicting the FIRST token from the prompt.")
        logging.debug("   This is what we observe in VLLM's chat.completions API.")
        logging.debug("=" * 80)
        
        logging.debug(f"✅ Extracted {len(response_tokens)} response tokens with logprobs (multimodal)")
        
        # ============================================================================
        # EXTRACT PROMPT TOKENS (if requested)
        # ============================================================================
        if prompt_tokens is not None and prompt_logprob is not None:
            try:
                # Check if prompt_logprobs are available in the response
                # The location depends on vLLM version, check multiple possible locations
                prompt_logprobs_data = None
                
                if hasattr(completion.choices[0].logprobs, 'prompt_logprobs'):
                    prompt_logprobs_data = completion.choices[0].logprobs.prompt_logprobs
                    logging.debug("✅ Found prompt_logprobs in logprobs.prompt_logprobs")
                elif hasattr(completion, 'prompt_logprobs'):
                    prompt_logprobs_data = completion.prompt_logprobs
                    logging.debug("✅ Found prompt_logprobs in completion.prompt_logprobs")
                
                if prompt_logprobs_data:
                    logging.debug(f"📊 Extracting {len(prompt_logprobs_data)} prompt tokens from vLLM")
                    
                    for idx, token_obj in enumerate(prompt_logprobs_data):
                        if token_obj is None:
                            continue
                        
                        # Extract token and logprob
                        if hasattr(token_obj, 'token'):
                            token = token_obj.token
                            logprob = token_obj.logprob if hasattr(token_obj, 'logprob') else 0.0
                        elif isinstance(token_obj, dict):
                            token = token_obj.get('token', '')
                            logprob = token_obj.get('logprob', 0.0)
                        else:
                            logging.warning(f"⚠️ Unexpected prompt_logprobs format: {type(token_obj)}")
                            continue
                        
                        prompt_tokens.append(token)
                        prompt_logprob.append(logprob)
                    
                    logging.debug(f"✅ Extracted {len(prompt_tokens)} prompt tokens from vLLM")
                    logging.debug(f"   First 10: {prompt_tokens[:10]}")
                    logging.debug(f"   Last 10: {prompt_tokens[-10:]}")
                else:
                    logging.warning("⚠️ prompt_logprobs requested but not found in vLLM response")
                    logging.warning("   This may indicate vLLM doesn't support prompt_logprobs for multimodal,")
                    logging.warning("   or the API version doesn't expose them for chat.completions")
                    
            except Exception as e:
                logging.error(f"❌ Error extracting prompt tokens: {e}")
                import traceback
                traceback.print_exc()

    def __call__(
        self,
        messages: list[dict],
        n_samples: int = 1,
        temperature: Optional[float] = None,
    ) -> Tuple[AIMessage, dict]:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        # Handle both Discussion objects (with .merge()) and plain lists
        if hasattr(messages, 'merge'):
            messages.merge()

        # Validate multimodal messages before calling the API.
        if MULTIMODAL_VALIDATION_AVAILABLE:
            try:
                # Convert messages to list format for validation
                message_list = []
                for msg in messages:
                    message_list.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                
                validation_result = validate_multimodal_message(message_list, self.model_name)
                
                if not validation_result['valid']:
                    # Log errors and warnings
                    for error in validation_result['errors']:
                        logging.error(f"🚨 VLLM Token Limit Error: {error}")
                    for warning in validation_result['warnings']:
                        logging.warning(f"⚠️ VLLM Token Warning: {warning}")
                    for suggestion in validation_result['suggestions']:
                        logging.debug(f"💡 Suggestion: {suggestion}")
                    
                    # Raise an informative error
                    token_info = validation_result['token_breakdown']
                    raise ValueError(
                        f"Message exceeds VLLM token limit: {token_info['total_tokens']} tokens "
                        f"(text: {token_info['text_tokens']}, images: {token_info['image_tokens']}) "
                        f"exceeds limit of {get_token_calculator(self.model_name).max_input_tokens - get_token_calculator(self.model_name).safety_buffer}. "
                        f"Suggestions: {'; '.join(validation_result['suggestions'])}"
                    )
                else:
                    # Log successful validation
                    token_info = validation_result['token_breakdown']
                    logging.debug(f"✅ VLLM Token Validation Passed: {token_info['total_tokens']} tokens "
                                f"(text: {token_info['text_tokens']}, images: {token_info['image_tokens']}, "
                                f"images: {token_info['image_count']})")
                    
                    # Log warnings if any
                    for warning in validation_result['warnings']:
                        logging.warning(f"⚠️ VLLM Token Warning: {warning}")
                        
            except Exception as e:
                logging.warning(f"⚠️ Could not validate message tokens: {e}")

        # Detect multimodal content (images) in messages.
        # If messages contain images, we MUST use chat.completions API instead of completions API
        # because apply_chat_template() strips out image data!
        has_images = False
        try:
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Check if any content item is an image
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_images = True
                            logging.debug("🖼️ Detected multimodal content with images in messages")
                            break
                if has_images:
                    break
        except Exception as e:
            logging.warning(f"⚠️ Error detecting multimodal content: {e}, assuming text-only")
        
        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            temperature = temperature if temperature is not None else self.temperature
            try:
                if has_images:
                    # 🖼️ MULTIMODAL PATH: Use chat.completions API for messages with images
                    # This endpoint preserves image_url content in the messages
                    logging.debug("🖼️ Using chat.completions API for multimodal content")
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # Pass messages directly without template processing
                        n=n_samples,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        logprobs=self.log_probs is not None,
                        top_logprobs=self.log_probs if self.log_probs is not None else None,
                        top_p=self.top_p,
                        extra_body={
                            "top_k": self.top_k,
                            "min_p": self.min_p,
                            "repetition_penalty": self.repetition_penalty,
                            # CRITICAL: Request token IDs for training consistency
                            # This ensures vLLM's tokenization matches what we use in training
                            "return_token_ids": True,
                        }
                    )
                else:
                    # 📝 TEXT-ONLY PATH: Use existing completions API with chat template
                    logging.debug("📝 Using completions API for text-only content")
                    
                    # Model-specific chat template handling
                    # - Gemma: supports enable_thinking=True to wrap thinking in <think> tags
                    # - Qwen: doesn't support enable_thinking parameter (will error if passed)
                    if self.model_family == "gemma":
                        chat_messages = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_special_tokens=False,       # bos and eos will be added by chat template
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    else:
                        # For Qwen and other models, don't use enable_thinking
                        # The model will output reasoning naturally (without explicit <think> tags)
                        chat_messages = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_special_tokens=False,
                            add_generation_prompt=True,
                        )
                    completion = self.client.completions.create(
                        model=self.model_name,
                        prompt=chat_messages,
                        n=n_samples,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        logprobs=self.log_probs,
                        top_p=self.top_p,
                        extra_body={
                            "top_k": self.top_k,
                            "min_p": self.min_p,
                            "repetition_penalty": self.repetition_penalty,
                            "return_tokens_as_token_ids": True,
                            "skip_special_tokens": False,
                            "spaces_between_special_tokens": True,
                            "include_stop_str_in_output": False,
                            "prompt_logprobs": self.log_probs,
                        }
                    )

                self.success = True
                break
            except openai.OpenAIError as e:
                # Handle context-length errors.
                error_message = str(e)
                if "context_length_exceeded" in error_message.lower() or "maximum context length" in error_message.lower():
                    logging.error(f"🚨 VLLM Context Length Exceeded: {error_message}")
                    if MULTIMODAL_VALIDATION_AVAILABLE:
                        logging.error("💡 This should have been caught by pre-flight validation!")
                        logging.error("   Consider updating token calculation parameters or image processing")
                    else:
                        logging.error("💡 Install multimodal token validation to prevent this: pip install Pillow")
                
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        if n_samples == 1:
            # Initialize output variables
            prompt_tokens = []
            prompt_logprob = []
            response_tokens = []
            response_logprob = []
            prompt_token_ids = []  # Token IDs returned by the backend.
            response_token_ids = []  # Token IDs returned by the backend.
            
            record_logprobs = self.log_probs is not None

            # ============================================================================
            # STEP 1: Extract response text from completion
            # ============================================================================
            if has_images:
                # MULTIMODAL: chat.completions endpoint returns message.content
                response_text = completion.choices[0].message.content
                logging.debug(f"📝 Extracted text from chat.completions (multimodal): {len(response_text)} chars")
            else:
                # TEXT-ONLY: completions endpoint returns .text
                response_text = completion.choices[0].text
                logging.debug(f"📝 Extracted text from completions (text-only): {len(response_text)} chars")

            # ============================================================================
            # STEP 2: Extract logprobs and token IDs if requested
            # ============================================================================
            if record_logprobs:
                if has_images:
                    # MULTIMODAL PATH: Extract logprobs from chat.completions format
                    # Also try to extract prompt tokens for verification
                    self._extract_multimodal_logprobs(
                        completion, 
                        response_tokens, 
                        response_logprob,
                        prompt_tokens,  # Pass to extract if available
                        prompt_logprob
                    )
                else:
                    # TEXT-ONLY PATH: Extract logprobs and token IDs from completions format
                    response_text = self._extract_text_only_logprobs(
                        completion,
                        prompt_tokens,
                        prompt_logprob,
                        response_tokens,
                        response_logprob,
                        prompt_token_ids,  # Token ID output list.
                        response_token_ids  # Token ID output list.
                    )

            res = AIMessage(response_text)

            # ============================================================================
            # STEP 3: Extract token IDs from vLLM response (for training consistency)
            # ============================================================================
            # For text-only, token IDs are already extracted from logprobs in STEP 2
            # For multimodal, we need to extract from completion.prompt_token_ids
            
            if has_images:
                # MULTIMODAL: Extract from chat.completions response attributes
                if hasattr(completion, 'prompt_token_ids') and completion.prompt_token_ids:
                    prompt_token_ids = completion.prompt_token_ids
                    logging.debug(f"✅ Extracted {len(prompt_token_ids)} prompt_token_ids from vLLM response")
                    logging.debug(f"   First 20 token IDs: {prompt_token_ids[:20]}")
                    # Count image soft tokens (ID: 262144)
                    image_token_count = sum(1 for tid in prompt_token_ids if tid == 262144)
                    logging.debug(f"   Including {image_token_count} image soft tokens (ID: 262144)")
                else:
                    logging.debug("⚠️  prompt_token_ids not available in vLLM response")
                
                if hasattr(completion.choices[0], 'token_ids') and completion.choices[0].token_ids:
                    response_token_ids = completion.choices[0].token_ids
                    logging.debug(f"✅ Extracted {len(response_token_ids)} response_token_ids from vLLM response")
                    logging.debug(f"   First 20 token IDs: {response_token_ids[:20]}")
                else:
                    logging.debug("⚠️  response_token_ids not available in vLLM response")
            else:
                # TEXT-ONLY: Token IDs already extracted from logprobs in STEP 2
                if prompt_token_ids:
                    logging.debug(f"✅ Extracted {len(prompt_token_ids)} prompt_token_ids from logprobs (text-only)")
                    logging.debug(f"   First 20 token IDs: {prompt_token_ids[:20]}")
                else:
                    logging.debug("⚠️  No prompt_token_ids extracted from logprobs")
                
                if response_token_ids:
                    logging.debug(f"✅ Extracted {len(response_token_ids)} response_token_ids from logprobs (text-only)")
                    logging.debug(f"   First 20 token IDs: {response_token_ids[:20]}")
                else:
                    logging.debug("⚠️  No response_token_ids extracted from logprobs")

            training_info = {
                "prompt_tokens": prompt_tokens,
                "prompt_logprob": prompt_logprob,
                "response_tokens": response_tokens,
                "response_logprob": response_logprob,
                # Token IDs from vLLM keep training aligned with generation.
                "prompt_token_ids": prompt_token_ids,
                "response_token_ids": response_token_ids,
            }

            return res, training_info
        else:
            raise NotImplementedError("n_samples > 1 is not supported for now")
