from typing import Optional
from transformers import AutoTokenizer
from agentlab.agents.dynamic_prompting import Shrinkable
from agentlab.llm.llm_utils  import BaseMessage, Discussion

import logging
from textwrap import dedent

# Import model family detection utility
from utils.vllm import get_model_family


def fit_tokens(
    system_prompt: BaseMessage,
    user_prompt: Shrinkable,
    max_prompt_tokens=None,
    max_iterations=20,
    model_name="openai/gpt-4",
    redundacy_ratio=0.05,
):
    """Shrink a prompt element until it fits `max_prompt_tokens`.

    Args:
        shrinkable (Shrinkable): The prompt element to shrink.
        max_prompt_tokens (int): The maximum number of tokens allowed.
        max_iterations (int, optional): The maximum number of shrink iterations, by default 20.
        model_name (str, optional): The name of the model used when tokenizing.
        additional_prompts (str or List[str], optional): Additional prompts to account for when shrinking, by default [""].

    Returns:
        str: the prompt after shrinking.

    Raises:
        ValueError: Unrecognized type for prompt
    """

    if max_prompt_tokens is None:
        return user_prompt.prompt

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Detect model family for model-specific chat template handling
    model_family = get_model_family(model_name)

    def count_tokens(user_prompt_msg):
        messages = Discussion([system_prompt, user_prompt_msg])
        messages.merge()
        
        # Model-specific chat template handling
        # - Gemma: supports enable_thinking=True
        # - Qwen: doesn't support enable_thinking parameter
        if model_family == "gemma":
            message_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_special_tokens=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            message_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_special_tokens=False,
                add_generation_prompt=True,
            )
        return len(message_tokens)

    for i in range(max_iterations):
        prompt = user_prompt.prompt

        # vllm tokenization may be off by several tokens, so we add a buffer
        if count_tokens(prompt) * (1 + redundacy_ratio) <= max_prompt_tokens:
            return prompt

        user_prompt.shrink()

    logging.warning(
        dedent(
            f"""\
            After {max_iterations} shrink iterations, the prompt is still
            {count_tokens(prompt)} tokens (greater than {max_prompt_tokens}). Returning the prompt as is."""
        )
    )
    return prompt
