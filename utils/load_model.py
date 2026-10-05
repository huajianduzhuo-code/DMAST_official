from typing import Any, Dict, Optional

import json
from pathlib import Path

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification


# ------------------------------- init value head's weight to 0 -------------------------------
class LinearZeroInit(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# -------------- Custom SequenceClassification wrappers for VLM models -------------- 
# These are needed because transformers doesn't provide ForSequenceClassification versions

# Gemma3 wrappers
class Gemma3ConditionalGenerationForSequenceClassification(Gemma3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.score = LinearZeroInit(config.text_config.hidden_size, config.num_labels, bias=False)

    def forward(self, *args, **kwargs):
        model_outputs = super().forward(*args, **kwargs)
        logits = self.score(model_outputs.hidden_states[-1])
        return logits


class Gemma3CausalLMForSequenceClassification(Gemma3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.score = LinearZeroInit(config.hidden_size, config.num_labels, bias=False)

    def forward(self, *args, **kwargs):
        model_outputs = self.model.forward(*args, **kwargs)
        logits = self.score(model_outputs.hidden_states[-1])
        return logits


# Qwen3-VL wrapper
class Qwen3VLForSequenceClassification(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Qwen3-VL uses hidden_size directly in config
        hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'text_config', {}).get('hidden_size', 4096)
        self.score = LinearZeroInit(hidden_size, config.num_labels, bias=False)

    def forward(self, *args, **kwargs):
        model_outputs = super().forward(*args, **kwargs)
        logits = self.score(model_outputs.hidden_states[-1])
        return logits


def set_special_tokens(model, tokenizer):

    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        print(f"====================================================")
        print(f"WARNING: the pad token of the tokenizer is None")
        # We do not resize the vocab embedding, since it ruins the KL value with the ref_model
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = tokenizer.decode(0)
        print(f"set pad token to {tokenizer.pad_token}")
        print(f"set pad token id to {tokenizer.pad_token_id}")
        print(f"====================================================")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model


def load_base_model(
    cls,
    model_name_or_path,
    tokenizer,
    zero_init_value_head=False,
    num_value_buckets: int = 1,
):

    # prepare model kwargs
    model_kwargs = {
        "trust_remote_code": True,
    }

    # Read model config to determine model type and architecture
    with open(Path(model_name_or_path) / "config.json", "r") as f:
        config = json.load(f)
        model_type = config["model_type"]
        architecture = config["architectures"][0]

    # Add model-specific kwargs
    if model_type.startswith("gemma3"):
        model_kwargs["attn_implementation"] = 'eager'
    # Qwen3-VL doesn't need special attention implementation

    # Handle VLM models that can't be loaded with AutoModelForCausalLM
    # These models use ForConditionalGeneration instead of ForCausalLM
    if cls == AutoModelForCausalLM:
        if architecture == "Qwen3VLForConditionalGeneration":
            # Qwen3-VL is a vision-language model, use its specific class
            cls = Qwen3VLForConditionalGeneration
            print(f"🔄 Using Qwen3VLForConditionalGeneration for VLM model")
        elif architecture == "Gemma3ForConditionalGeneration":
            # Gemma3 multimodal also needs special handling
            cls = Gemma3ForConditionalGeneration
            print(f"🔄 Using Gemma3ForConditionalGeneration for VLM model")
        # Other models can continue using AutoModelForCausalLM

    # init model
    if cls == AutoModelForSequenceClassification:
        model_kwargs["num_labels"] = num_value_buckets

        # Custom SequenceClassification wrappers for VLM models
        # (transformers doesn't provide ForSequenceClassification versions for these)
        if architecture == "Gemma3ForConditionalGeneration":
            cls = Gemma3ConditionalGenerationForSequenceClassification
        elif architecture == "Gemma3ForCausalLM":
            cls = Gemma3CausalLMForSequenceClassification
        elif architecture == "Qwen3VLForConditionalGeneration":
            cls = Qwen3VLForSequenceClassification
        else:
            # For other models, try using AutoModelForSequenceClassification
            # This may work for models that have native SequenceClassification support
            print(f"⚠️ Using AutoModelForSequenceClassification for {architecture}")
            print(f"   If this fails, you may need to add a custom wrapper for this architecture")

    model = cls.from_pretrained(
        model_name_or_path, # NOTE: this path is a path of a folder, which contains all the information of the model. A standard hugging face model folder contains the following files: config.json, model.safetensors, tokenizer.json, tokenizer.model, etc.
        **model_kwargs,
    )

    # set special tokens
    model = set_special_tokens(model, tokenizer)

    # zero_init_value_head only applies to SequenceClassification models
    if zero_init_value_head:
        value_head = model.score
        assert isinstance(value_head, nn.Linear), "value_head must be a linear layer"
        model.score = LinearZeroInit(value_head.in_features, value_head.out_features, bias=value_head.bias is not None)

    return model


def load_from_pretrained_model(
    cls,
    model_name_or_path,
    tokenizer,
    zero_init_value_head=False,
    num_value_buckets: int = 1,
    lora_config: Optional[LoraConfig] = None,
):
    model_name_or_path = Path(model_name_or_path)
    if cls in [PeftModelForCausalLM, PeftModelForSequenceClassification]:

        if not (model_name_or_path / "adapter_config.json").exists(): # NOTE: if adapter_config does not exist, this is the case when the model is not a LoRA model, so we need to load the base model.
            assert (model_name_or_path / "config.json").exists()
            if cls == PeftModelForCausalLM:
                cls = AutoModelForCausalLM
            elif cls == PeftModelForSequenceClassification:
                cls = AutoModelForSequenceClassification
            else:
                raise ValueError(f"Unsupported model class: {cls}")

            return load_from_pretrained_model(
                cls,
                model_name_or_path,
                tokenizer,
                zero_init_value_head=zero_init_value_head,
                num_value_buckets=num_value_buckets,
                lora_config=lora_config,
            )

        with open(model_name_or_path / "adapter_config.json", "r") as f:
            config = json.load(f)
            base_model_name_or_path = config["base_model_name_or_path"] # NOTE: this path is a path of a folder, which contains all the information of the model. A standard hugging face model folder contains the following files: config.json, model.safetensors, tokenizer.json, tokenizer.model, etc.

        if cls == PeftModelForCausalLM:
            base_model = load_base_model(AutoModelForCausalLM, base_model_name_or_path, tokenizer)
        elif cls == PeftModelForSequenceClassification:
            base_model = load_base_model(
                AutoModelForSequenceClassification,
                base_model_name_or_path,
                tokenizer,
                zero_init_value_head=zero_init_value_head,
                num_value_buckets=num_value_buckets,
            )

        # https://github.com/huggingface/peft/issues/137
        base_model.enable_input_require_grads()

        model = cls.from_pretrained(base_model, model_name_or_path, is_trainable=True) # Loads PEFT adapters (line 163): Uses PeftModelForCausalLM/SequenceClassification.from_pretrained() to load the saved LoRA adapter weights and attach them to the base model

        if cls == PeftModelForSequenceClassification:
            for param in model.base_model.score.parameters():
                param.requires_grad = True

    else: # NOTE: this is the case when the model class is not a LoRA model.

        model = load_base_model(
            cls,
            model_name_or_path, # NOTE: this path is a path of a folder, which contains all the information of the model. A standard hugging face model folder contains the following files: config.json, model.safetensors, tokenizer.json, tokenizer.model, etc.
            tokenizer,
            zero_init_value_head=zero_init_value_head,
            num_value_buckets=num_value_buckets,
        )

        if lora_config is not None: # Wrap the base model with trainable LoRA modules when configured.
            if cls == AutoModelForSequenceClassification:
                task_type = TaskType.SEQ_CLS
            elif cls == AutoModelForCausalLM:
                task_type = TaskType.CAUSAL_LM
            else:
                raise ValueError(f"Unsupported model class: {cls}")

            lora_config.task_type = task_type

            # https://github.com/huggingface/peft/issues/137
            model.enable_input_require_grads()

            model = get_peft_model(model, lora_config) # NOTE: get_peft_model is a function supported by peft, it returns the LoRA model by: first freeze the base model, then add the LoRA modules (requires grad) to the base model, finally return the LoRA model.

    return model
