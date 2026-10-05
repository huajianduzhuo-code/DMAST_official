from typing import List, Literal, Optional

from dataclasses import dataclass, field
from transformers.training_args import TrainingArguments
from peft import LoraConfig


@dataclass
class CustomTrainingArguments(TrainingArguments):
    debug_mode: bool = field(
        default=False,
        metadata={"help": "whether use the debug mode."}
    )

    # as wandb logging postfix
    roles: List[str] = field(
        default_factory=lambda: ["agent", "attacker"],
        metadata={
            "nargs": "+",
            "choices": ["agent", "attacker"],
            "help": "Which roles to assign (must be agent or attacker)",
        }
    )

    # for LR scheduler
    use_persistent_lr_scheduler: bool = field(
        default=False,
        metadata={"help": "whether use the persistent LR scheduler."}
    )

    num_total_train_epochs: int = field(
        default=1,
        metadata={"help": "the number of total training epochs."}
    )

    num_total_train_steps: int = field(
        default=-1,
        metadata={"help": "the number of total training steps."}
    )

    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    truncation_side: str = field(
        default="left",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    # model params
    model_name_or_path: str = field(
        default="",
        metadata={"help": "the path to load pretrained model."}
    )

    ref_model_name_or_path: str = field(
        default="",
        metadata={"help": "the path to load reference model."}
    )

    # lora params
    use_lora: bool = field(
        default=False,
        metadata={"help": "whether use the LoRA for the model."}
    )

    lora_r: int = field(
        default=64,
        metadata={"help": "the rank of the LoRA matrices."}
    )

    lora_alpha: int = field(
        default=128,
        metadata={"help": "the scaling factor for the LoRA matrices."}
    )

    lora_target_modules: str = field(
        default="all-linear",
        metadata={"help": "the target modules for the LoRA matrices."}
    )

    lora_bias: Literal["none", "all", "lora_only"] = field(
        default="none",
        metadata={"help": "the bias for the LoRA matrices."}
    )

    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "the modules to save for the LoRA matrices."}
    )

    # sampling params
    temperature: float = field(
        default=1.0,
        metadata={"help": "the temperature during sampling."}
    )

    # data params
    train_data_path: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "train datasets paths."}
    )

    train_method: str = field(
        default="sft",
        metadata={"help": "the LLM training method name."}
    )

    # PPO and GRPO params
    use_sampling_importance_weight: bool = field(
        default=True,
        metadata={"help": "whether use the importance sampling for the policy optimization."},
    )

    min_sampling_importance_weight: float = field(
        default=0.1,
        metadata={"help": "if the importance weight is less than this value, it will not be used for the policy optimization."},
    )

    min_policy_importance_weight: float = field(
        default=0.1,
        metadata={"help": "if the importance weight is less than this value, it will not be used for the policy optimization."},
    )

    use_ratio_clip: bool = field(
        default=True,
        metadata={"help": "whether use the ratio clipping for the policy optimization."},
    )

    low_clip_range: float = field(
        default=0.2,
        metadata={"help": "the lower range to clip the importance reweighting ratio for policy optimization."},
    )

    high_clip_range: float = field(
        default=0.2,
        metadata={"help": "the upper range to clip the importance reweighting ratio for policy optimization."},
    )

    value_clip_range: float = field(
        default=0.0,
        metadata={"help": "the range to clip the value prediction for value optimization. <= 0 means no clipping."},
    )

    num_value_buckets: int = field(
        default=255,
        metadata={"help": "the number of value buckets."}
    )

    value_activation: Literal["none", "sigmoid"] = field(
        default="none",
        metadata={"help": "the activation function for the value."}
    )

    value_min: float = field(
        default=-1,
        metadata={"help": "the minimum value."}
    )

    value_max: float = field(
        default=1,
        metadata={"help": "the maximum value."}
    )

    lm_entropy_coeff: float = field(
        default=0.0,
        metadata={"help": "the coefficient of entropy bonus."},
    )

    lm_kl_coeff: float = field(
        default=0.05,
        metadata={"help": "the coefficient of kl regularizer."},
    )

    lm_sft_coeff: float = field(
        default=0.0,
        metadata={"help": "the coefficient of SFT regularization (for SPAG). 0 = disabled."},
    )

    only_penalize_kl_when_smaller_than_ref: bool = field(
        default=False,
        metadata={"help": "whether only penalize the KL when the logprob is smaller than the reference logprob."},
    )

    use_data_weight: bool = field(
        default=False,
        metadata={"help": "whether use the data weight for the policy optimization."},
    )

    # save model params
    local_model_save_path: str = field(
        default="",
        metadata={"help": "the path to save the model."}
    )

    cloud_model_save_path: str = field(
        default="",
        metadata={"help": "the path to save the model."}
    )

    def __post_init__(self):
        super().__post_init__()

        if self.use_lora:
            if "," in self.lora_target_modules:
                target_modules = self.lora_target_modules.split(",")
            else:
                target_modules = self.lora_target_modules

            self.lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                bias=self.lora_bias,
                modules_to_save=self.lora_modules_to_save,
            )
        else:
            self.lora_config = None
