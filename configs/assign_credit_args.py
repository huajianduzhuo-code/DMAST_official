from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class AssignCreditArgs:
    # agent params
    roles: List[str] = field(
        metadata={
            "nargs": "+",
            "choices": ["agent", "attacker"],
            "help": "Which roles to assign credit to (must be agent or attacker)",
        }
    )
    rl_method: Literal["spag", "ppo", "grpo"] = field(
        metadata={"help": "the RL method to use."}
    )

    # model params
    model_name_or_path: str = field(
        metadata={"help": "the path to load pretrained model."}
    )

    ref_model_name_or_path: str = field(
        metadata={"help": "the path to load reference model."}
    )

    value_model_name_or_path: str = field(
        metadata={"help": "the path to load value model."}
    )

    is_multimodal: bool = field(
        default=True,
        metadata={"help": "whether the model is multimodal (requires processor and handles images)."}
    )

    num_value_buckets: int = field(
        default=1,
        metadata={"help": "the number of value buckets."}
    )

    # lora params
    use_lora: bool = field(
        default=False,
        metadata={"help": "whether use the LoRA for the model."}
    )
    ref_model_uses_lora: bool = field(
        default=False,
        metadata={"help": "whether the reference model is a LoRA finetuned model."}
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

    # data params
    reward_data_path: str = ""
    policy_data_path: str = ""
    value_data_path: str = ""

    # wandb logging params
    wandb_prefix: str = "credit_assignment"
    iteration: int = 1

    # sampling params
    temperature: float = 1.0

    # inference params
    per_device_batch_size: int = 2 # NOTE: never actually used

    # reward params
    length_penalty_type: Literal["none", "linear", "cosine"] = "none"
    length_penalty_coef: float = 0.1
    length_penalty_start: float = 0.5
    length_penalty_end: float = 1.0
    max_new_tokens: int = 4096

    # ppo params
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
    gamma: float = 0.9
    gae_lambda: float = 0.95

    # spag params
    spag_gamma: float = 1.0  # Paper uses γ=1.0 (no temporal discounting)
    spag_use_episode_filtering: bool = True  # Keep all real actions of winning role trajectories
    spag_reward_threshold: float = 0.0  # Each role wins when its own episode reward exceeds this

    # grpo params
    normalization_type: Literal["batch", "group"] = field(
        default="batch",
        metadata={"help": "the type of normalization to use."}
    )
