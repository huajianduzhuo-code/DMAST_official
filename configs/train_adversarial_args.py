from typing import List, Literal, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from utils.accelerate_config import update_accelerate_config
from utils.utils import assert_equal
from configs.task_splits import MINIWOB_TRAIN_TASKS, MINIWOB_VALID_TASKS, MINIWOB_EVAL_TASKS


CLOUD_STORAGE_PATH = Path.home() / "bucket"
LOCAL_STORAGE_PATH = Path.home()


@dataclass
class VLLMConfig:
    host: str = "127.0.0.1"
    api_key: str = "api_key"

    # model size related params
    max_model_len: int = 32768  # Increased to match model's full capacity
    gpu_memory_utilization: float = 0.8  # GPU memory utilization (0.7 recommended for multimodal with image attacks)

    def to_str_list(self):
        # helper function to gather commands for launching vLLM server
        return [
            "--host", self.host,
            "--api-key", self.api_key,
            "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--generation-config", "vllm",              # use vllm's generation config
            "--model-impl", "vllm",
            "--logprobs-mode", "processed_logprobs",
            "--max-logprobs", "0",                      # also return the logprobs of sampled tokens
            "--enable-sleep-mode",                      # enable sleep to release VRAM during training
            "--trust-remote-code",                      # required for Qwen-VL/Gemma3 custom vision processors
        ]


@dataclass
class DataCollectionConfig:
    """Configuration for data collection with HTML injection attacks.
    
    This version uses HTML injection instead of image editing for attacks.
    The attacker generates HTML/CSS that gets injected into the page via JavaScript,
    which automatically updates both the screenshot and accessibility tree.
    
    The two execution flags control what the agent sees after injection:
    - execute_text_injection: If True, agent sees modified AXTree; if False, agent sees original AXTree
    - execute_image_attacks: If True, agent sees modified screenshot; if False, agent sees original screenshot
    
    Benefits of HTML injection over image editing:
    - Much faster (no GPU needed for attack execution)
    - Both screenshot and axtree are automatically synchronized
    - Can be parallelized easily
    - More realistic (similar to real XSS attacks)
    """

    # environment params
    use_env_ids: bool = True
    benchmark: str = "miniwob"

    train_prop: float = 0.8
    valid_prop: float = 0.1
    eval_prop: float = 0.1

    use_valid: bool = False
    
    # 🎯 Multimodal Configuration
    enable_multimodal: bool = True  # Enable multimodal training (vision + language)
    enable_agent_vision: bool = True  # Agent can see screenshots
    
    # HTML Injection Attack Flags (independently control what agent sees)
    execute_text_injection: bool = True   # If True, agent sees modified AXTree after HTML injection
    execute_image_attacks: bool = True    # If True, agent sees modified screenshot after HTML injection
    
    max_image_size: list = (896, 896)  # Image resolution for VLM models (works for both Gemma and Qwen)

    # Shared task definitions keep all collectors on the same held-out split.
    train_env_ids: List[str] = field(
        default_factory=lambda: list(MINIWOB_TRAIN_TASKS),
    )
    valid_env_ids: List[str] = field(
        default_factory=lambda: list(MINIWOB_VALID_TASKS),
    )
    eval_env_ids: List[str] = field(
        default_factory=lambda: list(MINIWOB_EVAL_TASKS),
    )

    max_episode_steps: int = 5
    num_jobs: int = 64
    job_timeout: int = 600  # timeout for each episode collection job in seconds (0 = no timeout)

    # sampling params
    max_new_tokens: int = 4096
    temperature: float = 1.2
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    # attacker privilege
    attacker_uses_web_agent_history: bool = True
    attacker_uses_web_agent_action_history: bool = True
    attacker_uses_web_agent_think_history: bool = False
    attacker_uses_web_agent_memory: bool = False

    # initialize based on RLConfig
    num_train_envs_per_iter: int = 48
    num_train_responses_per_env: int = 16
    num_valid_responses_per_env: int = 5
    collect_responses_with_same_opponent: bool = True

    def __post_init__(self):
        if not self.use_env_ids:
            if not self.use_valid:
                self.valid_prop = 0.0
            assert self.train_prop > 0.0, "train_prop must be greater than 0.0"
            assert self.valid_prop >= 0.0, "valid_prop must be greater than or equal to 0.0"
            assert self.eval_prop >= 0.0, "eval_prop must be greater than or equal to 0.0"
            assert self.train_prop + self.valid_prop + self.eval_prop == 1.0

@dataclass
class ImitationConfig:
    raw_data_path: str = f"{LOCAL_STORAGE_PATH}/dmast_data/imitation/teacher_demos.json"
    data_dirname: str = "paper_imitation"  # Local postprocessed Stage 1 data
    ckpt_dirname: str = "paper_imitation"  # Local Stage 1 checkpoints

    data_base_path: str = field(init=False)
    ckpt_base_path: str = field(init=False)

    lr: float = 5e-6
    lm_kl_coeff: float = 0.05
    use_data_weight: bool = False
    
    # Imitation-specific batch size (can be larger than RL batch size)
    batch_size: int = 128
    per_device_train_batch_size: int = 1

    # ============================================================================
    # Data Format Options
    # ============================================================================
    # If True, raw_data_path is already in transitions format (from enhanced SFT pipeline)
    # and doesn't need postprocess_rollout. The data will be used directly for training.
    # If False (default), raw_data_path is in episodes format and needs postprocess_rollout.
    is_preprocessed: bool = False  # Set to True when using both_data.json directly
    
    # For preprocessed data with separate agent/attacker data files
    # Only used when is_preprocessed=True. If set, these override raw_data_path.
    # agent_data_path: str = f"{LOCAL_STORAGE_PATH}/dmast_data/imitation_qwen8b_gemma_help/both_data.json"  # Path to preprocessed agent data (transitions format)
    # attacker_data_path: str = f"{LOCAL_STORAGE_PATH}/dmast_data/imitation_qwen8b_gemma_help/both_data.json"  # Path to preprocessed attacker data (transitions format)

    def to_str_list(self):
        # helper function to gather commands for train_model.py
        return [
            "--train_method", "sft",
            "--num_train_epochs", "1",
            "--learning_rate", self.lr,
            "--lm_kl_coeff", self.lm_kl_coeff,
            "--use_data_weight", self.use_data_weight,
        ]


@dataclass
class EnhancedSFTConfig:
    """Configuration for Enhanced SFT stage (Stage 2 in 3-stage pipeline).
    
    This stage fine-tunes the model on oracle-guided, task-focused SFT data
    after the base imitation training (Stage 1).
    
    The 3-stage pipeline:
    1. Base Imitation: Train on teacher trajectories
    2. Enhanced SFT: Fine-tune on task-focused oracle CoT and clean/attacker samples
    3. RL: GRPO self-play
    """
    # Whether to enable this stage
    enabled: bool = True
    
    # Preprocessed transitions from the README's oracle data-collection command
    agent_data_path: str = f"{LOCAL_STORAGE_PATH}/dmast_data/enhanced_sft/final_sft_dataset_simplified.json"
    attacker_data_path: str = f"{LOCAL_STORAGE_PATH}/dmast_data/enhanced_sft/final_sft_dataset_simplified.json"
    
    # Checkpoint directory name
    ckpt_dirname: str = "paper_enhanced_sft_task_focused"  # Local Stage 2 checkpoints
    
    # Checkpoint base path (will be set in __post_init__ of TrainAdversarialConfig)
    ckpt_base_path: str = field(init=False)
    
    # Training hyperparameters
    lr: float = 2e-6  # Lower learning rate for fine-tuning
    lm_kl_coeff: float = 0.05
    use_data_weight: bool = False
    
    # Batch size settings
    batch_size: int = 128
    per_device_train_batch_size: int = 1
    
    def to_str_list(self):
        return [
            "--train_method", "sft",
            "--num_train_epochs", "1",
            "--learning_rate", self.lr,
            "--lm_kl_coeff", self.lm_kl_coeff,
            "--use_data_weight", self.use_data_weight,
        ]


@dataclass
class SPAGConfig:
    lr: float = 2e-6
    gamma: float = 1.0
    low_clip_range: float = 0.2
    high_clip_range: float = 0.3

    # number of batches = num_envs_per_iter * num_responses_per_env * num_repeats_per_episode / batch_size
    num_agent_repeats_per_episode: int = 4
    num_attacker_repeats_per_episode: int = 2

    def to_str_list(self):
        return [
            "--train_method", "spag",
            "--learning_rate", self.lr,
            "--low_clip_range", self.low_clip_range,
            "--high_clip_range", self.high_clip_range,
        ]


@dataclass
class PPOConfig:
    lr: float = 2e-6

    num_agent_policy_repeats_per_episode: int = 4
    num_agent_value_repeats_per_episode: int = 4

    num_attacker_policy_repeats_per_episode: int = 2
    num_attacker_value_repeats_per_episode: int = 2

    gamma: float = 0.8
    gae_lambda: float = 0.9

    use_sampling_importance_weight: bool = True
    min_sampling_importance_weight: float = 0.1
    min_policy_importance_weight: float = 0.1
    use_ratio_clip: bool = True
    low_clip_range: float = 0.2
    high_clip_range: float = 0.3

    value_clip_range: float = 0.0                           # <= 0 means no clipping

    num_value_buckets: int = 101
    value_activation: Literal["none", "sigmoid"] = "none"
    value_min: float = -1
    value_max: float = 1

    def to_str_list(self):
        return [
            # "--train_method" is added in train.py dynamically to specify whether to train value or policy
            "--learning_rate", self.lr,
            "--use_sampling_importance_weight", self.use_sampling_importance_weight,
            "--min_sampling_importance_weight", self.min_sampling_importance_weight,
            "--min_policy_importance_weight", self.min_policy_importance_weight,
            "--use_ratio_clip", self.use_ratio_clip,
            "--low_clip_range", self.low_clip_range,
            "--high_clip_range", self.high_clip_range,
            "--value_clip_range", self.value_clip_range,
            "--num_value_buckets", self.num_value_buckets,
            "--value_activation", self.value_activation,
            "--value_min", self.value_min,
            "--value_max", self.value_max,
        ]


@dataclass
class GRPOConfig:
    lr: float = 2e-6

    num_agent_repeats_per_episode: int = 8
    num_attacker_repeats_per_episode: int = 8

    use_sampling_importance_weight: bool = True
    min_sampling_importance_weight: float = 0.1
    min_policy_importance_weight: float = 0.1
    use_ratio_clip: bool = True
    low_clip_range: float = 0.1
    high_clip_range: float = 0.3

    # for advantage computation
    normalization_type: Literal["batch", "group"] = "group"

    def to_str_list(self):
        return [
            "--train_method", "grpo",
            "--learning_rate", self.lr,
            "--use_sampling_importance_weight", self.use_sampling_importance_weight,
            "--min_sampling_importance_weight", self.min_sampling_importance_weight,
            "--min_policy_importance_weight", self.min_policy_importance_weight,
            "--use_ratio_clip", self.use_ratio_clip,
            "--low_clip_range", self.low_clip_range,
            "--high_clip_range", self.high_clip_range,
        ]


@dataclass
class OnlineSFTConfig:
    """Configuration for online SFT (iterative adversarial SFT) training.
    
    In each iteration:
    1. Agent and attacker collect episodes (adversarial interaction)
    2. Filter successful episodes (where agent succeeds)
    3. SFT on agent using these successful episodes (like imitation training)
    4. Use SFT checkpoint for next iteration
    
    The attacker is NOT trained (similar to ART).
    """
    lr: float = 2e-6
    lm_kl_coeff: float = 0.05
    use_data_weight: bool = False

    # Number of training repeats per episode (controls training steps per iteration)
    num_agent_repeats_per_episode: int = 4

    def to_str_list(self):
        """Return training command arguments for SFT training."""
        return [
            "--train_method", "sft",
            "--num_train_epochs", "1",
            "--learning_rate", self.lr,
            "--lm_kl_coeff", self.lm_kl_coeff,
            "--use_data_weight", self.use_data_weight,
        ]


@dataclass
class RLConfig:
    train_method: str = "grpo"  # spag, ppo, grpo, online_sft
    num_iterations: int = 25  # Training cap and persistent scheduler horizon
    use_persistent_lr_scheduler: bool = True

    lm_entropy_coeff: float = 0.0
    lm_kl_coeff: float = 0.05
    lm_sft_coeff: float = 0.0  # SFT regularization coefficient (0 = disabled, typical: 0.1-0.5)
    only_penalize_kl_when_smaller_than_ref: bool = False
    use_data_weight: bool = False  # Optional loss weighting; does not control SPAG winner filtering

    # reward params
    length_penalty_type: Literal["none", "linear", "cosine"] = "cosine"
    length_penalty_coef: float = 0.1
    length_penalty_start: float = 0.75
    length_penalty_end: float = 1.0

    spag: SPAGConfig = field(default_factory=SPAGConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    online_sft: OnlineSFTConfig = field(default_factory=OnlineSFTConfig)

    ckpt_dirname: str = "rl"
    progress_fname: str = "rl_progress.txt"

    local_ckpt_base_path: str = field(init=False)
    cloud_ckpt_base_path: str = field(init=False)
    progress_file_path: str = field(init=False)

    # SPAG: role -> path
    # PPO: {role}_{module_name} -> path
    # online_sft: role -> path
    module_to_optimizer_save_path: Dict[str, str] = field(default_factory=dict)

    def to_str_list(self):
        base_commands = [
            "--lm_kl_coeff", self.lm_kl_coeff,
            "--lm_sft_coeff", self.lm_sft_coeff,
            "--only_penalize_kl_when_smaller_than_ref", self.only_penalize_kl_when_smaller_than_ref,
            "--lm_entropy_coeff", self.lm_entropy_coeff,
            "--use_data_weight", self.use_data_weight,
        ]

        if self.train_method == "spag":
            return base_commands + self.spag.to_str_list()
        elif self.train_method == "ppo":
            return base_commands + self.ppo.to_str_list()
        elif self.train_method == "grpo":
            return base_commands + self.grpo.to_str_list()
        elif self.train_method == "online_sft":
            return self.online_sft.to_str_list()
        else:
            raise ValueError(f"Unknown training method: {self.train_method}")


@dataclass
class LoraConfig:
    use_lora: bool = True
    r: int = 128
    lora_alpha: int = 256
    target_modules: Union[str, List[str]] = "all-linear"
    bias: str = "none"

    def to_str_list(self):
        if self.use_lora:
            if isinstance(self.target_modules, str):
                target_modules = self.target_modules
            elif isinstance(self.target_modules, list):
                target_modules = ",".join(self.target_modules)
            else:
                raise ValueError(f"Unknown target modules type: {type(self.target_modules)}")

            return [
                "--use_lora", self.use_lora,
                "--lora_r", self.r,
                "--lora_alpha", self.lora_alpha,
                "--lora_target_modules", target_modules,
                "--lora_bias", self.bias,
            ]
        else:
            return []

    @property
    def description(self):
        if self.use_lora:
            if isinstance(self.target_modules, str):
                target_modules = self.target_modules
            elif isinstance(self.target_modules, list):
                target_modules = "_".join(self.target_modules)
            else:
                raise ValueError(f"Unknown target modules type: {type(self.target_modules)}")

            return f"-lora_r_{self.r}_alpha_{self.lora_alpha}_modules_{target_modules}"
        else:
            return ""


@dataclass
class AgentConfig:
    model_path: str
    ref_model_path: Optional[str] = None

    train_imitation: bool = field(default=True)
    train_rl: bool = field(default=True)
    train_use_population: bool = field(default=False)

    # training config, adjust given model size and gpus
    batch_size: int = 128
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = field(init=False)
    zero_stage: int = 2

    lora: LoraConfig = field(default_factory=LoraConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    model_name: str = field(init=False)

    def __post_init__(self):
        if self.train_use_population:
            assert self.train_rl, "Must be trained with RL if using population"

        if self.ref_model_path is None:
            self.ref_model_path = self.model_path

        if (
            "pretrained" in self.model_path or
            "imitation" in self.model_path
        ):
            self.model_name = self.model_path.split("/")[-1]
        else:
            raise NotImplementedError("Only pretrained and imitation models are supported for now.")

        print(self.batch_size)
        print(self.per_device_train_batch_size)
        print(torch.cuda.device_count())

        # compute gradient accumulation steps
        assert self.batch_size % (self.per_device_train_batch_size * torch.cuda.device_count()) == 0, \
            "Batch size must be divisible by (number of devices * per device batch size)"
        self.gradient_accumulation_steps = self.batch_size // (self.per_device_train_batch_size * torch.cuda.device_count())


@dataclass
class TrainAdversarialConfig:
    seed: int = 0
    exp_name: str = "test" # placeholder, will be overwritten in __post_init__
    group_name: str = "dmast_paper"

    local_ckpt_base_path: str = f"{LOCAL_STORAGE_PATH}/dmast_ckpt"
    cloud_ckpt_base_path: str = f"{CLOUD_STORAGE_PATH}/ckpt"
    data_base_path: str = f"{LOCAL_STORAGE_PATH}/dmast_data"

    exp_local_ckpt_path: str = field(init=False)
    exp_cloud_ckpt_path: str = field(init=False)
    exp_data_path: str = field(init=False)

    # Paper pipeline: pretrained Gemma-3-12B-IT -> imitation -> oracle SFT -> GRPO
    agent: AgentConfig = field(default_factory=lambda: AgentConfig(
        model_path=f"{LOCAL_STORAGE_PATH}/dmast_ckpt/pretrained/gemma-3-12b-it",
        per_device_train_batch_size=1,
        train_imitation=True,
        train_rl=True,
        train_use_population=False, # NOTE: whether to use population-based training for the agent
        zero_stage=2,
        rl=RLConfig(
            lm_kl_coeff=0.05,
            lm_sft_coeff=0.0,      # Optional SFT regularization (can set to 0.1-0.2 if needed)
            use_data_weight=False,
        ),
    ))
    attacker: AgentConfig = field(default_factory=lambda: AgentConfig(
        model_path=f"{LOCAL_STORAGE_PATH}/dmast_ckpt/pretrained/gemma-3-12b-it",
        per_device_train_batch_size=1,
        train_imitation=True,
        train_rl=True,
        train_use_population=False, # NOTE: attacker should never use population
        zero_stage=2,
        rl=RLConfig(
            lm_kl_coeff=0.05,
            lm_sft_coeff=0.0,      # Optional SFT regularization (can set to 0.1-0.2 if needed)
            use_data_weight=False,
        ),
    ))
    self_play: bool = True # if true, the agent and attacker shares the same model. There is no distinction between agent's ckpt and attacker's ckpt. Data is processed together, training is unified. This must be set to False when using ART in the attacker.
    use_attacker: bool = True # if false, this trains an agent-only baseline where agent performs tasks in a clean environment. Must set train_rl in attacker to False
    attacker_use_art: bool = False # when use ART, it adds a MutationPrompt to the attacker's prompt. This must not be used together with RL
    
    # Population-based training: minimum ratio of GRPO groups allocated to the latest opponent
    # This ensures the agent always learns to counter the newest attacker strategies
    # Set to 0.5 means at least 50% of groups go to the latest attacker, remaining groups use recency-weighted sampling
    min_latest_opponent_ratio: float = 0.5
    
    # Recency weight decay for population-based training
    # Controls how much more weight recent opponents get vs older ones
    # Higher = more weight to recent opponents (each older opponent has exp(-decay) times the weight)
    # 0.7 means each older opponent has ~0.5x weight of the next newer one
    # 1.0 means each older opponent has ~0.37x weight of the next newer one
    recency_weight_decay: float = 0.7

    rl: RLConfig = field(init=False)

    # wandb config
    wandb_project: str = "agent_adversarial_learning"

    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    enhanced_sft: EnhancedSFTConfig = field(default_factory=EnhancedSFTConfig)

    def __post_init__(self):
        # set up experiment name
        self.exp_name = f"{self.agent.rl.train_method}"
        self.exp_name += f"_{self.agent.model_name.split('/')[-1].replace('-', '_')}"

        if self.self_play:
            self.exp_name += f"_self_play"
        
        if self.attacker_use_art:
            self.exp_name += f"_art"

        if self.agent.train_rl:
            if self.agent.train_use_population:
                self.exp_name += f"_agent_population"
            else:
                self.exp_name += f"_agent_iterative"

        if self.attacker.train_rl:
            if self.attacker.train_use_population:
                self.exp_name += f"_attacker_population"
            else:
                self.exp_name += f"_attacker_iterative"

        self.exp_name += f"_seed_{self.seed}"

        # train_rl arguments validation
        is_online_sft = self.agent.rl.train_method == "online_sft"

        if is_online_sft:
            # Online SFT: agent must be trained (train_rl=True drives the iterative loop),
            # but attacker is NOT trained (similar to ART)
            assert self.agent.train_rl, "For online_sft, the agent must have train_rl=True"
            assert not self.attacker.train_rl, "For online_sft, the attacker must NOT be trained"
            assert self.use_attacker, "For online_sft, the attacker must be used (adversarial collection)"
            assert not self.self_play, "For online_sft, self_play is not supported"
            assert not self.agent.train_use_population, "For online_sft, population training is not supported"
        else:
            assert self.agent.train_rl or self.attacker.train_rl, \
                "At least one of the agent or the attacker must be trained with RL"

        if self.attacker.train_rl:
            assert self.use_attacker, "When training the attacker with RL, it must be used"

            if self.attacker.train_use_population:
                assert self.agent.train_rl, "When training the attacker with a population of agents, the agent must be trained with RL"

        if self.agent.train_rl and self.agent.train_use_population:
            assert self.use_attacker, "When training the agent with a population of attackers, the attacker must be used"
            assert self.attacker.train_rl, "When training the agent with a population of attackers, the attacker must be trained with RL"

        if self.attacker_use_art:
            assert self.use_attacker, "When using ART in the attacker, it must be used"
            assert not self.attacker.train_rl
            assert not self.self_play

        # finalize paths, change to absolute paths
        self.local_ckpt_base_path = str(Path(self.local_ckpt_base_path).resolve())
        self.cloud_ckpt_base_path = str(Path(self.cloud_ckpt_base_path).resolve())
        self.data_base_path = str(Path(self.data_base_path).resolve())

        self.exp_local_ckpt_path = str(Path(self.local_ckpt_base_path) / self.group_name / self.exp_name) # NOTE: path to save the local checkpoints
        self.exp_cloud_ckpt_path = str(Path(self.cloud_ckpt_base_path) / self.group_name / self.exp_name) # NOTE: path to save the cloud checkpoints
        self.exp_data_path = str(Path(self.data_base_path) / self.group_name / self.exp_name) # NOTE: path to save the data

        self.imitation.ckpt_base_path = str(Path(self.local_ckpt_base_path) / self.imitation.ckpt_dirname) # This is a LOCAL PATH
        self.imitation.data_base_path = str(Path(self.data_base_path) / self.imitation.data_dirname) # This is a LOCAL PATH

        # set up Enhanced SFT checkpoint path
        self.enhanced_sft.ckpt_base_path = str(Path(self.local_ckpt_base_path) / self.enhanced_sft.ckpt_dirname) # This is a LOCAL PATH

        # set up RL paths for each agent
        for role, agent_config in self.agent_configs.items():
            agent_config.rl.local_ckpt_base_path = str(Path(self.exp_local_ckpt_path) / agent_config.rl.ckpt_dirname)
            agent_config.rl.cloud_ckpt_base_path = str(Path(self.exp_cloud_ckpt_path) / agent_config.rl.ckpt_dirname)
            agent_config.rl.progress_file_path = str(Path(self.exp_cloud_ckpt_path) / agent_config.rl.progress_fname)

            # Update data collection and ckpt save params
            if agent_config.rl.train_method == "ppo":
                for module_name in ["policy", "value"]:
                    if self.self_play:
                        agent_config.rl.module_to_optimizer_save_path[f"{role}_{module_name}"] = str(Path(self.exp_local_ckpt_path) / module_name)
                    else:
                        agent_config.rl.module_to_optimizer_save_path[f"{role}_{module_name}"] = str(Path(self.exp_local_ckpt_path) / role / module_name)

            elif agent_config.rl.train_method in ["spag", "grpo", "online_sft"]:
                if self.self_play:
                    agent_config.rl.module_to_optimizer_save_path[role] = str(Path(self.exp_local_ckpt_path))
                else:
                    agent_config.rl.module_to_optimizer_save_path[role] = str(Path(self.exp_local_ckpt_path) / role)

            else:
                raise ValueError(f"Unknown training method: {agent_config.rl.train_method}")

        # self play arguments validation
        if self.self_play:
            assert_equal(
                asdict(self.agent),
                asdict(self.attacker),
                ignore_keys=[
                    "train_use_population",
                    "rl.optim_state_paths",
                    "rl.module_to_optimizer_save_path",
                ],
            )
            assert self.use_attacker, "When self-playing, the attacker must be used"
            assert self.agent.train_rl, "When self-playing, the agent must be trained with RL"
            assert self.attacker.train_rl, "When self-playing, the attacker must be trained with RL"

        # check if shared RL config is same
        assert self.agent.rl.train_method == self.attacker.rl.train_method, "Agent and attacker must have the same RL training method"
        assert self.agent.rl.num_iterations == self.attacker.rl.num_iterations, "Agent and attacker must have the same number of RL iterations"
        assert self.agent.rl.use_persistent_lr_scheduler == self.attacker.rl.use_persistent_lr_scheduler, "Agent and attacker must have the same use_persistent_lr_scheduler"
        assert self.agent.rl.ckpt_dirname == self.attacker.rl.ckpt_dirname, "Agent and attacker must have the same ckpt_dirname"
        assert self.agent.rl.progress_fname == self.attacker.rl.progress_fname, "Agent and attacker must have the same progress_fname"

        self.rl = self.agent.rl

        # if using imitation, update the reference model from the base model to SFT model
        # Note: ref_model uses PeftModelForCausalLM which can auto-handle both LoRA adapter and merged model
        for role, agent_config in self.agent_configs.items():
            from utils.path import get_model_path_by_role_and_iteration
            if agent_config.train_imitation:
                # Determine which stage to use: enhanced_sft if enabled, otherwise imitation
                sft_stage = "enhanced_sft" if self.enhanced_sft.enabled else "imitation"
                if agent_config.rl.train_method == "ppo":
                    agent_config.ref_model_path = get_model_path_by_role_and_iteration(self, role, sft_stage)[f"{role}_policy"]
                elif agent_config.rl.train_method in ["spag", "grpo", "online_sft"]:
                    agent_config.ref_model_path = get_model_path_by_role_and_iteration(self, role, sft_stage)[role]
                else:
                    raise ValueError(f"Unknown training method: {self.rl.train_method}")

    @property
    def agent_configs(self):
        return {
            "agent": self.agent,
            "attacker": self.attacker,
        }

    @property
    def roles(self):
        return ["agent", "attacker"]

    # shared arguments for imitation and RL training
    def get_common_training_commands(self, role: str, is_imitation: bool, wandb_disabled: bool):
        agent_config = self.agent_configs[role]

        # Use imitation-specific batch size for imitation training, RL batch size for RL training
        if is_imitation:
            batch_size = self.imitation.batch_size
            per_device_batch_size = self.imitation.per_device_train_batch_size
            # Compute gradient accumulation for imitation
            gradient_accumulation_steps = batch_size // (per_device_batch_size * torch.cuda.device_count())
        else:
            batch_size = agent_config.batch_size
            per_device_batch_size = agent_config.per_device_train_batch_size
            gradient_accumulation_steps = agent_config.gradient_accumulation_steps

        update_accelerate_config(
            "configs/train_accelerate_config.yaml",
            gradient_accumulation_steps,
            agent_config.zero_stage,
        )

        return [
            "accelerate", "launch",
            "--config_file", "configs/train_accelerate_config.yaml",
            "tools/train_model.py",
            "--seed", self.seed,
            "--ref_model_name_or_path", agent_config.model_path if is_imitation else agent_config.ref_model_path,
            "--roles", *(self.roles if self.self_play else [role]),
            # save and logging params
            "--save_total_limit", 1,
            "--save_strategy", "no",
            "--logging_steps", 10,
            "--report_to", "none" if wandb_disabled else "wandb",
            # training params
            "--per_device_train_batch_size", per_device_batch_size,
            "--gradient_accumulation_steps", gradient_accumulation_steps,
            "--average_tokens_across_devices", True,
            "--gradient_checkpointing", True,
            "--lr_scheduler_type", "cosine",
            "--warmup_ratio", 0.03,
            "--temperature", self.data_collection.temperature,
            # misc
            "--remove_unused_columns", False,
            "--tf32", True,
            "--bf16", True,
            # lora params
            *agent_config.lora.to_str_list(),
        ]
