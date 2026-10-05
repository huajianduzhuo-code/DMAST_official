from pathlib import Path
from typing import Dict, List, Union

from configs.train_adversarial_args import TrainAdversarialConfig


def get_model_path_by_role_and_iteration(
    config: TrainAdversarialConfig,
    role: str,
    iteration: Union[str, int],
    cloud: bool = False,
    merged: bool = False,
) -> Dict[str, str]:
    """Get model path for a given role and iteration.
    
    Args:
        config: Training configuration
        role: "agent" or "attacker"
        iteration: "imitation", "enhanced_sft", or integer for RL iterations
        cloud: Whether to use cloud storage path (for RL only)
        merged: Whether to return the merged model path (adds _merged suffix for imitation/enhanced_sft)
    """

    agent_config = config.agent if role == "agent" else config.attacker

    if iteration == "imitation":
        imitation_ckpt_base_path = Path(config.imitation.ckpt_base_path)
        imitation_ckpt_path = str(imitation_ckpt_base_path / f"{agent_config.model_name}-{'both' if config.self_play else role}")

        if role == "agent" and not config.use_attacker:
            imitation_ckpt_path += "-task-only"
        
        if merged:
            imitation_ckpt_path += "_merged"

        iter_model_path = {}
        if config.rl.train_method in ["spag", "grpo", "online_sft"]:
            iter_model_path[role] = imitation_ckpt_path
        elif config.rl.train_method == "ppo":
            iter_model_path[f"{role}_policy"] = iter_model_path[f"{role}_value"] = imitation_ckpt_path
        else:
            raise ValueError(f"Unknown training method: {config.rl.train_method}")

    elif iteration == "enhanced_sft":
        enhanced_sft_ckpt_base_path = Path(config.enhanced_sft.ckpt_base_path)
        enhanced_sft_ckpt_path = str(enhanced_sft_ckpt_base_path / f"{agent_config.model_name}-{'both' if config.self_play else role}")

        if role == "agent" and not config.use_attacker:
            enhanced_sft_ckpt_path += "-task-only"
        
        if merged:
            enhanced_sft_ckpt_path += "_merged"

        iter_model_path = {}
        if config.rl.train_method in ["spag", "grpo", "online_sft"]:
            iter_model_path[role] = enhanced_sft_ckpt_path
        elif config.rl.train_method == "ppo":
            iter_model_path[f"{role}_policy"] = iter_model_path[f"{role}_value"] = enhanced_sft_ckpt_path
        else:
            raise ValueError(f"Unknown training method: {config.rl.train_method}")

    elif isinstance(iteration, int):
        rl_local_ckpt_base_path = Path(config.rl.local_ckpt_base_path)
        rl_cloud_ckpt_base_path = Path(config.rl.cloud_ckpt_base_path)

        if cloud:
            rl_ckpt_base_path = rl_cloud_ckpt_base_path
        else:
            rl_ckpt_base_path = rl_local_ckpt_base_path

        iter_postfix = f"iter_{iteration}"

        iter_model_path = {}
        role_ckpt_path = rl_ckpt_base_path if config.self_play else (rl_ckpt_base_path / role)
        if config.rl.train_method in ["spag", "grpo", "online_sft"]:
            iter_model_path[role] = role_ckpt_path / iter_postfix
        elif config.rl.train_method == "ppo":
            iter_model_path[f"{role}_policy"] = role_ckpt_path / "policy" / iter_postfix
            iter_model_path[f"{role}_value"] = role_ckpt_path / "value" / iter_postfix
        else:
            raise ValueError(f"Unknown training method: {config.rl.train_method}")

    else:
        raise ValueError(f"Unknown iteration: {iteration}")

    return {k: str(v) for k, v in iter_model_path.items()}


def get_current_policy_path_by_role(config: TrainAdversarialConfig, current_model_paths: Dict[str, str], role: str) -> str:
    if config.rl.train_method in ["spag", "grpo", "online_sft"]:
        return current_model_paths[role]
    elif config.rl.train_method == "ppo":
        return current_model_paths[f"{role}_policy"]
    else:
        raise ValueError(f"Unknown training method: {config.rl.train_method}")


# get all role's model paths for a given iteration
def get_model_path_by_iteration(
    config: TrainAdversarialConfig,
    iteration: Union[str, int],
    cloud: bool = False,
) -> Dict[str, str]:
    all_model_paths = {}
    for role in ["agent", "attacker"]:
        all_model_paths.update(get_model_path_by_role_and_iteration(config, role, iteration, cloud))
    return all_model_paths


# get a role's all existing model paths, used for population-based training
# used to launch vLLM server for population-based learning
def get_existing_model_paths_by_role(config: TrainAdversarialConfig, role: str, current_iteration: int) -> List[str]:
    agent_config = config.agent_configs[role]

    iterations = []
    model_paths = []
    if agent_config.train_imitation:
        # If enhanced_sft is enabled, use it instead of imitation (enhanced_sft supersedes imitation)
        if config.enhanced_sft.enabled:
            iterations.append("enhanced_sft")
        else:
            iterations.append("imitation")
    else:
        model_paths.append(agent_config.model_path)
    if agent_config.train_rl:
        iterations.extend(list(range(1, current_iteration)))

    for iteration in iterations:
        # Use merged=True for imitation/enhanced_sft since we need full models for vLLM
        use_merged = iteration in ["imitation", "enhanced_sft"]
        model_path = get_model_path_by_role_and_iteration(config, role, iteration, merged=use_merged)

        if config.rl.train_method in ["spag", "grpo", "online_sft"]:
            policy_path = model_path[role]
        elif config.rl.train_method == "ppo":
            policy_path = model_path[f"{role}_policy"]
        else:
            raise ValueError(f"Unknown training method: {config.rl.train_method}")

        model_paths.append(policy_path)

    return model_paths
