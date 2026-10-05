import torch
import yaml

def update_accelerate_config(
    config_path: str,
    gradient_accumulation_steps: int,
    zero_stage: int = 2,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["num_processes"] = torch.cuda.device_count()
    config["deepspeed_config"]["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["deepspeed_config"]["zero_stage"] = zero_stage
    if zero_stage == 3:
        config["deepspeed_config"]["offload_param_device"] = "cpu"
        config["deepspeed_config"]["zero3_save_16bit_model"] = True
    else:
        config["deepspeed_config"]["offload_param_device"] = "none"
        config["deepspeed_config"]["zero3_save_16bit_model"] = False

    with open(config_path, "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    update_accelerate_config("configs/train_accelerate_config.yaml", 8)
