import json
import libtmux
import os
import tempfile
import time
import requests
import shutil
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoConfig

from configs.train_adversarial_args import TrainAdversarialConfig
from utils.path import get_current_policy_path_by_role, get_existing_model_paths_by_role
from utils.utils import assert_equal


def get_vllm_available_models(url: str, api_key: str):
    """Waits for the vLLM server to become ready."""

    check_url = f"{url}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(check_url, headers=headers, timeout=5)
        if response.status_code == 200:
            model_info = response.json().get('data', [])
            available_model_names = [info["id"] for info in model_info]
            return available_model_names
        else:
            return []
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
        return []


def wait_for_vllm(url: str, api_key: str, timeout: int) -> None:
    """Waits for the vLLM server to become ready."""

    print(f"Waiting for vLLM server at {url} (timeout: {timeout}s)")
    start_time = time.time()

    while True:
        available_model_names = get_vllm_available_models(url, api_key)
        if len(available_model_names):
            return

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= timeout:
            raise Exception(f"ERROR: vLLM server at {url} failed to become ready. Stopping iteration.")

        print(".", end="", flush=True)
        time.sleep(5)


def close_vllm_session(session_name: str):
    # Attempts to kill the specified tmux session

    try:
        server = libtmux.Server()
        session = server.find_where({"session_name": session_name})

        if session:
            session.kill_session()
            print(f"Tmux session '{session_name}' killed successfully using libtmux.")
        else:
            print(f"Tmux session '{session_name}' does not exist or was already closed.")

    except Exception as e:
        print(f"An unexpected error occurred while using libtmux: {e}")


def post_request(url: str, headers: Dict, data: Optional[str | Dict] = None):
    if data is not None:
        assert isinstance(data, (str, dict))
        if isinstance(data, dict):
            data = json.dumps(data)

    response = requests.post(url, headers=headers, data=data, timeout=3600)

    if response.status_code == 200:
        return response
    else:
        error_message = f"Request failed with status code: {response.status_code}"
        try:
            error_body = json.dumps(response.json(), indent=4)
            error_message += f"\nResponse Body: {error_body}"
        except json.JSONDecodeError:
            error_body = response.text
            error_message += f"\nResponse Body: {error_body}"
        raise requests.exceptions.RequestException(error_message, response=response)


def is_vllm_sleeping(url: str, api_key: str):
    is_sleeping_url = f"{url}/is_sleeping"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(is_sleeping_url, headers=headers, timeout=5)
        if response.status_code == 200:
            is_sleeping = response.json()["is_sleeping"]
            return is_sleeping
        else:
            error_message = f"Request failed with status code: {response.status_code}"
            try:
                error_body = json.dumps(response.json(), indent=4)
                error_message += f"\nResponse Body: {error_body}"
            except json.JSONDecodeError:
                error_body = response.text
                error_message += f"\nResponse Body: {error_body}"
            raise requests.exceptions.RequestException(error_message, response=response)

    except requests.exceptions.ConnectionError:
        # Server is not yet accepting connections
        return False
    except requests.exceptions.Timeout:
        # Connection timed out
        return False
    except requests.exceptions.RequestException as e:
        # Other request errors (DNS, invalid URL, etc.)
        return False


def sleep_vllm(url: str, api_key: str):
    if is_vllm_sleeping(url, api_key):
        return

    sleep_url = f"{url}/sleep?level=1"
    headers = {"Authorization": f"Bearer {api_key}"}
    post_request(sleep_url, headers)


def wake_up_vllm(url: str, api_key: str):
    if not is_vllm_sleeping(url, api_key):
        return

    wake_up_url = f"{url}/wake_up"
    headers = {"Authorization": f"Bearer {api_key}"}
    post_request(wake_up_url, headers)


def load_lora_modules(url: str, api_key: str, lora_module_names: List[str]) -> None:
    available_model_names = get_vllm_available_models(url, api_key)
    for lora_module_name in lora_module_names:
        if lora_module_name in available_model_names:
            print(f"LoRA Module {lora_module_name} already available, skipping loading")
            continue

        load_url = f"{url}/v1/load_lora_adapter"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "lora_name": lora_module_name,
            "lora_path": lora_module_name
        }

        post_request(load_url, headers, data)
        print(f"Load LoRA Modules {lora_module_name} successful! Status Code: 200")


def unload_lora_modules(url: str, api_key: str, lora_module_names: List[str]) -> None:
    available_model_names = get_vllm_available_models(url, api_key)
    print("to unload")
    for lora_module_name in lora_module_names:
        print(lora_module_name)
    print("available")
    for lora_module_name in available_model_names:
        print(lora_module_name)
    for lora_module_name in lora_module_names:
        if lora_module_name not in available_model_names:
            print(f"LoRA Module {lora_module_name} not found, skipping unloading")
            continue

        unload_url = f"{url}/v1/unload_lora_adapter"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "lora_name": lora_module_name,
        }

        post_request(unload_url, headers, data)
        print(f"Unload LoRA Modules {lora_module_name} successful! Status Code: 200")


def is_multimodal_model(model_path: str) -> bool:
    """Check if model is a multimodal model (has vision components)"""
    try:
        config = AutoConfig.from_pretrained(model_path)
        # Check for multimodal architecture
        if hasattr(config, 'architectures') and len(config.architectures) > 0:
            arch = config.architectures[0]
            # Multimodal models often have "ForConditionalGeneration" or specific vision configs
            if "ConditionalGeneration" in arch:
                return True
            # Check for vision config attributes
            if hasattr(config, 'vision_config') or hasattr(config, 'vision_tower'):
                return True
        return False
    except Exception as e:
        print(f"Warning: Could not determine if model is multimodal: {e}")
        return False


def get_model_family(model_path: str) -> str:
    """
    Detect the model family from model path or config.
    
    Returns:
        str: "gemma", "qwen", or "unknown"
    
    This is critical for handling different model formats:
    - Gemma: Uses <think>...</think> tags with enable_thinking=True in chat template
    - Qwen: Outputs thinking directly without explicit tags (or uses different format)
    """
    model_path_lower = str(model_path).lower()
    
    # Check by model path name
    if "gemma" in model_path_lower:
        return "gemma"
    elif "qwen" in model_path_lower:
        return "qwen"
    
    # Try to detect from config
    try:
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, 'model_type'):
            model_type = config.model_type.lower()
            if "gemma" in model_type:
                return "gemma"
            elif "qwen" in model_type:
                return "qwen"
        
        if hasattr(config, 'architectures') and config.architectures:
            arch = config.architectures[0].lower()
            if "gemma" in arch:
                return "gemma"
            elif "qwen" in arch:
                return "qwen"
    except Exception as e:
        print(f"Warning: Could not detect model family from config: {e}")
    
    return "unknown"


def has_think_tags(response_text: str) -> bool:
    """Check if response contains <think>...</think> tags."""
    import re
    return bool(re.search(r'<think>.*?</think>', response_text, re.DOTALL))


def extract_think_content(response_text: str, model_family: str = None) -> Optional[str]:
    """
    Extract think/reasoning content from model response.
    
    Handles different model formats:
    - Gemma: <think>content</think>
    - Qwen: Content before the action (no explicit tags)
    
    Args:
        response_text: The full response text
        model_family: "gemma", "qwen", or None (auto-detect)
        
    Returns:
        The think content or None if not found
    """
    import re
    
    # First try to find explicit <think> tags (works for Gemma)
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    
    # If no explicit tags and model is Qwen (or unknown), try to extract thinking
    # For Qwen, the thinking is typically everything before the action
    if model_family in ["qwen", "unknown"]:
        # Try to find action at the end
        action_patterns = [
            r'(click\([^)]+\))',
            r'(fill\([^)]+\))',
            r'(scroll\([^)]+\))',
            r'(goto\([^)]+\))',
            r'(select_option\([^)]+\))',
            r'(send_msg_to_user\([^)]+\))',
            r'<action>(.*?)</action>',
            r'<target>(.*?)</target>',  # For attacker
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                action_start = match.start()
                think_content = response_text[:action_start].strip()
                if think_content:
                    return think_content
        
        # If no action found, treat entire response as think content
        if response_text.strip() and not response_text.strip().startswith('<'):
            return response_text.strip()
    
    return None




def merge_lora_weights_for_vllm(model_path: str, output_path: Optional[str] = None) -> str:
    """
    Merge LoRA weights into base model for vLLM compatibility with multimodal models.
    
    Since vLLM doesn't support LoRA on vision towers, we need to merge LoRA weights
    into the base model before loading it into vLLM. This function:
    1. Loads the base model with LoRA adapter
    2. Merges LoRA weights into the base model
    3. Saves the merged model to a new location
    
    Args:
        model_path: Path to LoRA adapter directory (contains adapter_config.json)
        output_path: Optional path to save merged model. If None, creates temp dir.
        
    Returns:
        Path to merged model directory
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    model_path = Path(model_path)
    
    # Check if this is a LoRA model
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"No adapter_config.json found at {model_path}, assuming base model (no merge needed)")
        return str(model_path)

    # Create output directory
    if output_path is None:
        # Use fixed merged directory name (no random suffix)
        # This ensures the same LoRA adapter always merges to the same path
        output_path = model_path.parent / f"{model_path.name}_merged"
    output_path = Path(output_path)
    
    # If merged model already exists, skip merging and return existing path
    if output_path.exists() and (output_path / "config.json").exists():
        print(f"✅ Merged model already exists at {output_path}, skipping merge")
        return str(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load adapter config to get base model path
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")
    
    if not base_model_path:
        raise ValueError(f"Could not find base_model_name_or_path in {adapter_config_path}")
    
    print(f"🔄 Merging LoRA adapter from {model_path} with base model {base_model_path}")
    
    try:
        # Load base model - use correct model class for VLMs
        print(f"📥 Loading base model from {base_model_path}")
        
        # Check if this is a VLM model that requires a specific model class
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        config_class_name = config.__class__.__name__
        
        if "Qwen3VL" in config_class_name or "Qwen2VL" in config_class_name:
            print(f"🔄 Detected Qwen VLM model, using Qwen3VLForConditionalGeneration")
            from transformers import Qwen3VLForConditionalGeneration
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation='eager'
            )
        elif "Gemma3" in config_class_name:
            print(f"🔄 Detected Gemma3 VLM model, using Gemma3ForConditionalGeneration")
            from transformers import Gemma3ForConditionalGeneration
            base_model = Gemma3ForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation='eager'
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation='eager'
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Load LoRA adapter
        print(f"📥 Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(base_model, str(model_path))
        
        # Merge LoRA weights into base model
        print(f"🔧 Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        print(f"💾 Saving merged model to {output_path}")
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        # Copy necessary config files (including video processor for Qwen VLM)
        config_files_to_copy = [
            "preprocessor_config.json",
            "generation_config.json",
            "video_preprocessor_config.json",  # Required for Qwen3-VL
            "chat_template.json",  # May be needed for chat formatting
        ]
        for config_file in config_files_to_copy:
            src_path = Path(base_model_path) / config_file
            dst_path = output_path / config_file
            if src_path.exists() and not dst_path.exists():
                shutil.copy(src_path, dst_path)
                print(f"   📄 Copied {config_file}")
        
        print(f"✅ Successfully merged LoRA weights to {output_path}")
        
        # Clean up
        del base_model, model, merged_model
        torch.cuda.empty_cache()
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Failed to merge LoRA weights: {e}")
        # Clean up failed merge attempt
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        raise


def get_tensor_parallel_size(model_path: str, num_gpus: int) -> int:
    config = AutoConfig.from_pretrained(model_path)

    assert len(config.architectures) == 1, "Only one architecture is supported"
    architecture = config.architectures[0]

    if architecture.endswith("ForCausalLM"):
        num_attention_heads = config.num_attention_heads
    elif architecture.endswith("ForConditionalGeneration"):
        num_attention_heads = config.text_config.num_attention_heads
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    if num_attention_heads > num_gpus:
        assert num_attention_heads % num_gpus == 0, "Number of attention heads must be divisible by number of GPUs"
        return num_gpus
    else:
        assert num_gpus % num_attention_heads == 0, "Number of GPUs must be divisible by number of attention heads"
        return num_attention_heads


def get_population_model_paths_for_multimodal(
    config: TrainAdversarialConfig,
    current_model_paths: Dict[str, str],
    on_policy_role: str,
    current_iter: int,
) -> Tuple[List[str], List[str]]:
    """
    Get all model paths for population-based training with multimodal models.
    
    For multimodal models, we need to merge LoRA weights before loading.
    This function returns lists of model paths for both roles, handling the
    merging of LoRA weights as needed.
    
    Returns:
        Tuple[List[str], List[str]]: (agent_model_paths, attacker_model_paths)
            - For on_policy_role="agent": agent_model_paths has 1 element (current), 
              attacker_model_paths has all historical attackers
            - For on_policy_role="attacker": attacker_model_paths has 1 element (current),
              agent_model_paths has 1 element (current) - no population for attacker
            - For on_policy_role="both": both have 1 element each
    """
    current_agent_policy_path = get_current_policy_path_by_role(config, current_model_paths, "agent")
    current_attacker_policy_path = get_current_policy_path_by_role(config, current_model_paths, "attacker")
    
    is_multimodal = is_multimodal_model(config.agent.model_path)
    use_lora = config.agent.lora.use_lora
    
    # For multimodal models with LoRA, we need to merge the weights
    if is_multimodal and use_lora:
        # Check if current paths have LoRA adapters
        agent_has_adapter = (Path(current_agent_policy_path) / "adapter_config.json").exists()
        attacker_has_adapter = (Path(current_attacker_policy_path) / "adapter_config.json").exists()
        
        if agent_has_adapter:
            current_agent_policy_path = merge_lora_weights_for_vllm(current_agent_policy_path)
        if attacker_has_adapter:
            current_attacker_policy_path = merge_lora_weights_for_vllm(current_attacker_policy_path)
    
    # Get existing model paths for population
    existing_agent_model_paths = get_existing_model_paths_by_role(config, "agent", current_iter)
    existing_attacker_model_paths = get_existing_model_paths_by_role(config, "attacker", current_iter)
    
    # For multimodal models, merge all historical LoRA adapters
    if is_multimodal and use_lora:
        merged_agent_paths = []
        for path in existing_agent_model_paths:
            if (Path(path) / "adapter_config.json").exists():
                merged_path = merge_lora_weights_for_vllm(path)
                merged_agent_paths.append(merged_path)
            else:
                merged_agent_paths.append(path)
        existing_agent_model_paths = merged_agent_paths
        
        merged_attacker_paths = []
        for path in existing_attacker_model_paths:
            if (Path(path) / "adapter_config.json").exists():
                merged_path = merge_lora_weights_for_vllm(path)
                merged_attacker_paths.append(merged_path)
            else:
                merged_attacker_paths.append(path)
        existing_attacker_model_paths = merged_attacker_paths
    
    # Determine which models to use based on on_policy_role
    if on_policy_role == "agent":
        # Agent is on-policy (current), attacker uses population
        agent_model_paths = [current_agent_policy_path]
        if config.agent.train_use_population and len(existing_attacker_model_paths) > 0:
            # Include current attacker in the population
            attacker_model_paths = existing_attacker_model_paths + [current_attacker_policy_path]
            # Remove duplicates while preserving order
            seen = set()
            attacker_model_paths = [x for x in attacker_model_paths if not (x in seen or seen.add(x))]
        else:
            attacker_model_paths = [current_attacker_policy_path]
    elif on_policy_role == "attacker":
        # Attacker is on-policy (current), agent is latest only (no population for attacker)
        attacker_model_paths = [current_attacker_policy_path]
        # Note: We don't use population when training attacker (asymmetric strategy)
        agent_model_paths = [current_agent_policy_path]
    elif on_policy_role == "both":
        # Both are on-policy (no population)
        agent_model_paths = [current_agent_policy_path]
        attacker_model_paths = [current_attacker_policy_path]
    else:
        raise ValueError(f"Unknown on-policy role: {on_policy_role}")
    
    return agent_model_paths, attacker_model_paths


def start_vllm_for_single_pair(
    config: TrainAdversarialConfig,
    agent_model_path: str,
    attacker_model_path: str,
    on_policy_role: str,
) -> Tuple[Dict[Literal["agent", "attacker"], Dict[str, str]], List[Dict[str, Any]]]:
    """
    Start vLLM server(s) for a single agent-attacker pair.
    
    This is used for population-based training with multimodal models where
    we can only load one model at a time. We iterate through different opponent
    models and collect episodes sequentially.
    
    Args:
        config: Training configuration
        agent_model_path: Path to the agent model (already merged if needed)
        attacker_model_path: Path to the attacker model (already merged if needed)
        on_policy_role: Which role is on-policy ("agent", "attacker", or "both")
        
    Returns:
        Same format as start_vllm - (vllm_model_arguments, vllm_server_args_list)
    """
    is_multimodal = is_multimodal_model(config.agent.model_path)
    use_lora = config.agent.lora.use_lora and not is_multimodal  # Disable LoRA for multimodal
    
    agent_configs = {
        "agent": config.agent,
        "attacker": config.attacker,
    }
    
    # Get GPU configuration
    port_offset = 1234
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        all_gpus = list(range(torch.cuda.device_count()))
    else:
        all_gpus = [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    
    available_gpus = all_gpus
    num_available_gpus = len(available_gpus)
    
    # Check if both models are the same (can share server)
    use_same_checkpoint = agent_model_path == attacker_model_path
    
    if use_same_checkpoint:
        # 1 server, shared by agent and attacker
        agent_config = config.agent
        
        tensor_parallel_size = get_tensor_parallel_size(config.agent.model_path, num_available_gpus)
        pipeline_parallel_size = num_available_gpus // tensor_parallel_size
        url = f"http://{agent_config.vllm.host}:{port_offset}"
        
        vllm_server_args_list = [
            {
                "role": "agent",
                "model_path": agent_model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "pipeline_parallel_size": pipeline_parallel_size,
                "url": url,
                "api_key": agent_config.vllm.api_key,
                "lora_module_names": set(),  # No LoRA for multimodal
            },
        ]
        
        vllm_model_arguments = {
            "agent": {
                "model_names": [agent_model_path],
                "urls": url,
                "api_key": agent_config.vllm.api_key,
            },
        }
        
        if config.use_attacker:
            vllm_model_arguments["attacker"] = {
                "model_names": [attacker_model_path],
                "urls": url,
                "api_key": agent_config.vllm.api_key,
            }
    else:
        # 2 servers, one for agent and one for attacker
        vllm_server_args_list = []
        vllm_model_arguments = {}
        
        roles_to_start = ["agent"]
        if config.use_attacker:
            roles_to_start.append("attacker")
        
        num_roles = len(roles_to_start)
        gpus_per_role = num_available_gpus // num_roles
        
        for j, role in enumerate(roles_to_start):
            agent_config = agent_configs[role]
            current_path = agent_model_path if role == "agent" else attacker_model_path
            
            # 🔧 Compute tensor_parallel_size per role (agent and attacker may have different model sizes)
            role_model_path = agent_config.model_path
            tensor_parallel_size = get_tensor_parallel_size(role_model_path, gpus_per_role)
            pipeline_parallel_size = gpus_per_role // tensor_parallel_size
            
            url = f"http://{agent_config.vllm.host}:{port_offset + j}"
            
            vllm_server_args_list.append({
                "role": role,
                "model_path": current_path,
                "tensor_parallel_size": tensor_parallel_size,
                "pipeline_parallel_size": pipeline_parallel_size,
                "url": url,
                "api_key": agent_config.vllm.api_key,
                "lora_module_names": set(),  # No LoRA for multimodal
            })
            
            vllm_model_arguments[role] = {
                "model_names": [agent_model_path if role == "agent" else attacker_model_path],
                "urls": url,
                "api_key": agent_config.vllm.api_key,
            }
    
    # Start/restart vLLM servers
    _start_or_restart_vllm_servers(
        config,
        vllm_server_args_list,
        available_gpus,
        use_lora=use_lora,
    )
    
    return vllm_model_arguments, vllm_server_args_list


def _start_or_restart_vllm_servers(
    config: TrainAdversarialConfig,
    vllm_server_args_list: List[Dict[str, Any]],
    available_gpus: List[int],
    use_lora: bool = False,
) -> None:
    """
    Helper function to start or restart vLLM servers based on configuration.
    
    This handles the logic of checking if existing servers can be reused
    or if they need to be restarted with different models.
    """
    agent_configs = {
        "agent": config.agent,
        "attacker": config.attacker,
    }
    
    port_offset = 1234
    api_key = config.agent.vllm.api_key
    
    # Check all possible vLLM sessions
    all_possible_servers = [
        {"session_name": "0_vllm_server_agent", "url": f"http://127.0.0.1:{port_offset}", "role": "agent"},
        {"session_name": "1_vllm_server_attacker", "url": f"http://127.0.0.1:{port_offset + 1}", "role": "attacker"},
    ]
    
    existing_servers_config = []
    for server_info in all_possible_servers:
        available_models = get_vllm_available_models(server_info["url"], api_key)
        if len(available_models) > 0:
            existing_servers_config.append({
                'session_name': server_info["session_name"],
                'url': server_info["url"],
                'role': server_info["role"],
                'has_server': True,
                'models': available_models,
            })
    
    # Check if we need to close all servers due to configuration mismatch
    num_existing_servers = len(existing_servers_config)
    num_needed_servers = len(vllm_server_args_list)
    need_restart_all = False
    
    if num_existing_servers > 0 and num_existing_servers != num_needed_servers:
        print(f"⚠️ Server configuration mismatch:")
        print(f"   Existing: {num_existing_servers} server(s)")
        print(f"   Needed: {num_needed_servers} server(s)")
        print(f"   Closing all existing servers to reconfigure...")
        need_restart_all = True
        
        for server_config in existing_servers_config:
            print(f"   Closing {server_config['session_name']}...")
            close_vllm_session(server_config['session_name'])
        
        # Wait for servers to shut down
        print("   Waiting for servers to shut down...", end='', flush=True)
        max_wait = 30
        for i in range(max_wait):
            time.sleep(1)
            all_down = True
            for server_info in all_possible_servers:
                try:
                    models = get_vllm_available_models(server_info["url"], api_key)
                    if len(models) > 0:
                        all_down = False
                        break
                except:
                    pass
            if all_down:
                print(f" Done ({i+1}s)")
                break
            if i % 5 == 4:
                print('.', end='', flush=True)
        else:
            print(f" Timeout ({max_wait}s), proceeding anyway")
    
    cuda_id_offset = 0
    
    for j, vllm_server_args in enumerate(vllm_server_args_list):
        role = vllm_server_args["role"]
        url = vllm_server_args["url"]
        api_key = vllm_server_args["api_key"]
        tensor_parallel_size = vllm_server_args["tensor_parallel_size"]
        
        available_model_names = get_vllm_available_models(url, api_key)
        agent_config = agent_configs[role]
        session_name = f"{j}_vllm_server_{vllm_server_args['role']}"
        
        need_new_server = False
        
        if len(available_model_names) > 0 and not need_restart_all:
            if vllm_server_args["model_path"] not in available_model_names:
                print(f"⚠️ vLLM server at {url} has models {available_model_names}")
                print(f"   but we need {vllm_server_args['model_path']}")
                print(f"   Closing old vLLM server and starting new one...")
                close_vllm_session(session_name)
                
                print("   Waiting for old server to shut down...", end='', flush=True)
                max_wait = 30
                for i in range(max_wait):
                    time.sleep(1)
                    try:
                        check_models = get_vllm_available_models(url, api_key)
                        if len(check_models) == 0:
                            print(f" Done ({i+1}s)")
                            break
                    except:
                        print(f" Done ({i+1}s)")
                        break
                    if i % 5 == 4:
                        print('.', end='', flush=True)
                else:
                    print(f" Timeout ({max_wait}s), proceeding anyway")
                
                need_new_server = True
            else:
                print(f"✅ vLLM server at {url} already has model {vllm_server_args['model_path']}")
                wake_up_vllm(url, api_key)
                cuda_id_offset += tensor_parallel_size
                continue
        else:
            need_new_server = True
        
        if need_new_server:
            print(f"🚀 Starting new vLLM server for {role} at {url}")
            
            cuda_ids = ",".join(str(available_gpus[k + cuda_id_offset]) for k in range(tensor_parallel_size))
            print(f"🎯 CUDA_VISIBLE_DEVICES: {cuda_ids}")
            
            vllm_command = [
                f"CUDA_VISIBLE_DEVICES={cuda_ids}",
                f"VLLM_ALLOW_RUNTIME_LORA_UPDATING={use_lora}",
                f"VLLM_SERVER_DEV_MODE=1",
                "vllm", "serve", vllm_server_args["model_path"],
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--port", str(port_offset + j),
            ]
            vllm_command += agent_config.vllm.to_str_list()
            
            if use_lora and agent_config.lora.use_lora:
                lora_command = [
                    "--enable-lora",
                    "--max-lora-rank", str(agent_config.lora.r),
                ]
                
                if len(vllm_server_args["lora_module_names"]):
                    lora_command.append("--lora-modules")
                
                for lora_module in vllm_server_args["lora_module_names"]:
                    lora_command.append(f"{lora_module}={lora_module}")
                
                vllm_command += lora_command
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".sh") as tmp:
                tmp.write("#!/bin/bash\n")
                tmp.write(" ".join(vllm_command) + '\n')
                script_path = tmp.name
                os.chmod(script_path, 0o755)
            
            server = libtmux.Server()
            old_session = server.find_where({"session_name": session_name})
            if old_session:
                print(f"   Killing old session {session_name} to start fresh...")
                old_session.kill_session()
            
            log_dir = Path.home() / "vllm_logs"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{session_name}_{timestamp}.log"
            
            session = server.new_session(session_name=session_name)
            pane = session.attached_pane
            pane.send_keys(f"bash {script_path} 2>&1 | tee {log_file}")
        
        cuda_id_offset += tensor_parallel_size
    
    # Wait for vLLM servers and load LoRA modules
    for vllm_server_args in vllm_server_args_list:
        url = vllm_server_args["url"]
        api_key = vllm_server_args["api_key"]
        wait_for_vllm(url, api_key, timeout=7200)
        if use_lora:
            load_lora_modules(url, api_key, vllm_server_args["lora_module_names"])


def start_vllm(
    config: TrainAdversarialConfig,
    current_model_paths: Dict[str, str],
    on_policy_role: str,
    current_iter: int,
) -> Tuple[Dict[Literal["agent", "attacker"], Dict[str, str]], List[Dict[str, Any]]]:

    use_lora = config.agent.lora.use_lora
    agent_configs = {
        "agent": config.agent,
        "attacker": config.attacker,
    }

    if config.agent.lora.use_lora != config.attacker.lora.use_lora:
        raise NotImplementedError("Agent and attacker must have the same lora config")

    # ================ determine the agent and attacker model ================
    current_agent_policy_path = get_current_policy_path_by_role(config, current_model_paths, "agent")
    current_attacker_policy_path = get_current_policy_path_by_role(config, current_model_paths, "attacker")
    
    # Check whether models are multimodal and merge LoRA if needed.
    # vLLM doesn't support LoRA on vision towers, so we merge LoRA weights first
    is_multimodal = is_multimodal_model(config.agent.model_path)
    has_lora_adapters = False  # Track if we actually have LoRA adapters
    
    if is_multimodal and use_lora:
        print("🎨 Detected multimodal model with LoRA enabled")
        
        # Check if we have actual LoRA adapters (not just base model)
        agent_has_adapter = (Path(current_agent_policy_path) / "adapter_config.json").exists()
        attacker_has_adapter = (Path(current_attacker_policy_path) / "adapter_config.json").exists()
        
        if agent_has_adapter or attacker_has_adapter:
            print("   Merging LoRA weights for vLLM compatibility...")
            has_lora_adapters = True
            
            # Merge LoRA for current agent policy if it's a LoRA adapter
            if agent_has_adapter:
                current_agent_policy_path = merge_lora_weights_for_vllm(current_agent_policy_path)
                
            # Merge LoRA for current attacker policy if it's a LoRA adapter
            if attacker_has_adapter:
                current_attacker_policy_path = merge_lora_weights_for_vllm(current_attacker_policy_path)
            
            # For multimodal models, we can't use LoRA in vLLM, so disable it
            use_lora = False
            print("   ⚠️ LoRA disabled for vLLM (multimodal model uses merged weights)")
        else:
            print("   No LoRA adapters found yet (using base model directly)")
            use_lora = False
    
    existing_agent_model_paths = get_existing_model_paths_by_role(config, "agent", current_iter)
    existing_attacker_model_paths = get_existing_model_paths_by_role(config, "attacker", current_iter)
    
    # Note: Population-based training for multimodal models is handled through
    # sequential collection in train.py's _collect_episodes_for_population function.
    # This start_vllm function is used for iterative training (non-population) mode.
    # When population training is enabled, start_vllm_for_single_pair is used instead.

    if on_policy_role == "agent":
        agent_model_names = [current_agent_policy_path]
        if config.agent.train_use_population and len(existing_attacker_model_paths) > 0:
            attacker_model_names = existing_attacker_model_paths
        else:
            attacker_model_names = [current_attacker_policy_path]
    elif on_policy_role == "attacker":
        attacker_model_names = [current_attacker_policy_path]
        if config.attacker.train_use_population and len(existing_agent_model_paths) > 0:
            agent_model_names = existing_agent_model_paths
        else:
            agent_model_names = [current_agent_policy_path]
    elif on_policy_role == "both":
        agent_model_names = [current_agent_policy_path]
        attacker_model_names = [current_attacker_policy_path]
    else:
        raise ValueError(f"Unknown on-policy role: {on_policy_role}")

    # ================ determine the vLLM server configuration ================
    port_offset = 1234
    
    # Allocate GPUs for multimodal training.
    # For multimodal models with image attacks, reserve GPUs for image editor
    # GPU allocation (8 GPUs total):
    #   - GPUs 0,1: Attacker vLLM
    #   - GPUs 2,3,4,5: Qwen Image Edit (reserved for data collection)
    #   - GPUs 6,7: Agent vLLM
    # This prevents OOM by separating vLLM and image editor onto different GPUs
    
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        all_gpus = list(range(torch.cuda.device_count()))
    else:
        all_gpus = [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    
    # HTML injection mode - no GPU reservation needed for attacks
    # Both execute_text_injection and execute_image_attacks are supported
    # but HTML injection doesn't require separate GPU resources
    execute_image_attacks = getattr(config.data_collection, 'execute_image_attacks', False) if hasattr(config, 'data_collection') else False
    execute_text_injection = getattr(config.data_collection, 'execute_text_injection', False) if hasattr(config, 'data_collection') else False
    
    # Use all available GPUs for vLLM (no GPU reservation for HTML injection)
    available_gpus = all_gpus
    if is_multimodal:
        attack_modes = []
        if execute_text_injection:
            attack_modes.append("AXTree")
        if execute_image_attacks:
            attack_modes.append("Screenshot")
        attack_str = " + ".join(attack_modes) if attack_modes else "None"
        print(f"🎯 Multimodal mode (HTML injection affects: {attack_str}): Using all GPUs {available_gpus} for vLLM")
    
    num_available_gpus = len(available_gpus)

    if use_lora:
        use_same_base_model = config.agent.model_path == config.attacker.model_path

        if use_same_base_model:
            # 1 server, shared by agent and attacker
            assert_equal(asdict(config.agent.vllm), asdict(config.attacker.vllm))
            agent_config = config.agent

            tensor_parallel_size = get_tensor_parallel_size(config.agent.model_path, num_available_gpus)
            pipeline_parallel_size = num_available_gpus // tensor_parallel_size
            url = f"http://{agent_config.vllm.host}:{port_offset}"

            # for launching vLLM server
            vllm_server_args_list = [
                {
                    "role": "agent",
                    "model_path": config.agent.model_path,
                    "tensor_parallel_size": tensor_parallel_size,
                    "pipeline_parallel_size": pipeline_parallel_size,
                    "url": url,
                    "api_key": agent_config.vllm.api_key,
                    # use set to avoid duplicate LoRA modules
                    "lora_module_names": set(agent_model_names + attacker_model_names) - {agent_config.model_path},
                },
            ]

            # for creating agent's and attacker's vLLM models
            vllm_model_arguments = {
                "agent": {
                    "model_names": agent_model_names,
                    "urls": url,
                    "api_key": agent_config.vllm.api_key,
                },
            }
            
            # Only add attacker if actually being used
            if config.use_attacker:
                vllm_model_arguments["attacker"] = {
                    "model_names": attacker_model_names,
                    "urls": url,
                    "api_key": agent_config.vllm.api_key,
                }

        else:
            # 2 servers, one for agent and one for attacker
            vllm_server_args_list = []
            vllm_model_arguments = {}

            # Determine which roles need servers
            roles_to_start = ["agent"]
            if config.use_attacker:
                roles_to_start.append("attacker")
            
            # Calculate GPU allocation based on number of roles
            num_roles = len(roles_to_start)
            gpus_per_role = num_available_gpus // num_roles

            for j, role in enumerate(roles_to_start):
                agent_config = agent_configs[role]

                if role == "agent":
                    lora_modules = agent_model_names
                else:
                    lora_modules = attacker_model_names
                lora_modules = set(lora_modules) - {agent_config.model_path}

                # 🔧 Compute tensor_parallel_size per role (agent and attacker may have different model sizes)
                role_model_path = agent_config.model_path
                tensor_parallel_size = get_tensor_parallel_size(role_model_path, gpus_per_role)
                pipeline_parallel_size = gpus_per_role // tensor_parallel_size

                url = f"http://{agent_config.vllm.host}:{port_offset + j}"

                # for launching vLLM server
                vllm_server_args_list.append({
                    "role": role,
                    "model_path": agent_config.model_path,
                    "tensor_parallel_size": tensor_parallel_size,
                    "pipeline_parallel_size": pipeline_parallel_size,
                    "url": url,
                    "api_key": agent_config.vllm.api_key,
                    "lora_module_names": lora_modules,
                })

                # for creating agent's and attacker's vLLM models
                vllm_model_arguments[role] = {
                    "model_names": agent_model_names if role == "agent" else attacker_model_names,
                    "urls": url,
                    "api_key": agent_config.vllm.api_key,
                }
    else:
        # No LoRA mode - for multimodal base models or non-LoRA training
        # This handles the first iteration with base model for multimodal cases
        print("📦 Using direct model loading (no LoRA)")
        
        # Share a server only when the current agent and attacker checkpoints match.
        # Role-specific training can produce distinct checkpoints from the same base model.
        use_same_checkpoint = current_agent_policy_path == current_attacker_policy_path
        print(f"   Current agent path: {current_agent_policy_path}")
        print(f"   Current attacker path: {current_attacker_policy_path}")
        print(f"   Use same checkpoint: {use_same_checkpoint}")
        
        if use_same_checkpoint:
            # 1 server, shared by agent and attacker
            assert_equal(asdict(config.agent.vllm), asdict(config.attacker.vllm))
            agent_config = config.agent
            
            tensor_parallel_size = get_tensor_parallel_size(config.agent.model_path, num_available_gpus)
            pipeline_parallel_size = num_available_gpus // tensor_parallel_size
            url = f"http://{agent_config.vllm.host}:{port_offset}"
            
            # for launching vLLM server (load current model paths directly, no LoRA)
            vllm_server_args_list = [
                {
                    "role": "agent",
                    "model_path": current_agent_policy_path,  # Use current path (base model for iter 1)
                    "tensor_parallel_size": tensor_parallel_size,
                    "pipeline_parallel_size": pipeline_parallel_size,
                    "url": url,
                    "api_key": agent_config.vllm.api_key,
                    "lora_module_names": set(),  # No LoRA modules
                },
            ]
            
            # for creating agent's and attacker's vLLM models
            # 🎯 Support population-based training for multimodal models
            vllm_model_arguments = {
                "agent": {
                    "model_names": agent_model_names,  # Use all agent models (current + population)
                    "urls": url,
                    "api_key": agent_config.vllm.api_key,
                },
            }
            
            # Only add attacker if actually being used
            if config.use_attacker:
                vllm_model_arguments["attacker"] = {
                    "model_names": attacker_model_names,  # Use all attacker models (current + population)
                    "urls": url,
                    "api_key": agent_config.vllm.api_key,
                }
        else:
            # 2 servers, one for agent and one for attacker
            print("   Using separate servers for agent and attacker")
            vllm_server_args_list = []
            vllm_model_arguments = {}
            
            # Determine which roles need servers
            roles_to_start = ["agent"]
            if config.use_attacker:
                roles_to_start.append("attacker")
            
            # Calculate GPU allocation based on number of roles
            num_roles = len(roles_to_start)
            gpus_per_role = num_available_gpus // num_roles
            
            for j, role in enumerate(roles_to_start):
                agent_config = agent_configs[role]
                current_path = current_agent_policy_path if role == "agent" else current_attacker_policy_path
                
                # 🔧 Compute tensor_parallel_size per role (agent and attacker may have different model sizes)
                role_model_path = agent_config.model_path
                tensor_parallel_size = get_tensor_parallel_size(role_model_path, gpus_per_role)
                pipeline_parallel_size = gpus_per_role // tensor_parallel_size
                
                url = f"http://{agent_config.vllm.host}:{port_offset + j}"
                
                # for launching vLLM server
                vllm_server_args_list.append({
                    "role": role,
                    "model_path": current_path,  # Use current path directly
                    "tensor_parallel_size": tensor_parallel_size,
                    "pipeline_parallel_size": pipeline_parallel_size,
                    "url": url,
                    "api_key": agent_config.vllm.api_key,
                    "lora_module_names": set(),  # No LoRA modules
                })
                
                # for creating agent's and attacker's vLLM models
                # 🎯 Support population-based training for multimodal models
                role_model_names = agent_model_names if role == "agent" else attacker_model_names
                vllm_model_arguments[role] = {
                    "model_names": role_model_names,  # Use all models (current + population)
                    "urls": url,
                    "api_key": agent_config.vllm.api_key,
                }

    # ================ wake up existing vLLM servers or start new vLLM servers ================

    # CRITICAL: Check if existing server configuration matches what we need
    # We need to restart if:
    # 1. Number of servers changed (1 vs 2)
    # 2. tensor_parallel_size changed (e.g., 4 GPUs vs 2 GPUs)
    # 3. Model changed
    
    # Inspect both agent and attacker sessions, including sessions omitted by the requested configuration.
    # A transition from two servers to one still requires inspecting both running sessions.
    all_possible_servers = [
        {"session_name": "0_vllm_server_agent", "url": f"http://127.0.0.1:{port_offset}", "role": "agent"},
        {"session_name": "1_vllm_server_attacker", "url": f"http://127.0.0.1:{port_offset + 1}", "role": "attacker"},
    ]
    
    api_key = config.agent.vllm.api_key
    existing_servers_config = []
    for server_info in all_possible_servers:
        available_models = get_vllm_available_models(server_info["url"], api_key)
        if len(available_models) > 0:
            existing_servers_config.append({
                'session_name': server_info["session_name"],
                'url': server_info["url"],
                'role': server_info["role"],
                'has_server': True,
                'models': available_models,
            })
    
    # Check if we need to close ALL servers due to configuration mismatch
    # (e.g., going from 1 server with 4 GPUs to 2 servers with 2 GPUs each)
    need_restart_all = False
    
    # Count how many existing servers are running (only running servers are in the list)
    num_existing_servers = len(existing_servers_config)
    num_needed_servers = len(vllm_server_args_list)
    
    if num_existing_servers > 0 and num_existing_servers != num_needed_servers:
        print(f"⚠️ Server configuration mismatch:")
        print(f"   Existing: {num_existing_servers} server(s)")
        print(f"   Needed: {num_needed_servers} server(s)")
        print(f"   Closing all existing servers to reconfigure...")
        need_restart_all = True
        
        # Close all existing servers
        for server_config in existing_servers_config:
            print(f"   Closing {server_config['session_name']}...")
            close_vllm_session(server_config['session_name'])
        
        # Wait for all servers to shut down (check all possible servers)
        import time
        print("   Waiting for servers to shut down...", end='', flush=True)
        max_wait = 30
        for i in range(max_wait):
            time.sleep(1)
            all_down = True
            for server_info in all_possible_servers:
                try:
                    models = get_vllm_available_models(server_info["url"], api_key)
                    if len(models) > 0:
                        all_down = False
                        break
                except:
                    pass
            if all_down:
                print(f" Done ({i+1}s)")
                break
            if i % 5 == 4:
                print('.', end='', flush=True)
        else:
            print(f" Timeout ({max_wait}s), proceeding anyway")

    cuda_id_offset = 0

    # for each server
    for j, vllm_server_args in enumerate(vllm_server_args_list):
        role = vllm_server_args["role"]
        url = vllm_server_args["url"]
        api_key = vllm_server_args["api_key"]
        tensor_parallel_size = vllm_server_args["tensor_parallel_size"]

        available_model_names = get_vllm_available_models(url, api_key)
        agent_config = agent_configs[role]
        session_name = f"{j}_vllm_server_{vllm_server_args['role']}"
        
        need_new_server = False

        if len(available_model_names) > 0 and not need_restart_all:
            # vLLM server is already running
            if vllm_server_args["model_path"] not in available_model_names:
                # Model mismatch: need to restart server
                print(f"⚠️ vLLM server at {url} has models {available_model_names}")
                print(f"   but we need {vllm_server_args['model_path']}")
                print(f"   Closing old vLLM server and starting new one...")
                close_vllm_session(session_name)
                
                # Wait for the old server to fully shut down
                import time
                print("   Waiting for old server to shut down...", end='', flush=True)
                max_wait = 30  # Maximum 30 seconds
                for i in range(max_wait):
                    time.sleep(1)
                    try:
                        # Check if server is still responding
                        check_models = get_vllm_available_models(url, api_key)
                        if len(check_models) == 0:
                            print(f" Done ({i+1}s)")
                            break
                    except:
                        # Server is down
                        print(f" Done ({i+1}s)")
                        break
                    if i % 5 == 4:
                        print('.', end='', flush=True)
                else:
                    print(f" Timeout ({max_wait}s), proceeding anyway")
                
                need_new_server = True
            else:
                # Model is already loaded, just wake up the server
                print(f"✅ vLLM server at {url} already has model {vllm_server_args['model_path']}")
                wake_up_vllm(url, api_key)
                # Update cuda_id_offset for next server
                cuda_id_offset += tensor_parallel_size
                continue
        else:
            # vLLM server is not running (or we restarted all servers)
            need_new_server = True

        # Start new vLLM server if needed
        if need_new_server:
            print(f"🚀 Starting new vLLM server for {role} at {url}")

            cuda_ids = ",".join(str(available_gpus[k + cuda_id_offset]) for k in range(tensor_parallel_size))
            print(f"🎯 CUDA_VISIBLE_DEVICES: {cuda_ids}")

            vllm_command = [
                f"CUDA_VISIBLE_DEVICES={cuda_ids}",
                f"VLLM_ALLOW_RUNTIME_LORA_UPDATING={use_lora}",
                f"VLLM_SERVER_DEV_MODE=1",
                "vllm", "serve", vllm_server_args["model_path"],
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--port", str(port_offset + j),
            ]
            vllm_command += agent_config.vllm.to_str_list()

            if agent_config.lora.use_lora:
                lora_command = [
                    "--enable-lora",
                    "--max-lora-rank", str(agent_config.lora.r),
                ]

                if len(vllm_server_args["lora_module_names"]):
                    lora_command.append("--lora-modules")

                for lora_module in vllm_server_args["lora_module_names"]:
                    lora_command.append(f"{lora_module}={lora_module}")

                vllm_command += lora_command

            # write vllm_command to a file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".sh") as tmp:
                tmp.write("#!/bin/bash\n")
                tmp.write(" ".join(vllm_command) + '\n')
                script_path = tmp.name
                os.chmod(script_path, 0o755)

            # CRITICAL: Kill any existing session with the same name to start fresh
            # This prevents reusing old, potentially corrupted sessions
            server = libtmux.Server()
            old_session = server.find_where({"session_name": session_name})
            if old_session:
                print(f"   Killing old session {session_name} to start fresh...")
                old_session.kill_session()
            
            # Create new session with logging to file
            # Save all vLLM output to a persistent log file.
            log_dir = Path.home() / "vllm_logs"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{session_name}_{timestamp}.log"
            
            session = server.new_session(session_name=session_name)
            pane = session.attached_pane
            # Redirect stdout and stderr to log file while still showing in tmux
            pane.send_keys(f"bash {script_path} 2>&1 | tee {log_file}")

        cuda_id_offset += tensor_parallel_size

    # Wait for vLLM server to become ready and load LoRA modules
    for vllm_server_args in vllm_server_args_list:
        url = vllm_server_args["url"]
        api_key = vllm_server_args["api_key"]
        wait_for_vllm(url, api_key, timeout=7200)
        load_lora_modules(url, api_key, vllm_server_args["lora_module_names"])

    return vllm_model_arguments, vllm_server_args_list
