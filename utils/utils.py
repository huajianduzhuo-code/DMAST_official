from typing import Any, Optional, List

import json
import logging

import torch

IGNORE_INDEX=-100


def print_separator(text, current_model_paths=None):
    print("-" * 50)
    print(text)
    if current_model_paths:
        for k, v in current_model_paths.items():
            print(f"{k}: {v}")

    print("-" * 50)


def read_json_or_jsonl_data(data_path):
    if data_path[-5:] == ".json":
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    else:
        with open(data_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]

    print(f">>> totally load {len(data_list)} data from {data_path}")
    return data_list


def setup_distributed_print(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def turn_off_deepspeed_info_logs():
    # turn off deepspeed INFO logs
    for name in logging.root.manager.loggerDict:
        if name.lower().startswith("deepspeed"):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING if logger.level <= logging.INFO else logger.level)


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def assert_equal(config1: Any, config2: Any, ignore_keys: Optional[List[str]] = None) -> None:
    """
    Recursively compare two data structures, handling nested dictionaries and lists.

    Prints detailed differences (types, keys, lengths, values) before raising an
    AssertionError when inequalities are found. Paths are dot-separated.

    Args:
        obj1: First data structure to compare
        obj2: Second data structure to compare
        ignore_keys: List of key paths to ignore during comparison, where keys are
            dot-separated paths (e.g., ["a.b.c"] ignores key "c" in path "a.b.c").
            If None, no keys are ignored.
    """
    if ignore_keys is None:
        ignore_keys = []

    differences: List[str] = []

    def should_ignore(path: str) -> bool:
        return any(path == ignore_path or path.startswith(ignore_path + ".") for ignore_path in ignore_keys)

    def get_path(current_path: str, key: str) -> str:
        assert "." not in key, "Key should not contain dot"
        return f"{current_path}.{key}" if current_path else key

    def compare(x1: Any, x2: Any, current_path: str = "") -> None:
        # Skip entire subtree if ignored
        if current_path and should_ignore(current_path):
            return

        if type(x1) != type(x2):
            # Allow tuple vs list mismatch (JSON converts tuples to lists)
            if isinstance(x1, (list, tuple)) and isinstance(x2, (list, tuple)):
                # Continue comparison as sequences
                pass
            else:
                differences.append(f"Type mismatch at {current_path}: {type(x1)} != {type(x2)}")
                return

        if isinstance(x1, dict):
            keys1 = set(x1.keys())
            keys2 = set(x2.keys())

            # Filter out ignored keys by full path
            keys1 = {k for k in keys1 if not should_ignore(get_path(current_path, k))}
            keys2 = {k for k in keys2 if not should_ignore(get_path(current_path, k))}

            missing_keys = keys1 - keys2
            unexpected_keys = keys2 - keys1
            if missing_keys or unexpected_keys:
                differences.append(
                    f"Keys mismatch at {current_path}: missing_keys={sorted(missing_keys)}, unexpected_keys={sorted(unexpected_keys)}"
                )

            for key in sorted(keys1 & keys2):
                new_path = get_path(current_path, key)
                compare(x1[key], x2[key], new_path)

        elif isinstance(x1, (list, tuple)):
            if len(x1) != len(x2):
                differences.append(f"Length mismatch at {current_path}: {len(x1)} != {len(x2)}")
                # Still compare up to the shortest length to surface inner diffs
            for i, (v1, v2) in enumerate(zip(x1, x2)):
                compare(v1, v2, get_path(current_path, str(i)))

        else:
            if x1 != x2 and not should_ignore(current_path):
                differences.append(f"Value mismatch at {current_path}: {x1} != {x2}")

    compare(config1, config2)
    if differences:
        header = [
            "Configurations are not equal.",
            ("Ignored keys: " + ", ".join(ignore_keys)) if ignore_keys else "Ignored keys: (none)",
            f"Total differences found: {len(differences)}",
            f"differences details: {differences}",
        ]
        # Print all differences (or could limit if desired)
        print("\n".join(header))
        for diff in differences:
            print(" - " + diff)
        raise AssertionError("Configuration mismatch. See differences printed above.")
