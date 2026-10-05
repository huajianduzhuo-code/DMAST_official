"""Validate SFT data provenance before filtering, mixing, or cache reuse."""

import json
from pathlib import Path

from configs.task_splits import validate_training_tasks


_RECORD_KEYS = ("golden_steps", "attacked_steps", "cot_steps", "samples", "attacker_steps")


def validate_training_records(records, *, context):
    """Reject malformed or held-out records without changing or filtering them.

    Empty record lists are valid: a collection stage may have no successes.
    Every nonempty record must retain its task name so its split can be checked.
    """
    if not isinstance(records, list):
        raise ValueError(f"{context}: expected a list of training records")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "task_name" not in record:
            raise ValueError(f"{context}: record {index} must be an object with task_name")
    if records:
        task_names = validate_training_tasks(
            [record["task_name"] for record in records], context=context
        )
        placeholders = sorted(set(task_names) & {"miniwob.unknown", "miniwob.placeholder"})
        if placeholders:
            raise ValueError(f"{context}: missing task identity: {', '.join(placeholders)}")


def validate_training_data_file(path):
    """Load and validate a known SFT JSON schema, returning the original data.

    Accept raw episode/sample lists or one recognized record list in a wrapper.
    Missing and ambiguous schemas fail closed instead of being treated as empty.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        keys = [key for key in _RECORD_KEYS if key in data]
        if len(keys) != 1:
            raise ValueError(
                f"{path}: expected exactly one record list from {_RECORD_KEYS}; found {keys}"
            )
        records = data[keys[0]]
    else:
        raise ValueError(f"{path}: expected a training record list or a known SFT wrapper")
    validate_training_records(records, context=str(path))
    return data
