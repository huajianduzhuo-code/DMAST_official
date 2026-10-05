from typing import Dict, Any

import os
import wandb


def retrieve_existing_run_id(project_name, group, run_name):
    """
    Checks for an interrupted run with the given project and name, and resumes it if found.
    Otherwise, creates a new run.
    """

    api = wandb.Api()

    # check if project_name exists
    projects = api.projects()
    if project_name not in [p.name for p in projects]:
        return None

    runs = api.runs(project_name)
    previous_run_id = None

    for run in runs:
        if run.group == group and run.name == run_name:
            previous_run_id = run.id
            break

    return previous_run_id


def resume_wandb_from_env_vars(is_main_process):
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    WANDB_GROUP = os.getenv("WANDB_GROUP")
    WANDB_NAME = os.getenv("WANDB_NAME")
    WANDB_RESUME = os.getenv("WANDB_RESUME")
    WANDB_RUN_ID = os.getenv("WANDB_RUN_ID")
    WANDB_DISABLED = os.getenv("WANDB_DISABLED", "true")

    assert WANDB_DISABLED in ["true", "false"], f"WANDB_DISABLED must be 'true' or 'false', got {WANDB_DISABLED}"
    WANDB_DISABLED = WANDB_DISABLED == "true"

    if (
        is_main_process
        and not WANDB_DISABLED
        and WANDB_RESUME == "must"
        and WANDB_RUN_ID is not None
    ):
        wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=WANDB_NAME,
            resume=True,
            id=WANDB_RUN_ID,
        )

def wandb_log(logs: Dict[str, Any], step_metric: str, step_value: int, prefix: str=None):
    if wandb.run is not None:
        wandb.define_metric(step_metric)

        logs = {f"{prefix}/{k}" if prefix else k: v for k, v in logs.items()}

        for k in logs.keys():
            wandb.define_metric(k, step_metric=step_metric)

        wandb.log({**logs, step_metric: step_value})
