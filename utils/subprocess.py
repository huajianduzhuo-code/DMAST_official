from typing import Any, Dict, List, Optional

import os
import wandb
import subprocess

from utils.wandb import resume_wandb_from_env_vars


def run_commands(commands: List[Any], env_vars: Optional[Dict[str, str]]=None, description: Optional[str] = None):
    commands = list(map(str, commands))

    if env_vars is None:
        env_vars = {}

    for k, v in env_vars.items():
        if isinstance(v, bool):
            # True -> "true", False -> "false"
            env_vars[k] = str(v).lower()

        # add wandb env vars so resume_wandb_from_env_vars could retrieve them
        os.environ[k] = env_vars[k]

    # only one process can connect to the same wandb run
    # so we need to stop the connect from the main process
    if wandb.run is not None:
        wandb.finish()

    print(f"Running commands:")
    for cmd in commands:
        print(f"  {cmd}")

    subprocess.run(
        commands,
        check=True,
        env=os.environ | env_vars,
    )
    print(f"{description} finished")

    resume_wandb_from_env_vars(is_main_process=True)
