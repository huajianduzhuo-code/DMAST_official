from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class PostprocessRolloutArg:
    # if "imitation":
    #   remove failure episodes
    # if "rl_on_policy":
    #   remove off-policy samples
    # if "eval":
    #   keep all episodes
    mode: Literal["imitation", "rl_on_policy", "rl_off_policy", "eval"]
    roles: List[str] = field(
        metadata={
            "nargs": "+",
            "choices": ["agent", "attacker"],
            "help": "Which roles to postprocess rollout for (must be agent or attacker)",
        }
    )
    use_attacker: bool = True

    input_data_path: str = ""
    output_data_path: str = ""
