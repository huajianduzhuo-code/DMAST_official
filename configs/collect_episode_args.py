from typing import Any, Literal, List, Optional
from dataclasses import dataclass, field
import logging

from copy import deepcopy

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags

import bgym
from browsergym.experiments.loop import EnvArgs

from agents.llm.vllm_model import VllmModelArgs


@dataclass
class AttackerPrivilegeObsFlags(dp.Flags):
    attacker_uses_web_agent_history: bool = False
    attacker_uses_web_agent_action_history: bool = False
    attacker_uses_web_agent_think_history: bool = False
    attacker_uses_web_agent_memory: bool = False

    def __post_init__(self):
        if self.attacker_uses_web_agent_history:
            assert (
                self.attacker_uses_web_agent_action_history or
                self.attacker_uses_web_agent_think_history or
                self.attacker_uses_web_agent_memory
            ), "use at least one of web agent's history"

        if self.attacker_uses_web_agent_action_history:
            assert self.attacker_uses_web_agent_history

        if self.attacker_uses_web_agent_think_history:
            assert self.attacker_uses_web_agent_history

        if self.attacker_uses_web_agent_memory:
            assert self.attacker_uses_web_agent_history


@dataclass
class ObsFlags(dp.ObsFlags):
    attacker_privilege: AttackerPrivilegeObsFlags = field(init=False)


@dataclass
class VllmModelArgsList:
    model_names: List[str]
    base_urls: List[str] | str
    max_input_tokens: int = 4096


@dataclass
class CollectEpisodeArgs:
    seed: int = 0
    mode: Literal["train", "valid", "eval"] = "train"

    use_env_ids: bool = True
    benchmark: Literal["miniwob"] = field(
        default="miniwob",
    )
    env_ids: List[str] = field(
        default_factory=lambda: [],
    )

    train_prop: float = 0.8                             # proportion of env_ids for training
    valid_prop: float = 0.1                             # proportion of env_ids for validation
    eval_prop: float = 0.1                              # proportion of env_ids for evaluation

    num_envs_per_iter: int = 0                          # if 0, use all env_ids
    num_responses_per_env: int = 0                      # collect this many iterations of all envs in the benchmark

    # misc
    num_jobs: int = 128                                  # number of envs to run in parallel
    save_every_n_episodes: int = 128                    # save collected data every n episodes

    env_step_timeout: int = 30                          # timeout for environment.step in seconds
    job_timeout: int = 300                              # timeout for each episode collection job in seconds (0 = no timeout)
    fix_seed: bool = False                              # whether to fix the seed for environment.reset.
                                                        # If so, the environment will use the same observation for all episodes of the same task.
                                                        # This is useful for GRPO training.
    collect_responses_with_same_opponent: bool = False

    on_policy_role: Literal["agent", "attacker", "both"] = "agent"
    use_attacker: bool = False                          # whether to use attacker
    attacker_use_art: bool = False                      # whether to use ART in attacker
    
    # Skip model validation when using population-based collection (models will be set later)
    skip_model_validation: bool = False

    default_model_args: VllmModelArgs = field(
        default_factory=lambda: VllmModelArgs(
            model_name="placeholder",                   # to be overwritten by self.create_agent_args()
            base_url="placeholder",                     # to be overwritten by self.create_agent_args()
            temperature=1.0,
            max_new_tokens=4096,
            max_input_tokens=4096,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
        ),
    )

    agent_args_list: VllmModelArgsList = field(
        default_factory=lambda: VllmModelArgsList(
            model_names=[],
            base_urls=[],
        ),
    )
    attacker_args_list: VllmModelArgsList = field(
        default_factory=lambda: VllmModelArgsList(
            model_names=[],
            base_urls=[],
        ),
    )

    agent_args: List[VllmModelArgs] = field(init=False)
    attacker_args: List[VllmModelArgs] = field(init=False)

    attacker_privilege: AttackerPrivilegeObsFlags = field(
        default_factory=lambda: AttackerPrivilegeObsFlags(
            attacker_uses_web_agent_history=True,
            attacker_uses_web_agent_action_history=True,
            attacker_uses_web_agent_think_history=True,
            attacker_uses_web_agent_memory=False,
        ),
    )

    prompt_flags: GenericPromptFlags = field(
        default_factory=lambda: GenericPromptFlags(
            obs=ObsFlags(
                use_html=False,
                use_ax_tree=True,
                use_focused_element=True,
                use_error_logs=False,
                use_history=True,
                use_past_error_logs=False,
                use_action_history=True,
                use_think_history=True,
                use_diff=False,
                html_type="pruned_html",
                use_screenshot=True,  # Include screenshots in multimodal observations.
                use_som=False,
                extract_visible_tag=True,
                extract_clickable_tag=False,
                extract_coords="False",
                filter_visible_elements_only=False,
            ),
            action=dp.ActionFlags(
                action_set=bgym.HighLevelActionSetArgs(
                    subsets=("bid",),
                    multiaction=False,
                ),
                long_description=False,
                individual_examples=True,
                multi_actions=False,
                is_strict=False,
            ),
            use_plan=False,
            use_criticise=False,
            use_thinking=True,
            use_memory=False,
            use_concrete_example=True,
            use_abstract_example=True,
            use_hints=True,
            enable_chat=False,
            be_cautious=True,
            extra_instructions=None,
            add_missparsed_messages=True,
            max_trunc_itr=50,
        )
    )

    env_args: EnvArgs = field(
        default_factory=lambda: EnvArgs(
            task_name="placeholder",
            max_steps=2,  # Default step limit; collection entry points set their own limits.
        ),
    )

    output_path: str = "data/data.json"

    def create_agent_args(self, args_list: VllmModelArgsList):
        output_args = []

        assert isinstance(args_list.model_names, list)
        if isinstance(args_list.base_urls, str):
            args_list.base_urls = [args_list.base_urls] * len(args_list.model_names)
        elif isinstance(args_list.base_urls, list):
            assert len(args_list.base_urls) == len(args_list.model_names)
        else:
            raise ValueError(f"Invalid URLs: {args_list.base_urls}")

        for model_name, base_url in zip(args_list.model_names, args_list.base_urls):
            if not self.skip_model_validation:
                assert model_name != "placeholder", f"Model name cannot be 'placeholder' unless skip_model_validation=True"
            assert isinstance(base_url, str)

            model_args = deepcopy(self.default_model_args)
            model_args.model_name = model_name
            model_args.base_url = base_url
            model_args.max_input_tokens = args_list.max_input_tokens

            output_args.append(model_args)

        return output_args

    def __post_init__(self):
        assert self.num_responses_per_env > 0, "num_responses_per_env must be greater than 0"

        assert self.train_prop + self.valid_prop + self.eval_prop == 1.0, \
            "train_prop + valid_prop + eval_prop should sum to 1.0"

        # overwrite obs prompt flags with attacker privilege
        self.prompt_flags.obs.attacker_privilege = self.attacker_privilege

        # check if self_play is consistent with player_args.model_name and opponent_args.model_names
        if self.mode in ["train", "valid"]:
            if self.on_policy_role == "agent":
                assert len(self.agent_args_list.model_names) == 1, "agent_args_list should have a single model"
                if self.use_attacker:
                    assert len(self.attacker_args_list.model_names) >= 1, "attacker_args_list should have more than one model"
            elif self.on_policy_role == "attacker":
                assert len(self.agent_args_list.model_names) >= 1, "agent_args_list should have more than one model"
                if self.use_attacker:
                    assert len(self.attacker_args_list.model_names) == 1, "attacker_args_list should have a single model"
            elif self.on_policy_role == "both":
                assert len(self.agent_args_list.model_names) == 1, "agent_args_list should have a single model"
                if self.use_attacker:
                    assert len(self.attacker_args_list.model_names) == 1, "attacker_args_list should have a single model"
            else:
                raise ValueError(f"Invalid on_policy_role: {self.on_policy_role}")

        elif self.mode == "eval":
            # assert self.on_policy_role == "both", "when mode == 'eval', on_policy_role must be 'both'"
            assert len(self.agent_args_list.model_names) == 1, "agent_args_list should have a single model"
            if self.use_attacker:
                assert len(self.attacker_args_list.model_names) == 1, "attacker_args_list should have a single model"

        else:
            raise ValueError(f"Invalid mode: {self.mode}")                 

        self.agent_args = self.create_agent_args(self.agent_args_list)
        if self.use_attacker:
            self.attacker_args = self.create_agent_args(self.attacker_args_list)
        else:
            self.attacker_args = []


@dataclass
class MultimodalCollectEpisodeArgs(CollectEpisodeArgs):
    """Enhanced arguments for multimodal episode collection with HTML injection attacks.
    
    This version uses HTML injection instead of image editing for attacks.
    The attacker generates HTML/CSS that gets injected into the page via JavaScript,
    which automatically updates both the screenshot and accessibility tree.
    
    The two execution flags independently control what the agent sees:
    - execute_text_injection: If True, agent sees modified AXTree; if False, agent sees original
    - execute_image_attacks: If True, agent sees modified screenshot; if False, agent sees original
    """
    
    # Agent vision capabilities
    enable_agent_vision: bool = True
    max_image_size: tuple = (896, 896)  # Default resolution for VLM models (works for both Gemma and Qwen)
    
    # HTML injection attack configuration
    # These flags independently control what the agent sees after HTML injection
    execute_text_injection: bool = True   # If True, agent sees modified AXTree
    execute_image_attacks: bool = True    # If True, agent sees modified screenshot
    
    # Enhanced SFT pipeline option
    # When True, saves raw observation data (clean_obs, attacked_obs) in step_data
    # This significantly increases file size due to base64 screenshots
    # Only enable for enhanced SFT pipeline; keep False for RL training
    save_raw_observations: bool = False
    
    # Agent defense level: 0=basic, 1=enhanced (detailed attack pattern warnings), 2=maximum
    defense_level: int = 0
    # Optional system-prompt security addendum override for prompt comparisons.
    # When set, replaces the hardcoded `security_rules_addendum` regardless of defense_level.
    system_prompt_addendum_override: str | None = None
    #   "leakage"        — paper default; attacker tries to extract user PII
    #   "misinformation" — attacker plants a per-task wrong value from
    #                      configs/misinformation_wrong_values.py (Table 4).
    attack_goal: str = "leakage"

    def __post_init__(self):
        super().__post_init__()
        
        # Ensure vision support is enabled for VLM models when using vision
        if self.enable_agent_vision:
            logging.info(f"🔧 Setting vision_support=True for agent models (enable_agent_vision={self.enable_agent_vision})")
            for i, model_args in enumerate(self.agent_args):
                if hasattr(model_args, 'vision_support'):
                    old_value = model_args.vision_support
                    model_args.vision_support = True
                    logging.info(f"  Agent model {i}: vision_support changed from {old_value} to {model_args.vision_support}")
                else:
                    logging.warning(f"  Agent model {i}: No vision_support attribute found!")
