from functools import partial
from agentlab.agents import dynamic_prompting as dp

from agents.prompt_utils.agent_history import shrink_words

class AttackerHistoryStep(dp.Shrinkable):
    def __init__(
        self,
        previous_obs,
        current_obs,
        attacker_thought,
        attacker_action,
        attacker_memory,
        web_agent_thought,
        web_agent_action,
        web_agent_memory,
        flags: dp.ObsFlags,
        shrink_speed=0.5,
    ) -> None:
        super().__init__()

        self.previous_obs = previous_obs
        self.current_obs = current_obs
        self.attacker_thought = attacker_thought
        self.attacker_action = attacker_action
        self.attacker_memory = attacker_memory
        self.web_agent_thought = web_agent_thought
        self.web_agent_action = web_agent_action
        self.web_agent_memory = web_agent_memory
        self.flags = flags

        self.attacker_uses_web_agent_history = flags.attacker_privilege.attacker_uses_web_agent_history
        self.attacker_uses_web_agent_action_history = flags.attacker_privilege.attacker_uses_web_agent_action_history
        self.attacker_uses_web_agent_think_history = flags.attacker_privilege.attacker_uses_web_agent_think_history
        self.attacker_uses_web_agent_memory = flags.attacker_privilege.attacker_uses_web_agent_memory

        self.shrink_words = partial(shrink_words, shrink_speed=shrink_speed)
        self.deleted_attacker_thought_word_count = 0
        self.deleted_attacker_action_word_count = 0
        self.deleted_web_agent_thought_word_count = 0
        self.deleted_web_agent_action_word_count = 0

    @property
    def _prompt(self) -> str:
        prompt = ""

        if self.flags.use_think_history:
            prompt += f"\n<think>\n{self.attacker_thought}\n</think>\n"

        if self.flags.use_action_history:
            prompt += f"\n<action>\n{self.attacker_action}\n</action>\n"

        if self.attacker_memory is not None:
            prompt += f"\n<memory>\n{self.attacker_memory}\n</memory>\n"

        if self.attacker_uses_web_agent_history:
            if self.attacker_uses_web_agent_think_history:
                prompt += f"\n<web agent's think>\n{self.web_agent_thought}\n</web agent's think>\n"

            if self.attacker_uses_web_agent_action_history:
                prompt += f"\n<web agent's action>\n{self.web_agent_action}\n</web agent's action>\n"

            if self.attacker_uses_web_agent_memory:
                prompt += f"\n<web agent's memory>\n{self.web_agent_memory}\n</web agent's memory>\n"

        return prompt

    def shrink(self):
        if self.flags.use_think_history and self.attacker_thought is not None:
            self.attacker_thought, self.deleted_attacker_thought_word_count = self.shrink_words(self.attacker_thought, self.deleted_attacker_thought_word_count)
        if self.flags.use_action_history and self.attacker_action is not None:
            self.attacker_action, self.deleted_attacker_action_word_count = self.shrink_words(self.attacker_action, self.deleted_attacker_action_word_count)

        if self.attacker_uses_web_agent_history:
            if self.attacker_uses_web_agent_think_history and self.web_agent_thought is not None:
                self.web_agent_thought, self.deleted_web_agent_thought_word_count = self.shrink_words(self.web_agent_thought, self.deleted_web_agent_thought_word_count)
            if self.attacker_uses_web_agent_action_history and self.web_agent_action is not None:
                self.web_agent_action, self.deleted_web_agent_action_word_count = self.shrink_words(self.web_agent_action, self.deleted_web_agent_action_word_count)


class AttackerHistory(dp.History):
    def __init__(
        self,
        history_obs,
        attacker_thoughts,
        attacker_actions,
        attacker_memories,
        web_agent_thoughts,
        web_agent_actions,
        web_agent_memories,
        flags: dp.ObsFlags,
        shrink_speed=1,
    ) -> None:
        super().__init__(history_obs, attacker_actions, attacker_memories, attacker_thoughts, flags, shrink_speed)

        self.history_steps: list[AttackerHistoryStep] = []

        attacker_uses_web_agent_action_history = flags.attacker_privilege.attacker_uses_web_agent_action_history
        attacker_uses_web_agent_think_history = flags.attacker_privilege.attacker_uses_web_agent_think_history
        attacker_uses_web_agent_memory = flags.attacker_privilege.attacker_uses_web_agent_memory

        if attacker_uses_web_agent_action_history:
            assert len(web_agent_actions) == len(attacker_actions)

        if attacker_uses_web_agent_think_history:
            assert len(web_agent_thoughts) == len(attacker_thoughts)

        if attacker_uses_web_agent_memory:
            assert len(web_agent_memories) == len(attacker_memories)

        for i in range(1, len(history_obs)):
            self.history_steps.append(
                AttackerHistoryStep(
                    history_obs[i - 1],
                    history_obs[i],
                    attacker_thoughts[i - 1],
                    attacker_actions[i - 1],
                    attacker_memories[i - 1],
                    web_agent_thoughts[i - 1] if attacker_uses_web_agent_think_history else None,
                    web_agent_actions[i - 1] if attacker_uses_web_agent_action_history else None,
                    web_agent_memories[i - 1] if attacker_uses_web_agent_memory else None,
                    flags,
                )
            )
        self.shrink_calls = 0

    def shrink(self):
        if not self.history_steps:
            return

        # shrink the history step in order, from the oldest to the newest
        step_idx = self.shrink_calls % len(self.history_steps)
        self.history_steps[step_idx].shrink()
        self.shrink_calls += 1
