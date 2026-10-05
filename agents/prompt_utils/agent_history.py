import re
from functools import partial
from agentlab.agents import dynamic_prompting as dp


def shrink_words(text, deleted_word_count, shrink_speed):
    # if the text is shrinked before, it will end with "... Deleted X words to reduce prompt size."
    # we need to remove this part before shrinking
    parts = re.split(r"\n\.\.\. Deleted \d+ words to reduce prompt size\.$", text, 1)
    text = parts[0]

    words = text.split(" ")
    if len(words) <= 5:
        return text, deleted_word_count

    new_words_count = int(len(words) * (1 - shrink_speed))
    deleted_word_count += len(words) - new_words_count
    text = " ".join(words[:new_words_count])
    text += f"\n... Deleted {deleted_word_count} words to reduce prompt size."
    return text, deleted_word_count


class HistoryStep(dp.HistoryStep):
    def __init__(
        self, previous_obs, current_obs, action, memory, thought, flags: dp.ObsFlags, shrink_speed=0.5,
    ) -> None:
        super().__init__(previous_obs, current_obs, action, memory, thought, flags)

        self.deleted_thought_word_count = 0
        self.deleted_action_word_count = 0
        self.shrink_words = partial(shrink_words, shrink_speed=shrink_speed)

    def shrink(self):
        self.thought, self.deleted_thought_word_count = self.shrink_words(self.thought, self.deleted_thought_word_count)
        self.action, self.deleted_action_word_count = self.shrink_words(self.action, self.deleted_action_word_count)


class History(dp.History):
    def __init__(
        self, history_obs, actions, memories, thoughts, flags: dp.ObsFlags, shrink_speed=1
    ) -> None:
        super().__init__(history_obs, actions, memories, thoughts, flags, shrink_speed)

        self.history_steps: list[HistoryStep] = []

        for i in range(1, len(history_obs)):
            self.history_steps.append(
                HistoryStep(
                    history_obs[i - 1],
                    history_obs[i],
                    actions[i - 1],
                    memories[i - 1],
                    thoughts[i - 1],
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
