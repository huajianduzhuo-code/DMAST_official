"""Canonical MiniWob++ task splits shared by training data collectors.

Importing this module requires only the Python standard library. BrowserGym and
Gymnasium are imported lazily only when selecting all registered training tasks.
"""

import re
from typing import Iterable, List


MINIWOB_TRAIN_TASKS = (
    "miniwob.buy-ticket",
    "miniwob.circle-center",
    "miniwob.click-button",
    "miniwob.click-checkboxes-soft",
    "miniwob.click-collapsible",
    "miniwob.click-collapsible-nodelay",
    "miniwob.click-menu",
    "miniwob.click-menu-2",
    "miniwob.click-tab",
    "miniwob.click-tab-2-medium",
    "miniwob.copy-paste",
    "miniwob.count-sides",
    "miniwob.drag-items",
    "miniwob.drag-items-grid",
    "miniwob.drag-single-shape",
    "miniwob.drag-sort-numbers",
    "miniwob.draw-line",
    "miniwob.email-inbox",
    "miniwob.email-inbox-delete",
    "miniwob.enter-text",
    "miniwob.enter-text-dynamic",
    "miniwob.find-midpoint",
    "miniwob.find-word",
    "miniwob.generate-number",
    "miniwob.identify-shape",
    "miniwob.multi-layouts",
    "miniwob.multi-orderings",
    "miniwob.navigate-tree",
    "miniwob.odd-or-even",
    "miniwob.phone-book",
    "miniwob.read-table",
    "miniwob.read-table-2",
    "miniwob.sign-agreement",
    "miniwob.simple-algebra",
    "miniwob.simple-arithmetic",
    "miniwob.social-media",
    "miniwob.use-autocomplete",
    "miniwob.use-autocomplete-nodelay",
    "miniwob.use-colorwheel",
    "miniwob.use-colorwheel-2",
    "miniwob.use-spinner",
    "miniwob.visual-addition",
)

MINIWOB_VALID_TASKS = (
    "miniwob.click-option",
    "miniwob.click-checkboxes",
    "miniwob.focus-text",
    "miniwob.click-widget",
    "miniwob.click-scroll-list",
    "miniwob.login-user-popup",
    "miniwob.click-test-2",
    "miniwob.count-shape",
    "miniwob.form-sequence-2",
    "miniwob.click-checkboxes-transfer",
    "miniwob.click-test",
    "miniwob.click-dialog",
)

MINIWOB_EVAL_TASKS = (
    "miniwob.enter-password",
    "miniwob.copy-paste-2",
    "miniwob.login-user",
    "miniwob.click-test-transfer",
    "miniwob.bisect-angle",
    "miniwob.form-sequence-3",
    "miniwob.click-dialog-2",
    "miniwob.enter-text-2",
    "miniwob.unicode-test",
    "miniwob.scroll-text",
    "miniwob.choose-list",
    "miniwob.click-button-sequence",
    "miniwob.email-inbox-nl-turk",
    "miniwob.guess-number",
    "miniwob.focus-text-2",
    "miniwob.email-inbox-noscroll",
)

MINIWOB_HELD_OUT_TASKS = frozenset(MINIWOB_VALID_TASKS + MINIWOB_EVAL_TASKS)


# Evaluation-only task list for the 56-task attack-modality study.
# This list includes held-out tasks and must never select training data.
MINIWOB_ATTACK_STUDY_TASKS = (
    "miniwob.login-user",
    "miniwob.click-collapsible-nodelay",
    "miniwob.drag-sort-numbers",
    "miniwob.use-spinner",
    "miniwob.click-menu-2",
    "miniwob.click-test-2",
    "miniwob.generate-number",
    "miniwob.click-option",
    "miniwob.count-sides",
    "miniwob.visual-addition",
    "miniwob.navigate-tree",
    "miniwob.buy-ticket",
    "miniwob.circle-center",
    "miniwob.click-button",
    "miniwob.click-button-sequence",
    "miniwob.click-checkboxes",
    "miniwob.click-checkboxes-soft",
    "miniwob.click-collapsible",
    "miniwob.click-dialog",
    "miniwob.click-menu",
    "miniwob.click-tab",
    "miniwob.click-tab-2-medium",
    "miniwob.click-test-transfer",
    "miniwob.click-widget",
    "miniwob.copy-paste-2",
    "miniwob.count-shape",
    "miniwob.drag-shapes",
    "miniwob.draw-line",
    "miniwob.email-inbox",
    "miniwob.email-inbox-important",
    "miniwob.enter-password",
    "miniwob.enter-text",
    "miniwob.enter-text-2",
    "miniwob.enter-text-dynamic",
    "miniwob.focus-text-2",
    "miniwob.form-sequence-2",
    "miniwob.form-sequence-3",
    "miniwob.login-user-popup",
    "miniwob.multi-layouts",
    "miniwob.read-table",
    "miniwob.read-table-2",
    "miniwob.scroll-text",
    "miniwob.simple-algebra",
    "miniwob.stock-market",
    "miniwob.unicode-test",
    "miniwob.use-autocomplete",
    "miniwob.click-dialog-2",
    "miniwob.click-scroll-list",
    "miniwob.focus-text",
    "miniwob.right-angle",
    "miniwob.simple-arithmetic",
    "miniwob.text-transform",
    "miniwob.email-inbox-noscroll",
    "miniwob.phone-book",
    "miniwob.sign-agreement",
    "miniwob.use-colorwheel-2",
)


def normalize_task_name(task_name: str) -> str:
    """Return ``miniwob.<task>`` for a bare or BrowserGym-qualified task name.

    This checks the name's format, not whether an environment is installed.
    """
    if not isinstance(task_name, str):
        raise ValueError(f"Expected a MiniWob task name string, got {task_name!r}")
    name = task_name.strip()
    if name.startswith("browsergym/"):
        name = name[len("browsergym/"):]
    if name.startswith("miniwob."):
        name = name[len("miniwob."):]
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(f"Invalid MiniWob task name: {task_name!r}")
    return f"miniwob.{name}"


def validate_training_tasks(
    task_names: Iterable[str], *, context: str = "training"
) -> List[str]:
    """Normalize a nonempty task list and reject any canonical held-out task.

    Return a new list preserving input order and repetitions. Registry membership
    is not required, allowing this validation before browser dependencies load.
    """
    if isinstance(task_names, (str, bytes)):
        raise ValueError(f"{context}: expected a collection of task names, not one string")
    try:
        tasks = [normalize_task_name(name) for name in task_names]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid training task list: {exc}") from exc
    if not tasks:
        raise ValueError(f"{context}: training task list must not be empty")
    overlap = sorted(set(tasks) & MINIWOB_HELD_OUT_TASKS)
    if overlap:
        raise ValueError(f"{context}: held-out tasks cannot be used for training: {', '.join(overlap)}")
    return tasks


def select_sft_tasks(
    num_tasks: int = len(MINIWOB_TRAIN_TASKS), all_tasks: bool = False
) -> List[str]:
    """Select a fixed-prefix SFT subset, or every registered non-held-out task.

    ``all_tasks=True`` ignores ``num_tasks`` and returns the actual registered
    count after normalization, deduplication and held-out exclusion. No browser
    environment is created or executed.
    """
    if all_tasks:
        import bgym  # noqa: F401 -- registers BrowserGym environments
        import gymnasium as gym

        tasks = sorted({
            normalize_task_name(env_id)
            for env_id in gym.envs.registry
            if isinstance(env_id, str) and env_id.startswith("browsergym/miniwob.")
        } - MINIWOB_HELD_OUT_TASKS)
        return validate_training_tasks(tasks, context="SFT registered tasks")

    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int):
        raise ValueError("SFT num_tasks must be an integer")
    if not 1 <= num_tasks <= len(MINIWOB_TRAIN_TASKS):
        raise ValueError(f"SFT num_tasks must be between 1 and {len(MINIWOB_TRAIN_TASKS)}")
    return validate_training_tasks(MINIWOB_TRAIN_TASKS[:num_tasks], context="SFT tasks")


def select_attack_study_tasks(
    num_tasks: int = len(MINIWOB_ATTACK_STUDY_TASKS), all_tasks: bool = False
) -> List[str]:
    """Select evaluation tasks, retaining held-out tasks and registry order.

    This selector is exclusively for evaluation, never for SFT or RL training.
    ``all_tasks=True`` ignores ``num_tasks`` and includes all registered MiniWob
    tasks, deduplicated after normalization without creating an environment.
    """
    if all_tasks:
        import bgym  # noqa: F401 -- registers BrowserGym environments
        import gymnasium as gym

        tasks = list(dict.fromkeys(
            normalize_task_name(env_id)
            for env_id in gym.envs.registry
            if isinstance(env_id, str) and env_id.startswith("browsergym/miniwob.")
        ))
        if not tasks:
            raise ValueError("Attack study: no registered MiniWob tasks found")
        return tasks

    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int):
        raise ValueError("Attack-study num_tasks must be an integer")
    if not 1 <= num_tasks <= len(MINIWOB_ATTACK_STUDY_TASKS):
        raise ValueError(
            f"Attack-study num_tasks must be between 1 and {len(MINIWOB_ATTACK_STUDY_TASKS)}"
        )
    return list(MINIWOB_ATTACK_STUDY_TASKS[:num_tasks])
