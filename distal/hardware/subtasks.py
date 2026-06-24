"""Placeholder task / subtask decomposition for the real-robot CycleVLA pipeline.

The sim CycleVLA dataset (`cyclevla/libero_decomposed_progress`) trains on
SUBTASK-level language: each LIBERO task is split into an ordered list of short
imperative subtasks, and every frame's `task` is the current subtask string (not
the high-level instruction). To match that schema for real-robot teleop without
touching the read-only `cyclevla_code` repo, we:

  1. record a per-frame `subtask_index` (operator presses 'y' to advance it,
     see `lerobot_robot_piper/piper.py`), and
  2. map `subtask_index -> subtask string` offline in
     `distal/hardware/convert_to_cyclevla.py` using the tables below.

Lookup is keyed by the episode's HIGH-LEVEL instruction — i.e. whatever
`record.yaml single_task` was set to while recording, which LeRobot stores as the
per-frame `task`. So you can record different tasks in different sessions (each
with its own `single_task`) and append them into one dataset; the convert maps
each episode correctly via its stored high-level instruction.

IMPORTANT conventions (mirroring the sim Stage-3 builder
`LIBERO_Decomposed_Progress_dataset_builder.py`):
  * Subtask strings are lowercase imperative phrases.
  * A subtask that actuates the gripper MUST contain the word "gripper"
    ("close the gripper ...", "open the gripper ..."). The offline tail-
    oversampling keys off this word: gripper subtasks repeat the last frame x8,
    non-gripper subtasks repeat the last 3 frames x4. Keep the wording when you
    revise these.

These three tasks (4 / 8 / 8 subtasks) are PLACEHOLDERS chosen only to exercise
the pipeline end-to-end and to cover the gripper / non-gripper oversample branches.
Replace the instructions and subtasks with the real ones before collecting data;
the rest of the pipeline is agnostic to the count and wording.
"""

# High-level instruction -> ordered subtask strings. The high-level key is what
# you put in `record.yaml single_task` while teleoperating that task.
TASKS: dict[str, list[str]] = {
    # 4 subtasks — simple reach / grasp / transport / release.
    "pick up the black bowl and place it on the plate": [
        "reach for the black bowl",
        "close the gripper to grasp the black bowl",
        "move the black bowl over the plate",
        "open the gripper to release the black bowl",
    ],
    # 8 subtasks — two-stage manipulation.
    "put the red mug on the shelf and open the cabinet drawer": [
        "reach for the red mug",
        "close the gripper to grasp the red mug",
        "lift the red mug off the table",
        "move the red mug onto the shelf",
        "open the gripper to release the red mug",
        "reach for the cabinet drawer handle",
        "close the gripper to grasp the cabinet drawer handle",
        "pull the cabinet drawer open",
    ],
    # 8 subtasks — sequential placement of two objects.
    "put the bread on the plate and move the butter knife beside it": [
        "reach for the bread",
        "close the gripper to grasp the bread",
        "lift the bread off the table",
        "move the bread over the plate",
        "open the gripper to release the bread onto the plate",
        "reach for the butter knife",
        "close the gripper to grasp the butter knife",
        "move the butter knife beside the plate",
    ],
}


def normalize(instruction: str) -> str:
    """Normalize a high-level instruction for case/whitespace-insensitive lookup."""
    return " ".join(instruction.strip().lower().split())


# Lowercased lookup so a `single_task` with different casing/spacing still matches.
TASKS_NORMALIZED: dict[str, list[str]] = {
    normalize(instr): subtasks for instr, subtasks in TASKS.items()
}


def get_subtasks(instruction: str) -> list[str]:
    """Return the ordered subtask list for a high-level instruction.

    Raises KeyError (listing the known instructions) if the instruction is not a
    registered placeholder task — the convert needs an explicit mapping to assign
    per-subtask language, so silently falling back would corrupt the dataset.
    """
    key = normalize(instruction)
    if key not in TASKS_NORMALIZED:
        known = "\n  - ".join(TASKS.keys())
        raise KeyError(
            f"No subtask decomposition registered for instruction "
            f"{instruction!r}. Set record.yaml single_task to one of:\n  - {known}\n"
            f"or add it to distal/hardware/subtasks.py."
        )
    return TASKS_NORMALIZED[key]


def subtask_for_index(instruction: str, subtask_index: int) -> str:
    """Map a (high-level instruction, subtask_index) to its subtask string.

    `subtask_index` is clamped to the last subtask: if the operator pressed 'y'
    more times than there are subtasks (e.g. an extra press after the final
    subtask), those trailing frames are attributed to the last subtask rather
    than crashing. The convert logs when clamping happens.
    """
    subtasks = get_subtasks(instruction)
    idx = max(0, min(int(subtask_index), len(subtasks) - 1))
    return subtasks[idx]
