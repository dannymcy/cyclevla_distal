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
  * Every subtask is phrased around "the gripper" — both the actuation steps
    ("close the gripper ...", "open the gripper ...") and the motion steps
    ("move the gripper above ..."). This matches sim, where all subtasks contain
    the word "gripper". The offline tail-oversampling in
    `convert_to_cyclevla.py` keys off this word, so with this wording EVERY
    subtask takes the gripper branch: repeat the last frame x8.

These three tasks (4 / 8 / 8 subtasks) are the real-robot tasks. The rest of the
pipeline is agnostic to the count and wording; lookup is keyed by the high-level
instruction (see below).
"""

# High-level instruction -> ordered subtask strings. The high-level key is what
# you put in `record.yaml single_task` while teleoperating that task.
TASKS: dict[str, list[str]] = {
    # 4 subtasks — single object placement.
    "hang the green teapot on the mug holder": [
        "move the gripper above the green teapot",
        "close the gripper to grasp the green teapot",
        "move the gripper toward the middle peg of the mug holder while holding the green teapot",
        "open the gripper to hang the green teapot on the mug holder",
    ],
    # 8 subtasks — sequential placement of two objects.
    "place the grape on the red plate and the apple on the green plate": [
        "move the gripper above the grape",
        "close the gripper to grasp the grape",
        "move the gripper above the red plate while holding the grape",
        "open the gripper to release the grape",
        "move the gripper above the apple",
        "close the gripper to grasp the apple",
        "move the gripper above the green plate while holding the apple",
        "open the gripper to release the apple",
    ],
    # 8 subtasks — sequential placement of two objects.
    "place the can of corn on the pan and the milk carton in the pot": [
        "move the gripper above the can of corn",
        "close the gripper to grasp the can of corn",
        "move the gripper above the pan while holding the can of corn",
        "open the gripper to release the can of corn",
        "move the gripper above the milk carton",
        "close the gripper to grasp the milk carton",
        "move the gripper above the pot while holding the milk carton",
        "open the gripper to release the milk carton",
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
