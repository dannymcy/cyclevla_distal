from dataclasses import dataclass, field

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_teleop")
@dataclass
class PiperTeleoperatorConfig(TeleoperatorConfig):
    can_interface_left: str = "can_arm_left"
    can_interface_right: str = "can_arm_right"
    # Active teleop arm set(s); must match the robot's `sides` so recorded actions
    # line up. ["right"] disables the left set, ["left"] disables the right set.
    sides: list[str] = field(default_factory=lambda: ["right"])
    # Derived from `sides` in __post_init__ when left empty.
    joint_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        invalid = [s for s in self.sides if s not in ("left", "right")]
        if invalid:
            raise ValueError(f"sides must be 'left'/'right', got: {invalid}")

        if not self.joint_names:
            self.joint_names = [
                f"{side}_joint_{i + 1}" for side in self.sides for i in range(6)
            ]
