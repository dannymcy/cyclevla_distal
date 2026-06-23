from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface_left: str = "can_arm_left"
    can_interface_right: str = "can_arm_right"
    # Active arm set(s). The rig has two PiPER sets (one per side); list only the
    # side(s) to drive. Set to ["right"] to disable the left set, ["left"] to
    # disable the right set, or ["left", "right"] for both. Everything downstream
    # (CAN interfaces opened, joint_names, cameras) is derived from this so the
    # idle set is simply never connected or commanded.
    sides: list[str] = field(default_factory=lambda: ["right"])
    # Derived from `sides` in __post_init__ when left empty. Provide explicitly
    # only to override the default per-side joint naming.
    joint_names: list[str] = field(default_factory=list)
    # Per-side, per-joint home target for distal/hardware/zero.py, in milli-degrees
    # (0.001 deg), fed straight into JointCtrl with no conversion. Each side's [0]*6
    # keeps the all-zeros home. Joints only — gripper homing is the home_gripper knob
    # below. Capture a hand-set pose with read_pose.py.
    home_pose: dict[str, list[int]] = field(
        default_factory=lambda: {"left": [0] * 6, "right": [0] * 6}
    )
    # Per-side gripper home state for zero.py: "open" (full open) or "closed".
    home_gripper: dict[str, str] = field(
        default_factory=lambda: {"left": "open", "right": "open"}
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left_wrist": RealSenseCameraConfig(
                serial_number_or_name="335122272969",
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist": RealSenseCameraConfig(
                serial_number_or_name="123622270993",
                fps=30,
                width=640,
                height=480,
            ),
            "top": RealSenseCameraConfig(
                serial_number_or_name="323622271046",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    teleop_mode: bool = True
    action_bias_path: str | None = None
    apply_bias_to_obs: bool = False
    # EMA smoothing on outgoing actions: smoothed = alpha * new + (1 - alpha) * prev.
    # None disables smoothing. Lower alpha = more smoothing. Only applied in policy
    # rollout (teleop_mode=False).
    action_ema_alpha: float | None = None
    # Clip outgoing actions to the [q01, q99] range from the current policy's
    # postprocessor unnormalizer safetensors. Only applied in policy rollout
    # (teleop_mode=False). The policy path is resolved from --policy.path.
    clip_action: bool = False

    def __post_init__(self):
        invalid = [s for s in self.sides if s not in ("left", "right")]
        if invalid:
            raise ValueError(f"sides must be 'left'/'right', got: {invalid}")

        # Validate here so a bad record.yaml fails fast when lerobot-record loads
        # the config, even though the robot itself never consumes home_pose.
        for side, pose in self.home_pose.items():
            if side not in ("left", "right"):
                raise ValueError(f"home_pose keys must be 'left'/'right', got: {side}")
            if len(pose) != 6 or not all(isinstance(v, int) for v in pose):
                raise ValueError(
                    f"home_pose[{side}] must be 6 integer milli-degree values, "
                    f"got: {pose}"
                )

        for side, state in self.home_gripper.items():
            if side not in ("left", "right"):
                raise ValueError(
                    f"home_gripper keys must be 'left'/'right', got: {side}"
                )
            if state not in ("open", "closed"):
                raise ValueError(
                    f"home_gripper[{side}] must be 'open'/'closed', got: {state}"
                )

        # Derive joint names from the active sides unless explicitly provided.
        if not self.joint_names:
            self.joint_names = [
                f"{side}_joint_{i + 1}" for side in self.sides for i in range(6)
            ]

        # Drop wrist cameras belonging to inactive sides; keep shared cameras
        # (e.g. "top"). A camera is side-specific iff its key is "{side}_wrist".
        self.cameras = {
            key: cam
            for key, cam in self.cameras.items()
            if not (key.endswith("_wrist") and key[: -len("_wrist")] not in self.sides)
        }

        # Base RobotConfig validates camera width/height/fps; run after filtering.
        super().__post_init__()
