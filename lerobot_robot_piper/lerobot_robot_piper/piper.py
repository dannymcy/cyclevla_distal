import os
import time
from typing import Any

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from piper_sdk import C_PiperInterface_V2

from .config_piper import PiperConfig

POSTPROCESSOR_STATS_FILENAME = (
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
)


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        # Only open CAN interfaces for the active side(s); the idle set is never
        # connected, so connect()'s EnablePiper() loop won't hang on a missing arm.
        interface_by_side = {
            "left": self.config.can_interface_left,
            "right": self.config.can_interface_right,
        }
        self.arms = {
            side: C_PiperInterface_V2(interface_by_side[side])
            for side in self.config.sides
        }
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_piper_connected = False
        self.action_bias = load_action_bias(config.action_bias_path)
        self.action_clip = (
            load_action_clip_stats(list(self.action_features.keys()))
            if config.clip_action
            else None
        )
        self.prev_action: dict[str, float] | None = None
        # Live subtask index, advanced by the operator pressing 'y' during teleop
        # (wired in distal/hardware/record.py). Recorded per frame as the
        # `subtask_index` observation so the offline CycleVLA convert can segment
        # subtasks and assign per-subtask language + stop/progress signals. Plain
        # int increment is GIL-safe across the keyboard-listener thread.
        self.current_subtask = 0
        # Set of subtask indices that actually received >=1 frame this episode.
        # record.py validates this against the task's expected subtask count to
        # DISCARD episodes where the operator pressed 'y' the wrong number of times
        # (under/over-press, or a 0-frame subtask from a rapid double-'y'). Only the
        # main loop writes it (in get_observation); the keyboard thread touches the
        # int, never the set.
        self.subtask_indices_seen: set[int] = set()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.cameras.items()}

    @property
    def _eef_ft(self) -> dict[str, type]:
        # 6-DOF end-effector pose per active side: [x, y, z, rx, ry, rz]. Recorded
        # ALONGSIDE the joints (we keep both) so the offline CycleVLA convert can
        # build the EEF state + delta-EEF action that the sim `pi05_libero_cyclevla`
        # config expects (its state is absolute EEF, its action is delta EEF).
        return {
            f"{side}_eef_{ax}.pos": float
            for side in self.arms
            for ax in ("x", "y", "z", "rx", "ry", "rz")
        }

    @property
    def observation_features(self) -> dict:
        # LeRobot concatenates all scalar-float observation features into a single
        # `observation.state` vector (in this insertion order); images stay
        # separate. Order: joints(6/side), gripper(1/side), EEF(6/side),
        # subtask_index(1). The convert indexes by the recorded `names`, not
        # position, so this order is for readability only.
        ft: dict = {**self._motors_ft}
        for side in self.arms:
            ft[f"{side}_gripper.pos"] = float
        ft.update(self._eef_ft)
        # Per-frame subtask index (operator presses 'y' to advance). Drives the
        # offline subtask segmentation + per-subtask language / stop-progress
        # signals; not used at inference.
        ft["subtask_index"] = float
        ft.update(self._cameras_ft)
        return ft

    @property
    def action_features(self) -> dict:
        ft = {f"{name}.pos": float for name in self.config.joint_names}
        for side in self.arms:
            ft[f"{side}_gripper.pos"] = float
        return ft

    @property
    def is_connected(self) -> bool:
        return self._is_piper_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        for arm in self.arms.values():
            arm.ConnectPort()
        time.sleep(0.1)

        for arm in self.arms.values():
            while not arm.EnablePiper():
                time.sleep(0.01)

        self._is_piper_connected = True

        self.min_pos = []
        self.max_pos = []
        for arm in self.arms.values():
            limits = arm.GetAllMotorAngleLimitMaxSpd()
            # Convert from 0.1 deg to deg
            self.min_pos.extend(
                [
                    pos.min_angle_limit / 10.0
                    for pos in limits.all_motor_angle_limit_max_spd.motor[1:7]
                ]
                + [0.0]
            )
            self.max_pos.extend(
                [
                    pos.max_angle_limit / 10.0
                    for pos in limits.all_motor_angle_limit_max_spd.motor[1:7]
                ]
                + [10.0]
            )

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for side, arm in self.arms.items():
            js = arm.GetArmJointMsgs().joint_state
            g = arm.GetArmGripperMsgs()
            for i in range(1, 7):
                key = f"{side}_joint_{i}.pos"
                value = getattr(js, f"joint_{i}") / 1000.0
                if self.config.apply_bias_to_obs:
                    value -= self.action_bias.get(key, 0.0)
                obs[key] = value
            obs[f"{side}_gripper.pos"] = g.gripper_state.grippers_angle / 10000.0

            # End-effector pose feedback (CAN 0x152-0x154). SDK units are
            # 0.001 mm (X/Y/Z) and 0.001 deg (RX/RY/RZ); convert to SI (m, rad)
            # so the recorded EEF is physically meaningful and matches the inverse
            # conversion an EndPoseCtrl-based inference path would apply.
            ep = arm.GetArmEndPoseMsgs().end_pose
            obs[f"{side}_eef_x.pos"] = ep.X_axis * 1e-6
            obs[f"{side}_eef_y.pos"] = ep.Y_axis * 1e-6
            obs[f"{side}_eef_z.pos"] = ep.Z_axis * 1e-6
            obs[f"{side}_eef_rx.pos"] = float(np.deg2rad(ep.RX_axis * 1e-3))
            obs[f"{side}_eef_ry.pos"] = float(np.deg2rad(ep.RY_axis * 1e-3))
            obs[f"{side}_eef_rz.pos"] = float(np.deg2rad(ep.RZ_axis * 1e-3))

        # Stamp the current subtask index onto every frame (see __init__) and record
        # that this index was actually observed (for record.py's per-episode count
        # validation).
        obs["subtask_index"] = float(self.current_subtask)
        self.subtask_indices_seen.add(self.current_subtask)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def bump_subtask(self) -> None:
        """Advance the live subtask index by one. Called when the operator presses
        'y' during teleop to mark the END of the current subtask; the next frames
        are then stamped with the incremented index. See `current_subtask`."""
        self.current_subtask += 1

    def reset_subtask(self) -> None:
        """Reset the subtask index to 0 and clear the seen-set. Called at the start
        of every episode so each episode's subtask numbering begins at the first
        subtask and the count validation only sees this episode's marks."""
        self.current_subtask = 0
        self.subtask_indices_seen = set()

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # In teleop mode, the hardware handles control - just return the action
        if not self.config.teleop_mode:
            if self.action_clip is not None:
                action = {
                    k: max(self.action_clip[k][0], min(self.action_clip[k][1], v))
                    if k in self.action_clip
                    else v
                    for k, v in action.items()
                }

            alpha = self.config.action_ema_alpha
            if alpha is not None:
                if self.prev_action is None:
                    smoothed = dict(action)
                else:
                    smoothed = {
                        k: alpha * action[k] + (1.0 - alpha) * self.prev_action[k]
                        for k in action
                    }
                self.prev_action = smoothed
                action = smoothed

            for side, arm in self.arms.items():
                j_ints = [
                    int(
                        round(
                            (
                                action[f"{side}_joint_{i}.pos"]
                                - self.action_bias.get(f"{side}_joint_{i}.pos", 0.0)
                            )
                            * 1000.0
                        )
                    )
                    for i in range(1, 7)
                ]
                gripper_mm = int(round(action[f"{side}_gripper.pos"] * 10000.0))

                arm.JointCtrl(*j_ints)
                arm.GripperCtrl(gripper_mm, 1000, 0x01, 0)

        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        for arm in self.arms.values():
            arm.DisconnectPort()

        for cam in self.cameras.values():
            cam.disconnect()

        self.prev_action = None


def load_action_bias(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = np.load(path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    n_done = int(data["n_done"])
    diag_bias = (data["live_state"][:n_done] - data["recorded_state"][:n_done]).mean(
        axis=0
    )
    bias = {n: float(b) for n, b in zip(names, diag_bias) if "gripper" not in n}
    print(f"Piper: loaded action_bias from {path}")
    for n, b in bias.items():
        print(f"  {n:24s} bias={b:+.3f}")
    return bias


def load_action_clip_stats(action_keys: list[str]) -> dict[str, tuple[float, float]]:
    from lerobot.configs import parser
    from safetensors import safe_open

    policy_path = parser.get_path_arg("policy")
    if not policy_path:
        raise ValueError("clip_action=True requires --policy.path to be set")

    if os.path.isdir(policy_path):
        stats_file = os.path.join(policy_path, POSTPROCESSOR_STATS_FILENAME)
    else:
        from huggingface_hub import hf_hub_download

        stats_file = hf_hub_download(
            repo_id=policy_path, filename=POSTPROCESSOR_STATS_FILENAME
        )

    with safe_open(stats_file, framework="pt") as f:
        q01 = f.get_tensor("action.q01").tolist()
        q99 = f.get_tensor("action.q99").tolist()

    if len(q01) != len(action_keys):
        raise ValueError(
            f"Policy action dim ({len(q01)}) does not match "
            f"Piper action_features ({len(action_keys)} keys: {action_keys})"
        )

    clip = {k: (float(q01[i]), float(q99[i])) for i, k in enumerate(action_keys)}
    print(f"Piper: loaded action clip stats from {policy_path}")
    for k, (lo, hi) in clip.items():
        print(f"  {k:24s} [{lo:+.3f}, {hi:+.3f}]")
    return clip
