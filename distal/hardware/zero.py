import argparse
import time

import yaml
from piper_sdk import C_PiperInterface_V2

# Gripper angle in 0.001 mm units. 70 mm is the Piper gripper's full open; 0 closed.
OPEN_POSITION = 70_000
CLOSED_POSITION = 0

# Streamed control parameters for the "normal" (inference / CAN-control) homing
# path. The Piper firmware needs the 0x151 motion-control + joint commands sent
# continuously or it drops the motion, so we re-send every cycle at CONTROL_HZ for
# SETTLE_SECONDS (matches piper_sdk demo piper_ctrl_moveJ.py).
SPEED = 30  # % of max joint speed; gentle for homing
CONTROL_HZ = 100
SETTLE_SECONDS = 5

# Master-slave (teleop) homing settle time. ReqMasterArmMoveToHome is a single
# fire-and-forget request the firmware executes internally, so we just wait for the
# motion to finish rather than streaming anything.
MASTER_HOME_SETTLE_SECONDS = 6

# ReqMasterArmMoveToHome (CAN 0x191) was added in firmware V1.7-4. Older firmware
# silently ignores it, so we parse the reported version and warn rather than
# leaving the user with a silent no-op (the observed can_arm_right symptom).
MIN_FIRMWARE_FOR_HOME = (1, 7, 4)
MIN_FIRMWARE_LABEL = "V1.7-4"

# Firmware version request needs a trigger frame; without SearchPiperFirmwareVersion
# the read returns the error code -0x4AF until a feedback frame arrives.
FIRMWARE_READ_ATTEMPTS = 40
FIRMWARE_READ_INTERVAL = 0.025

# GetArmStatus().arm_status.ctrl_mode values (CAN ID 0x2A1):
CTRL_MODE_STANDBY = 0x00
CTRL_MODE_CAN = 0x01
CTRL_MODE_TEACH = 0x02  # master-slave master / drag-teach: ignores JointCtrl
CTRL_MODE_NAMES = {
    CTRL_MODE_STANDBY: "standby",
    CTRL_MODE_CAN: "CAN-control",
    CTRL_MODE_TEACH: "teach/master",
}


def interface_side(interface: str) -> str:
    """Map a CAN interface name to its side, matching PiperConfig's defaults
    (can_interface_left="can_arm_left", can_interface_right="can_arm_right")."""
    if "left" in interface:
        return "left"
    if "right" in interface:
        return "right"
    raise ValueError(f"cannot infer side ('left'/'right') from interface: {interface}")


def load_home_pose(config_path: str) -> dict[str, list[int]]:
    """Read robot.home_pose (per-side, per-joint milli-degrees) from record.yaml.

    Read straight from YAML rather than via draccus/the lerobot robot so homing
    doesn't open cameras or register plugins. Each side falls back to all-zeros if
    absent. Values are milli-degrees fed directly to JointCtrl (no conversion).
    Only used by the --mode normal path; master-slave homing always targets zero.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    raw = (cfg.get("robot") or {}).get("home_pose") or {}
    home_pose = {"left": [0] * 6, "right": [0] * 6}
    for side, pose in raw.items():
        if side not in home_pose:
            raise ValueError(f"home_pose side must be 'left'/'right', got: {side}")
        if len(pose) != 6:
            raise ValueError(f"home_pose[{side}] must be 6 values, got: {pose}")
        home_pose[side] = [int(v) for v in pose]
    return home_pose


def load_home_gripper(config_path: str) -> dict[str, int]:
    """Read robot.home_gripper ('open'/'closed' per side) from record.yaml, mapped
    to GripperCtrl angle units. Each side defaults to 'open' if absent. Only used by
    the --mode normal path."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    raw = (cfg.get("robot") or {}).get("home_gripper") or {}
    angle_for = {"open": OPEN_POSITION, "closed": CLOSED_POSITION}
    gripper = {"left": OPEN_POSITION, "right": OPEN_POSITION}
    for side, state in raw.items():
        if side not in gripper:
            raise ValueError(f"home_gripper side must be 'left'/'right', got: {side}")
        if state not in angle_for:
            raise ValueError(
                f"home_gripper[{side}] must be 'open'/'closed', got: {state}"
            )
        gripper[side] = angle_for[state]
    return gripper


def read_firmware(arm) -> str | None:
    """Return the firmware version string (e.g. 'S-V1.7-4'), or None if it never
    arrives. GetPiperFirmwareVersion returns the int error code -0x4AF until a
    firmware feedback frame is received, so we trigger one with
    SearchPiperFirmwareVersion and poll (per the piper_read_firmware demo)."""
    for _ in range(FIRMWARE_READ_ATTEMPTS):
        arm.SearchPiperFirmwareVersion()
        time.sleep(FIRMWARE_READ_INTERVAL)
        fw = arm.GetPiperFirmwareVersion()
        if isinstance(fw, str) and "V" in fw:
            return fw
    return None


def parse_firmware(fw: str | None) -> tuple[int, int, int] | None:
    """Parse 'S-V1.7-4' -> (1, 7, 4). Returns None if it doesn't match."""
    if not fw:
        return None
    import re

    m = re.search(r"V(\d+)\.(\d+)-(\d+)", fw)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def report_arm_state(iface: str, arm) -> tuple[str | None, int]:
    """Print firmware, control mode, and live joint positions for one arm, and
    return (firmware_string, ctrl_mode).

    Why: the can_arm_right home is a silent no-op and we must tell apart the two
    causes. Firmware < V1.7-4 means ReqMasterArmMoveToHome (0x191) is unsupported;
    non-zero/responsive joint readings + a valid ctrl_mode prove the arm is
    actually communicating (so it's a firmware issue, not a dead bus). A
    master/teach arm reports ctrl_mode == 0x02; normal CAN control == 0x01.
    """
    fw = read_firmware(arm)
    ctrl_mode = int(arm.GetArmStatus().arm_status.ctrl_mode)
    mode_name = CTRL_MODE_NAMES.get(ctrl_mode, f"0x{ctrl_mode:02X}")
    js = arm.GetArmJointMsgs().joint_state
    joints = [getattr(js, f"joint_{i}") for i in range(1, 7)]
    print(
        f"  {iface}: firmware={fw or '<unavailable>'} "
        f"ctrl_mode={mode_name} (0x{ctrl_mode:02X}) joints={joints}"
    )

    version = parse_firmware(fw)
    if version is not None and version < MIN_FIRMWARE_FOR_HOME:
        print(
            f"    WARNING: firmware {fw} < {MIN_FIRMWARE_LABEL}; "
            f"ReqMasterArmMoveToHome (0x191) is UNSUPPORTED on this arm and "
            f"homing will have no effect."
        )
    elif fw is None:
        print(
            f"    WARNING: no firmware feedback from {iface}; the arm may not be "
            f"powered/communicating on this bus — homing will have no effect."
        )
    return fw, ctrl_mode


def home_master_slave(iface: str, arm, gripper_angle: int | None) -> None:
    """Home a master-slave teleop pair using the firmware-native return-to-zero.

    During teleoperation the arm is left in AgileX master-slave mode (the master is
    set via MasterSlaveConfig(0xFA)). In that state the master SILENTLY IGNORES
    JointCtrl/MotionCtrl_2 (the firmware drives it), and forcing it out of that mode
    DEPOWERS the motors -> the arm free-falls and self-collides. Both failure modes
    were observed on hardware.

    ReqMasterArmMoveToHome (CAN ID 0x191, firmware >= V1.7-4) avoids both: it homes
    the pair in-mode with motors powered, and the master actually moves because the
    firmware executes the move. mode=2 sends master AND slave to zero together;
    mode=0 then restores master-slave mode so teleoperation resumes cleanly.

    Sending 0x191 to the master homes both arms via the linkage; if `iface` is the
    slave the command is simply a no-op there, which the diagnostic print above
    makes visible.

    The 0x191 home moves the 6 joints only, not the gripper. When `gripper_angle`
    is not None we send a direct GripperCtrl afterwards: the gripper is outside the
    joint master-slave linkage, so it may accept a direct command even while the
    arm is in master-slave mode (verified by observation on hardware).
    """
    print(f"  {iface}: requesting master+slave return-to-zero (0x191 mode=2)...")
    arm.ReqMasterArmMoveToHome(2)
    time.sleep(MASTER_HOME_SETTLE_SECONDS)
    if gripper_angle is not None:
        arm.GripperCtrl(gripper_angle, 1000, 0x01, 0)
        print(f"  {iface}: commanded gripper to {gripper_angle} (0.001mm)")
    # Restore master-slave mode so the operator can teleoperate again immediately.
    arm.ReqMasterArmMoveToHome(0)
    print(f"  {iface}: restored master-slave mode (0x191 mode=0)")


def home_normal(arms: dict, sides: dict, home_pose: dict, home_gripper: dict) -> None:
    """Home arms that are in normal CAN-control mode (inference / rollout context,
    NOT master-slave). Here JointCtrl works, so we can hit the configured per-side
    home_pose. Stream the 0x151 motion-control + joint commands continuously: the
    firmware drops motion if it isn't re-sent every cycle.
    """
    for arm in arms.values():
        while not arm.EnablePiper():
            time.sleep(0.01)

    print(f"Driving to home pose ({SETTLE_SECONDS}s)...")
    for _ in range(int(SETTLE_SECONDS * CONTROL_HZ)):
        for iface, arm in arms.items():
            side = sides[iface]
            arm.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
            arm.JointCtrl(*home_pose[side])
            arm.GripperCtrl(home_gripper[side], 1000, 0x01, 0)
        time.sleep(1.0 / CONTROL_HZ)


def main():
    parser = argparse.ArgumentParser(description="Home Piper arm(s) to a rest pose.")
    # Home BOTH sets by default: each is on its own CAN interface. In master-slave
    # mode a single 0x191 to a master homes that master+slave pair via the linkage,
    # so listing both interfaces covers whichever is the master on this rig.
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=["can_arm_left", "can_arm_right"],
        help="CAN interface name(s) to home (default: can_arm_left can_arm_right)",
    )
    parser.add_argument(
        "--mode",
        choices=["teleop", "normal"],
        default="teleop",
        help=(
            "teleop: firmware return-to-zero for a master-slave pair (default; the "
            "only mode that won't free-fall and that moves the master). "
            "normal: stream JointCtrl to the per-side home_pose for an arm already "
            "in CAN-control mode (inference/rollout)."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/record.yaml",
        help="YAML providing robot.home_pose (default: configs/record.yaml)",
    )
    parser.add_argument(
        "--home-gripper",
        action="store_true",
        help=(
            "In --mode teleop, also command the gripper to the per-side "
            "home_gripper state after the joint home (0x191 homes joints only)."
        ),
    )
    args = parser.parse_args()

    # Map each interface to its side up front so the side is available for prints and
    # for the per-side pose in --mode normal.
    sides = {iface: interface_side(iface) for iface in args.interfaces}
    arms = {iface: C_PiperInterface_V2(iface) for iface in args.interfaces}

    print("Connecting...")
    for arm in arms.values():
        arm.ConnectPort()
    time.sleep(0.1)

    # Diagnostic: firmware + control mode + joints per arm (see report_arm_state).
    print(f"Arm state ({len(arms)} interface(s)):")
    for iface, arm in arms.items():
        report_arm_state(iface, arm)

    if args.mode == "teleop":
        # Gripper home state is per-side; only resolve it when requested.
        home_gripper = load_home_gripper(args.config) if args.home_gripper else None
        print("Homing (master-slave firmware return-to-zero)...")
        for iface, arm in arms.items():
            gripper_angle = home_gripper[sides[iface]] if home_gripper else None
            home_master_slave(iface, arm, gripper_angle)
    else:
        home_pose = load_home_pose(args.config)
        home_gripper = load_home_gripper(args.config)
        print(f"Homing (normal CAN control) using poses from {args.config}:")
        for iface, side in sides.items():
            gripper = "open" if home_gripper[side] == OPEN_POSITION else "closed"
            print(f"  {iface} ({side}): joints={home_pose[side]} gripper={gripper}")
        home_normal(arms, sides, home_pose, home_gripper)

    print("Done.")


if __name__ == "__main__":
    main()
