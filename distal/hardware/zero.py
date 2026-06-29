"""Home Piper master-slave teleop pair(s) to zero via the firmware return-to-zero.

Each "set" is an AgileX master-slave pair: you move the master/leader by hand and
the slave/follower mirrors it through a firmware-level linkage. The ONLY safe way
to home such a pair is the firmware command `ReqMasterArmMoveToHome` (CAN 0x191),
which drives both arms back to zero while staying in master-slave mode, motors
powered. Plain JointCtrl is deliberately NOT used: it only commands the slave-side
arm and switches it into pure CAN-control, which BREAKS the master-slave pairing
(teleop and 0x191 both die until a power cycle) — observed on hardware.

`ReqMasterArmMoveToHome` requires arm firmware >= V1.7-4; older firmware silently
ignores it, so we read and warn on the version.
"""

import argparse
import re
import time

import yaml
from piper_sdk import C_PiperInterface_V2

# Gripper angle in 0.001 mm units. 70 mm is the Piper gripper's full open; 0 closed.
OPEN_POSITION = 70_000
CLOSED_POSITION = 0

# Map a config `sides` entry to its CAN interface, matching PiperConfig's defaults
# (can_interface_left="can_arm_left", can_interface_right="can_arm_right").
SIDE_TO_INTERFACE = {"left": "can_arm_left", "right": "can_arm_right"}

# ReqMasterArmMoveToHome is a single fire-and-forget request the firmware executes
# internally, so we just wait for the motion to finish rather than streaming.
MASTER_HOME_SETTLE_SECONDS = 6

# ReqMasterArmMoveToHome (CAN 0x191) was added in firmware V1.7-4. Older firmware
# silently ignores it, so we parse the reported version and warn rather than
# leaving the user with a silent no-op (the observed can_arm_right symptom).
MIN_FIRMWARE_FOR_HOME = (1, 7, 4)
MIN_FIRMWARE_LABEL = "V1.7-4"

# Firmware version request needs a trigger frame (SearchPiperFirmwareVersion), which
# ALSO resets the SDK's firmware buffer. So call Search once per round, then poll a
# few times to let the full version string assemble before re-Searching. Re-Searching
# every poll (the old bug) wipes the partial data and truncates the read to "S-V1".
FIRMWARE_SEARCH_ROUNDS = 8
FIRMWARE_POLL_PER_ROUND = 40
FIRMWARE_POLL_INTERVAL = 0.025

# GetArmStatus().arm_status.ctrl_mode values (CAN ID 0x2A1):
CTRL_MODE_NAMES = {
    0x00: "standby",
    0x01: "CAN-control",
    0x02: "teach/master",  # master-slave master / drag-teach: ignores JointCtrl
}


def interface_side(interface: str) -> str:
    """Map a CAN interface name to its side, matching PiperConfig's defaults."""
    if "left" in interface:
        return "left"
    if "right" in interface:
        return "right"
    raise ValueError(f"cannot infer side ('left'/'right') from interface: {interface}")


def load_sides(config_path: str) -> list[str]:
    """Read robot.sides (the active arm set) from record.yaml so homing targets the
    set you actually record with. Read straight from YAML rather than via draccus so
    homing doesn't open cameras or register plugins. Defaults to PiperConfig's
    ["right"] if absent."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    sides = (cfg.get("robot") or {}).get("sides") or ["right"]
    invalid = [s for s in sides if s not in SIDE_TO_INTERFACE]
    if invalid:
        raise ValueError(f"robot.sides must be 'left'/'right', got: {invalid}")
    return list(sides)


def load_home_gripper(config_path: str) -> dict[str, int]:
    """Read robot.home_gripper ('open'/'closed' per side) from record.yaml, mapped
    to GripperCtrl angle units. Each side defaults to 'open' if absent. Used only by
    the opt-in --home-gripper (0x191 homes the 6 joints but not the gripper)."""
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


def parse_firmware(fw: str | None) -> tuple[int, int, int] | None:
    """Parse 'S-V1.7-4' -> (1, 7, 4). Returns None if it doesn't match."""
    if not fw:
        return None
    m = re.search(r"V(\d+)\.(\d+)-(\d+)", fw)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())  # ty: ignore[invalid-return-type]


def read_firmware(arm) -> str | None:
    """Return the full firmware version string (e.g. 'S-V1.8-7'), or None if it
    never arrives. GetPiperFirmwareVersion returns the int error code -0x4AF until a
    feedback frame is received. SearchPiperFirmwareVersion triggers that frame but
    also resets the buffer, so call it once per round and poll for the COMPLETE
    'V<major>.<minor>-<patch>' before returning (a partial read yields 'S-V1')."""
    for _ in range(FIRMWARE_SEARCH_ROUNDS):
        arm.SearchPiperFirmwareVersion()
        for _ in range(FIRMWARE_POLL_PER_ROUND):
            time.sleep(FIRMWARE_POLL_INTERVAL)
            fw = arm.GetPiperFirmwareVersion()
            if isinstance(fw, str) and parse_firmware(fw) is not None:
                return fw
    fw = arm.GetPiperFirmwareVersion()
    return fw if isinstance(fw, str) else None


def report_arm_state(iface: str, arm) -> None:
    """Print firmware, control mode, and live joint positions for one arm, and warn
    if the firmware predates the home command.

    Why: a failed home must be diagnosable. Firmware < V1.7-4 means
    ReqMasterArmMoveToHome (0x191) is unsupported (the right set's S-V1.7-3 case);
    a missing firmware read / dead joints means the bus isn't communicating. A
    master/teach arm reports ctrl_mode == 0x02, normal CAN control == 0x01.
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
            f"homing will have no effect (flash this set or record with the other)."
        )
    elif fw is None:
        print(
            f"    WARNING: no firmware feedback from {iface}; the arm may not be "
            f"powered/communicating on this bus — homing will have no effect."
        )


def home_master_slave(
    iface: str, arm, gripper_angle: int | None, restore: bool = True
) -> None:
    """Home a master-slave teleop pair using the firmware-native return-to-zero.

    ReqMasterArmMoveToHome (CAN 0x191, firmware >= V1.7-4) homes the pair in-mode
    with motors powered: mode=2 sends master AND slave to zero together; mode=0 then
    restores master-slave mode so teleoperation resumes cleanly. Sending 0x191 to
    the master homes both via the linkage; on the slave it is a no-op (the
    diagnostic print above makes that visible).

    `restore` (default True) controls the final `ReqMasterArmMoveToHome(0)` that
    RE-ENABLES the master-slave linkage. record/teleop want that so the operator can
    teleoperate again. AUTONOMOUS EVAL passes restore=False: re-linking would make
    the leader keep driving the follower during the rollout (the "leader also moves"
    bug); skipping it leaves the pair homed but unlinked, so the follower can then be
    put in CAN control (set_eef_mode) and driven ALONE via EndPoseCtrl.

    The 0x191 home moves the 6 joints only, not the gripper. When `gripper_angle`
    is not None we send a direct GripperCtrl afterwards: the gripper is outside the
    joint master-slave linkage, so it may accept a direct command even in
    master-slave mode (opt-in via --home-gripper; unverified on hardware).
    """
    print(f"  {iface}: requesting master+slave return-to-zero (0x191 mode=2)...")
    arm.ReqMasterArmMoveToHome(2)
    time.sleep(MASTER_HOME_SETTLE_SECONDS)
    if gripper_angle is not None:
        arm.GripperCtrl(gripper_angle, 1000, 0x01, 0)
        print(f"  {iface}: commanded gripper to {gripper_angle} (0.001mm)")
    if restore:
        # Restore master-slave mode so the operator can teleoperate again.
        arm.ReqMasterArmMoveToHome(0)
        print(f"  {iface}: restored master-slave mode (0x191 mode=0)")
    else:
        print(f"  {iface}: left UNLINKED (no 0x191 mode=0) — autonomous eval.")


def main():
    parser = argparse.ArgumentParser(
        description="Home Piper master-slave teleop pair(s) to zero (firmware 0x191)."
    )
    # Default to the active set from record.yaml `sides` so a bare run homes the set
    # you record with; pass --interfaces to override (e.g. home both, or the idle set).
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=None,
        help="CAN interface(s) to home (default: derived from record.yaml sides)",
    )
    parser.add_argument(
        "--config",
        default="configs/record.yaml",
        help="YAML providing robot.sides / robot.home_gripper (default: record.yaml)",
    )
    parser.add_argument(
        "--home-gripper",
        action="store_true",
        help=(
            "Also command the gripper to the per-side home_gripper state after the "
            "joint home (0x191 homes joints only)."
        ),
    )
    args = parser.parse_args()

    interfaces = args.interfaces
    if interfaces is None:
        interfaces = [SIDE_TO_INTERFACE[s] for s in load_sides(args.config)]

    sides = {iface: interface_side(iface) for iface in interfaces}
    arms = {iface: C_PiperInterface_V2(iface) for iface in interfaces}

    print(f"Homing set(s): {', '.join(interfaces)}")
    print("Connecting...")
    for arm in arms.values():
        arm.ConnectPort()
    time.sleep(0.1)

    # Diagnostic: firmware + control mode + joints per arm (see report_arm_state).
    print(f"Arm state ({len(arms)} interface(s)):")
    for iface, arm in arms.items():
        report_arm_state(iface, arm)

    # Gripper home state is per-side; only resolve it when requested.
    home_gripper = load_home_gripper(args.config) if args.home_gripper else None
    print("Homing (master-slave firmware return-to-zero)...")
    for iface, arm in arms.items():
        gripper_angle = home_gripper[sides[iface]] if home_gripper else None
        home_master_slave(iface, arm, gripper_angle)

    print("Done.")


if __name__ == "__main__":
    main()
