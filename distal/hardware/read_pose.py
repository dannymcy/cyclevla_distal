"""Read and print the Piper arm(s)' current joint pose.

Hand-pose the arm, run this, and paste the printed `home_pose` line into
configs/record.yaml so zero.py homes to that pose.
"""

import argparse
import time

from piper_sdk import C_PiperInterface_V2


def interface_side(interface: str) -> str:
    """Map a CAN interface name to its side, matching PiperConfig's defaults."""
    if "left" in interface:
        return "left"
    if "right" in interface:
        return "right"
    raise ValueError(f"cannot infer side ('left'/'right') from interface: {interface}")


def main():
    parser = argparse.ArgumentParser(description="Print Piper arm(s)' current pose.")
    # Mirror zero.py's default so the same active set is read by default.
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=["can_arm_right"],
        help="CAN interface name(s) to read (default: can_arm_right)",
    )
    args = parser.parse_args()

    arms = {iface: C_PiperInterface_V2(iface) for iface in args.interfaces}

    # ConnectPort() starts the CAN read thread; encoder feedback then streams in
    # regardless of motor state. We deliberately do NOT call EnablePiper() here:
    # enabling energizes the motors, which would hold/fight a pose set by hand.
    for arm in arms.values():
        arm.ConnectPort()
    # Give the read thread a moment to receive the first feedback frames.
    time.sleep(0.2)

    for iface, arm in arms.items():
        js = arm.GetArmJointMsgs().joint_state
        # joint_i is in milli-degrees (0.001 deg); dataset "pos" units are / 1000.
        joints = [getattr(js, f"joint_{i}") for i in range(1, 7)]
        # grippers_angle is in 0.0001 units; dataset "pos" units are / 10000.
        gripper = arm.GetArmGripperMsgs().gripper_state.grippers_angle

        print(f"\n{iface}:")
        for i, milli_deg in enumerate(joints, start=1):
            print(f"  joint_{i}: {milli_deg:>8d}  ({milli_deg / 1000.0:+8.3f} deg)")
        print(f"  gripper: {gripper:>8d}  ({gripper / 10000.0:+8.3f} mm)")
        # Ready-to-paste line: nests under robot.home_pose in configs/record.yaml.
        print(f"  {interface_side(iface)}: {joints}")

    for arm in arms.values():
        arm.DisconnectPort()


if __name__ == "__main__":
    main()
