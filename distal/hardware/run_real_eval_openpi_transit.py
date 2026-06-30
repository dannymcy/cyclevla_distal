"""Real-robot transit-only CycleVLA eval for the pi05 / openpi policy.

Hardware counterpart of
`cyclevla_code/experiments/robot/libero/run_libero_eval_openpi_transit.py`:
the same subtask-transit protocol (no VLM, no backtrack, no MBR) — drive each
subtask until the policy's stop signal confirms it, then advance — but the robot
is the real AgileX Piper instead of LIBERO sim, and the 9-D policy is served by
the openpi WebSocket server (run separately in cyclevla_code/openpi).

Per control step: read the arm + cameras, build the 8-D EEF state + two images,
query the server, apply the returned ΔEEF via Cartesian EndPoseCtrl, and watch
the stop signal. Both-camera rollout videos (raw + subtitled with the live
subtask) are saved to a gitignored dir. There is no automatic success check —
the operator inspects the saved video (per the eval plan).

Run (with the policy server already up — see PI.md / serve_policy.py):
  pixi run python -m distal.hardware.run_real_eval_openpi_transit \
      --config_path configs/real_eval.yaml
Dry-run the schema + connectivity with no arm:
  pixi run python -m distal.hardware.run_real_eval_openpi_transit \
      --config_path configs/real_eval.yaml --dry_run true
"""

import logging
import time
from collections import deque

import draccus
from lerobot.utils.robot_utils import precise_sleep

from distal.hardware import subtasks
from distal.hardware.real_eval_common import (
    AsyncFetch,
    CameraReadError,
    PolicyClient,
    RealEvalConfig,
    RolloutVideoWriter,
    SignalConfirmer,
    active_side,
    apply_delta,
    build_observation,
    clip_delta,
    command_eef,
    eef_from_obs,
    home_robot,
    init_eval_keyboard_listener,
    make_robot,
    read_observation,
    reconnect_cameras,
    report_ctrl_mode,
    resolve_gripper_command,
    run_dry_run,
    set_eef_mode,
    setup_logging,
    split_action,
    standby,
    wait_for_start,
)

logger = logging.getLogger(__name__)

# Third-person camera key on the Piper rig (see config_piper.py). Mapped to the
# policy's `image`; the wrist camera ({side}_wrist) maps to `wrist_image`.
TOP_KEY = "top"


def run_episode_transit(cfg, robot, arm, client, states, events, writer):
    """Drive one episode, advancing through `states` on the stop signal.

    Same semantics as the sim transit eval (query → execute N open-loop → requery
    on empty), but the requery runs in a BACKGROUND thread (AsyncFetch) so the main
    loop reads cameras EVERY tick at steady fps and never blocks on the network.
    Frames are STREAMED to disk via `writer` (RolloutVideoWriter) each tick — never
    held in RAM (holding them starves the RealSense read thread → camera drop).

    Ends on natural completion, max_steps, OR the operator's `->`/`<-`/Esc.
    """
    side = active_side(cfg)
    wrist_key = f"{side}_wrist"
    period = 1.0 / cfg.fps
    fetch = AsyncFetch(client, cfg.num_open_loop_steps)
    t = 0  # global step counter (bounds the whole episode)

    try:
        for subtask_idx, current_state in enumerate(states):
            if events["exit_early"]:
                break
            logger.info(f"Starting subtask {subtask_idx}: {current_state}")
            stop_confirm = SignalConfirmer(threshold=0.5)
            queue: deque = deque()
            pending = False  # a background fetch is in flight for this subtask
            cmd_eef = None  # accumulating command pose for delta_base == "chunk"
            # Per-subtask step budget mirrors the sim transit script.
            subtask_budget = int(
                cfg.max_steps // len(states) * 1.5 + cfg.num_steps_wait
            )
            t_sub = 0

            while t_sub < subtask_budget and t < cfg.max_steps:
                if events["exit_early"]:  # `->` save / `<-` redo / Esc stop
                    logger.info(f"Operator ended episode at step {t}.")
                    return
                loop_start = time.perf_counter()

                # Read cameras + arm EVERY tick, and STREAM frames to disk (never
                # hold them in RAM — accumulation starves the camera read thread).
                obs = read_observation(robot)
                writer.add(
                    {"image": obs[TOP_KEY], "wrist_image": obs[wrist_key]},
                    f"{subtask_idx}:{current_state}",
                )

                # Queue empty → requery in the background (non-blocking).
                if not queue and not pending:
                    obs_dict = build_observation(obs, current_state, side)
                    if fetch.request(obs_dict):
                        pending = True
                # Chunk arrived → load it; re-ground the chunk-start command pose.
                if pending and fetch.ready():
                    queue.extend(fetch.take())
                    pending = False
                    cmd_eef = eef_from_obs(obs, side)

                if queue:
                    delta6, g, stop_signal, progress_signal = split_action(
                        queue.popleft()
                    )
                    delta6 = clip_delta(delta6, cfg.max_pos_step, cfg.max_rot_step)
                    # Per-step action print (debug the stop/progress signals).
                    logger.info(
                        "t=%d sub=%d Δp=%s Δr=%s grip=%.3f s_t=%.2f p_t=%.2f",
                        t,
                        subtask_idx,
                        [round(float(x), 4) for x in delta6[:3]],
                        [round(float(x), 4) for x in delta6[3:6]],
                        g,
                        stop_signal,
                        progress_signal,
                    )
                    if cfg.delta_base == "live":
                        target = apply_delta(eef_from_obs(obs, side), delta6)
                    else:  # "chunk": accumulate onto the per-chunk base
                        cmd_eef = apply_delta(cmd_eef, delta6)
                        target = cmd_eef
                    if not cfg.no_arm:
                        command_eef(arm, target, resolve_gripper_command(cfg, g))
                    t += 1
                    t_sub += 1
                    if stop_confirm.update(stop_signal):
                        logger.info(
                            f"Finished subtask {subtask_idx} ({current_state}) at "
                            f"step {t} (stop={stop_signal:.2f}, "
                            f"progress={progress_signal:.2f})."
                        )
                        break
                # else: holding (waiting for the first chunk) — cameras keep streaming.

                # Pace the loop to the control fps (precise_sleep matches the rollout).
                precise_sleep(max(0.0, period - (time.perf_counter() - loop_start)))

            if t >= cfg.max_steps:
                logger.warning(f"Hit max_steps ({cfg.max_steps}); ending episode.")
                break
    finally:
        fetch.stop()


@draccus.wrap()
def main(cfg: RealEvalConfig):
    setup_logging(cfg, "transit")

    # Subtask decomposition (same table the training convert used). The prompt
    # for each subtask is the bare lowercased subtask string.
    states = [s.lower() for s in subtasks.get_subtasks(cfg.single_task)]
    logger.info(f"Task: {cfg.single_task!r} -> {len(states)} subtasks: {states}")

    # Connect to the policy server (blocks until reachable).
    logger.info(
        f"Connecting to openpi policy server ws://{cfg.host}:{cfg.port} "
        f"(start it first if this hangs)..."
    )
    client = PolicyClient(host=cfg.host, port=cfg.port)

    if cfg.dry_run:
        run_dry_run(cfg, client, states)
        return

    robot = make_robot(cfg)
    side = active_side(cfg)
    arm = robot.arms[side]
    if cfg.no_arm:
        logger.info("[no_arm] arm disabled — cameras + server only (isolation test).")
    # record.py-style episode loop: home → SPACE start → `->` save → `<-` redo →
    # Esc stop. Homing happens at the TOP of each episode (like record.py).
    listener, events = init_eval_keyboard_listener()
    try:
        ep = 0
        while ep < cfg.num_episodes and not events["stop_recording"]:
            # Home to zero at the start of each episode (record.py-style), WITHOUT
            # re-linking master-slave (home_robot uses restore=False) so the leader
            # stays idle during the rollout.
            if cfg.home:
                home_robot(robot, cfg)
            wait_for_start(events, listener)  # "press SPACE to start"
            if events["stop_recording"]:
                break

            # Re-warm the cameras AFTER the idle gap (homing + SPACE wait + scene
            # reset) so the episode starts from a fresh, streaming pipeline.
            reconnect_cameras(robot)

            # Switch the follower to CAN command + MOVE_P so the policy drives ONLY
            # it via EndPoseCtrl; verify the linkage actually broke (ctrl_mode 0x01,
            # not teach 0x02). Skipped in --no_arm (we issue no arm commands).
            if not cfg.no_arm:
                set_eef_mode(arm, cfg.eef_speed_rate)
                report_ctrl_mode(arm, "after set_eef_mode")
            events["exit_early"] = False
            events["rerecord_episode"] = False
            # Stream frames straight to disk (bounded memory). Opened per episode.
            writer = RolloutVideoWriter(
                cfg.video_dir, ep + 1, cfg.single_task, cfg.fps, "transit"
            )
            logger.info(f"=== Episode {ep + 1}/{cfg.num_episodes} START ===")
            try:
                run_episode_transit(cfg, robot, arm, client, states, events, writer)
            except KeyboardInterrupt:
                logger.warning("Interrupted mid-episode; saving partial rollout.")
                standby(arm)
                writer.close()
                raise
            except CameraReadError as e:
                logger.error(f"{e} — camera bus dropped; saving partial & stopping.")
                logger.error("Check USB (see REAL_EVAL.md 'Cameras drop mid-rollout').")
                standby(arm)
                writer.close()
                break

            standby(arm)

            if events["rerecord_episode"]:  # `<-`: discard + re-record this episode
                logger.info(f"Episode {ep + 1} discarded; re-recording.")
                writer.discard()
                continue  # loop top re-homes before the next SPACE

            writer.close()
            logger.info(
                f"Episode {ep + 1} saved & encoded ({writer.num_frames} frames)."
            )
            ep += 1
    except KeyboardInterrupt:
        logger.warning("Stopping eval.")
    finally:
        standby(arm)
        # Eval ran in CONTROL mode, which breaks the master-slave pairing until a
        # power cycle — so we do NOT try to restore teleop here.
        logger.info(
            "Eval used control mode; POWER-CYCLE the arm before `pixi run record`."
        )
        robot.disconnect()
        if listener is not None:
            listener.stop()
        logger.info("Disconnected.")


if __name__ == "__main__":
    main()
