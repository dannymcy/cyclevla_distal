"""Camera-only soak test — isolate the RealSense cameras from the policy server
and the arm.

During a real-robot eval rollout both cameras sometimes drop together
(`try_wait_for_frames -> status=False`). `pixi run record` and `pixi run rollout`
stream the same cameras fine, so this script reproduces JUST the camera read at
the eval's rate WITHOUT contacting the openpi server and WITHOUT commanding the
arm. If the cameras stream cleanly here, the drop is triggered by the eval loop's
blocking server query (fix: mirror the RTC async loop); if they drop here too,
it's a USB/driver issue (fix: USB topology / --camera-fps / cable — see
REAL_EVAL.md).

Run:
  pixi run camera-check -- --seconds 60
  pixi run camera-check -- --seconds 60 --camera-fps 15
  pixi run camera-check -- --seconds 60 --home      # also test the homing path
"""

import argparse
import logging
import threading
import time

from distal.hardware import subtasks
from distal.hardware.real_eval_common import (
    PolicyClient,
    RealEvalConfig,
    RolloutVideoWriter,
    active_side,
    build_observation,
    home_robot,
    init_eval_keyboard_listener,
    make_robot,
    setup_logging,
    standby,
    synthetic_observation,
)

logger = logging.getLogger(__name__)

TOP_KEY = "top"


def start_server_load_thread(cfg, side, stop_event):
    """Background thread that drives REAL server inference load (mirrors the eval's
    query) WITHOUT touching the cameras, so the main loop can read cameras steadily
    and we can see whether server GPU load alone drops frames.

    Returns the thread (already started). Uses a fixed synthetic observation — the
    image content is irrelevant; the point is to make the server run pi0.5 inference
    at the eval's cadence."""
    client = PolicyClient(host=cfg.host, port=cfg.port)
    prompt = subtasks.get_subtasks(cfg.single_task)[0].lower()
    obs_dict = build_observation(synthetic_observation(side), prompt, side)
    period = 1.0 / max(0.1, cfg.query_hz)
    n_queries = [0]

    def loop():
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                client.get_action(obs_dict, cfg.num_open_loop_steps)
                n_queries[0] += 1
            except Exception as e:  # noqa: BLE001 — keep loading even if one call fails
                logger.warning(f"server query failed: {e}")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    th = threading.Thread(target=loop, daemon=True, name="server-load")
    th.start()
    logger.info(
        f"Server-load thread started: querying ws://{cfg.host}:{cfg.port} at "
        f"~{cfg.query_hz} Hz while the main loop reads cameras."
    )
    return th, n_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0, help="Soak duration.")
    parser.add_argument("--fps", type=int, default=20, help="Read rate (Hz).")
    parser.add_argument(
        "--sides", nargs="+", default=["left"], help="Active arm set (cameras follow)."
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=None,
        help="Override RealSense fps (e.g. 15) to cut USB bandwidth; default keeps 30.",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Also run the firmware home before soaking (tests the homing path).",
    )
    parser.add_argument(
        "--query-server",
        action="store_true",
        help="Also hammer the openpi server in a BACKGROUND thread (real inference "
        "load) while reading cameras steadily — isolates server-GPU-load vs the "
        "eval's blocking cadence as the camera-drop cause.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="openpi server host.")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port.")
    parser.add_argument(
        "--query-hz",
        type=float,
        default=4.0,
        help="Server query rate for --query-server (eval ≈ fps/num_open_loop_steps).",
    )
    # --- Bisect toggles: add the eval-only elements one at a time onto the soak ---
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=0.0,
        help="After connect (+ --home), sit idle this long WITHOUT reading cameras "
        "before the loop — mimics the eval's homing + SPACE-wait gap.",
    )
    parser.add_argument(
        "--listener",
        action="store_true",
        help="Start the pynput keyboard listener (the eval has one; the plain soak "
        "does not) — rule out the X11/pynput thread.",
    )
    parser.add_argument(
        "--live-obs",
        action="store_true",
        help="Each tick, also build_observation() from the LIVE frames (mirrors the "
        "eval's per-tick PIL resize on the main thread) — rule that out.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Append every full-res frame to in-RAM lists like the eval's OLD "
        "frames_by_cam (never freed) — reproduces the drop (accumulation starves "
        "the read thread).",
    )
    parser.add_argument(
        "--stream-frames",
        action="store_true",
        help="STREAM every frame to disk via RolloutVideoWriter (the eval's fix) "
        "instead of accumulating — should stay CLEAN under --query-server, proving "
        "the streaming eval won't drop.",
    )
    args = parser.parse_args()

    # Reuse the eval config + robot construction so cameras/fps match the eval.
    cfg = RealEvalConfig(
        sides=list(args.sides),
        fps=args.fps,
        camera_fps=args.camera_fps,
        home=args.home,
        host=args.host,
        port=args.port,
    )
    cfg.query_hz = args.query_hz  # camera_check-only knob (not a RealEvalConfig field)
    setup_logging(cfg, "camera_check")
    side = active_side(cfg)
    wrist_key = f"{side}_wrist"

    mode = "cameras + BACKGROUND server load" if args.query_server else "cameras only"
    logger.info(
        f"Camera soak: {args.seconds:.0f}s @ {args.fps}Hz, sides={args.sides}, "
        f"camera_fps={args.camera_fps}, home={args.home}, query_server="
        f"{args.query_server}, save_frames={args.save_frames}. Mode: {mode}. "
        f"NO arm commands."
    )
    robot = make_robot(cfg)
    arm = robot.arms[side]
    prompt = subtasks.get_subtasks(cfg.single_task)[0].lower()

    # Optional bisect: pynput keyboard listener (eval-only element).
    listener = None
    if args.listener:
        listener, _ = init_eval_keyboard_listener()
        logger.info("Keyboard listener started (bisect).")

    # Optionally drive real server inference load in the background.
    stop_load = threading.Event()
    load_thread, load_count = None, [0]
    if args.query_server:
        load_thread, load_count = start_server_load_thread(cfg, side, stop_load)
    period = 1.0 / args.fps
    ok = 0
    dropped = 0
    cur_streak = 0
    max_streak = 0
    shapes_logged = False
    # Mimic the eval's frames_by_cam: accumulate every full-res frame, never freed.
    saved_frames = {"image": [], "wrist_image": []} if args.save_frames else None
    # Mimic the eval's FIX: stream frames to disk (bounded memory). Discarded at end.
    stream_writer = (
        RolloutVideoWriter(
            cfg.video_dir, 0, "camera_check_soak", args.fps, "camera_check"
        )
        if args.stream_frames
        else None
    )
    try:
        if args.home:
            home_robot(robot, cfg)
        if args.idle_seconds > 0:
            logger.info(
                f"Idle {args.idle_seconds:.0f}s WITHOUT reading cameras "
                f"(mimics homing + SPACE gap)..."
            )
            time.sleep(args.idle_seconds)

        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            try:
                obs = (
                    robot.get_observation()
                )  # raw read (no reconnect) to see true drops
                ok += 1
                cur_streak = 0
                if saved_frames is not None:
                    # Mirror the eval's frames_by_cam accumulation (held in RAM).
                    saved_frames["image"].append(obs[TOP_KEY])
                    saved_frames["wrist_image"].append(obs[wrist_key])
                if stream_writer is not None:
                    # Mirror the eval's streaming fix (write + drop, nothing held).
                    stream_writer.add(
                        {"image": obs[TOP_KEY], "wrist_image": obs[wrist_key]}, "soak"
                    )
                if args.live_obs:
                    # Mirror the eval's per-tick PIL resize on the main thread.
                    build_observation(obs, prompt, side)
                if not shapes_logged:
                    logger.info(
                        f"First frames OK: top={obs[TOP_KEY].shape}, "
                        f"{wrist_key}={obs[wrist_key].shape}"
                    )
                    shapes_logged = True
            except (TimeoutError, RuntimeError) as e:
                dropped += 1
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
                logger.warning(f"DROP #{dropped} (streak {cur_streak}): {e}")
            time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
    finally:
        stop_load.set()
        if load_thread is not None:
            load_thread.join(timeout=2.0)
        if listener is not None:
            listener.stop()
        if stream_writer is not None:
            stream_writer.discard()  # soak video not needed
        standby(arm)
        robot.disconnect()

    total = ok + dropped
    rate = (100.0 * dropped / total) if total else 0.0
    logger.info(
        f"=== Camera soak summary === reads={total} ok={ok} dropped={dropped} "
        f"({rate:.1f}%) longest_drop_streak={max_streak} "
        f"server_queries={load_count[0]}"
    )
    if args.query_server:
        # This run reads cameras STEADILY while the server runs real inference.
        if dropped == 0:
            logger.info(
                "CLEAN with background server load → server GPU inference does NOT "
                "disrupt the cameras; the eval drop was the BLOCKING query cadence → "
                "the RTC async refactor will fix it (plan Follow-up 5 D)."
            )
        else:
            logger.info(
                "DROPPED with background server load (cameras read steadily) → server "
                "inference disrupts the USB cameras regardless of client threading → "
                "async won't help; run the server on another GPU/host or fix USB "
                "topology / --camera-fps (Follow-up 4)."
            )
    elif dropped == 0:
        logger.info(
            "Cameras streamed CLEAN with no server/arm → re-run with --query-server "
            "to test whether server inference load is what drops them."
        )
    else:
        logger.info(
            "Cameras dropped even without server/arm → USB/driver issue; see "
            "REAL_EVAL.md 'Cameras drop mid-rollout' (USB topology / fps / cable)."
        )


if __name__ == "__main__":
    main()
