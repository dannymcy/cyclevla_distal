"""Custom Piper teleop record loop: home-between-episodes + SPACE-gated start +
stable, resumable dataset.

This is a thin local replacement for LeRobot's stock ``lerobot-record`` entry
point (`lerobot/scripts/lerobot_record.py`). It reuses LeRobot's importable
helpers (`record_loop`, processor/feature builders, `VideoEncodingManager`, the
`RecordConfig` schema) and only rewrites the outer episode loop. We need a local
script because the two knobs we want are impossible through config alone:

  * Stock controls live in `init_keyboard_listener()` (only ->/<-/Esc), which we
    can't extend; and
  * `record()` is `@parser.wrap()`-decorated, so it can't be imported and
    overridden.

Three changes vs. stock `record()`:

1. Stable, unstamped, create-or-resume dataset. Stock calls
   `cfg.dataset.stamp_repo_id()` on every fresh run, appending a timestamp to
   `repo_id` so each session makes a new dataset under a different on-disk path.
   We instead resolve the dataset path and AUTO-DETECT: if it already exists we
   `LeRobotDataset.resume()` (append with continuing episode indices); otherwise
   we `LeRobotDataset.create()` WITHOUT stamping. So `pixi run record` always
   reads/writes the same `data/<repo_id>`, and rerunning the command simply
   appends — ending with one coherent dataset that `compute_norm_stats` resolves
   directly. `num_episodes` is the count to ADD this session.

2. Our own keyboard listener adds SPACE as a "start next episode" gate, keeping
   ->/<-/Esc semantics identical so `record_loop` is unchanged.

3. Between episodes we home the active arm set to zero exactly like
   `distal/hardware/zero.py` (firmware return-to-zero `ReqMasterArmMoveToHome`,
   joints only), then BLOCK until the operator presses SPACE — replacing stock's
   blind, timed `reset_time_s` window. Homing reuses the robot's already-open CAN
   handles (`robot.arms[side]`, the same `can_arm_left` bus `zero.py` homes), so
   there is no second connection and no bus contention (homing runs between
   `record_loop` calls, when nothing else drives the bus).

The gripper is NOT homed: in master-slave teleop the gripper sits outside the
joint linkage and software gripper-open is unverified, so the operator opens the
leader gripper by hand (the slave mirrors it) before pressing SPACE.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

from lerobot.common.control_utils import (
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

from distal.hardware import subtasks
from distal.hardware.decompose_videos import export_clips_for_dataset
from distal.hardware.zero import SIDE_TO_INTERFACE, home_master_slave

# One mp4 per episode (per camera) for easy debugging. LeRobot rolls a new video
# file only when the current one would exceed `video_files_size_in_mb`; its
# documented minimum is int 1 MB, but our episodes are sub-MB (≈0.02–0.4 MB each),
# so 1 MB would still concatenate several into one file. A tiny threshold (~1 KB)
# is smaller than any real clip, so the size check trips on EVERY episode and each
# episode gets its own file — via LeRobot's normal roll path (no overwrite). This
# is stored in info.json at create, so resume honors it too. `data_files_size_in_mb`
# is left at its default (100 MB) so the data parquet keeps concatenating.
PER_EPISODE_VIDEO_FILE_SIZE_MB = 0.001


def init_keyboard_listener(robot=None):
    """Like LeRobot's `init_keyboard_listener` but adds two extra keys on top of
    the stock ->/<-/Esc:

      * SPACE -> `start_next`, used to gate the start of the next episode after
        homing; and
      * 'y' -> `robot.bump_subtask()`, marking the END of the current subtask
        live during teleop. Because the bump increments the robot's
        `current_subtask` immediately and every frame stamps `subtask_index`, the
        boundary is frame-accurate with no sidecar. Press 'y' after each subtask
        EXCEPT the last (the episode's save closes the final subtask), so a task
        with K subtasks needs K-1 presses.

    ->/<-/Esc keep their stock meaning so `record_loop` (which only reads
    `exit_early`) is unchanged. Returns `(listener, events)`; `listener` is None
    when headless."""
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_next"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. Keyboard inputs will not be available; "
            "episode gating (SPACE), subtask marking ('y') and ->/<-/Esc controls "
            "are disabled."
        )
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow pressed. Ending and saving this episode...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow pressed. Discarding and re-recording this episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                print("Space pressed. Starting next episode...")
                events["start_next"] = True
            elif hasattr(key, "char") and key.char == "y" and robot is not None:
                # Mark the end of the current subtask; subsequent frames are
                # stamped with the incremented index.
                robot.bump_subtask()
                print(
                    f"'y' pressed. Subtask boundary -> index {robot.current_subtask}."
                )
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def home_active_sides(robot, play_sounds: bool) -> None:
    """Home every active arm set to zero via the firmware return-to-zero, reusing
    the robot's already-open CAN handles. Joints only (gripper is opened by hand
    in teleop mode); identical to `distal/hardware/zero.py::home_master_slave`."""
    log_say("Homing the arm", play_sounds)
    for side, arm in robot.arms.items():
        home_master_slave(SIDE_TO_INTERFACE[side], arm, gripper_angle=None)


def wait_for_start(events: dict, play_sounds: bool) -> None:
    """Block until the operator presses SPACE (`start_next`) or Esc
    (`stop_recording`). Replaces stock's timed `reset_time_s` window: the arm is
    already homed, so this is the moment to open the gripper by hand and reset the
    scene. Used both before the first episode and between episodes. Clears
    `start_next`/`exit_early` afterwards so an accidental arrow tap during the wait
    doesn't immediately skip the upcoming episode."""
    if events["stop_recording"]:
        return
    log_say("Open the gripper, then press space to start recording", play_sounds)
    print(
        "Homed. Open the gripper by hand, then press SPACE to start recording "
        "(Esc to stop)."
    )
    while not events["start_next"] and not events["stop_recording"]:
        time.sleep(0.05)
    events["start_next"] = False
    events["exit_early"] = False


def resolve_dataset_root(cfg: RecordConfig) -> Path:
    """Resolve the on-disk dataset directory the same way LeRobotDataset does:
    explicit `dataset.root`, else `$HF_LEROBOT_HOME/<repo_id>`. The `record` pixi
    task sets HF_LEROBOT_HOME to the project `data/` dir, so this lands under
    `data/<repo_id>`."""
    if cfg.dataset.root:
        return Path(cfg.dataset.root)
    return HF_LEROBOT_HOME / cfg.dataset.repo_id


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Resolve how many subtasks this task should have, BEFORE touching hardware, so
    # a misconfigured `single_task` fails fast. get_subtasks raises (listing the
    # valid tasks) if `single_task` isn't registered in distal/hardware/subtasks.py
    # — this also catches the stale `single_task: Test`. Used to validate, per
    # episode, that the operator pressed 'y' the right number of times.
    expected_subtasks = len(subtasks.get_subtasks(cfg.dataset.single_task))

    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (
            cfg.display_data
            and cfg.display_ip is not None
            and cfg.display_port is not None
        )
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    )

    # Identity pipelines (record has no policy/processors of its own).
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        # Create-or-resume on a STABLE, unstamped path so reruns append into one
        # dataset (see module docstring). Existence of meta/info.json marks an
        # already-created dataset.
        root = resolve_dataset_root(cfg)
        repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
        if repo_name.startswith("eval_"):
            raise ValueError(
                "Dataset names starting with 'eval_' are reserved for policy "
                "evaluation. Use lerobot-rollout for policy deployment."
            )

        # robot.cameras is dynamically typed (base Robot has no `cameras` attr).
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0  # ty: ignore[invalid-argument-type]
        if (root / "meta" / "info.json").exists():
            logging.info(f"Resuming existing dataset at {root} (appending episodes).")
            # resume() requires an explicit root (writing into the Hub snapshot
            # cache would corrupt it).
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=str(root),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes
                if num_cameras > 0
                else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(
                dataset, robot, cfg.dataset.fps, dataset_features
            )
        else:
            logging.info(f"Creating new dataset at {root}.")
            # NOTE: deliberately NOT calling cfg.dataset.stamp_repo_id() — we want
            # the bare repo_id path so the dataset stays resumable and resolvable
            # by compute_norm_stats / openpi.
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=str(root),
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * num_cameras,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                # One mp4 per episode (video only); see PER_EPISODE_VIDEO_FILE_SIZE_MB.
                # Float into an int-typed param is fine: the value is only used in a
                # >= size comparison, never cast to int.
                video_files_size_in_mb=PER_EPISODE_VIDEO_FILE_SIZE_MB,  # ty: ignore[invalid-argument-type]
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener(robot=robot)

        # Remember how many episodes already existed so we only export the ones
        # added this session as per-subtask debug clips (see finally block).
        episodes_before = dataset.num_episodes

        with VideoEncodingManager(dataset):
            # Home before the FIRST episode too (safety): the arm starts at a known
            # zero and the operator presses SPACE only once ready — same gate as
            # between episodes. Applies on fresh-create and on resume.
            home_active_sides(robot, cfg.play_sounds)
            wait_for_start(events, cfg.play_sounds)

            recorded_episodes = 0
            while (
                recorded_episodes < cfg.dataset.num_episodes
                and not events["stop_recording"]
            ):
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                print(
                    f">>> Task has {expected_subtasks} subtask(s): press 'y' "
                    f"{expected_subtasks - 1} time(s) (at the end of each subtask "
                    f"except the last)."
                )
                # Each episode's subtask numbering starts at 0 (first subtask);
                # 'y' presses during record_loop advance it. Reset here so it
                # covers the first episode, between-episode, and re-record paths
                # (all of which re-enter this record_loop call).
                robot.reset_subtask()
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                # <- : discard and re-record. Home + gate so the redo starts from
                # zero, just like a normal between-episode transition.
                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if not events["stop_recording"]:
                        home_active_sides(robot, cfg.play_sounds)
                        wait_for_start(events, cfg.play_sounds)
                    continue

                # Subtask-count validation: the operator must have marked EXACTLY
                # this task's subtasks — final index == expected-1 AND every index
                # 0..expected-1 actually got >=1 frame. This catches under-press
                # (missing high indices), over-press (extra index / final too high),
                # and a 0-frame middle subtask from a rapid double-'y'. A bad episode
                # is DISCARDED here so it never reaches the dataset (the convert can't
                # fix a wrong number of subtask marks). Mirrors the <- re-record path.
                subtasks_ok = (
                    robot.current_subtask == expected_subtasks - 1
                    and robot.subtask_indices_seen == set(range(expected_subtasks))
                )
                if not subtasks_ok:
                    log_say("Wrong subtask count, discarding episode", cfg.play_sounds)
                    events["exit_early"] = False
                    seen = sorted(robot.subtask_indices_seen)
                    print(
                        f"\n>>> DISCARDED: marked subtask indices {seen} (final "
                        f"index {robot.current_subtask}), but the task needs all of "
                        f"{list(range(expected_subtasks))}. Press 'y' exactly "
                        f"{expected_subtasks - 1} time(s), once at the end of each "
                        f"subtask except the last.\n"
                    )
                    dataset.clear_episode_buffer()
                    if not events["stop_recording"]:
                        home_active_sides(robot, cfg.play_sounds)
                        wait_for_start(events, cfg.play_sounds)
                    continue

                # save_episode() blocks until the episode's video is fully encoded
                # (streaming_encoding finishes/joins its encoder threads here), so
                # once it returns the episode is committed and on disk. Print a
                # clear confirmation BEFORE homing + the SPACE prompt so the
                # operator knows it's safe to continue. (Only the metadata-buffer
                # flush is deferred to finalize(), which runs on Esc / loop end.)
                dataset.save_episode()
                recorded_episodes += 1
                log_say("Episode saved and encoded", cfg.play_sounds)
                print(
                    f"\n>>> Episode saved & encoded — "
                    f"{dataset.num_episodes} episode(s) now in the dataset.\n"
                )

                # Home + SPACE gate between episodes (skipped after the final
                # episode or once Esc was pressed).
                if (
                    recorded_episodes < cfg.dataset.num_episodes
                    and not events["stop_recording"]
                ):
                    home_active_sides(robot, cfg.play_sounds)
                    wait_for_start(events, cfg.play_sounds)
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

            # Best-effort: after the dataset is finalized (videos encoded, parquet
            # flushed) export one debug mp4 per subtask per episode/camera for the
            # episodes added this session, split on the `subtask_index` column. A
            # failure here must never lose the just-recorded data, so it is wrapped.
            try:
                added = list(range(episodes_before, dataset.num_episodes))
                if added and not is_headless():
                    out_dir = resolve_dataset_root(cfg) / "videos_decomposed"
                    written = export_clips_for_dataset(
                        dataset, out_dir, label_source="subtask_index", episodes=added
                    )
                    print(
                        f">>> Wrote {len(written)} per-subtask debug clips to {out_dir}"
                    )
            except Exception as e:
                logging.warning(f"Per-subtask clip export failed (non-fatal): {e}")

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    # cfg is injected by @parser.wrap() from --config_path, so it has no arg here.
    record()  # ty: ignore[missing-argument]


if __name__ == "__main__":
    main()
