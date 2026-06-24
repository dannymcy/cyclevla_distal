"""Stage A of the real-robot CycleVLA convert: raw teleop -> NEUTRAL intermediate.

This is the real-robot analogue of the sim
`openpi/examples/libero/convert_libero_data_to_lerobot_cyclevla.py` PLUS the
Stage-3 RLDS builder's progress / tail-oversampling logic
(`LIBERO_Decomposed_Progress_dataset_builder.py`). It reshapes the data we record
with `pixi run record` (joint + EEF + gripper + per-frame `subtask_index`) into the
CycleVLA schema (state 8-D EEF+gripper, actions 9-D EEF-delta+gripper+s_t+p_t,
per-subtask language).

WHY TWO STAGES: openpi's environment runs **LeRobot v2.1**, but the distal hardware
stack (and this raw dataset) is **LeRobot v3.0** — a v2.1 reader cannot load a v3.0
dataset (it looks for `meta/tasks.jsonl` and 401s on the Hub). So this Stage A does
all the processing here (in the distal v3.0 env) and writes a small, version-neutral
intermediate; `intermediate_to_v21.py` (run in the openpi v2.1 env) then writes the
actual v2.1 LeRobot dataset that `compute_norm_stats`/training read.

Intermediate layout written to --out-root (compact; no per-frame PNG folder):
  meta.json                 fps, cam->shape map, cam list, repo_id hint, totals
  manifest.parquet          one row/frame: episode_index, frame, state(8),
                            actions(9), task  (NO image bytes/paths)
  videos/ep{e:04d}_{key}.mp4   per-episode processed video per camera; frame `i`
                            of the manifest == frame `i` of this mp4
  videos_decomposed/        per-subtask debug clips (optional)

Per-episode processing order:
  1. segment frames by `subtask_index` (the operator's 'y' marks);
  2. per subtask: DROID-style no-op filter (drop frames whose motion-to-next is
     below threshold), keeping the first and last frame;
  3. per subtask: fractional progress p_t (0.1..0.9 over the body) + tail
     oversampling (gripper subtask: last frame x8; else last 3 frames x4), tail
     frames flagged s_t=1, p_t=1.0 — exactly the sim builder's scheme;
  4. EEF-delta actions computed within the subtask (orientation deltas wrapped to
     (-pi, pi]); the gripper is kept RAW (no inversion), matching sim.

Run (distal env):
  pixi run python -m distal.hardware.convert_to_cyclevla \
      --src-root data/cyclevla/real_robot_decomposed_progress
then Stage B (openpi env):
  cd <openpi> && HF_LEROBOT_HOME=<distal>/data uv run python \
      <distal>/distal/hardware/intermediate_to_v21.py \
      --intermediate <distal>/data/cyclevla/real_robot_decomposed_progress_intermediate
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from distal.hardware import subtasks
from distal.hardware.decompose_videos import (
    export_subtask_clips_from_frames,
    write_clip,
)

# Default camera key -> CycleVLA image key. `top` is the third-person/base view,
# `left_wrist` the wrist view (sim uses base_0_rgb / left_wrist_0_rgb downstream).
DEFAULT_CAM_MAP = {
    "observation.images.top": "image",
    "observation.images.left_wrist": "wrist_image",
}

EEF_AXES = ("x", "y", "z", "rx", "ry", "rz")


def flatten_names(names):
    """LeRobot may store feature `names` nested one level ([[...]]); flatten it."""
    if names and isinstance(names[0], (list, tuple)):
        return list(names[0])
    return list(names)


def resolve_state_indices(state_names, side):
    """Locate the EEF (6), gripper (1) and subtask_index (1) columns inside the raw
    `observation.state` vector by name. Returns (eef_idx[6], grip_idx, subtask_idx)."""
    names = flatten_names(state_names)
    eef_idx = [names.index(f"{side}_eef_{ax}.pos") for ax in EEF_AXES]
    grip_idx = names.index(f"{side}_gripper.pos")
    subtask_idx = names.index("subtask_index")
    return eef_idx, grip_idx, subtask_idx


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi] so orientation deltas don't blow up at the
    +/-pi seam."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def segment_runs(labels):
    """Split per-frame labels into consecutive equal-value runs: (start, end, label)."""
    runs = []
    start = 0
    n = len(labels)
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def noop_keep_mask(eef, grip, eps_pos, eps_ori, eps_grip):
    """DROID-style idle filter within one subtask: drop a frame whose motion FROM
    the previous KEPT frame is below threshold (pos AND ori AND gripper), so idle
    holds collapse. The first and last frames are always kept (the last anchors the
    oversampled tail). Returns a boolean keep-mask over the subtask frames.

    Compared against the previous KEPT frame (not index-1) so a long slow drift
    isn't dropped frame-by-frame while a true static hold still collapses.
    """
    n = len(eef)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    last = 0
    for k in range(1, n):
        dpos = np.linalg.norm(eef[k, :3] - eef[last, :3])
        dori = np.linalg.norm(wrap_angle(eef[k, 3:6] - eef[last, 3:6]))
        dgrip = abs(grip[k] - grip[last])
        if dpos >= eps_pos or dori >= eps_ori or dgrip >= eps_grip:
            keep[k] = True
            last = k
    keep[-1] = True  # always anchor the tail
    return keep


def compute_actions(eef, grip):
    """Per-frame action over a (filtered) subtask: 6D EEF delta to the NEXT frame
    (orientation wrapped) + next-frame gripper (raw, absolute). The final frame has
    no successor, so its delta/gripper are forward-filled from the previous step
    (matches the sim builder copying the last demonstrated action). Returns
    (deltas[L,6], gripper_action[L])."""
    n = len(eef)
    deltas = np.zeros((n, 6), dtype=np.float32)
    grip_act = np.asarray(grip, dtype=np.float32).copy()
    for k in range(n - 1):
        d = eef[k + 1] - eef[k]
        d[3:6] = wrap_angle(d[3:6])
        deltas[k] = d
        grip_act[k] = grip[k + 1]
    if n >= 2:
        deltas[n - 1] = deltas[n - 2]  # forward-fill last
        grip_act[n - 1] = grip[n - 1]  # last commanded gripper = its own state
    return deltas, grip_act


def body_progress(k, frames_for_progress):
    """Fractional progress p_t for body frame k (sim builder formula): rises 0.1..0.9
    over the body, discretized in 0.1 bins."""
    if frames_for_progress <= 0:
        return 0.5
    raw = (k + 1) / frames_for_progress
    return float(min(0.9, max(0.1, round(raw * 10) / 10)))


def state_vector(eef_row, grip_value):
    """8-D state = [6D EEF pose, 2D gripper]. Piper has one gripper DOF; we fill the
    2D slot as [g, -g] to mirror LIBERO's symmetric two-finger gripper state.
    Polarity vs sim is recomputed by norm stats and adapted by finetuning."""
    g = float(grip_value)
    return np.concatenate(
        [np.asarray(eef_row, dtype=np.float32), np.array([g, -g], dtype=np.float32)]
    ).astype(np.float32)


def action_vector(delta6, grip_act, s_t, p_t):
    """9-D action = [6D EEF delta, gripper, s_t, p_t]."""
    return np.concatenate(
        [
            np.asarray(delta6, dtype=np.float32),
            np.array([float(grip_act), float(s_t), float(p_t)], dtype=np.float32),
        ]
    ).astype(np.float32)


def process_subtask(eef, grip, images, subtask_str, eps_pos, eps_ori, eps_grip):
    """Turn one subtask's frames into the CycleVLA output frames (body + oversampled
    tail). `images` is {out_image_key: [HWC uint8 frame, ...]} aligned with eef/grip.
    Returns a list of output frame dicts ready for `add_frame` (minus `task`)."""
    keep = noop_keep_mask(eef, grip, eps_pos, eps_ori, eps_grip)
    idx = np.nonzero(keep)[0]
    eef_f = eef[idx]
    grip_f = grip[idx]
    img_f = {key: [frames[i] for i in idx] for key, frames in images.items()}
    L = len(eef_f)

    deltas, grip_act = compute_actions(eef_f, grip_f)

    is_gripper = "gripper" in subtask_str.lower()
    frames_to_oversample = 1 if is_gripper else min(3, L)
    frames_for_progress = L - frames_to_oversample

    out = []

    def emit(k, s_t, p_t):
        frame = {key: frames[k] for key, frames in img_f.items()}
        frame["state"] = state_vector(eef_f[k], grip_f[k])
        frame["actions"] = action_vector(deltas[k], grip_act[k], s_t, p_t)
        out.append(frame)

    # Body: progress 0.1..0.9, not a stop.
    for k in range(frames_for_progress):
        emit(k, s_t=0.0, p_t=body_progress(k, frames_for_progress))

    # Oversampled tail: stop=1, progress=1.0. gripper subtask repeats the last
    # frame x8; otherwise repeats the last `frames_to_oversample` frames x4 each.
    if is_gripper:
        for _ in range(8):
            emit(L - 1, s_t=1.0, p_t=1.0)
    else:
        for k in range(L - frames_to_oversample, L):
            for _ in range(4):
                emit(k, s_t=1.0, p_t=1.0)

    return out


def meta_scalar(x):
    """Unwrap a possibly list-wrapped v3.0 metadata value."""
    return x[0] if isinstance(x, (list, tuple, np.ndarray)) else x


def decode_video(path):
    """Sequentially decode an mp4 -> list of HWC uint8 RGB frames (one fast pass,
    no per-frame seeking)."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def load_raw_states(src_root):
    """Read `observation.state` for every frame straight from the raw data
    parquet(s), grouped by episode and ordered by `frame_index`. Returns
    {episode_index: (N, D) float32}. This avoids LeRobot's per-frame __getitem__,
    which decodes the video on every index (the Stage-A bottleneck)."""
    import pyarrow.parquet as pq

    files = sorted((src_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet under {src_root / 'data'}")
    by_ep = {}
    for f in files:
        t = pq.read_table(
            f, columns=["observation.state", "episode_index", "frame_index"]
        )
        for ep, fi, st in zip(
            t.column("episode_index").to_pylist(),
            t.column("frame_index").to_pylist(),
            t.column("observation.state").to_pylist(),
        ):
            by_ep.setdefault(int(ep), []).append((int(fi), st))
    out = {}
    for ep, rows in by_ep.items():
        rows.sort(key=lambda r: r[0])
        out[ep] = np.asarray([r[1] for r in rows], dtype=np.float32)
    return out


def build_episode_frames(src_root, ep_meta, ep, state_arr, cam_map, idxs, eps, fps):
    """Build one episode's processed CycleVLA frames from pre-read state + a single
    sequential video decode per camera. Returns (frames, n_raw, n_subtasks). Raises
    if a video is missing/corrupt so the caller can skip that episode.

    `idxs` = (eef_idx[6], grip_idx, subtask_idx) into `observation.state`;
    `eps` = (eps_pos, eps_ori, eps_grip)."""
    eef_idx, grip_idx, subtask_idx = idxs
    n = len(state_arr)
    eef = state_arr[:, list(eef_idx)]
    grip = state_arr[:, grip_idx]
    sub = np.rint(state_arr[:, subtask_idx]).astype(np.int64)

    tasks = ep_meta["tasks"]
    high_level = tasks[0] if isinstance(tasks, (list, tuple, np.ndarray)) else tasks

    # One sequential decode per camera; slice this episode's frames by its
    # from_timestamp offset (== 0 for our one-file-per-episode recordings).
    images = {}
    for sk, ok in cam_map.items():
        vc = int(meta_scalar(ep_meta[f"videos/{sk}/chunk_index"]))
        vf = int(meta_scalar(ep_meta[f"videos/{sk}/file_index"]))
        from_ts = float(meta_scalar(ep_meta[f"videos/{sk}/from_timestamp"]))
        vpath = src_root / "videos" / sk / f"chunk-{vc:03d}" / f"file-{vf:03d}.mp4"
        frames = decode_video(vpath)
        off = int(round(from_ts * fps))
        frames = frames[off : off + n]
        if len(frames) != n:
            raise ValueError(
                f"ep{ep}: {sk} yielded {len(frames)} frames for {n} state rows "
                f"(video {vpath} inconsistent)."
            )
        images[ok] = frames

    runs = segment_runs(list(sub))
    ep_frames = []
    for s, e, label in runs:
        if label >= len(subtasks.get_subtasks(high_level)):
            logging.warning(
                f"ep{ep}: subtask_index {label} >= #subtasks for "
                f"{high_level!r}; clamping to last subtask."
            )
        subtask_str = subtasks.subtask_for_index(high_level, label)
        seg_imgs = {key: frames[s:e] for key, frames in images.items()}
        for fr in process_subtask(eef[s:e], grip[s:e], seg_imgs, subtask_str, *eps):
            fr["task"] = subtask_str
            ep_frames.append(fr)
    return ep_frames, n, len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root", required=True, help="Raw teleop dataset root (contains meta/)."
    )
    parser.add_argument(
        "--src-repo-id",
        default=None,
        help="Defaults to the last two path components of --src-root.",
    )
    parser.add_argument(
        "--out-repo-id",
        default="cyclevla/libero_decomposed_progress",
        help="Target repo_id hint stored in the intermediate meta.json (Stage B "
        "writes the v2.1 dataset under this id).",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Intermediate output dir. Defaults to a sibling of --src-root named "
        "<src-name>_intermediate.",
    )
    parser.add_argument(
        "--side", default="left", help="Active arm side whose EEF/gripper to use."
    )
    parser.add_argument(
        "--eps-pos", type=float, default=5e-4, help="No-op position threshold (m)."
    )
    parser.add_argument(
        "--eps-ori", type=float, default=5e-3, help="No-op orientation threshold (rad)."
    )
    parser.add_argument(
        "--eps-grip", type=float, default=1e-3, help="No-op gripper threshold."
    )
    parser.add_argument(
        "--no-clips", action="store_true", help="Skip per-subtask debug clip export."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output dataset first.",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        help="LeRobot video decode backend for reading the source/output videos "
        "(e.g. 'pyav' or 'torchcodec'). Default auto-selects; use 'pyav' on hosts "
        "without the CUDA libs torchcodec needs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets import LeRobotDataset

    src_root = Path(args.src_root)
    src_repo_id = args.src_repo_id or "/".join(src_root.parts[-2:])
    src = LeRobotDataset(
        src_repo_id, root=str(src_root), video_backend=args.video_backend
    )

    cam_map = {k: v for k, v in DEFAULT_CAM_MAP.items() if k in src.meta.features}
    missing = [k for k in DEFAULT_CAM_MAP if k not in src.meta.features]
    if missing:
        logging.warning(
            f"Source missing expected cameras {missing}; available: "
            f"{[k for k in src.meta.features if k.startswith('observation.images.')]}"
        )
    if not cam_map:
        raise ValueError("No mappable cameras found in the source dataset.")

    eef_idx, grip_idx, subtask_idx = resolve_state_indices(
        src.meta.features["observation.state"]["names"], args.side
    )

    # Default intermediate dir: a sibling of the raw dataset (never nested inside it).
    out_root = (
        Path(args.out_root)
        if args.out_root
        else src_root.parent / f"{src_root.name}_intermediate"
    )

    # Hard safety guard: NEVER let --overwrite delete the raw teleop dataset. Refuse
    # if the output resolves to the source, or either path is nested inside the
    # other (e.g. a mistaken --out-root pointing at/inside the raw dataset).
    src_resolved = src_root.resolve()
    out_resolved = out_root.resolve()
    overlaps = (
        out_resolved == src_resolved
        or src_resolved in out_resolved.parents
        or out_resolved in src_resolved.parents
    )
    if overlaps:
        raise ValueError(
            f"Refusing to run: --out-root ({out_resolved}) overlaps the source "
            f"dataset ({src_resolved}). Choose an output path outside the raw "
            f"teleop dataset so it can never be deleted."
        )

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_root)
    vid_root = out_root / "videos"
    vid_root.mkdir(parents=True, exist_ok=True)

    cam_shapes = {
        out_key: list(src.meta.features[src_key]["shape"])
        for src_key, out_key in cam_map.items()
    }
    manifest_rows = []  # one dict per output frame
    total_in = total_out = 0

    eps = (args.eps_pos, args.eps_ori, args.eps_grip)
    idxs = (eef_idx, grip_idx, subtask_idx)
    # Read all per-frame state once from the data parquet (no video decode), then
    # decode each episode's video in a single sequential pass below.
    raw_states = load_raw_states(src_root)
    skipped = 0
    for ep in range(src.num_episodes):
        try:
            if ep not in raw_states:
                raise ValueError("no state rows in the data parquet")
            ep_frames, n_raw, n_subs = build_episode_frames(
                src_root, src.meta.episodes[ep], ep, raw_states[ep],
                cam_map, idxs, eps, src.fps,
            )
        except Exception as e:
            # One corrupt/missing video (e.g. a dropped recording) shouldn't kill
            # the whole convert — skip it and keep the good episodes.
            logging.warning(f"ep{ep}: failed to read/process ({e}); skipping.")
            skipped += 1
            continue
        total_in += n_raw

        # Write the processed frames as ONE mp4 per camera (compact; no PNG folder)
        # plus a numeric-only manifest row per frame. Manifest frame `i` lines up
        # with frame `i` of each per-episode mp4, which Stage B decodes back.
        for out_key in cam_map.values():
            write_clip(
                [fr[out_key] for fr in ep_frames],
                src.fps,
                vid_root / f"ep{ep:04d}_{out_key}.mp4",
            )
        for i, fr in enumerate(ep_frames):
            manifest_rows.append(
                {
                    "episode_index": ep,
                    "frame": i,
                    "state": [float(x) for x in fr["state"]],
                    "actions": [float(x) for x in fr["actions"]],
                    "task": fr["task"],
                }
            )
            total_out += 1

        # Per-subtask debug clips from the processed frames (task = subtask label).
        if not args.no_clips and ep_frames:
            imgs_by_cam = {
                out_key: [fr[out_key] for fr in ep_frames]
                for out_key in cam_map.values()
            }
            labels = [fr["task"] for fr in ep_frames]
            export_subtask_clips_from_frames(
                imgs_by_cam, labels, src.fps, out_root / "videos_decomposed",
                prefix=f"ep{ep:04d}",
            )

        logging.info(
            f"ep{ep}: {n_raw} raw frames -> {len(ep_frames)} processed "
            f"({n_subs} subtasks)"
        )

    # Write the manifest parquet + meta.json (the version-neutral intermediate that
    # Stage B `intermediate_to_v21.py` consumes in the openpi v2.1 env).
    pq.write_table(
        pa.Table.from_pylist(manifest_rows), str(out_root / "manifest.parquet")
    )
    (out_root / "meta.json").write_text(
        json.dumps(
            {
                "fps": int(src.fps),
                "cams": list(cam_map.values()),
                "cam_shapes": cam_shapes,
                "out_repo_id": args.out_repo_id,
                "total_episodes": int(src.num_episodes - skipped),
                "total_frames": int(total_out),
            },
            indent=2,
        )
    )
    logging.info(
        f"[convert] Stage A done: {src.num_episodes - skipped}/{src.num_episodes} "
        f"episodes ({skipped} skipped), {total_in} raw -> {total_out} processed "
        f"frames. Intermediate at {out_root}\n"
        f"  Next (openpi env): uv run python "
        f"distal/hardware/intermediate_to_v21.py --intermediate {out_root}"
    )


if __name__ == "__main__":
    main()
