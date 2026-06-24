"""Export per-subtask video clips from a recorded/converted LeRobot dataset.

CycleVLA decomposes each episode into ordered subtasks. For debugging we want to
SEE each subtask on its own — one mp4 per subtask per episode per camera — at two
points:

  * raw teleop data: subtask boundaries come from the per-frame `subtask_index`
    column (operator pressed 'y'); and
  * post-processed convert output: boundaries come from the per-frame `task`
    string (which the convert sets to the subtask language). Watching these
    confirms the DROID-style no-op drops (idle motion vanishes) and the tail
    oversampling (last frame held/repeated).

The clips are written to a SEPARATE `videos_decomposed/` dir next to the dataset's
canonical `videos/` so LeRobot's strict `videos/<key>/chunk-*/file-*.mp4` layout
(and its loader) are untouched.

Encoding goes through the conda-provided `ffmpeg` binary via a rawvideo stdin
pipe, so there is no extra Python video dependency.

Run standalone on a finalized dataset:

  pixi run python -m distal.hardware.decompose_videos \
      --root data/cyclevla/real_robot_decomposed_progress --label-source subtask_index
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np


def scalar(x):
    """Unwrap a possibly list-wrapped metadata value (v3.0 stores some as [v])."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0]
    return x


def to_hwc_uint8(img) -> np.ndarray:
    """Coerce a LeRobot image (CHW float[0,1] torch tensor, or HWC uint8/float) to
    a contiguous HWC uint8 RGB numpy array suitable for rawvideo encoding."""
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    # CHW -> HWC when the leading axis is the channel one.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:  # grayscale -> RGB
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def write_clip(frames, fps, out_path) -> bool:
    """Encode HWC uint8 RGB frames to an mp4 via an ffmpeg rawvideo pipe.

    Returns True on success. Skips (returns False) when there are no frames."""
    frames = [to_hwc_uint8(f) for f in frames]
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(f.tobytes())
    finally:
        proc.stdin.close()
        ret = proc.wait()
    return ret == 0


def segment_runs(labels):
    """Split a per-frame label sequence into consecutive equal-label runs.

    Returns a list of (start, end_exclusive, label)."""
    runs = []
    start = 0
    n = len(labels)
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def safe_name(text: str, limit: int = 40) -> str:
    """Filesystem-safe token from an arbitrary label."""
    token = "".join(c if c.isalnum() else "_" for c in str(text))
    return token[:limit].strip("_") or "x"


def export_subtask_clips_from_frames(
    images_by_cam, labels, fps, out_dir, prefix, label_to_name=None
):
    """Write one mp4 per subtask run per camera.

    images_by_cam: {cam_name: [HWC-uint8-or-tensor frame, ...]} (all len == #frames).
    labels:        per-frame subtask label (int index or task string).
    Returns the list of written file paths.
    """
    out_dir = Path(out_dir)
    written = []
    runs = segment_runs(list(labels))
    for k, (s, e, lab) in enumerate(runs):
        name = label_to_name(lab) if label_to_name else lab
        for cam, frames in images_by_cam.items():
            out = out_dir / f"{prefix}_sub{k:02d}_{safe_name(name)}_{cam}.mp4"
            if write_clip(frames[s:e], fps, out):
                written.append(out)
    return written


def subtask_index_label_fn(dataset):
    """Build a per-frame label function reading `subtask_index` out of the
    `observation.state` vector (raw teleop datasets store it there)."""
    names = dataset.meta.features["observation.state"]["names"]
    # names may be nested like [["a", "b", ...]]; flatten one level if so.
    flat = names[0] if names and isinstance(names[0], (list, tuple)) else names
    idx = list(flat).index("subtask_index")

    def label_fn(frame):
        return int(round(float(frame["observation.state"][idx])))

    return label_fn


def task_label_fn(frame):
    """Per-frame label = the `task` string (converted datasets store the subtask
    language there)."""
    return frame["task"]


def export_clips_for_dataset(
    dataset, out_dir, label_source="subtask_index", episodes=None
):
    """Export per-subtask clips for (a subset of) episodes of a finalized dataset.

    label_source: "subtask_index" (raw teleop) or "task" (converted output)."""
    # Select image/video features by dtype so this works for both the raw dataset
    # (observation.images.top/left_wrist, video) and the converted one
    # (image/wrist_image, image).
    cam_keys = [
        k
        for k, v in dataset.meta.features.items()
        if v.get("dtype") in ("image", "video")
    ]
    if label_source == "subtask_index":
        label_fn = subtask_index_label_fn(dataset)
    elif label_source == "task":
        label_fn = task_label_fn
    else:
        raise ValueError(
            f"label_source must be 'subtask_index' or 'task', got {label_source!r}"
        )

    ep_indices = range(dataset.num_episodes) if episodes is None else episodes
    all_written = []
    for ep in ep_indices:
        meta = dataset.meta.episodes[ep]
        start = int(scalar(meta["dataset_from_index"]))
        end = int(scalar(meta["dataset_to_index"]))
        images_by_cam = {c: [] for c in cam_keys}
        labels = []
        for i in range(start, end):
            frame = dataset[i]
            labels.append(label_fn(frame))
            for c in cam_keys:
                # short cam name in the filename (drop the observation.images. prefix)
                images_by_cam[c].append(frame[c])
        short_cams = {c.split(".")[-1]: imgs for c, imgs in images_by_cam.items()}
        written = export_subtask_clips_from_frames(
            short_cams, labels, dataset.fps, out_dir, prefix=f"ep{ep:04d}"
        )
        all_written.extend(written)
    return all_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="Dataset root dir (contains meta/)."
    )
    parser.add_argument(
        "--label-source",
        choices=["subtask_index", "task"],
        default="subtask_index",
        help="Where subtask boundaries come from.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for clips (default: <root>/videos_decomposed).",
    )
    args = parser.parse_args()

    from lerobot.datasets import LeRobotDataset

    root = Path(args.root)
    # repo_id is only used for metadata bookkeeping; derive it from the path tail.
    repo_id = "/".join(root.parts[-2:])
    dataset = LeRobotDataset(repo_id, root=str(root))
    out_dir = Path(args.out_dir) if args.out_dir else root / "videos_decomposed"
    written = export_clips_for_dataset(dataset, out_dir, label_source=args.label_source)
    print(f"[decompose_videos] wrote {len(written)} clips to {out_dir}")


if __name__ == "__main__":
    main()
