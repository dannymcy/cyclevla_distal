"""Stage B of the real-robot CycleVLA convert: NEUTRAL intermediate -> LeRobot v2.1.

Run this in the OPENPI environment (`uv run`), NOT the distal pixi env. openpi ships
LeRobot **v2.1** (`lerobot.common.datasets`), which cannot read the distal-produced
v3.0 dataset — so Stage A (`convert_to_cyclevla.py`, distal env) does all the
processing and writes a version-neutral intermediate (PNGs + manifest.parquet), and
this script writes the actual **v2.1** LeRobot dataset that
`compute_norm_stats --config-name pi05_libero_cyclevla` and training read.

This file is intentionally self-contained: it imports ONLY `lerobot.common.datasets`
(v2.1) + pyarrow + numpy + PIL, with NO `distal.*` imports, so it runs cleanly under
openpi's interpreter.

It mirrors the sim convert
(`openpi/examples/libero/convert_libero_data_to_lerobot_cyclevla.py`): same feature
schema (image/wrist_image as `image` dtype, state(8), actions(9), per-frame `task`)
and the same `create -> add_frame -> save_episode` flow. `image` dtype writes PNG
files (NOT embedded in parquet), so there is no parquet bloat.

Output goes to `$HF_LEROBOT_HOME/<repo_id>` (like the sim convert), so run with
HF_LEROBOT_HOME pointing at the distal data dir, e.g.:

  cd <openpi>
  HF_LEROBOT_HOME=<distal>/data uv run python \
      <distal>/distal/hardware/intermediate_to_v21.py \
      --intermediate <distal>/data/cyclevla/real_robot_decomposed_progress_intermediate
"""

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def decode_mp4(path):
    """Decode an mp4 into an ordered list of HWC uint8 RGB frames (via pyav)."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intermediate",
        required=True,
        help="Stage-A intermediate dir (meta.json + manifest.parquet + images/).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Output repo_id. Defaults to the intermediate meta.json `out_repo_id` "
        "(typically cyclevla/libero_decomposed_progress).",
    )
    parser.add_argument("--robot-type", default="piper")
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    args = parser.parse_args()

    # v2.1 LeRobot (openpi env). Importing here makes the failure message obvious if
    # this is accidentally run in the distal v3.0 env.
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    inter = Path(args.intermediate)
    meta = json.loads((inter / "meta.json").read_text())
    repo_id = args.repo_id or meta["out_repo_id"]
    fps = int(meta["fps"])
    cam_shapes = meta["cam_shapes"]  # {out_key: [H, W, 3]}

    features = {}
    for out_key, shape in cam_shapes.items():
        # VIDEO dtype (NOT image): v2.1 stores image dtype as PNG bytes embedded in
        # the parquet (struct<bytes,path>), which bloated the dataset to ~55 MB for
        # 200 frames. Video dtype writes one mp4 per episode and keeps the parquet
        # numeric-only. openpi/LeRobot decode video features back to frames on read.
        features[out_key] = {
            "dtype": "video",
            "shape": tuple(shape),
            "names": ["height", "width", "channel"],
        }
    features["state"] = {"dtype": "float32", "shape": (8,), "names": ["state"]}
    features["actions"] = {"dtype": "float32", "shape": (9,), "names": ["actions"]}
    cam_keys = list(cam_shapes.keys())

    # Write to $HF_LEROBOT_HOME/<repo_id>, replacing any prior attempt (mirrors sim).
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"[stage-b] removing existing {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=args.robot_type,
        fps=fps,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    # Group manifest rows by episode (preserve frame order within each episode).
    rows = pq.read_table(str(inter / "manifest.parquet")).to_pylist()
    by_ep = OrderedDict()
    for r in rows:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)

    total = 0
    for ep, ep_rows in by_ep.items():
        ep_rows.sort(key=lambda r: int(r["frame"]))
        # Decode this episode's per-camera mp4 back into frame lists; manifest frame
        # `i` == decoded frame `i` (Stage A wrote them in the same order).
        decoded = {
            cam: decode_mp4(inter / "videos" / f"ep{ep:04d}_{cam}.mp4")
            for cam in cam_keys
        }
        for cam, frames in decoded.items():
            if len(frames) != len(ep_rows):
                raise ValueError(
                    f"ep{ep}: {cam} mp4 has {len(frames)} frames but manifest has "
                    f"{len(ep_rows)} rows — intermediate is inconsistent."
                )
        for i, r in enumerate(ep_rows):
            frame = {
                "state": np.asarray(r["state"], dtype=np.float32),
                "actions": np.asarray(r["actions"], dtype=np.float32),
                "task": r["task"],
            }
            for cam in cam_keys:
                frame[cam] = decoded[cam][i]
            dataset.add_frame(frame)
            total += 1
        dataset.save_episode()
        print(f"[stage-b] ep{ep}: wrote {len(ep_rows)} frames")

    print(
        f"[stage-b] done: {len(by_ep)} episodes, {total} frames -> {output_path}\n"
        f"  Next: uv run scripts/compute_norm_stats.py "
        f"--config-name pi05_libero_cyclevla"
    )


if __name__ == "__main__":
    main()
