# Real-robot CycleVLA eval (transit + full method)

Evaluate the trained PI0.5 / openpi CycleVLA policy on the real AgileX PiPER arm.
The policy is served by the **openpi WebSocket server** (run in
`cyclevla_code/openpi`, no edits there); the two distal scripts here are clients
that build observations from the arm + cameras, query the server, and apply the
9‑D `[ΔEEF, gripper, s_t, p_t]` actions via Cartesian `EndPoseCtrl`.

- `run_real_eval_openpi_transit.py` — transit-only baseline (drive each subtask,
  advance on the stop signal). Mirrors `run_libero_eval_openpi_transit.py`.
- `run_real_eval_openpi_cyclevla.py` — full method: 90% VLM check →
  `transit | backtrack`, with joint-replay backtrack + MBR re-decode. Mirrors
  `run_libero_eval_openpi_cyclevla.py`.

There is **no automatic success check** — inspect the saved rollout videos.

## Prerequisites

- The policy server is up (Step 1 below).
- **Power OFF (or unplug) the LEADER/master arm of the active set.** Eval drives
  only the follower (standalone CAN control). The AgileX master-slave pairing can't
  be cleared in software without a power-cycle, so a powered leader keeps pushing
  linkage frames and **fights the policy** (the two arms collide). Inference uses no
  master arm. (After eval, power-cycle the follower to restore teleop for `record`.)
- `single_task` in `configs/real_eval.yaml` is one of the keys in
  `distal/hardware/subtasks.py`.
- **Full method only:** `OPENAI_API_KEY` set in `.env` (repo root) for the VLM.

## Step 1 — serve the checkpoint (openpi `uv` env, GPU)

```bash
cd /home/kai/Projects/cyclevla_code/openpi
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=0 PORT=8000 \
CKPT_DIR=/home/kai/Projects/cyclevla_code/openpi/checkpoints/pi05_libero_cyclevla/CycleVLA_real_robot_decomposed_progress_pi05_A100/15000 \
scripts/serve_openpi_cyclevla.sh
```

The trailing `\` on each line are required — they join everything into ONE command
so the env vars apply to it. Without them, `XLA_...=...` on its own line is a
no-op assignment that never reaches the serve command. `CKPT_DIR` must be the
**step dir** (it contains `params/` + `assets/`); `20000` is the latest step
(`15000` shown here is also valid).

openpi serves via JAX, which preallocates ~75% of VRAM by default. `PREALLOCATE=false`
grows VRAM on demand and `MEM_FRACTION=0.9` caps the JAX pool at 90% of the GPU
(both inherited by the script's `uv run`). For the smallest footprint instead, use
`XLA_PYTHON_CLIENT_ALLOCATOR=platform`, which allocates/frees on demand and
overrides `MEM_FRACTION`.

## Step 2 — dry-run (no arm): connectivity + schema + action math

```bash
cd /home/kai/Projects/cyclevla_distal
pixi run python -m distal.hardware.run_real_eval_openpi_transit --config_path=configs/real_eval.yaml --dry_run true
```

Builds a synthetic observation, queries the server, and prints the parsed 9‑D
action + the EEF target it implies. Run this before touching hardware.

## Step 3 — on the arm

```bash
pixi run real-eval-transit        # baseline
pixi run real-eval-cyclevla       # full method (needs OPENAI_API_KEY)
```

(Equivalently: `pixi run python -m distal.hardware.run_real_eval_openpi_transit
--config_path=configs/real_eval.yaml`.)

### Per-episode loop (same controls as `pixi run record`)

1. At the start of each episode the arm **homes to joint-zero in CONTROL mode**
   (`ModeCtrl(CAN, MOVE_J)` + `JointCtrl(0…)`, NOT the teleop 0x191) — only the
   follower moves — then prints **"press SPACE to start"** (like `record.py`).
2. **Reset the scene**, then press **SPACE**. The follower switches to CAN /
   `MOVE_P` control (a `ctrl_mode` line logs `CAN-control 0x01` to confirm) and the
   policy drives **only the follower**; the leader stays idle.
3. The episode runs the subtask state machine (a per-step
   `t=… Δp=… Δr=… grip=… s_t=… p_t=…` line prints for debugging):
   - **transit**: each subtask completes on the robust stop-signal confirmation.
   - **cyclevla**: at ~90% progress a VLM decides `transit` (continue) or
     `backtrack` (replay recorded joints back to the subtask start, then retry
     with an MBR-ranked chunk); completion is the stop signal.
4. Press **`→`** to stop early (it also ends on completion or `max_steps`). The
   both-camera videos are saved and "saved & encoded" is logged.
5. Press **SPACE** to run the next episode (which re-homes first).

Other keys: **`←`** discards and re-records the current episode (no save);
**`Esc`** ends the whole session. **Ctrl-C** is a hard stop that still saves the
partial rollout.

**Only the follower moves.** The eval runs entirely in **control mode** (never
teleop) — it homes via `ModeCtrl(CAN)+JointCtrl(0)` and drives the follower via
`EndPoseCtrl`. The `[after set_eef_mode] ctrl_mode` line confirms the follower is in
`CAN-control (0x01)`. BUT a **powered leader still broadcasts master-slave linkage
frames** that the follower's CAN mode does NOT suppress (the `0xFC` slave pairing
can't be cleared in software without a power-cycle) — so the leader **must be powered
off** during eval (see Prerequisites), or the arms fight. After an eval session,
**power-cycle the follower before `pixi run record`** to restore teleop.
(`lerobot-rollout` drives the follower the same standalone-CAN way.)

> Headless (e.g. SSH with no display): the keyboard is unavailable — episodes
> start on **ENTER** and end on natural completion / `max_steps` / Ctrl-C.

## Outputs

- **Videos** (gitignored): `data/rollouts/<variant>/<task>/<stamp>--ep<N>--{image,wrist_image}_{raw,subtitled}.mp4`
  — `image` = top camera, `wrist_image` = wrist; `_subtitled` overlays the live
  subtask + transition tag (e.g. `VLM_90%_check`, `backtrack_retry1`).
- **Logs** (gitignored, sim-eval style): `data/rollouts/logs/real_eval_<variant>_<stamp>.log`
  — subtask transitions, stop/VLM/MBR decisions, backtracks, and per-episode
  "saved & encoded" summaries.

## Key knobs (`configs/real_eval.yaml`)

| Key | Meaning |
| --- | --- |
| `single_task` | High-level task; **must** be a `subtasks.py` key |
| `num_episodes` | Episodes to run this session |
| `home` | Home to joint-zero in CONTROL mode (ModeCtrl(CAN)+JointCtrl, no teleop) at each episode start |
| `host` / `port` | openpi policy server address |
| `num_open_loop_steps` | Actions executed before requerying (≤10) |
| `fps` / `max_steps` | Control rate / per-episode step budget |
| `eef_speed_rate` | `MotionCtrl_2` speed % — keep low at first |
| `delta_base` | `chunk` (accumulate Δ from EEF at requery) or `live` |
| `max_pos_step` / `max_rot_step` | Per-step ΔEEF safety clamps (m / rad). `max_rot_step≈1.0` — 0.3 over-clips the policy's rotation deltas → poor wrist motion |
| `invert_gripper` / `gripper_open` | Gripper polarity flip (see caveats) |
| `no_arm` | Diagnostic: cameras + server only, never command the arm |
| `camera_fps` | Override RealSense fps (e.g. 15) to cut USB bandwidth; `null` = 30 |
| `progress_threshold` | p_t level that triggers the VLM check (cyclevla) |
| `max_subtask_retries` | Backtrack/retry cap per subtask (cyclevla) |
| `vlm_model` / `mbr_num_seeds` | VLM model / MBR candidate count (cyclevla) |

## Cameras dropping (RealSense `status=False` / `TimeoutError`) — FIXED

**Root cause (bisected & confirmed):** the eval used to hold every full-res frame
in RAM for the end-of-episode video (`frames_by_cam` lists). That unbounded
accumulation starved the RealSense background read thread → `async_read` 200 ms
`TimeoutError` + `read failed (status=False)` (+ a `'NoneType' ... is_set` storm
from the reconnect race). Proven with `camera-check --save-frames` (drops) vs
`--stream-frames` (clean); every other suspect (homing, server load, idle gap,
listener) stayed clean in `camera-check`.

**Fix (applied):** the eval **streams frames straight to disk** each tick via
`RolloutVideoWriter` (raw + subtitled mp4 per camera, encode-and-drop) — nothing is
accumulated, so memory stays flat and the read thread is never starved. Confirm:
```bash
pixi run camera-check -- --seconds 30 --query-server --save-frames    # DROPS (old behavior)
pixi run camera-check -- --seconds 30 --query-server --stream-frames  # CLEAN (the fix)
```

**Deeper hardware note:** the wrist camera is on a **USB 2.0** link — the fragile
link that the old memory pressure tipped over. For durable robustness:
```bash
lsusb -t                                 # cameras vs the CAN adapter / link speed
pixi run lerobot-find-cameras realsense  # confirm each camera negotiates USB 3.x (not 2.1)
```
Prefer the **wrist RealSense on a USB3 port/controller**; or set `camera_fps: 15`
in `configs/real_eval.yaml` to halve its bandwidth.

## Confirmed working

A follower-only rollout works with: **leader arm powered off** + **`max_rot_step:
1.0`**. `[after set_eef_mode] ctrl_mode = CAN-control (0x01)` confirms the follower is
standalone; only the follower moves; cameras stream with no drops.

## Caveats (first runs)

- **Rotation clamp**: keep `max_rot_step ≈ 1.0`. The policy's rotation deltas are
  larger than position; `0.3` saturates rx/rz every step → erratic/under-rotated
  wrist. (Resolved by raising the clamp; no convention bug.)
- **Gripper polarity** vs sim is unverified (norm stats absorb scale). If
  open/close is reversed on the first rollout, set `invert_gripper: true`.
- **EEF tracking at 20 Hz**: start with a low `eef_speed_rate` and confirm the
  arm tracks the deltas smoothly before raising it.
- **Safety**: the follower runs autonomously in CAN-control during an episode;
  the leader must be powered off (see Prerequisites); keep an e-stop within reach.
