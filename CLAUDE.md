# CycleVLA — Real Robot Experiment

Real-robot experiment for CycleVLA, built on top of **DistAL**. Goal: teleoperate
three tasks on the AgileX PiPER arm, train PI0.5 on the collected data, and eval
on the real robot.

The rig has two PiPER sets (each = leader + follower); we record with the
**left** set (`sides: [left]` in `record.yaml`) and leave the right set idle.
Reason: the left set runs firmware `S-V1.8-7`, while the right runs `S-V1.7-3`,
which predates the `ReqMasterArmMoveToHome` home command (CAN `0x191`, needs
`≥ V1.7-4`); the right set cannot be auto-homed until its firmware is flashed
(Windows `ArmRobotUA.exe` → Upgrade). Left stays the default until then.

**Read first:** `/home/kai/Projects/cyclevla_code/README.md` (CycleVLA entry
point), then `/home/kai/Projects/cyclevla_code/PI.md` (training steps).

## Workflow

1. Teleoperate three tasks with the AgileX PiPER arm; data auto-saves as LeRobot
   format to `/home/kai/Projects/cyclevla_distal/data`.
2. Run **Step 2 — Compute Normalization Statistics** in `PI.md` directly on the
   saved dataset, then continue from there to train PI0.5.
3. Eval on the real robot. Inference logic must match exactly:
   - `experiments/robot/libero/run_libero_eval_openpi_transit.py`
   - `experiments/robot/libero/run_libero_eval_openpi_cyclevla.py`

## Open TODOs

1. **Disable the second PiPER set.** [done]
   *Config-driven `sides` knob on `PiperConfig`/`PiperTeleoperatorConfig`; the
   idle set's CAN interface is never opened. `record.yaml` uses `sides: [left]`
   (right idle); flip to switch. Software-disable only.*
2. **Homing works.** [done]
   *`distal/hardware/zero.py` homes the active `sides` set to zero via the
   firmware return-to-zero `ReqMasterArmMoveToHome` (CAN `0x191`), which keeps
   the master-slave pairing intact. JointCtrl homing was removed (it breaks the
   pairing). Needs firmware `≥ V1.7-4`; the right set (`S-V1.7-3`) is too old —
   hence the left default above.*
3. **Verify teleop data capture.** [done]
   *`pixi run record` runs the local `distal/hardware/record.py` (not stock
   `lerobot-record`). It saves a LeRobot v3.0 dataset to
   `data/cyclevla/real_robot_decomposed_progress` (`data/` is gitignored). The
   custom loop: homes the active set and waits for SPACE before the first episode
   and between episodes (→ saves, ← discards+re-records, Esc stops+finalizes);
   prints a "saved & encoded" confirmation per episode. The dataset uses a
   stable, unstamped `repo_id` and auto create-or-resumes, so reruns append into
   one dataset (`num_episodes` = episodes to add this session) that
   compute_norm_stats resolves directly. Videos are one mp4 per episode per
   camera (`video_files_size_in_mb` set tiny for debuggability); the data parquet
   stays concatenated. NOTE: set `single_task` to the real instruction before
   collecting real data — it is baked into every frame.*
4. **Align the teleop dataset to the CycleVLA schema + PI0.5 finetune.** [done]
   *Feeds the read-only `pi05_libero_cyclevla` config (norm-stats → finetune) with
   ZERO `cyclevla_code` edits. Sim schema = absolute-EEF state(8) + delta-EEF
   action(9) `[ΔEEF(6),gripper,s_t,p_t]` + per-frame SUBTASK language; our teleop was
   joint-space(7) single-task. All changes distal-side:*
   - *`lerobot_robot_piper/piper.py`: record live EEF (`GetArmEndPoseMsgs`, →m/rad)
     + per-frame `subtask_index` (both fold into `observation.state`); joints kept.*
   - *`distal/hardware/record.py`: `y` = end-of-current-subtask (frame-accurate
     `bump_subtask`); episodes whose `y`-marks ≠ all K subtasks auto-discard +
     re-prompt. Exports raw per-subtask debug clips after each episode.*
   - *`distal/hardware/subtasks.py`: 3 PLACEHOLDER tasks (4/8/8 subtasks — revise
     before real data). `record.yaml single_task` must be one of these keys; a
     subtask whose text contains "gripper" triggers the ×8 tail-oversample.*
   - *Stage A `convert_to_cyclevla.py` (distal): raw→CycleVLA math — DROID no-op
     filter → fractional p_t 0.1→0.9 → tail oversample (gripper ×8 / else last-3 ×4,
     s_t=1/p_t=1.0) → EEF-delta actions + `[g,-g]` gripper, 20 fps. Reads raw via
     parquet (state) + ONE sequential video decode/episode (fast); writes a
     version-NEUTRAL intermediate (per-episode mp4 + `manifest.parquet` + `meta.json`),
     read-only on the source, skipping corrupt episodes. Exports post-process clips.*
   - *Stage B `intermediate_to_v21.py` (openpi env): writes the **LeRobot v2.1**
     dataset at repo_id `cyclevla/libero_decomposed_progress` under `$HF_LEROBOT_HOME`,
     `image`/`wrist_image` as **video dtype** (v2.1 image dtype embeds PNG bytes in
     the parquet → bloat; video keeps the parquet tiny).*
   - *`distal/hardware/decompose_videos.py`: shared per-subtask mp4 export.*
   *WHY TWO STAGES: openpi runs **LeRobot v2.1** (`tasks.jsonl`), distal records
   **v3.0** — a v2.1 reader can't load v3.0. So: process in distal (v3.0) → neutral
   intermediate → write v2.1 in openpi. Verified end-to-end (norm-stats + finetune).*

   **Workflow** (`<distal>`=this repo, `<openpi>`=cyclevla_code/openpi):
   1. *RECORD: set `record.yaml single_task` to a `subtasks.py` key → `pixi run
      record`. SPACE start; **`y` = end current subtask** (K subtasks → K-1 presses,
      in order); `→` save, `←` redo, `Esc` stop.*
   2. *STAGE A: `pixi run python -m distal.hardware.convert_to_cyclevla --src-root
      data/cyclevla/real_robot_decomposed_progress` → `..._intermediate/`
      (`--video-backend pyav` on hosts without CUDA/torchcodec).*
   3. *STAGE B: `cd <openpi> && HF_LEROBOT_HOME=<distal>/data uv run python
      <distal>/distal/hardware/intermediate_to_v21.py --intermediate
      <distal>/data/cyclevla/real_robot_decomposed_progress_intermediate` → v2.1
      `data/cyclevla/libero_decomposed_progress`.*
   4. *NORM-STATS + TRAIN (openpi, same `HF_LEROBOT_HOME`): `uv run
      scripts/compute_norm_stats.py --config-name pi05_libero_cyclevla`, then PI.md
      Step 3 with a DISTINCT `<exp_name>`; keep `action_horizon=10`.*

   **Gotchas:** *(1) video dtype needs a working video decoder at TRAIN time; the
   box's `torchcodec` fails if FFmpeg shared libs (libavutil.so.56–59 = FFmpeg 4–7,
   NOT 8) are missing — install a supported FFmpeg (`apt install ffmpeg`, or
   conda-forge `ffmpeg=7.*` on `LD_LIBRARY_PATH`). `uv pip uninstall torchcodec` does
   NOT stick (uv re-syncs); sim's image-dtype dataset never decodes video so it's
   unaffected. (2) `asset_id=repo_id`, so real `compute_norm_stats` OVERWRITES the sim
   norm-stats — back up the sim asset + use a distinct train run name. (3) Gripper
   open/close polarity vs sim unverified (norm stats absorb scale; confirm at eval).*

5. **Write evaluation scripts and launch on the real robot.** [done — transit
   baseline + cyclevla full method both run end-to-end]
   *Two distal-side clients to the openpi WebSocket server (ZERO `cyclevla_code`
   edits), mirroring `experiments/robot/libero/run_libero_eval_openpi_{transit,
   cyclevla}.py`:*
   - *`distal/hardware/run_real_eval_openpi_transit.py` — drive each subtask, advance
     on the robust stop-signal confirmation.*
   - *`distal/hardware/run_real_eval_openpi_cyclevla.py` — two-phase 90% VLM check →
     `transit`/`backtrack` (joint-replay rewind + MBR re-decode); needs
     `OPENAI_API_KEY` in `.env`.*
   - *`distal/hardware/real_eval_common.py` — shared client: builds the 8-D EEF state
     + two UNROTATED 224 images, queries the server, applies 9-D `[ΔEEF,grip,s_t,p_t]`
     via Cartesian `EndPoseCtrl` (`target=current+Δ`, additive-euler — exact inverse
     of `convert_to_cyclevla`). `configs/real_eval.yaml` + `pixi run real-eval-{transit,
     cyclevla}`; `pixi run camera-check` is the camera diagnostic.*
   - *Run from `cyclevla_code/openpi` first: `serve_openpi_cyclevla.sh` (CKPT_DIR =
     `.../CycleVLA_real_robot_decomposed_progress_pi05_A100/<step>`). Full operator
     guide: `distal/hardware/REAL_EVAL.md`.*

   **HARD-WON GOTCHAS (all resolved):**
   - ***Stream rollout frames to disk*** *(`RolloutVideoWriter`), NEVER accumulate in
     RAM — holding full-res frames starved the RealSense read thread → both cameras
     `status=False`. (Wrist cam is on a USB-2 link; prefer USB3 / `camera_fps:15`.)*
   - ***Control mode, not teleop.*** *Eval homes the follower via `ModeCtrl(CAN)+
     JointCtrl(0)` and drives `EndPoseCtrl` — it must NOT call the 0x191 master-slave
     home (that re-engages teleop; record keeps it). `report_ctrl_mode` logs
     `CAN-control (0x01)` to confirm. **POWER OFF the leader arm** during eval (the
     `0xFC` slave linkage can't be software-disabled without a power-cycle, so a
     powered leader fights the policy); power-cycle the follower before `record` again.*
   - ***`max_rot_step ≈ 1.0`*** *(default raised from 0.3, which over-clipped the
     policy's rx/rz rotation deltas).*
   - *Console+file logging; per-step action print; record.py-style SPACE/→/←/Esc loop.*

   **Norm-stats name-collision caveat still applies** (see TODO 4): real
   `compute_norm_stats` overwrites the sim asset (`asset_id=repo_id`); back up the sim
   asset + use a distinct train run name.

## DOs (Very Important)

1. Make a clear plan first, and ask questions before making changes.
2. This codebase is complex and tightly coupled.  
   Read and understand all relevant files before attempting any optimization or refactoring.
3. When modifying anything related to official specifications (e.g., robot parameters, hardware limits, kinematics), always verify using official documentation or reliable online resources.
4. Write clear and explicit comments in the code explaining **why** something is done a certain way—not just **what** is done.

---

# DistAL (Base Pipeline)
DistAL: a RECAP-style RL pipeline for fine-tuning Pi0.5 with advantage
conditioning and Mahalanobis-distance-based rewards, built on a fork of
HuggingFace LeRobot. Primary evaluation target is LIBERO simulation; also
supports a physical Piper arm.

**Python 3.12** (`>=3.12,<3.13`). **pixi** manages everything: conda-forge
system binaries (ffmpeg, imagemagick, python), PyPI deps (incl. git/path/index
overrides), environments (`default`, `hardware`), and tasks — see `pixi.toml`.
The pixi-managed env lives at `.pixi/envs/default`. `pyproject.toml` is a stub
for hatchling (build backend for the editable `distal` install) plus ruff
config.

## Common Commands

```bash
# Base policy training & evaluation
pixi run train                           # lerobot-train using configs/train.yaml
pixi run eval                            # lerobot-eval in LIBERO sim (pi05-libero default)

# RECAP pipeline (run directly via pixi, not lerobot-train)
pixi run python distal/collect.py               # rollouts → LeRobot dataset
pixi run python distal/collect_libero_plus.py
pixi run python distal/rewards/maha_stats.py    # mean / cov_inv from base-dataset embeddings
pixi run python distal/train_value.py           # distributional value network
pixi run python distal/train_pi_star.py         # advantage-conditioned Pi0.5 fine-tune
pixi run python distal/auroc.py                 # Mahalanobis / kNN AUROC vs episode success
pixi run python distal/eval_guidance.py         # sweep guidance scales

# Hardware (Piper)
pixi run record                          # teleop demos
pixi run rollout                         # play trained policy on the arm

# Cluster / cloud
pixi run sky [cluster_id]                # launch on Vast via SkyPilot, or sky exec on existing
pixi run sky-ssh [cluster_id]            # same but via SSH cloud
pixi run container                       # build container.sif and scp to HTC
pixi run slurm run                       # SLURM submit (slurm-tools git dep)
pixi run slurm gui [stop]                # Flask job-monitor daemon

# Quality
pixi run pre-commit run --all-files      # ruff (E,F,I + format), check-toml/yaml, mdformat --wrap 80, ty
```

No formal test suite — verification is via `pixi run eval` and the `auroc`
diagnostic.

## Conventions

- **Never start function or variable names with underscores.** Use plain names.
- **Don't add `Usage:` sections to module docstrings** — entry points use
  `draccus`/`lerobot.configs.parser`, which are self-documenting.
- **Never use OSMesa for MuJoCo rendering. Always EGL** (`MUJOCO_GL=egl`).
  OSMesa is too slow for policy evaluation.
- `slurm-tools` is a separate git repo (pulled as a git dependency); push
  changes to it from its own checkout.
- **Cluster jobs go through `pixi run slurm run`.** Override configs on the CLI
  rather than editing `configs/slurm.yaml`. Check job status at
  `localhost:5000/jobs` and view logs at `localhost:5000/logs/<job_id>` (both
  served by `pixi run slurm gui`).
- **SkyPilot API server caches code.** After patching SkyPilot source under
  `.pixi/envs/default/`, run `pixi run sky api stop` before retrying — the
  daemon keeps stale modules loaded.
- **PRs to external repos** (LeRobot fork etc.): check
  `.github/pull_request_template.md` and `CONTRIBUTING.md` first and follow
  their format.
- **Isambard AI Phase 2 (`u6jz.aip2.isambard`) home is 100 GiB; large dirs are
  symlinked to `$SCRATCHDIR=/scratch/u6jz/reece.u6jz` (5 TiB).** Active links:
  `~/distal/{.pixi,outputs,wandb}` and `~/.cache/{huggingface,rattler,triton}`.
  `.pixi` lives on scratch so pixi can hardlink (rather than copy) into it.
  Write new bulky outputs to scratch (or under an existing symlinked path) —
  never to a fresh dir under `~`.

## Architecture

### Pipeline

The system is a multi-stage pipeline; each stage produces an artifact consumed
by the next.

1. **Collect** (`distal/collect.py`, `distal/collect_libero_plus.py`) — Roll out
   a base policy in LIBERO via LeRobot's `eval_policy()`, save observations,
   actions, and per-episode `success` into a LeRobot dataset.
2. **Maha stats** (`distal/rewards/maha_stats.py`) — From the base dataset the
   policy was trained on, fit Ledoit-Wolf mean / inverse covariance over
   mean-pooled VLM image-token embeddings. Saved as safetensors and cached on
   the HF Hub.
3. **Train value** (`distal/train_value.py`) — Distributional value model
   (`RECAPValueNetwork` in `distal/value_model.py`: SmolVLM + expert + learned
   value query token + categorical head, vision encoder frozen). Reward signal
   is either fixed `-1` per step or `distal/rewards/maha.py` (Mahalanobis-based
   `[-1, 0]` rewards). Adapted from the upstream LeRobot
   `jv/recap-value-network` PR.
4. **Train PiStar06** (`distal/train_pi_star.py`) — Advantage-conditioned Pi0.5
   fine-tune. **Advantages are pre-computed in this script** by running the
   frozen value network once over the dataset, then injected into batches via a
   frame-index → advantage dict. Caching is content-addressed by
   `distal/advantage_cache.py` (key = dataset + VN commit SHAs +
   hyperparameters), with cache files mirrored to a HF Hub `dataset` repo. There
   is no separate `compute_advantage_labels` step — it lives inside this script.

### PiStar06 Plugin (`lerobot_policy_pistar06/`)

LeRobot plugin registering the **`pistar06`** policy type. PiStar06 = Pi0.5
(`PI05Config`) extended with binary advantage conditioning injected via
text/embedding into the action expert (`embed_suffix`). Built with flat
`nn.Module` composition rather than the deep PaliGemma inheritance chain to
avoid ~3× peak memory during init. Key config knobs: `value_network_checkpoint`,
`enable_advantage_conditioning` (master switch, persisted in `config.json` so
inference matches training), `advantage_threshold` (resolved scalar, typically
auto-set to a per-task percentile during training), `advantage_dropout` (CFG).

### Supporting modules (`distal/`)

- `value_model.py` — `RECAPValueNetwork` (SmolVLM + expert backbone, value query
  token, categorical head over discretized return bins).
- `rewards/maha.py` — Loads stats from `rewards/maha_stats.py`, computes per-
  frame Mahalanobis distances on a value-training dataset, min-max normalizes to
  `[-1, 0]` for use as per-step rewards. Local content-addressed cache under
  `HF_ASSETS_CACHE/distal/rewards/`.
- `rewards/knn.py` — Same shape as `rewards/maha.py`, but per-frame score is the
  mean L2/cosine distance to the k nearest base-policy demo embeddings (demo
  embeddings cached under `HF_ASSETS_CACHE/distal/demo_embs/`).
- `auroc.py` — Evaluates Mahalanobis or kNN distance as a failure predictor:
  per-frame distances → episode-mean → AUROC vs `success` labels.
- `advantage_cache.py` — Content-addressed cache for precomputed advantages,
  Hub-mirrored.
- `eval_guidance.py` — Sweeps classifier-free guidance scales by shelling out to
  `lerobot-eval`.
- `push_to_hub.py` — Upload checkpoints / value networks to HF Hub.
- `plotting/` — Diagnostic scripts: `plot_rewards.py`, `plot_returns.py`.
  `plot_returns.py` mirrors the exact reward/return construction in
  `train_value._build_frame_targets` so the plot reflects what the model
  actually trains against.
- `hardware/zero.py`, `hardware/can_activate.py` — Piper init / CAN bring-up.

### Hardware Plugins

- `lerobot_robot_piper/` — Piper arm (6-DOF + gripper, CAN bus) + 2× Intel
  RealSense D435 (wrist + scene, 640×480 @ 30fps). Platform-specific RealSense
  variants for macOS vs Linux.
- `lerobot_teleoperator_piper/` — Piper teleop interface for `pixi run record`.

Both packages are commented out from the `dev` group in `pyproject.toml`; sync
locally only when working on hardware.

### Configs (`configs/`)

YAML configs drive workflows via draccus / LeRobot config parsers:

- `train.yaml` — base Pi0.5 training (`pixi run train`).
- `eval.yaml` — LIBERO eval; **policy args must come from CLI**, e.g.
  `pixi run eval` overrides `--policy.path` and `--policy.n_action_steps`.
- `sky.yaml` / `sky-ssh.yaml` — SkyPilot launch configs (Vast / generic SSH),
  including LIBERO-plus assets bootstrap.
- `slurm.yaml` — HTC SLURM submission with Singularity bind mounts.
- `record.yaml` / `play.yaml` — Hardware workflows.

### Deployment

- **Singularity** (`container.def` → `container.sif`) for the HTC cluster
  (L40S/H100). Built and uploaded via `pixi run container`.
- **SkyPilot** (`configs/sky*.yaml`) targets Vast / RunPod / etc. The setup
  block bootstraps `pixi`, runs `pixi install`, and downloads LIBERO assets from
  the `Sylvest/LIBERO-plus` HF dataset.

### LeRobot fork

`pyproject.toml` pins `lerobot` to a custom fork:
`reeceomahoney/lerobot @ distal`. The fork exposes two simulation extras
side-by-side: `libero` (base suites via `hf-libero`, providing the `libero`
Python module) and `libero-plus` (perturbation suites via
`reeceomahoney/LIBERO-plus@distal-deps`, renamed to the `libero_plus` Python
module so both packages coexist). `lerobot/envs/libero.py::_libero_backend`
dispatches between them based on the `is_libero_plus` flag at the call site. The
two packages each read their own `~/.libero/config.yaml` /
`~/.libero_plus/config.yaml`, so concurrent slurm jobs (one base, one
libero-plus) no longer clobber each other.
