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
3. **Verify teleop data capture.** Confirm teleoperation saves correctly to
   `/home/kai/Projects/cyclevla_distal/data`. Check the record command and
   whether that path is gitignored. [ ]

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
1. **Maha stats** (`distal/rewards/maha_stats.py`) — From the base dataset the
   policy was trained on, fit Ledoit-Wolf mean / inverse covariance over
   mean-pooled VLM image-token embeddings. Saved as safetensors and cached on
   the HF Hub.
1. **Train value** (`distal/train_value.py`) — Distributional value model
   (`RECAPValueNetwork` in `distal/value_model.py`: SmolVLM + expert + learned
   value query token + categorical head, vision encoder frozen). Reward signal
   is either fixed `-1` per step or `distal/rewards/maha.py` (Mahalanobis-based
   `[-1, 0]` rewards). Adapted from the upstream LeRobot
   `jv/recap-value-network` PR.
1. **Train PiStar06** (`distal/train_pi_star.py`) — Advantage-conditioned Pi0.5
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
