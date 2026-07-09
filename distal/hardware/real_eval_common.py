"""Shared library for the real-robot CycleVLA eval (transit + cyclevla).

Real-robot counterpart of the read-only sim eval glue in
`cyclevla_code/experiments/robot/openpi_utils.py` +
`.../libero/libero_utils.py::save_rollout_video_decomposed`. The 9-dim PI0.5
policy is served by the openpi WebSocket server (run inside
`cyclevla_code/openpi`, with NO edits there); everything here is the CLIENT side
that runs on the Piper rig:

  * builds the observation the policy was trained on (8-D EEF state + two camera
    images) and queries the server for action chunks,
  * applies the returned 9-D ``[ΔEEF(6), gripper, s_t, p_t]`` actions on the arm
    via Cartesian ``EndPoseCtrl`` (``target = current_eef + Δ``; no IK — the
    firmware solves it),
  * records per-step joint positions so the cyclevla variant can BACKTRACK
    deterministically by replaying joints in reverse (``JointCtrl``),
  * saves both-camera rollout videos (raw + subtitled) to a gitignored dir.

CRITICAL schema facts — these MUST mirror
`distal/hardware/convert_to_cyclevla.py` exactly, because the policy was trained
on data that converter produced:

  * State (8-D) = ``[eef_x, eef_y, eef_z (m), eef_rx, eef_ry, eef_rz (euler rad),
    g, -g]`` (see ``state_vector`` there). The six EEF fields and the gripper
    already exist in ``robot.get_observation()`` (piper.py converts the raw SDK
    units to m / rad; gripper to its /1e4 unit).
  * Action (9-D) = ``[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper, s_t, p_t]``. The deltas
    were built as consecutive-frame differences with the ORIENTATION delta a
    plain wrapped euler difference (``compute_actions`` -> ``wrap_angle``), NOT a
    rotation composition. So the inverse at inference is additive euler:
    ``target = current + Δ``, ``target[3:6] = wrap_angle(target[3:6])``.
  * Images: ``top`` -> ``image``, ``{side}_wrist`` -> ``wrist_image``, resized
    with aspect-pad to 224. Fed UNROTATED: the 180° flip in the sim
    ``get_libero_image`` is a LIBERO artifact; the real training videos were
    stored unrotated (the converter applies no rotation).

The openpi gripper / stop / progress conventions (raw RLDS gripper, raw stop
``~{0,1}``, raw progress ``~[0.1, 1.0]``, thresholded ``> 0.5`` / ``>= 0.9``)
match `openpi_utils.py` and are reproduced by ``split_action`` below.
"""

import datetime
import logging
import os
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# openpi-client import. The package is pure-Python (websockets.sync + a bundled
# msgpack_numpy that only needs `msgpack` + numpy). Its pyproject pins
# numpy<2.0.0, but the distal env runs numpy 2.x, so we deliberately DO NOT add
# it as a managed dependency (that would force a numpy downgrade and break
# lerobot/torch). Instead we import its source directly, reusing the distal env's
# numpy. Override the location with OPENPI_CLIENT_SRC if the checkout moves.
# ---------------------------------------------------------------------------
DEFAULT_OPENPI_CLIENT_SRC = (
    "/home/kai/Projects/cyclevla_code/openpi/packages/openpi-client/src"
)


def ensure_openpi_client() -> None:
    """Make `openpi_client` importable (installed, or via OPENPI_CLIENT_SRC)."""
    try:
        import openpi_client  # noqa: F401

        return
    except ImportError:
        pass
    src = os.environ.get("OPENPI_CLIENT_SRC", DEFAULT_OPENPI_CLIENT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import openpi_client  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Could not import `openpi_client`. Looked on sys.path and at "
            f"{src!r}. Install the openpi-client package or set OPENPI_CLIENT_SRC "
            f"to <openpi>/packages/openpi-client/src."
        ) from e


# ---------------------------------------------------------------------------
# Constants mirroring openpi_utils.py / convert_to_cyclevla.py.
# ---------------------------------------------------------------------------
RESIZE_SIZE = 224  # pi05 LIBERO trains on 224x224 images
ACTION_DIM = 9  # [ΔEEF(6), gripper, s_t, p_t]
EEF_AXES = ("x", "y", "z", "rx", "ry", "rz")

# Piper SDK control constants (see piper_sdk interface_v2 + piper.py send_action).
CTRL_MODE_STANDBY = 0x00
CTRL_MODE_CAN = 0x01  # CAN command control (required for autonomous EndPose/Joint)
MOVE_MODE_P = 0x00  # point-to-point pose (MOVE_P)
MOVE_MODE_J = 0x01  # joint interpolation (MOVE_J)

# Unit conversions. piper.py reads EEF as 0.001mm->m (*1e-6) and 0.001deg->rad;
# we invert that to command EndPoseCtrl, and scale the gripper exactly like
# piper.py send_action (action * 1e4 -> GripperCtrl units).
M_TO_ENDPOSE = 1e6  # meters -> 0.001 mm
GRIPPER_SCALE = 1e4  # obs/action gripper unit -> GripperCtrl units
# GripperCtrl effort (0.001 N·m; SDK range 0-5000 = 0-5 N·m). Raised from the old
# hardcoded 1000 (1 N·m), which under-grips (partial close on objects) and
# under-opens on reset. Config `gripper_effort` overrides this per-run. Defined here
# (above command_eef/command_joints) so it is in scope for their default args.
GRIPPER_EFFORT = 3000


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi] — identical to convert_to_cyclevla.wrap_angle."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Observation construction (real-robot analogue of build_openpi_observation).
# ---------------------------------------------------------------------------
def eef_from_obs(obs, side):
    """6-D absolute EEF pose [x,y,z (m), rx,ry,rz (euler rad)] from a robot obs."""
    return np.array([obs[f"{side}_eef_{ax}.pos"] for ax in EEF_AXES], dtype=np.float64)


def gripper_from_obs(obs, side):
    """Scalar gripper position in the same unit the converter/training used."""
    return float(obs[f"{side}_gripper.pos"])


def joints_from_obs(obs, side):
    """6 joint positions in degrees (piper.py reports SDK 0.001deg -> deg)."""
    return np.array(
        [obs[f"{side}_joint_{i}.pos"] for i in range(1, 7)], dtype=np.float64
    )


def build_state(eef, gripper):
    """8-D state = [EEF(6), g, -g] — mirrors convert_to_cyclevla.state_vector.

    The [g, -g] fills LIBERO's symmetric two-finger slot; norm stats absorb the
    scale and finetuning adapts polarity, so we pass the raw gripper value.
    """
    g = float(gripper)
    return np.concatenate(
        [np.asarray(eef, dtype=np.float32), np.array([g, -g], dtype=np.float32)]
    ).astype(np.float32)


def build_observation(
    obs, prompt, side, top_key="top", wrist_key=None, resize=RESIZE_SIZE
):
    """Build the dict the openpi LIBERO policy server expects.

    Keys match `LiberoInputs`: observation/image, observation/wrist_image,
    observation/state, prompt. Images are NOT rotated (unlike the sim helper) —
    the real training videos were stored unrotated.
    """
    from openpi_client import image_tools

    wrist_key = wrist_key or f"{side}_wrist"
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs[top_key], resize, resize)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs[wrist_key], resize, resize)
    )
    state = build_state(eef_from_obs(obs, side), gripper_from_obs(obs, side))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(prompt),
    }


def split_action(action):
    """Split a 9-D openpi action into (delta6, gripper, stop, progress).

    Counterpart of openpi_utils.split_openpi_action, but keeps the EEF delta (6)
    and gripper (1) separate since we command the arm in Cartesian space rather
    than passing a 7-D vector to env.step(). No gripper inversion (openpi trains
    on the raw gripper); dims 7/8 are the raw stop / progress floats.
    """
    a = np.asarray(action, dtype=np.float32)
    assert a.shape[-1] == ACTION_DIM, (
        f"Expected a {ACTION_DIM}-dim openpi action (serve config "
        f"pi05_libero_cyclevla, action_dim=9), got shape {a.shape}."
    )
    return a[:6].astype(np.float64), float(a[6]), float(a[7]), float(a[8])


def apply_delta(base_eef, delta6):
    """target = base + Δ, with the orientation additive-euler wrapped to (-pi, pi].

    This is the exact inverse of the converter's delta construction (additive
    euler difference, wrapped), so commanding `target` reconstructs the absolute
    EEF trajectory the policy's deltas encode.
    """
    target = np.asarray(base_eef, dtype=np.float64).copy()
    target[:3] += np.asarray(delta6[:3], dtype=np.float64)
    target[3:6] = wrap_angle(target[3:6] + np.asarray(delta6[3:6], dtype=np.float64))
    return target


def clip_delta(delta6, max_pos, max_rot):
    """Per-dim safety clamp on a single ΔEEF step (m / rad). Defaults are generous
    (real deltas are tiny); this only guards against an occasional wild output
    yanking the arm. Set max_pos/max_rot <= 0 to disable."""
    d = np.asarray(delta6, dtype=np.float64).copy()
    if max_pos and max_pos > 0:
        d[:3] = np.clip(d[:3], -max_pos, max_pos)
    if max_rot and max_rot > 0:
        d[3:6] = np.clip(d[3:6], -max_rot, max_rot)
    return d


# ---------------------------------------------------------------------------
# Policy client.
# ---------------------------------------------------------------------------
class PolicyClient:
    """Thin wrapper over openpi's WebsocketClientPolicy.

    Connecting BLOCKS until the policy server (serve_policy.py / the
    serve_openpi_cyclevla.sh helper, config pi05_libero_cyclevla) is reachable.
    """

    def __init__(self, host="0.0.0.0", port=8000):
        ensure_openpi_client()
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port)
        logger.info(f"[openpi] Connected to policy server at ws://{host}:{port}")

    def get_action(self, obs_dict, n):
        """Query the server with an already-built obs dict; return up to n actions.

        The server returns an `action_horizon` (10) chunk; we return the first n
        (== num_open_loop_steps) so the caller's queue fills exactly. The server
        is stochastic per call, so repeated calls on the same obs yield diverse
        chunks (used by the cyclevla MBR sampler).
        """
        chunk = np.asarray(self._client.infer(obs_dict)["actions"])
        n = min(n, len(chunk))
        return [chunk[i] for i in range(n)]


class AsyncFetch:
    """Single-flight background `get_action`, so the control loop never blocks on
    the network.

    The eval keeps the reference's exact query → execute N open-loop → requery
    semantics; the ONLY change is that the requery happens in a background thread
    while the main loop keeps reading cameras (the arm just holds) — which is what
    stops the RealSense frames from dropping (a blocking main loop made the camera
    reads irregular). At most one fetch is in flight at a time.
    """

    def __init__(self, client: PolicyClient, n: int):
        self._client = client
        self._n = n
        self._lock = threading.Lock()
        self._result = None  # list[np.ndarray] | Exception | None
        self._busy = False
        self._thread = None

    def request(self, obs_dict) -> bool:
        """Start a fetch if none is in flight. Returns True if it started."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._result = None
        self._thread = threading.Thread(
            target=self._run, args=(obs_dict,), daemon=True, name="async-fetch"
        )
        self._thread.start()
        return True

    def _run(self, obs_dict):
        try:
            res = self._client.get_action(obs_dict, self._n)
        except Exception as e:  # noqa: BLE001 — surfaced to the caller via take()
            res = e
        with self._lock:
            self._result = res
            self._busy = False

    def ready(self) -> bool:
        with self._lock:
            return self._result is not None

    def take(self):
        """Return the fetched chunk (and clear it); re-raises a fetch error."""
        with self._lock:
            res = self._result
            self._result = None
        if isinstance(res, Exception):
            raise res
        return res

    def stop(self):
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Hardware control helpers. `arm` is a piper_sdk C_PiperInterface_V2 instance
# (robot.arms[side]).
# ---------------------------------------------------------------------------
def set_eef_mode(arm, speed_rate):
    """Put the arm in CAN-command + MOVE_P so EndPoseCtrl drives it.

    NOTE: this BREAKS the master-slave teleop linkage (the follower switches to
    pure CAN control) — which is exactly what we want for autonomous eval. Home
    via zero.home_master_slave (which needs master-slave mode) BEFORE calling
    this; restore_teleop() puts it back afterwards.
    """
    arm.MotionCtrl_2(CTRL_MODE_CAN, MOVE_MODE_P, int(speed_rate), 0x00)


def set_joint_mode(arm, speed_rate):
    """Put the arm in CAN-command + MOVE_J so JointCtrl drives it (backtrack)."""
    arm.MotionCtrl_2(CTRL_MODE_CAN, MOVE_MODE_J, int(speed_rate), 0x00)


def command_eef(
    arm, target_eef, gripper, gripper_clip=(0, 100000), effort=GRIPPER_EFFORT
):
    """Command an absolute EEF pose (m / euler rad) + gripper.

    Inverts piper.py's obs scaling: m -> 0.001mm (*1e6), rad -> deg -> 0.001deg
    (*1e3). Gripper scaled exactly like send_action (action * 1e4).
    """
    x = int(round(target_eef[0] * M_TO_ENDPOSE))
    y = int(round(target_eef[1] * M_TO_ENDPOSE))
    z = int(round(target_eef[2] * M_TO_ENDPOSE))
    rx = int(round(np.rad2deg(target_eef[3]) * 1e3))
    ry = int(round(np.rad2deg(target_eef[4]) * 1e3))
    rz = int(round(np.rad2deg(target_eef[5]) * 1e3))
    arm.EndPoseCtrl(x, y, z, rx, ry, rz)
    g_units = int(round(float(gripper) * GRIPPER_SCALE))
    g_units = max(gripper_clip[0], min(gripper_clip[1], g_units))
    arm.GripperCtrl(g_units, effort, 0x01, 0)


def command_joints(
    arm, joints_deg, gripper=None, gripper_clip=(0, 100000), effort=GRIPPER_EFFORT
):
    """Command 6 joint angles (deg -> SDK 0.001deg), optionally the gripper too.

    Mirrors piper.py send_action's JointCtrl scaling (no action_bias applied —
    we replay measured joints, which already include any bias)."""
    j = [int(round(float(v) * 1000.0)) for v in joints_deg]
    arm.JointCtrl(*j)
    if gripper is not None:
        g_units = int(round(float(gripper) * GRIPPER_SCALE))
        g_units = max(gripper_clip[0], min(gripper_clip[1], g_units))
        arm.GripperCtrl(g_units, effort, 0x01, 0)


def restore_teleop(arm):
    """Best-effort return to master-slave teleop mode (mirrors zero.py mode=0)."""
    try:
        arm.ReqMasterArmMoveToHome(0)
    except Exception as e:  # noqa: BLE001 — cleanup must not mask the real error
        logger.warning(f"restore_teleop failed: {e}")


def standby(arm):
    """Best-effort stop: put the arm in standby so it holds position."""
    try:
        arm.MotionCtrl_2(CTRL_MODE_STANDBY, MOVE_MODE_P, 0, 0x00)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"standby failed: {e}")


def backtrack_joints(
    arm, joint_hist, target_idx, fps, speed_rate, grip_hist=None, on_step=None
):
    """Rewind the arm by replaying recorded joints in reverse to `target_idx`.

    `joint_hist[k]` is the joint vector (deg) measured just before executing step
    k; `grip_hist[k]` the matching gripper. We step from the latest recorded pose
    back down to (and including) `target_idx`, the recorded START of the subtask
    we are backtracking to. Deterministic and safe: every pose was physically
    visited during this rollout. Leaves the arm in EEF (MOVE_P) mode for the
    retry.

    `on_step`, if given, is called once per reverse step (after the move + pace
    sleep) — the cyclevla eval uses it to stream a camera frame per step so the
    physical rewind motion is captured in the rollout video instead of appearing
    as a jump.
    """
    if target_idx < 0:
        target_idx = 0
    set_joint_mode(arm, speed_rate)
    period = 1.0 / fps
    for k in range(len(joint_hist) - 1, target_idx - 1, -1):
        g = grip_hist[k] if grip_hist is not None else None
        command_joints(arm, joint_hist[k], gripper=g)
        time.sleep(period)
        if on_step is not None:
            on_step()
    set_eef_mode(arm, speed_rate)


# ---------------------------------------------------------------------------
# Rollout video saving (real-robot analogue of save_rollout_video_decomposed):
# both cameras, each as a raw + subtitled mp4, to a gitignored dir.
# ---------------------------------------------------------------------------
_FONT_CACHE: dict = {}


def _get_font(size):
    from PIL import ImageFont

    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        except Exception:  # noqa: BLE001
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def overlay_subtitle(frame, text):
    """Return an RGB ndarray with `text` drawn in a bottom-right box (same layout
    as the sim libero_utils overlay). Used per-frame by the streaming writer."""
    from PIL import Image, ImageDraw

    arr = np.asarray(frame)
    w = arr.shape[1]
    scale = w / 256.0
    font = _get_font(max(10, int(10 * scale)))
    max_width_px = int(150 * scale)
    padding = int(5 * scale)
    pil = Image.fromarray(arr).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    lines = textwrap.wrap(str(text), width=25)
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2
    total_h = line_h * max(1, len(lines))
    x = pil.width - max_width_px - padding
    y = pil.height - total_h - padding
    draw.rectangle(
        [x - 2, y - 2, pil.width - padding + 2, pil.height - padding + 2],
        fill=(255, 255, 255, 180),
    )
    for j, line in enumerate(lines):
        draw.text((x, y + j * line_h), line, fill=(0, 0, 0, 255), font=font)
    return np.array(Image.alpha_composite(pil, overlay).convert("RGB"))


def _write_video(path, frames, fps, subtitles=None):
    """Write one mp4 from a LIST of frames (used by the legacy non-streaming
    save_rollout_videos). The eval uses RolloutVideoWriter (streaming) instead."""
    import imageio

    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    for i, img in enumerate(frames):
        if subtitles:
            writer.append_data(overlay_subtitle(img, subtitles[i]))
        else:
            writer.append_data(np.asarray(img))
    writer.close()


class RolloutVideoWriter:
    """STREAM rollout frames straight to disk — never accumulate in RAM.

    Holding every full-res frame in RAM (the old `frames_by_cam` lists) starves the
    RealSense read thread and drops the cameras (confirmed via camera_check
    --save-frames). This writer opens a raw + subtitled mp4 per camera up front and
    encodes each frame as it arrives, so at most one frame is in flight.
    """

    def __init__(
        self,
        save_dir,
        episode_idx,
        task,
        fps,
        variant,
        cam_keys=("image", "wrist_image"),
    ):
        import imageio

        task_clean = task.lower().replace(" ", "_").replace(".", "_")[:50]
        stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        out_dir = os.path.join(save_dir, variant, task_clean)
        os.makedirs(out_dir, exist_ok=True)
        self._writers = {}
        self._paths = []
        for cam in cam_keys:
            base = os.path.join(out_dir, f"{stamp}--ep{episode_idx}--{cam}")
            for kind in ("raw", "subtitled"):
                path = f"{base}_{kind}.mp4"
                self._writers[(cam, kind)] = imageio.get_writer(
                    path, fps=fps, macro_block_size=1
                )
                self._paths.append(path)
        self._closed = False
        self.num_frames = 0

    def add(self, frames: dict, subtitle: str):
        """Encode one tick: raw frame + subtitled frame per camera. Holds nothing."""
        for cam, frame in frames.items():
            arr = np.asarray(frame)
            self._writers[(cam, "raw")].append_data(arr)
            self._writers[(cam, "subtitled")].append_data(
                overlay_subtitle(arr, subtitle)
            )
        self.num_frames += 1

    def _close_writers(self):
        if self._closed:
            return
        for w in self._writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        self._closed = True

    def close(self):
        self._close_writers()
        for p in self._paths:
            logger.info(f"Saved rollout video: {p}")
        return self._paths

    def discard(self):
        """Close + delete the files (for the `<-` rerecord path)."""
        self._close_writers()
        for p in self._paths:
            try:
                os.remove(p)
            except OSError:
                pass
        logger.info("Discarded rollout videos (rerecord).")


def save_rollout_videos(
    frames_by_cam, subtitles, save_dir, episode_idx, task, fps, variant
):
    """Save raw + subtitled mp4s for every camera in `frames_by_cam`.

    Args:
        frames_by_cam: {"image": [HWC uint8...], "wrist_image": [...]} (full-res
            camera frames, NOT the 224 policy input).
        subtitles: per-frame caption list (current subtask + transition tag).
        save_dir: root dir (gitignored, e.g. data/rollouts).
        variant: "transit" | "cyclevla", folded into the filename.
    """
    task_clean = task.lower().replace(" ", "_").replace(".", "_")[:50]
    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = os.path.join(save_dir, variant, task_clean)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for cam_key, frames in frames_by_cam.items():
        if not frames:
            continue
        base = os.path.join(out_dir, f"{stamp}--ep{episode_idx}--{cam_key}")
        raw_path = f"{base}_raw.mp4"
        _write_video(raw_path, frames, fps)
        written.append(raw_path)
        if subtitles:
            sub_path = f"{base}_subtitled.mp4"
            _write_video(sub_path, frames, fps, subtitles=subtitles)
            written.append(sub_path)
    for p in written:
        logger.info(f"Saved rollout video: {p}")
    return written


# ---------------------------------------------------------------------------
# Stop / progress robust confirmation (shared by both scripts). Mirrors the
# counter logic in the reference run loops exactly.
# ---------------------------------------------------------------------------
class SignalConfirmer:
    """Robust thresholded-signal confirmation used for stop (s_t) and the 90%
    progress (p_t) checks. Returns True once the signal is CONFIRMED: two
    consecutive highs, OR a high recurring after >=2 low steps. Reset per
    subtask / per phase."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.first_high_seen = False
        self.consecutive_high = 0
        self.steps_since_high = 0

    def update(self, value):
        if value > self.threshold:
            self.consecutive_high += 1
            if not self.first_high_seen:
                self.first_high_seen = True
            confirmed = self.consecutive_high >= 2 or (
                self.first_high_seen and self.steps_since_high >= 2
            )
            self.steps_since_high = 0
            return confirmed
        self.consecutive_high = 0
        if self.first_high_seen:
            self.steps_since_high += 1
        return False


# ---------------------------------------------------------------------------
# Shared config (draccus). transit ignores the cyclevla-only fields.
# ---------------------------------------------------------------------------
@dataclass
class RealEvalConfig:
    # fmt: off
    # ---- policy server ----
    host: str = "0.0.0.0"                 # openpi websocket policy server host
    port: int = 8000                      # ... port
    num_open_loop_steps: int = 5          # actions executed before requery (<=10)

    # ---- task / robot ----
    # single_task MUST be a key in distal/hardware/subtasks.py.
    single_task: str = "hang the green teapot on the mug holder"
    sides: list[str] = field(default_factory=lambda: ["left"])  # sides[0] driven
    fps: int = 20                         # control loop rate (matches training fps)
    max_steps: int = 600                  # hard per-episode step budget
    num_steps_wait: int = 0               # settle steps before first action (~0 real)
    num_episodes: int = 1                 # episodes to run this session
    home: bool = True                     # firmware home (master-slave) before run
    restore_teleop_on_exit: bool = True   # return to master-slave teleop at the end

    # ---- EEF control / safety ----
    eef_speed_rate: int = 30              # MotionCtrl_2 speed % (0-100); low at first
    # delta_base: "chunk" = accumulate Δ from EEF at each requery; "live" = on live EEF.
    delta_base: str = "chunk"
    max_pos_step: float = 0.05            # per-step ΔEEF pos clamp (m); <=0 off
    # per-step ΔEEF rot clamp (rad); 1.0 verified — 0.3 over-clips the policy's
    # rotation deltas (saturates rx/rz) → poor wrist motion. <=0 disables.
    max_rot_step: float = 1.0
    invert_gripper: bool = False  # flip gripper polarity (verify at first run)
    # Measured gripper obs range in the training data (data/cyclevla/
    # libero_decomposed_progress, confirmed by norm_stats.json): bimodal — CLOSED/grasp
    # ≈ 4.3, FULLY OPEN ≈ 9.8, valley at ~7.0. The policy outputs an absolute stroke
    # that GripperCtrl drives the gripper TO (there is no ±1 command).
    gripper_open: float = 10.0  # obs FULL OPEN (~9.8); hysteresis open target
    gripper_closed: float = 0.0  # obs CLOSE target (0 = firm full close)
    # binarize close trigger: g <= this → full close, g > this → pos-control.
    # Sits in the antimode (~7.0) between the close (~4.3) / open (~9.8) clusters.
    # Hysteresis uses gripper_open_above/gripper_close_below instead.
    gripper_close_threshold: float = 7.0
    # binarize = FIRM-CLOSE-ONLY hybrid: below gripper_close_threshold → full close
    # (gripper_closed); above → normal pos-control (raw stroke, opening reaches ~9.8).
    gripper_binarize: bool = True
    # Third mode (precedence over gripper_binarize): dual-threshold HYSTERESIS.
    # Latch OPEN when g >= gripper_open_above, CLOSED when g <= gripper_close_below,
    # else HOLD the last state. Mirrors LIBERO's binarize_gripper_actions (threshold at
    # the extremes, carry state between): robust and non-brittle (a hold-band, not one
    # point) — unlike a prev-vs-current delta, which drifts on gradual reopens.
    # Precedence: hysteresis > binarize > raw. See GripperResolver.
    gripper_hysteresis: bool = False
    gripper_open_above: float = 8.5  # hyst: latch OPEN at/above (~9.8 cluster)
    gripper_close_below: float = 5.5  # hyst: latch CLOSED at/below (~4.3 cluster)
    # Reset/home open width (obs), decoupled from gripper_open so the home pose and the
    # per-step open target tune separately. Training full-open (~9.8) so the arm STARTS
    # open (the state the policy expects); see home_robot.
    gripper_reset_open: float = 10.0
    # GripperCtrl effort (0.001 N·m, 0-5000); raised from the old 1000 (1 N·m) which
    # under-grips / under-opens. See GRIPPER_EFFORT.
    gripper_effort: int = 3000
    # Diagnostic: run cameras + server queries but NEVER command the arm (no
    # set_eef_mode / EndPoseCtrl / backtrack). Isolates whether the arm's USB-CAN
    # writes are what drops the RealSense cameras.
    no_arm: bool = False
    # Override the RealSense capture fps (e.g. 15) to cut USB bandwidth; None keeps
    # the configured 30. Only affects eval, not the record defaults.
    camera_fps: Optional[int] = None

    # ---- output ----
    video_dir: str = "data/rollouts"      # gitignored (data/ is in .gitignore)
    log_dir: str = "data/rollouts/logs"

    # ---- dry run (no arm): synthetic obs -> query server -> print action ----
    dry_run: bool = False

    # ---- cyclevla-only (ignored by the transit script) ----
    progress_threshold: float = 0.90      # p_t level that triggers the VLM check
    # Seconds to HOLD the idle arm after the 90% check fires, BEFORE capturing the
    # image sent to the VLM. Lets the operator manually perturb the object to force
    # a backtrack while debugging. 0 = immediate capture (no hold). The hold is
    # recorded to the rollout video.
    vlm_check_delay_sec: float = 2.0
    max_subtask_retries: int = 3          # backtrack/retry cap per subtask
    vlm_model: str = "gpt-5.5"            # VLM for the transit/backtrack decision
    vlm_temperature: float = 1.0
    mbr_num_seeds: int = 8                # candidate chunks sampled per backtrack
    mbr_distance_metric: str = "l2"       # l2 | l1 | cosine | correlation | chebyshev
    mbr_use_failed_repulsion: bool = False
    mbr_r_neighborhood: Optional[int] = None
    mbr_vanilla: bool = False
    # fmt: on


def active_side(cfg: RealEvalConfig) -> str:
    return cfg.sides[0]


def resolve_gripper_command(cfg: RealEvalConfig, gripper_value: float) -> float:
    """Map the policy's raw gripper output to a GripperCtrl command (obs units).

    Two optional transforms, applied in order so they compose:
      1. invert (--invert-gripper): flip open/close polarity vs sim as
         g = gripper_open - g. Sim vs real sign is unverified (norm stats absorb
         scale); off by default.
      2. binarize (--gripper-binarize): FIRM-CLOSE-ONLY hybrid. Below the close
         threshold (policy clearly closing) snap to a full close (gripper_closed);
         ABOVE it fall through to normal pos-control (the raw stroke). Pos-control alone
         only reaches the ~4.3 grasp-width close, so this forces a firm full grasp while
         opening stays faithful (pos-control reaches ~9.8 ≈ full on its own). Threshold
         sits in the ~7.0 valley between the close (~4.3) / open (~9.8) clusters."""
    g = float(gripper_value)
    if cfg.invert_gripper:
        g = cfg.gripper_open - g
    if cfg.gripper_binarize and g <= cfg.gripper_close_threshold:
        g = cfg.gripper_closed  # firm full close; above threshold → raw pos-control
    return g


class GripperResolver:
    """Per-episode, stateful gripper open/close resolver — the single choke point both
    eval loops call each step. Three modes (precedence hysteresis > binarize > raw):

      - raw / threshold: delegate to the stateless resolve_gripper_command (unchanged).
      - hysteresis (cfg.gripper_hysteresis): latch OPEN when g >= gripper_open_above,
        CLOSED when g <= gripper_close_below, else HOLD the latched state, then command
        gripper_open / gripper_closed. A hold-band (not a single midpoint) keyed off the
        clean bimodal absolute signal — robust and drift-free (validated on eval logs:
        99.2% agreement with the absolute signal, 0% drift), unlike a prev-vs-current
        delta which sticks closed on gradual reopens.

    Seed `open` per episode: the arm is homed open, so the latch starts open."""

    def __init__(self, cfg: RealEvalConfig, seed_open: bool = True):
        self.cfg = cfg
        self.open = seed_open

    def resolve(self, gripper_value: float) -> float:
        cfg = self.cfg
        if not cfg.gripper_hysteresis:
            return resolve_gripper_command(cfg, gripper_value)
        g = float(gripper_value)
        if cfg.invert_gripper:
            g = cfg.gripper_open - g
        if g >= cfg.gripper_open_above:
            self.open = True
        elif g <= cfg.gripper_close_below:
            self.open = False
        # else: within the band → hold the latched state
        return cfg.gripper_open if self.open else cfg.gripper_closed


def setup_logging(cfg: RealEvalConfig, variant: str):
    """Log INFO to BOTH the console and a per-run file; returns the file path.

    The console handler is what makes eval progress (and openpi's "Waiting for
    server..." while it connects) visible on the terminal — without it the root
    logger suppresses INFO on screen and everything would only land in the file.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path = os.path.join(cfg.log_dir, f"real_eval_{variant}_{stamp}.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Console handler (a StreamHandler that is NOT a FileHandler) — add once so
    # repeated setup_logging calls don't double-print.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return path


def make_robot(cfg: RealEvalConfig):
    """Construct + return a connected Piper in autonomous (non-teleop) mode.

    teleop_mode=False is set for correctness, but we never call send_action (it is
    joint-only); we drive the arm via the SDK directly (EndPoseCtrl / JointCtrl).
    If cfg.camera_fps is set, the RealSense cameras are reconfigured to that fps to
    reduce USB bandwidth (eval-only; does not change the record defaults).
    """
    import dataclasses

    from lerobot_robot_piper.config_piper import PiperConfig
    from lerobot_robot_piper.piper import Piper

    piper_cfg = PiperConfig(sides=list(cfg.sides), teleop_mode=False)
    if cfg.camera_fps is not None:
        piper_cfg.cameras = {
            key: dataclasses.replace(cam, fps=int(cfg.camera_fps))
            for key, cam in piper_cfg.cameras.items()
        }
        logger.info(f"Cameras set to {cfg.camera_fps} fps (USB bandwidth relief).")
    robot = Piper(piper_cfg)
    robot.connect()
    return robot


class CameraReadError(RuntimeError):
    """Raised when robot.get_observation() keeps failing (camera bus dropped) even
    after reconnect attempts — caught by the episode loop to save the partial
    rollout and stop cleanly instead of crashing with a traceback."""


def reconnect_cameras(robot):
    """Disconnect + reconnect every camera so the next read starts from a fresh,
    streaming pipeline.

    The eval's idle gap (homing + the SPACE wait + the operator resetting the
    scene) leaves the cameras unserviced by the foreground; the RealSense pipeline
    stalls and the FIRST read of the episode then times out on both cameras at
    once. `connect()` warms up (pulls frames) before returning, which mirrors the
    known-clean soak pattern (fresh connect -> immediate continuous reads).

    Called at EPISODE START while the read threads are still HEALTHY, so the
    disconnect's join succeeds cleanly and we never hit the wedged-thread race in
    `read_observation` (a thread stuck in `try_wait_for_frames` outlives the 2 s
    join, leaving `stop_event = None` -> the `'NoneType'.is_set()` storm)."""
    for key, cam in getattr(robot, "cameras", {}).items():
        try:
            cam.disconnect()
        except Exception:  # noqa: BLE001 — already-disconnected is fine
            pass
        try:
            cam.connect()  # warms up (pulls frames) before returning
            logger.info(f"Camera {key} re-warmed for episode start.")
        except Exception as ce:  # noqa: BLE001
            logger.warning(f"Camera {key} re-warm failed: {ce}")


def read_observation(robot, retries=3, reconnect=True):
    """robot.get_observation() with a retry-first recovery for RealSense hiccups.

    A USB hiccup makes the background read thread miss a frame and `async_read`
    times out. The thread normally resumes delivering frames on its own once the
    bus settles, so we RETRY the plain read first (no reconnect). Only if every
    retry still fails do we fall back to a disconnect+reconnect (`reconnect_cameras`).

    Why retry-first: reconnecting while a read thread is wedged inside
    `try_wait_for_frames(timeout_ms=10000)` races the upstream 2 s join — it sets
    `stop_event = None` under a live thread and stacks a second pipeline on the
    device, producing the `AttributeError: 'NoneType' ... is_set` storm and more
    `status=False`. Retrying first lets transient drops self-heal without ever
    triggering that path. Raises CameraReadError if it never recovers."""
    last_err = None
    for attempt in range(retries):
        try:
            return robot.get_observation()
        except (TimeoutError, RuntimeError) as e:
            last_err = e
            logger.warning(
                f"get_observation failed (attempt {attempt + 1}/{retries}): {e}"
            )
            # Retry the plain read first; only reconnect as a last resort, once the
            # cheap retries are exhausted (the thread usually self-heals by then).
            if reconnect and attempt == retries - 2:
                reconnect_cameras(robot)
            time.sleep(0.2)
    raise CameraReadError(f"Camera read failed after {retries} attempts: {last_err}")


def record_while_busy(
    robot, writer, top_key, wrist_key, subtitle, fps, fn, *args, **kwargs
):
    """Run a slow BLOCKING `fn(*args)` in a background thread while STREAMING live
    camera frames to `writer`, so idle time (the VLM query, MBR sampling) is
    captured in the rollout video instead of showing as a freeze. Returns fn's
    result (re-raises its exception on the caller's thread).

    The arm is NOT commanded here — it holds its last pose, which is exactly what
    we want to record. Thread-safety: the caller is blocked inside this function,
    so the main loop issues no camera/client calls meanwhile, and `fn` (VLM or
    MBR) never touches the cameras or the arm — the worker and this frame-recording
    loop share no mutable state.
    """
    result, error = {}, {}

    def worker():
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 — surfaced to the caller below
            error["value"] = e

    thread = threading.Thread(target=worker, daemon=True)
    period = 1.0 / fps
    thread.start()
    while thread.is_alive():
        loop_start = time.perf_counter()
        try:
            o = read_observation(robot)
            writer.add({"image": o[top_key], "wrist_image": o[wrist_key]}, subtitle)
        except CameraReadError as e:  # a transient drop must not abort the wait
            logger.warning(f"Camera read failed during idle recording: {e}")
        time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
    thread.join()
    if "value" in error:
        raise error["value"]
    return result.get("value")


# Fallback gripper open on reset, in GripperCtrl units (0.001 mm). home_robot passes
# cfg.gripper_open*GRIPPER_SCALE instead (single source of truth) so reset opens as wide
# as the training data's full-open (~9.8 → 98000), not the old conservative 70 mm.
HOME_GRIPPER_OPEN = 100_000


def home_follower(
    arm, speed, repeats=5, settle=0.5, open_gripper=True, effort=GRIPPER_EFFORT,
    open_units=HOME_GRIPPER_OPEN,
):
    """Home ONE arm to joint-zero in CONTROL mode (NOT teleop / 0x191).

    Mirrors the rollout / user homing snippet: ModeCtrl(CAN, MOVE_J) then repeated
    JointCtrl(0,...) + GripperCtrl. This drives ONLY this arm and BREAKS the
    master-slave pairing (so the leader stays idle during eval) — the opposite of
    zero.py's `home_master_slave`, which keeps teleop for record. Note: control mode
    breaks the pairing until a power cycle, so power-cycle before `pixi run record`.
    """
    arm.ModeCtrl(0x01, 0x01, int(speed), 0x00)  # CAN control + MOVE_J
    for _ in range(repeats):
        arm.JointCtrl(0, 0, 0, 0, 0, 0)  # home to joint zero (this arm only)
        if open_gripper:
            arm.GripperCtrl(int(open_units), effort, 0x01, 0)
        time.sleep(settle)


def home_robot(robot, cfg: RealEvalConfig):
    """Home the active arm(s) to joint-zero in CONTROL mode (no teleop / 0x191).

    Eval runs entirely in control mode so ONLY the follower moves; the leader stays
    idle (the 0x191 master-slave home re-engages teleop and makes the leader drive
    the follower → collision). record keeps the teleop home (zero.py)."""
    for side, arm in robot.arms.items():
        logger.info(f"Homing {side} in control mode (no teleop) ...")
        home_follower(
            arm, cfg.eef_speed_rate, effort=cfg.gripper_effort,
            # full open per training data (~9.8)
            open_units=int(cfg.gripper_reset_open * GRIPPER_SCALE),
        )


# CTRL_MODE codes from GetArmStatus().arm_status.ctrl_mode (CAN 0x2A1); mirrors
# distal/hardware/zero.py::CTRL_MODE_NAMES.
CTRL_MODE_NAMES = {0x00: "standby", 0x01: "CAN-control", 0x02: "teach/master"}


def report_ctrl_mode(arm, label):
    """Log the arm's firmware control mode. After set_eef_mode it should read
    CAN-control (0x01); if it still reads teach/master (0x02) the master-slave
    linkage was NOT broken (the leader will move) — log a loud warning."""
    try:
        mode = int(arm.GetArmStatus().arm_status.ctrl_mode)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{label}] could not read ctrl_mode: {e}")
        return None
    name = CTRL_MODE_NAMES.get(mode, f"0x{mode:02X}")
    logger.info(f"[{label}] arm ctrl_mode = {name} (0x{mode:02X})")
    if mode == 0x02:
        logger.warning(
            f"[{label}] arm is in TEACH/MASTER mode (0x02) — master-slave NOT "
            f"broken; the leader will move. EndPoseCtrl may be ignored."
        )
    return mode


# ---------------------------------------------------------------------------
# Keyboard control of the episode loop. Mirrors distal/hardware/record.py so the
# eval feels identical to `pixi run record`: SPACE starts the next episode, `->`
# ends + saves it, `<-` discards + re-records it, Esc ends the session. (Unlike
# record.py there is no `y` subtask-bump — eval advances subtasks automatically.)
# ---------------------------------------------------------------------------
def init_eval_keyboard_listener():
    """Start a pynput listener; return (listener, events).

    events = {start_next, exit_early, rerecord_episode, stop_recording}. Headless
    (e.g. over SSH) -> (None, events) + a warning, and wait_for_start falls back to
    an ENTER prompt while `->`/`<-`/Esc are unavailable (episodes then end on
    natural completion / max_steps / Ctrl-C)."""
    events = {
        "start_next": False,
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    # Reuse LeRobot's headless check so behavior matches record.py exactly.
    try:
        from lerobot.common.control_utils import is_headless

        headless = is_headless()
    except Exception:  # noqa: BLE001 — fall back to assuming a display is present
        headless = False

    if headless:
        logger.warning(
            "Headless environment detected. Keyboard controls (SPACE/->/<-/Esc) "
            "are disabled; episodes start on ENTER and end on completion/Ctrl-C."
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
                print("Escape pressed. Stopping evaluation...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                print("Space pressed. Starting next episode...")
                events["start_next"] = True
        except Exception as e:  # noqa: BLE001
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def wait_for_start(events, listener):
    """Block until the operator presses SPACE (start_next) or Esc (stop_recording),
    then clear start_next/exit_early so a stray arrow during the wait doesn't
    immediately end the upcoming episode. Headless (listener None) -> one ENTER."""
    if events["stop_recording"]:
        return
    if listener is None:
        try:
            input("Reset the scene, then press ENTER to start (Ctrl-C to stop)...")
        except EOFError:
            events["stop_recording"] = True
        events["exit_early"] = False
        return
    print("Homed. Reset the scene, then press SPACE to start ('<-' redo, Esc stop).")
    while not events["start_next"] and not events["stop_recording"]:
        time.sleep(0.05)
    events["start_next"] = False
    events["exit_early"] = False


def synthetic_observation(side, top_key="top"):
    """A plausible zero-ish observation for --dry-run (no hardware). Mid-workspace
    EEF, mid gripper, blank images — enough to exercise the schema + server."""
    obs = {}
    mid_eef = [0.21, -0.03, 0.21, -2.10, 0.72, -2.03]  # ~ sim state means
    for ax, v in zip(EEF_AXES, mid_eef):
        obs[f"{side}_eef_{ax}.pos"] = v
    obs[f"{side}_gripper.pos"] = 3.5
    for i in range(1, 7):
        obs[f"{side}_joint_{i}.pos"] = 0.0
    obs[top_key] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs[f"{side}_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


def run_dry_run(cfg: RealEvalConfig, client: PolicyClient, states):
    """Build a synthetic observation, query the server, and print the parsed
    action + the EEF target it implies. Verifies connectivity + schema before
    touching hardware."""
    side = active_side(cfg)
    obs = synthetic_observation(side)
    prompt = states[0]
    obs_dict = build_observation(obs, prompt, side)
    logger.info(
        f"[dry-run] obs keys={list(obs_dict)} state={obs_dict['observation/state']} "
        f"image={obs_dict['observation/image'].shape} prompt={prompt!r}"
    )
    actions = client.get_action(obs_dict, cfg.num_open_loop_steps)
    base = eef_from_obs(obs, side)
    for i, a in enumerate(actions):
        delta6, g, s_t, p_t = split_action(a)
        target = apply_delta(
            base, clip_delta(delta6, cfg.max_pos_step, cfg.max_rot_step)
        )
        logger.info(
            f"[dry-run] action[{i}] Δpos={np.round(delta6[:3], 4)} "
            f"Δrot={np.round(delta6[3:6], 4)} grip={g:.3f} s_t={s_t:.2f} p_t={p_t:.2f} "
            f"-> target_eef={np.round(target, 4)}"
        )
        base = target  # accumulate, as in the chunk delta-base
    logger.info("[dry-run] OK: server reachable, schema valid, actions parsed.")
