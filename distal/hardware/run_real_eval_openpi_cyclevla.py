"""Real-robot full-method CycleVLA eval for the pi05 / openpi policy.

Hardware counterpart of
`cyclevla_code/experiments/robot/libero/run_libero_eval_openpi_cyclevla.py`:
the full proactive self-correction loop, on the real AgileX Piper.

Per subtask, two phases (exactly like the sim version):
  * "to_check": drive the subtask until the policy reports ~90% progress, then
    query a VLM for a `transit | backtrack` decision.
  * "to_complete": drive to completion on the stop signal, then advance.

On `backtrack`, the sim teleports MuJoCo state back; we cannot teleport hardware,
so we BACKTRACK DETERMINISTICALLY by replaying the recorded joint positions in
reverse (JointCtrl) to the start of the target subtask, then retry with an
MBR-selected action chunk (the openpi server is stochastic per call, so N queries
on the same observation give N diverse candidates to rank).

Differences vs the sim script, by necessity:
  * No two-stage "rerun only the failed episodes" gating (that reads sim transit
    videos + uses sim initial states). Here every requested episode is run; the
    operator inspects the saved videos for success.
  * The MBR trajectory features are integrated from the live EEF (read off the
    arm) instead of the sim proprio.

Both-camera rollout videos (raw + subtitled with subtask + transition tag) are
saved to a gitignored dir. Needs OPENAI_API_KEY (in .env) for the VLM and the
policy server up (serve_policy.py, config pi05_libero_cyclevla).

Run:
  pixi run python -m distal.hardware.run_real_eval_openpi_cyclevla \
      --config_path configs/real_eval.yaml
"""

# E501 disabled for this file: it contains the VLM prompt reproduced VERBATIM from
# the read-only cyclevla_code reference (long lines are intentional — rewrapping
# would diverge the prompt the method was tuned with).
# ruff: noqa: E501

import base64
import io
import logging
import os
import time
from collections import deque

import cv2
import draccus
import numpy as np
from lerobot.utils.robot_utils import precise_sleep
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from distal.hardware import subtasks
from distal.hardware.real_eval_common import (
    CameraReadError,
    GripperResolver,
    PolicyClient,
    RealEvalConfig,
    RolloutVideoWriter,
    SignalConfirmer,
    active_side,
    apply_delta,
    backtrack_joints,
    build_observation,
    clip_delta,
    command_eef,
    eef_from_obs,
    gripper_from_obs,
    home_robot,
    init_eval_keyboard_listener,
    joints_from_obs,
    make_robot,
    read_observation,
    reconnect_cameras,
    record_while_busy,
    report_ctrl_mode,
    run_dry_run,
    set_eef_mode,
    setup_logging,
    split_action,
    standby,
    wait_for_start,
)

logger = logging.getLogger(__name__)

TOP_KEY = "top"  # third-person camera (see run_real_eval_openpi_transit.py)


# ---------------------------------------------------------------------------
# VLM detector. Vendored from
# cyclevla_code/.../run_libero_eval_decomposed_progress_mbr.py::VLMDetector so
# the distal env does not import the libero-heavy reference module. The prompt is
# reproduced verbatim (it is what the method was tuned with).
# ---------------------------------------------------------------------------
class VLMDetector:
    def __init__(self, model_name="gpt-5.5", temperature=1.0):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env for the VLM detector."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def encode_image(self, input_img):
        if input_img is None:
            raise ValueError("Image loading failed.")
        # cameras give RGB; cv2 imencode wants BGR.
        input_img = cv2.cvtColor(np.asarray(input_img), cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".png", input_img)
        if not success:
            raise ValueError("Image encoding failed.")
        image_bytes = io.BytesIO(encoded_image).read()
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    def extract_res(self, output_text):
        lines = output_text.strip().splitlines()
        subtask, type_str, reason = None, None, None
        for line in lines:
            if line.lower().startswith("next_subtask:"):
                subtask = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("type:"):
                type_str = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        if not subtask or not type_str or not reason:
            raise ValueError("Failed to extract subtask or type from the response.")
        return subtask, type_str, reason

    def detect_subtask(
        self,
        current_subtask,
        all_subtasks,
        subtask_history,
        language_instruction,
        obv,
        obv_wrist,
    ):
        obv_encoded = self.encode_image(obv)
        wrist_encoded = self.encode_image(obv_wrist)

        output_format = """
        ""Write in the following format. Output nothing else:
        next_subtask: <exact subtask from subtasks list>
        type: <transit / backtrack>
        reason: <explanation>
        """

        prompt = f"""
        You are an expert robot behavior annotator. Decide what the robot should do next given it is ~90% through the current subtask.
        Your job is to FORECAST whether the current subtask will likely succeed if we continue without corrective repositioning.

        Inputs:
        1) Task instruction: {language_instruction}
        2) Subtask list: {all_subtasks}
        3) Current subtask: {current_subtask}
        4) Visual inputs (two synchronized views):
        - FRONT: third-person view (global alignment, object identity, spatial relations)
        - WRIST: close-up gripper view (detailed contact, local geometry, physical affordances)

        Decision rule (forecasting at ~90%):
        - **Transit** when success appears reasonably likely within the next few actions **without** corrective repositioning.
        - Choose **backtrack** if strong, unambiguous visual evidence indicates that the subtask will fail without repositioning.

        View-specific fusion instruction:
        - FRONT view provides **global context**: object identity, pose, global alignment, reachability, and path clearance.
        - WRIST view provides **local interaction cues**: gripper orientation, contact points, slip, stability, and detailed positioning relative to affordances.
        - Combine both views to reason about **functional success**: whether the current configuration supports the intended physical interaction (e.g., grasping, pulling, pushing).
        - FRONT dominates for global spatial reasoning and goal reachability.
        - WRIST dominates for local contact accuracy and grasp quality.

        Affordance reasoning guidance:
        - Aim for the center of the object.
        - For the subtask "move the gripper toward the middle peg of the mug holder while holding the green teapot", the peg needs to be inside the handle of the teapot (CHECK CAREFULLY), so that when released, the teapot will hang on it.
        - At subtask "move the gripper toward the middle peg of the mug holder while holding the green teapot", always backtrack and retry for one time.

        Wrong object or wrong subtask detection:
        In addition to misalignment, detect late-stage "silent failures" involving **wrong object engagement or wrong subtask execution**.
        If visual evidence indicates the gripper is interacting with an unintended object, target, or affordance
        (e.g., lifting or contacting a distractor, manipulating the wrong receptacle, or committing to a different subtask's goal),
        or that the intended object/site remains unaffected while another changes,
        output `type: backtrack` and set `next_subtask` to the earliest subtask that restores correct target selection
        and preconditions (typically a reach, align, or target-identification step, NOT a trivial open/close gripper).

        Backtracking target:
        - Do NOT backtrack to a trivial "open gripper" or "close gripper" subtask.
        - Backtrack to the **earliest** subtask that restores the missing precondition
        (typically a positioning or alignment step that enables correct affordance engagement).

        Output format (STRICT; keep keys exactly):
        next_subtask: <exact subtask from subtasks list>
        type: <transit / backtrack>
        reason: <explain in a concise paragraph, justifying the decision based on predicted execution success and task/subtask correctness>

        front_view_evidence:
        - <concise observable cue 1>
        - <concise observable cue 2>
        - <concise observable cue 3>
        - <concise observable cue 4>

        wrist_view_evidence:
        - <concise observable cue 1>
        - <concise observable cue 2>
        - <concise observable cue 3>
        - <concise observable cue 4>

        assessment:
        - success_likelihood: <high | medium | low>
        - key_risks: <comma-separated brief phrases>
        - view_agreement: <agree | partial | disagree>; <short phrase on which view dominates and why>
        - decision_basis: <short phrase linking likelihood + dominant cues to decision>

        Constraints:
        - Focus strictly on **observable** visual and physical evidence.
        - Keep each bullet concise (<=12 words).
        - Use **exact** strings from `subtasks` for `next_subtask`.
        - `type` must be either transit or backtrack.
        - Return only the specified fields; no extra commentary.

        Now produce the decision.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": obv_encoded, "detail": "high"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": wrist_encoded, "detail": "high"},
                    },
                    {"type": "text", "text": output_format},
                ],
            }
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=self.temperature
        )
        return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# MBR sampling + ranking (vendored / adapted from the sim cyclevla script's
# sample_and_rank_chunks_mbr). Integrates each candidate chunk's deltas into a
# predicted EEF trajectory and ranks by r-NN density (or vanilla average).
# ---------------------------------------------------------------------------
def chunk_features(chunk, current_eef, n_steps):
    """Integrate a candidate chunk's ΔEEF into a predicted absolute-pose feature
    vector (per step: xyz + euler), exactly as the sim MBR sampler does."""
    cumulative_pos = np.asarray(current_eef[:3], dtype=np.float64).copy()
    cumulative_rot = R.from_euler("xyz", np.asarray(current_eef[3:6], dtype=np.float64))
    feats = []
    for action in chunk:
        delta6, _, _, _ = split_action(action)
        cumulative_pos = cumulative_pos + np.asarray(delta6[:3])
        cumulative_rot = cumulative_rot * R.from_euler("xyz", np.asarray(delta6[3:6]))
        feats.extend(cumulative_pos.tolist())
        feats.extend(cumulative_rot.as_euler("xyz").tolist())
    expected = n_steps * 6
    while len(feats) < expected:
        feats.extend([0, 0, 0, 0, 0, 0])
    return np.array(feats[:expected])


def executed_trajectory_features(eef_hist, start_idx, n_steps):
    """Absolute-pose features of the executed (failed) attempt, comparable to
    chunk_features, for optional MBR failed-repulsion."""
    expected = n_steps * 6
    feats = []
    for k in range(start_idx, min(start_idx + n_steps, len(eef_hist))):
        feats.extend(np.asarray(eef_hist[k][:3]).tolist())
        feats.extend(np.asarray(eef_hist[k][3:6]).tolist())
    while len(feats) < expected:
        feats.extend([0, 0, 0, 0, 0, 0])
    return np.array(feats[:expected])


def sample_and_rank_chunks_mbr(
    cfg, client, obs_dict, current_eef, failed_trajectories, selection_mode="rep"
):
    """Sample N candidate chunks from the (stochastic) server and MBR-rank them.
    Returns a list of chunks, best-first. Direct port of the sim version."""
    num_seeds = cfg.mbr_num_seeds
    metric_map = {
        "l2": "euclidean",
        "l1": "cityblock",
        "cosine": "cosine",
        "correlation": "correlation",
        "chebyshev": "chebyshev",
    }
    metric = metric_map.get(cfg.mbr_distance_metric, "euclidean")
    expected = cfg.num_open_loop_steps * 6

    sampled_chunks, sampled_feats = [], []
    for _ in range(num_seeds):
        chunk = client.get_action(obs_dict, cfg.num_open_loop_steps)
        sampled_chunks.append(chunk)
        sampled_feats.append(
            chunk_features(chunk, current_eef, cfg.num_open_loop_steps)
        )

    X = np.stack(sampled_feats)
    N = X.shape[0]
    dist_mat = cdist(X, X, metric=metric)

    if cfg.mbr_vanilla:
        avg = dist_mat.mean(axis=1)
        order = np.argsort(avg)[::-1] if selection_mode == "away" else np.argsort(avg)
        return [sampled_chunks[i] for i in order]

    r = (
        cfg.mbr_r_neighborhood
        if cfg.mbr_r_neighborhood is not None
        else max(2, min(4, int(np.sqrt(N))))
    )
    r_eff = min(r, max(1, N - 1))
    rnn_radius = np.partition(dist_mat, r_eff, axis=1)[:, r_eff]
    center_idx = int(np.argmin(rnn_radius))
    cluster_idx = np.argsort(dist_mat[center_idx])[:r_eff]
    intra = dist_mat[np.ix_(cluster_idx, cluster_idx)]
    medoid_local = cluster_idx[int(np.argmin(intra.mean(axis=1)))]
    d_to_medoid = dist_mat[medoid_local]

    def robust_norm(v):
        v = np.asarray(v)
        if len(v) < 2:
            return np.zeros_like(v)
        lo, hi = np.percentile(v, [10, 90])
        v_clip = np.clip(v, lo, hi)
        med = np.median(v_clip)
        iqr = (np.percentile(v_clip, 75) - np.percentile(v_clip, 25)) + 1e-8
        return (v - med) / iqr

    if cfg.mbr_use_failed_repulsion and failed_trajectories:
        valid = [
            ft
            for ft in failed_trajectories
            if isinstance(ft, np.ndarray) and ft.shape[0] == expected
        ]
        d_fail = (
            cdist(X, np.stack(valid), metric=metric).min(axis=1)
            if valid
            else np.full((N,), np.median(rnn_radius) if N > 0 else 1.0)
        )
    else:
        d_fail = np.full((N,), np.median(rnn_radius) if N > 0 else 1.0)

    rnn_norm = robust_norm(rnn_radius)
    dmed_norm = robust_norm(d_to_medoid)
    dfail_norm = robust_norm(d_fail)
    repulse = 1.0 / (1.0 + np.exp(-dfail_norm))
    lambda_fail = 0.5
    if selection_mode == "away":
        scores = dmed_norm + lambda_fail * repulse
    else:
        scores = -rnn_norm + lambda_fail * repulse
    order = np.argsort(scores)[::-1]
    return [sampled_chunks[i] for i in order]


# ---------------------------------------------------------------------------
# Episode loop.
# ---------------------------------------------------------------------------
def run_episode_cyclevla(cfg, robot, arm, client, states, vlm, events, writer):
    """Run one episode with the two-phase VLM/MBR + joint-replay backtrack loop.

    STREAMS frames to disk via `writer` (RolloutVideoWriter) each tick — never holds
    them in RAM (accumulation starves the RealSense read thread). Mirrors the sim
    run_episode's state machine. Ends on natural completion, max_steps, OR the
    operator's `->`/`<-`/Esc (`events["exit_early"]`).
    """
    side = active_side(cfg)
    wrist_key = f"{side}_wrist"
    period = 1.0 / cfg.fps
    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    # One stateful gripper resolver for the whole episode (hysteresis latch persists across
    # subtasks/backtracks); arm is homed open so seed the latch open.
    gripper_resolver = GripperResolver(cfg)
    current_state = states[0]
    subtask_hist, exe_type_hist = [current_state], ["init"]

    # Per-step executed-state history (recorded BEFORE executing each action) for
    # deterministic joint-replay backtracking + MBR failed features.
    joint_hist, grip_hist, eef_hist = [], [], []
    # Where (index into joint_hist) each subtask started — the backtrack target.
    subtask_start_idx = {current_state: 0}

    subtask_phase = "to_check"
    # Long/multi-stage tasks use robust (consecutive/recurring) 90% confirmation,
    # like the sim libero_goal / libero_10 path.
    robust_progress = len(states) >= 6
    progress_confirm = SignalConfirmer(threshold=cfg.progress_threshold)
    count_check_signals = 0
    stop_confirm = SignalConfirmer(threshold=0.5)

    subtask_retry_count, subtask_chunk_rankings = {}, {}
    current_chunk_index, subtask_failed_trajectories = {}, {}

    cmd_eef = None
    max_steps = int(cfg.max_steps * 1.5 + cfg.num_steps_wait)

    def reset_phase_counters():
        progress_confirm.reset()
        stop_confirm.reset()

    while t < max_steps:
        if events["exit_early"]:  # `->` save / `<-` redo / Esc stop
            logger.info(f"Operator ended episode at step {t}.")
            break
        loop_start = time.perf_counter()

        obs = read_observation(robot)
        writer.add(
            {"image": obs[TOP_KEY], "wrist_image": obs[wrist_key]},
            f"{exe_type_hist[-1]}: {current_state}",
        )

        if len(action_queue) == 0:
            obs_dict = build_observation(obs, current_state, side)
            action_queue.extend(client.get_action(obs_dict, cfg.num_open_loop_steps))
            cmd_eef = eef_from_obs(obs, side)

        # Record the executed-state snapshot before stepping.
        joint_hist.append(joints_from_obs(obs, side))
        grip_hist.append(gripper_from_obs(obs, side))
        eef_hist.append(eef_from_obs(obs, side))

        delta6, g, stop_signal, progress_signal = split_action(action_queue.popleft())
        delta6 = clip_delta(delta6, cfg.max_pos_step, cfg.max_rot_step)
        # Per-step action print (debug the stop/progress signals).
        logger.info(
            "t=%d sub=`%s` Δp=%s Δr=%s grip=%.3f s_t=%.2f p_t=%.2f",
            t,
            current_state,
            np.round(delta6[:3], 4),
            np.round(delta6[3:6], 4),
            g,
            stop_signal,
            progress_signal,
        )
        if cfg.delta_base == "live":
            target = apply_delta(eef_from_obs(obs, side), delta6)
        else:
            cmd_eef = apply_delta(cmd_eef, delta6)
            target = cmd_eef
        if not cfg.no_arm:
            command_eef(
                arm, target, gripper_resolver.resolve(g), effort=cfg.gripper_effort
            )
        t += 1
        # Pace the control loop before any (slow) VLM call below.
        precise_sleep(max(0.0, period - (time.perf_counter() - loop_start)))

        # ===== PHASE 1: 90% progress -> VLM decision =====
        if subtask_phase == "to_check":
            # Only actual gripper ACTUATION subtasks (open/close) are trivial and
            # skip the VLM; reach/motion subtasks ("move the gripper above ...")
            # must still go through the VLM transit/backtrack decision. Mirrors the
            # sim (run_libero_eval_openpi_cyclevla.py), but keys off the leading
            # verb so it also catches "open the gripper to hang ..." (teapot task).
            # NOTE: distinct from the convert_to_cyclevla.py oversampling test,
            # where a broad "gripper" substring is intentional (every subtask).
            cs = current_state.lower()
            # if cs.startswith("close the gripper to") or cs.startswith("open the gripper to"):
            #     logger.info(f"Gripper subtask `{current_state}` - skipping VLM check.")
            #     subtask_phase = "to_complete"
            #     reset_phase_counters()
                # continue
            
            # Never skip
            if cs.startswith("Never skip"):
                logger.info(f"Gripper subtask `{current_state}` - skipping VLM check.")
                subtask_phase = "to_complete"
                reset_phase_counters()
                continue

            vlm_check_needed = False
            if robust_progress:
                if progress_confirm.update(progress_signal):
                    vlm_check_needed = True
            else:
                if progress_signal >= cfg.progress_threshold:
                    count_check_signals += 1
                    logger.info(
                        f"90% signal #{count_check_signals} at step {t} "
                        f"(progress={progress_signal:.2f})"
                    )
                    if count_check_signals >= 2:
                        vlm_check_needed = True

            if not vlm_check_needed:
                continue

            # ===== 90% reached: VLM decision point =====
            # Prominent banner so the operator knows a check was hit (and can nudge
            # the object to inject an error during the hold below).
            delay = cfg.vlm_check_delay_sec
            logger.info("=" * 70)
            logger.info(
                f">>> VLM CHECK REACHED @ step {t} — subtask `{current_state}` (p_t>=0.90)"
            )
            if delay > 0:
                logger.info(
                    f">>> Holding {delay:.1f}s before capturing the VLM image — "
                    f"ADJUST THE OBJECT NOW to inject an error."
                )
            logger.info("=" * 70)

            # Hold the arm idle for `delay`s, streaming live frames (so the
            # perturbation window is on the video) with a 1 Hz countdown log.
            period = 1.0 / cfg.fps
            n_frames = max(0, int(round(delay * cfg.fps)))
            last_logged_sec = None
            for i in range(n_frames):
                loop_start = time.perf_counter()
                remaining_sec = int(round((n_frames - i) * period))
                if remaining_sec != last_logged_sec and remaining_sec > 0:
                    logger.info(f">>> capturing VLM image in ~{remaining_sec}s ...")
                    last_logged_sec = remaining_sec
                try:
                    o = read_observation(robot)
                    writer.add(
                        {"image": o[TOP_KEY], "wrist_image": o[wrist_key]},
                        f"HUMAN MAY INJECT ERROR: {current_state}",
                    )
                except CameraReadError as e:  # a drop must not abort the hold
                    logger.warning(f"Camera read failed during VLM-check delay: {e}")
                precise_sleep(max(0.0, period - (time.perf_counter() - loop_start)))

            # Capture the (possibly perturbed) image for the VLM, then STREAM live
            # frames of the idle robot during the (multi-second) blocking call —
            # otherwise the recording shows a freeze while the decision is made.
            obs = read_observation(robot)
            top_img, wrist_img = obs[TOP_KEY], obs[wrist_key]
            try:
                res = record_while_busy(
                    robot,
                    writer,
                    TOP_KEY,
                    wrist_key,
                    f"VLM_90%_check: {current_state}",
                    cfg.fps,
                    vlm.detect_subtask,
                    current_state,
                    states,
                    subtask_hist,
                    cfg.single_task,
                    top_img,
                    wrist_img,
                )
                detected_state, exe_type, reason = vlm.extract_res(res)
            except Exception as e:  # noqa: BLE001 — a flaky VLM call should not kill the run
                logger.warning(f"VLM call/parse failed ({e}); treating as transit.")
                detected_state, exe_type, reason = current_state, "transit", "vlm error"
            logger.info(
                f"VLM @90% `{current_state}` -> next={detected_state}, type={exe_type}, reason={reason}"
            )

            if exe_type == "backtrack" and detected_state not in states:
                logger.info(
                    f"Backtrack target `{detected_state}` not a known subtask; treating as transit."
                )
                exe_type = "transit"

            if exe_type == "backtrack":
                subtask_retry_count[current_state] = (
                    subtask_retry_count.get(current_state, 0) + 1
                )
                retry_num = subtask_retry_count[current_state]
                if retry_num >= cfg.max_subtask_retries:
                    logger.info(
                        f"Max retries ({cfg.max_subtask_retries}) for `{current_state}`; "
                        f"forcing completion."
                    )
                    subtask_phase = "to_complete"
                    reset_phase_counters()
                    continue

                logger.info(
                    f"Backtrack from `{current_state}` to `{detected_state}` (retry #{retry_num})."
                )
                # Record failed-run features under the FAILING subtask (MBR repulsion).
                subtask_failed_trajectories.setdefault(current_state, []).append(
                    executed_trajectory_features(
                        eef_hist,
                        subtask_start_idx.get(current_state, 0),
                        cfg.num_open_loop_steps,
                    )
                )

                current_state = detected_state
                subtask_hist.append(current_state)
                exe_type_hist.append(f"backtrack_retry{retry_num}")
                action_queue.clear()

                # Physically rewind to the recorded start of the target subtask
                # (skipped in --no_arm). Backtracking to the FIRST subtask (idx 0)
                # re-homes to the canonical zero start pose instead of replaying the
                # whole trajectory in reverse.
                target_idx = subtask_start_idx.get(current_state, 0)
                if not cfg.no_arm:

                    # Stream a frame per motion step so the physical return is
                    # captured in the video instead of appearing as a jump.
                    def capture_frame(subtitle):
                        try:
                            o = read_observation(robot)
                            writer.add(
                                {"image": o[TOP_KEY], "wrist_image": o[wrist_key]},
                                subtitle,
                            )
                        except CameraReadError as e:  # a drop must not abort the motion
                            logger.warning(f"Camera read failed during backtrack: {e}")

                    if target_idx == 0:
                        # First subtask == episode start: a clean re-home to joint
                        # zero (same as the per-episode home) beats retracing every
                        # recorded step in reverse.
                        logger.info(
                            "Backtrack target is subtask 0 — re-homing to zero start "
                            "pose instead of joint replay."
                        )
                        home_robot(
                            robot,
                            cfg,
                            on_step=lambda: capture_frame(
                                f"re-homing to start: {current_state}"
                            ),
                        )
                        # home leaves the arm in MOVE_J; restore EEF/MOVE_P for the retry.
                        set_eef_mode(arm, cfg.eef_speed_rate)
                    else:
                        logger.info(f"Joint-replay backtrack to index {target_idx} ...")
                        backtrack_joints(
                            arm,
                            joint_hist,
                            target_idx,
                            cfg.fps,
                            cfg.eef_speed_rate,
                            grip_hist,
                            on_step=lambda: capture_frame(
                                f"backtracking: {current_state}"
                            ),
                        )
                # Truncate histories to where we physically returned.
                joint_hist = joint_hist[:target_idx]
                grip_hist = grip_hist[:target_idx]
                eef_hist = eef_hist[:target_idx]

                # Re-read the arm at the rewound pose; show a backtrack frame.
                obs = read_observation(robot)
                writer.add(
                    {"image": obs[TOP_KEY], "wrist_image": obs[wrist_key]},
                    f"backtracked_to: {current_state}",
                )
                cmd_eef = eef_from_obs(obs, side)
                obs_dict = build_observation(obs, current_state, side)

                # MBR decoding: sample + rank on the first backtrack to this
                # subtask; reuse the ranking (next-best chunk) on later ones.
                if current_state not in subtask_chunk_rankings:
                    logger.info(
                        f"Sampling {cfg.mbr_num_seeds} chunks for MBR of `{current_state}`..."
                    )
                    # STREAM live frames while the (multi-call, blocking) MBR
                    # sampling runs — the arm sits idle at the rewound pose.
                    subtask_chunk_rankings[current_state] = record_while_busy(
                        robot,
                        writer,
                        TOP_KEY,
                        wrist_key,
                        f"MBR_sampling: {current_state}",
                        cfg.fps,
                        sample_and_rank_chunks_mbr,
                        cfg,
                        client,
                        obs_dict,
                        eef_from_obs(obs, side),
                        subtask_failed_trajectories.get(current_state, []),
                    )
                    current_chunk_index[current_state] = 0
                else:
                    current_chunk_index[current_state] += 1
                    if current_chunk_index[current_state] >= len(
                        subtask_chunk_rankings[current_state]
                    ):
                        current_chunk_index[current_state] = 0
                chunk_idx = current_chunk_index[current_state]
                action_queue.extend(subtask_chunk_rankings[current_state][chunk_idx])
                logger.info(
                    f"Retrying `{current_state}` with MBR chunk "
                    f"#{chunk_idx + 1}/{len(subtask_chunk_rankings[current_state])}"
                )

                # This retry restarts the subtask here.
                subtask_start_idx[current_state] = len(joint_hist)
                subtask_phase = "to_check"
                count_check_signals = 0
                reset_phase_counters()
            else:
                logger.info(f"VLM transit: continue `{current_state}` to completion.")
                exe_type_hist.append("transit")
                subtask_phase = "to_complete"
                count_check_signals = 0
                reset_phase_counters()

        # ===== PHASE 2: 100% completion via the stop signal =====
        elif subtask_phase == "to_complete":
            if stop_confirm.update(stop_signal):
                idx = states.index(current_state)
                logger.info(f"Subtask `{current_state}` completed at step {t}.")
                if idx + 1 < len(states):
                    current_state = states[idx + 1]
                    subtask_hist.append(current_state)
                    exe_type_hist.append("continue")
                    action_queue.clear()
                    cmd_eef = None
                    subtask_start_idx[current_state] = len(joint_hist)
                    subtask_phase = "to_check"
                    count_check_signals = 0
                    reset_phase_counters()
                else:
                    logger.info(f"All subtasks completed at step {t}.")
                    break

    if t >= max_steps:
        logger.warning(f"Hit max_steps ({max_steps}); ending episode.")


@draccus.wrap()
def main(cfg: RealEvalConfig):
    # Load OPENAI_API_KEY (and any other secrets) from .env, like the sim eval.
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging(cfg, "cyclevla")
    states = [s.lower() for s in subtasks.get_subtasks(cfg.single_task)]
    logger.info(f"Task: {cfg.single_task!r} -> {len(states)} subtasks: {states}")

    logger.info(
        f"Connecting to openpi policy server ws://{cfg.host}:{cfg.port} "
        f"(start it first if this hangs)..."
    )
    client = PolicyClient(host=cfg.host, port=cfg.port)
    if cfg.dry_run:
        run_dry_run(cfg, client, states)
        return

    # Construct the VLM up front so a missing OPENAI_API_KEY fails before homing.
    vlm = VLMDetector(model_name=cfg.vlm_model, temperature=cfg.vlm_temperature)

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
            # Home to zero at the start of each episode WITHOUT re-linking
            # master-slave (restore=False) so the leader stays idle during the run.
            if cfg.home:
                home_robot(robot, cfg)
            wait_for_start(events, listener)  # "press SPACE to start"
            if events["stop_recording"]:
                break

            # Re-warm the cameras AFTER the idle gap (homing + SPACE wait + scene
            # reset) so the episode starts from a fresh, streaming pipeline.
            reconnect_cameras(robot)

            # CAN/MOVE_P so the policy drives ONLY the follower; verify the linkage
            # broke (ctrl_mode 0x01, not teach 0x02).
            if not cfg.no_arm:
                set_eef_mode(arm, cfg.eef_speed_rate)
                report_ctrl_mode(arm, "after set_eef_mode")
            events["exit_early"] = False
            events["rerecord_episode"] = False
            # Stream frames straight to disk (bounded memory). Opened per episode.
            writer = RolloutVideoWriter(
                cfg.video_dir, ep + 1, cfg.single_task, cfg.fps, "cyclevla"
            )
            logger.info(f"=== Episode {ep + 1}/{cfg.num_episodes} START ===")
            try:
                run_episode_cyclevla(
                    cfg, robot, arm, client, states, vlm, events, writer
                )
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
