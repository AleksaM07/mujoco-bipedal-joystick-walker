"""Render BVH reference clips to short MP4 videos for manual inspection."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import sys
from pathlib import Path

import jax

if platform.system() == "Windows" and os.environ.get("MUJOCO_GL") == "egl":
    print("Ignoring MUJOCO_GL=egl on Windows; using MuJoCo default OpenGL backend.")
    os.environ.pop("MUJOCO_GL", None)

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv  # noqa: E402
from bvh_reference import LoopMode, expand_motion_paths  # noqa: E402
from config import DEFAULT_BVH_REFERENCE_LIST, expand_reference_gait_files  # noqa: E402
from debug_reference_playback import build_full_state, query_reference_np, torso_up  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("bvh_provera"))
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--reference-gait-list", type=Path, action="append")
    parser.add_argument(
        "--arms-on",
        dest="arm_actuators",
        action="store_true",
        help="Render the experimental 26-DOF model with arm actuators.",
    )
    parser.add_argument(
        "--arms-off",
        dest="arm_actuators",
        action="store_false",
        help="Render the stable 18-DOF locomotion model without arm actuators.",
    )
    parser.set_defaults(arm_actuators=False)
    parser.add_argument(
        "--reference-loop-mode",
        choices=["auto", "wrap", "clamp"],
        default="auto",
        help="Override reference loop classification for preview renders.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Limit expanded BVH source files before environment loading.",
    )
    parser.add_argument(
        "--mode",
        choices=["kinematic", "pd", "compare"],
        default="kinematic",
        help="kinematic=ideal reference, pd=physics tracking, compare=left/right.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=1.0,
        help=(
            "Scale MuJoCo gravity for diagnostic renders. 0.0 isolates joint "
            "tracking without balance/contact loading."
        ),
    )
    parser.add_argument(
        "--pin-root-to-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force the free root state to follow the reference during PD "
            "playback. This isolates joint-target tracking from balance/root "
            "dynamics."
        ),
    )
    parser.add_argument(
        "--pin-root-position-to-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force only root XYZ position/velocity to the reference during PD "
            "playback."
        ),
    )
    parser.add_argument(
        "--pin-root-xy-to-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force only root horizontal XY position/velocity to the reference "
            "during PD playback."
        ),
    )
    parser.add_argument(
        "--pin-root-z-to-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force only root vertical Z position/velocity to the reference "
            "during PD playback."
        ),
    )
    parser.add_argument(
        "--pin-root-rotation-to-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force only root orientation/angular velocity to the reference during "
            "PD playback."
        ),
    )
    parser.add_argument(
        "--actuator-force-scale",
        type=float,
        default=1.0,
        help="Scale diagnostic actuator force limits before playback.",
    )
    parser.add_argument(
        "--actuator-kp-scale",
        type=float,
        default=1.0,
        help="Scale diagnostic position-actuator stiffness before playback.",
    )
    parser.add_argument(
        "--trunk-kp-scale",
        type=float,
        default=1.0,
        help="Extra stiffness scale for abdomen/trunk actuators only.",
    )
    parser.add_argument(
        "--pelvis-kp-scale",
        type=float,
        default=1.0,
        help="Extra stiffness scale for pelvis actuators only.",
    )
    parser.add_argument(
        "--ankle-kp-scale",
        type=float,
        default=1.0,
        help="Extra stiffness scale for ankle actuators only.",
    )
    parser.add_argument(
        "--hip-kp-scale",
        type=float,
        default=1.0,
        help="Extra stiffness scale for hip actuators only.",
    )
    parser.add_argument(
        "--trunk-passive-scale",
        type=float,
        default=1.0,
        help="Scale passive abdomen joint stiffness/damping/frictionloss.",
    )
    parser.add_argument(
        "--pelvis-passive-scale",
        type=float,
        default=1.0,
        help="Scale passive pelvis joint stiffness/damping/frictionloss.",
    )
    parser.add_argument(
        "--head-passive-scale",
        type=float,
        default=1.0,
        help="Scale passive head/neck joint stiffness/damping/frictionloss.",
    )
    parser.add_argument(
        "--hip-passive-scale",
        type=float,
        default=1.0,
        help="Scale passive hip joint damping/frictionloss.",
    )
    parser.add_argument(
        "--contact-friction-scale",
        type=float,
        default=1.0,
        help="Scale floor and sole contact friction for diagnostic playback.",
    )
    parser.add_argument(
        "--root-z-assist-kp",
        type=float,
        default=0.0,
        help="Diagnostic vertical root support force gain toward reference height.",
    )
    parser.add_argument(
        "--root-z-assist-kd",
        type=float,
        default=0.0,
        help="Diagnostic vertical root support damping toward reference vertical velocity.",
    )
    parser.add_argument(
        "--root-z-assist-max-force",
        type=float,
        default=0.0,
        help="Optional absolute clip for diagnostic root-Z assist force. 0 disables clipping.",
    )
    parser.add_argument(
        "--root-pitch-assist-kp",
        type=float,
        default=0.0,
        help="Diagnostic root pitch torque gain toward reference orientation.",
    )
    parser.add_argument(
        "--root-pitch-assist-kd",
        type=float,
        default=0.0,
        help="Diagnostic root pitch torque damping toward reference angular rate.",
    )
    parser.add_argument(
        "--root-pitch-assist-max-torque",
        type=float,
        default=0.0,
        help="Optional absolute clip for diagnostic root-pitch assist torque. 0 disables clipping.",
    )
    parser.add_argument(
        "--trace-dt",
        type=float,
        default=0.01,
        help="Detailed trace sampling interval in wall-clock seconds.",
    )
    parser.add_argument(
        "--start-phase",
        type=float,
        default=0.0,
        help="Normalized clip phase in [0, 1] where the diagnostic window starts.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Optional clip sub-window length in motion seconds before any slow-down.",
    )
    parser.add_argument(
        "--reference-speed-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for reference motion progression during rendering. "
            "0.25 means the clip target advances 4x slower than real time."
        ),
    )
    parser.add_argument(
        "--min-video-seconds",
        type=float,
        default=4.0,
        help="Minimum preview length. Wrap clips replay for multiple loops.",
    )
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=None,
        help="Exact preview length override.",
    )
    parser.add_argument("--end-hold-seconds", type=float, default=0.35)
    parser.add_argument(
        "--clip-id",
        type=int,
        default=None,
        help="Render only one loaded clip id.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--training-filtered",
        action="store_true",
        help="Render only clips kept by the current training filter.",
    )
    parser.add_argument(
        "--debug-body-floor-collision",
        action="store_true",
        help=(
            "Viewer-only readable falls: make all body geoms collide with the "
            "floor. This changes PD diagnostic physics, not training."
        ),
    )
    parser.add_argument(
        "--reference-lock-stance-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anchor low stance foot XY while rendering retargeted references.",
    )
    parser.add_argument(
        "--reference-action-mode",
        choices=["mimickit", "residual"],
        default="residual",
        help="Env-side policy-to-PD mapping used when emulating training control.",
    )
    parser.add_argument(
        "--reference-action-center",
        choices=["default", "joint_midpoint"],
        default="default",
        help="Action-space zero point for MimicKit-style control emulation.",
    )
    parser.add_argument(
        "--reference-action-range",
        choices=["action_scale", "joint_limits", "reference_targets"],
        default="reference_targets",
        help="Half-range convention for MimicKit-style control emulation.",
    )
    parser.add_argument(
        "--reference-action-range-scale",
        type=float,
        default=1.1,
        help="Multiplier for --reference-action-range during control emulation.",
    )
    parser.add_argument(
        "--reference-residual-scale",
        type=float,
        default=1.0,
        help="Residual action scale for residual-mode control emulation.",
    )
    parser.add_argument(
        "--reference-replay-target-step",
        type=int,
        default=0,
        help="Reference step offset used by residual-mode control emulation.",
    )
    parser.add_argument(
        "--control-source",
        choices=["direct_reference", "zero_policy", "encoded_reference"],
        default="direct_reference",
        help=(
            "direct_reference bypasses the policy map; zero_policy and "
            "encoded_reference emulate the training action-to-PD map."
        ),
    )
    parser.add_argument(
        "--overlay-title",
        type=str,
        default=None,
        help="Optional short text banner burned into the output video.",
    )
    parser.add_argument(
        "--trace-actuator-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append per-actuator torque/ratio/target/actual/error columns to the "
            "trace CSVs for forensic debugging."
        ),
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Write an RGB frame list to video using whichever writer is installed."""
    try:
        import mediapy as media

        media.write_video(path, frames, fps=fps)
        return
    except Exception as exc:
        print(f"mediapy video writer failed, trying imageio | {exc}", flush=True)

    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frames), fps=fps)
        return
    except Exception as exc:
        raise RuntimeError(
            "Install mediapy or imageio to write MP4 files: "
            "pip install mediapy imageio imageio-ffmpeg"
        ) from exc


def apply_actuator_scales(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    kp_scale: float,
    force_scale: float,
    trunk_kp_scale: float,
    pelvis_kp_scale: float,
    ankle_kp_scale: float,
    hip_kp_scale: float,
) -> None:
    """Scale diagnostic actuator stiffness and force limits in-place."""
    if kp_scale <= 0.0:
        raise ValueError(f"--actuator-kp-scale must be positive, got {kp_scale}.")
    if force_scale <= 0.0:
        raise ValueError(
            f"--actuator-force-scale must be positive, got {force_scale}."
        )
    for extra_scale, name in (
        (trunk_kp_scale, "--trunk-kp-scale"),
        (pelvis_kp_scale, "--pelvis-kp-scale"),
        (ankle_kp_scale, "--ankle-kp-scale"),
        (hip_kp_scale, "--hip-kp-scale"),
    ):
        if extra_scale <= 0.0:
            raise ValueError(f"{name} must be positive, got {extra_scale}.")
    model.actuator_gainprm[:, 0] *= float(kp_scale)
    model.actuator_biasprm[:, 1] *= float(kp_scale)
    model.actuator_forcerange[:] *= float(force_scale)

    joint_names = tuple(env._actuator_joint_names)
    trunk_mask = np.array(
        [name.startswith("abdomen_") for name in joint_names],
        dtype=bool,
    )
    pelvis_mask = np.array(
        [name.startswith("pelvis_") for name in joint_names],
        dtype=bool,
    )
    ankle_mask = np.array(
        [("ankle_" in name) for name in joint_names],
        dtype=bool,
    )
    hip_mask = np.array(
        [("hip_" in name) for name in joint_names],
        dtype=bool,
    )

    for mask, scale in (
        (trunk_mask, float(trunk_kp_scale)),
        (pelvis_mask, float(pelvis_kp_scale)),
        (ankle_mask, float(ankle_kp_scale)),
        (hip_mask, float(hip_kp_scale)),
    ):
        if not np.any(mask) or scale == 1.0:
            continue
        model.actuator_gainprm[mask, 0] *= scale
        model.actuator_biasprm[mask, 1] *= scale


def apply_contact_friction_scale(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    friction_scale: float,
) -> None:
    """Scale floor/foot friction terms in-place for contact diagnostics."""
    if friction_scale <= 0.0:
        raise ValueError(
            "--contact-friction-scale must be positive, got "
            f"{friction_scale}."
        )
    geom_ids = [
        env._floor_geom_id,
        env._left_foot_sole_geom_id,
        env._right_foot_sole_geom_id,
    ]
    for geom_id in geom_ids:
        model.geom_friction[geom_id, :] *= float(friction_scale)


def apply_passive_joint_scales(
    model: mujoco.MjModel,
    *,
    trunk_passive_scale: float,
    pelvis_passive_scale: float,
    head_passive_scale: float,
    hip_passive_scale: float,
) -> None:
    """Scale passive joint stiffness, damping, and frictionloss by joint group."""
    for scale, flag_name in (
        (trunk_passive_scale, "--trunk-passive-scale"),
        (pelvis_passive_scale, "--pelvis-passive-scale"),
        (head_passive_scale, "--head-passive-scale"),
        (hip_passive_scale, "--hip-passive-scale"),
    ):
        if scale <= 0.0:
            raise ValueError(f"{flag_name} must be positive, got {scale}.")

    joint_groups = {
        "head_": float(head_passive_scale),
        "abdomen_": float(trunk_passive_scale),
        "pelvis_": float(pelvis_passive_scale),
        "left_hip_": float(hip_passive_scale),
        "right_hip_": float(hip_passive_scale),
    }
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name or ""
        scale = 1.0
        for prefix, candidate_scale in joint_groups.items():
            if joint_name.startswith(prefix):
                scale = candidate_scale
                break
        if scale == 1.0:
            continue
        model.jnt_stiffness[joint_id] *= scale
        dof_adr = int(model.jnt_dofadr[joint_id])
        if dof_adr >= 0:
            model.dof_damping[dof_adr] *= scale
            model.dof_frictionloss[dof_adr] *= scale


def root_z_assist_force(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
    target_ref: dict[str, np.ndarray],
    *,
    kp: float,
    kd: float,
    max_force: float,
) -> float:
    """Return one diagnostic external vertical support force on the root body."""
    if kp < 0.0 or kd < 0.0:
        raise ValueError("--root-z-assist gains must be non-negative.")
    target_z = float(np.asarray(target_ref["root_pos"], dtype=np.float64)[2])
    target_vz = float(np.asarray(target_ref["root_vel"], dtype=np.float64)[2])
    current_z = float(data.qpos[2])
    current_vz = float(data.qvel[2])
    force = (kp * (target_z - current_z)) + (kd * (target_vz - current_vz))
    if max_force > 0.0:
        force = float(np.clip(force, -max_force, max_force))
    return force


def root_pitch_assist_torque(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
    target_ref: dict[str, np.ndarray],
    *,
    kp: float,
    kd: float,
    max_torque: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return one diagnostic pitch-stabilizing root torque and scalar diagnostics."""
    if kp < 0.0 or kd < 0.0:
        raise ValueError("--root-pitch-assist gains must be non-negative.")
    current_rotation = body_rotation_matrix(data, env._reference_anchor_body_id)
    ref_rotation = quat_to_rotmat(np.asarray(target_ref["root_quat"], dtype=np.float64))
    _, current_left = heading_frame_axes_from_rotation(current_rotation)
    _, ref_left = heading_frame_axes_from_rotation(ref_rotation)
    current_pitch = rotation_pitch_rad(current_rotation)
    ref_pitch = rotation_pitch_rad(ref_rotation)
    current_pitch_rate = float(
        np.dot(np.asarray(data.qvel[3:6], dtype=np.float64), current_left)
    )
    ref_pitch_rate = float(
        np.dot(np.asarray(target_ref["root_angvel"], dtype=np.float64), ref_left)
    )
    torque_scalar = (kp * (ref_pitch - current_pitch)) + (
        kd * (ref_pitch_rate - current_pitch_rate)
    )
    if max_torque > 0.0:
        torque_scalar = float(np.clip(torque_scalar, -max_torque, max_torque))
    torque_world = torque_scalar * current_left
    return torque_world, {
        "pd_root_pitch_assist_torque": float(torque_scalar),
        "pd_root_pitch_deg": float(np.degrees(current_pitch)),
        "ref_root_pitch_deg": float(np.degrees(ref_pitch)),
        "pd_root_pitch_rate": current_pitch_rate,
        "ref_root_pitch_rate": ref_pitch_rate,
    }


def overlay_frame_title(
    frame: np.ndarray,
    title: str | None,
    mode: str,
) -> np.ndarray:
    """Add a compact diagnostic banner to the top of a rendered frame."""
    if not title:
        return frame

    banner_height = 44
    image = Image.fromarray(frame)
    canvas = Image.new(
        "RGB",
        (image.width, image.height + banner_height),
        color=(18, 18, 18),
    )
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, banner_height), fill=(18, 18, 18))
    draw.text((12, 13), title, fill=(245, 245, 245), font=font)
    if mode == "compare":
        midpoint = canvas.width // 2
        draw.line((midpoint, 0, midpoint, banner_height), fill=(80, 80, 80), width=1)
        draw.text((12, 28), "LEFT: KINEMATIC REFERENCE", fill=(180, 180, 180), font=font)
        draw.text(
            (midpoint + 12, 28),
            "RIGHT: PD TRACKING",
            fill=(180, 180, 180),
            font=font,
        )
    return np.asarray(canvas)


def default_reference_files(args: argparse.Namespace) -> list[str]:
    files = expand_reference_gait_files(
        args.reference_gait_file,
        args.reference_gait_list,
    )
    if files is None:
        files = expand_reference_gait_files(reference_gait_lists=[DEFAULT_BVH_REFERENCE_LIST])
    if args.source_limit is not None:
        files = [
            path.as_posix()
            for path in expand_motion_paths(tuple(files))[: max(0, args.source_limit)]
        ]
    return [Path(path).as_posix() for path in files]


def make_env(args: argparse.Namespace) -> BiomechanicsJoystickEnv:
    config_overrides = {
        "impl": "jax",
        "physics_backend": "mjx_jax",
        "reference_gait": "bvh",
        "arm_actuators": args.arm_actuators,
        "reference_gait_file": default_reference_files(args),
        "reference_loop_mode": args.reference_loop_mode,
        "reference_target_observation": False,
        "reference_action_mode": args.reference_action_mode,
        "reference_action_center": args.reference_action_center,
        "reference_action_range": args.reference_action_range,
        "reference_action_range_scale": args.reference_action_range_scale,
        "reference_residual_scale": args.reference_residual_scale,
        "reference_replay_target_step": args.reference_replay_target_step,
        "deepmimic_reward_mode": "pure",
        "pose_termination": False,
        "enable_erfi": False,
        "reference_lock_stance_feet": args.reference_lock_stance_feet,
    }
    if not args.training_filtered:
        config_overrides["reference_min_motion_length"] = 0.0
    return BiomechanicsJoystickEnv(config_overrides=config_overrides)


def clip_label(loop_mode: int) -> str:
    if loop_mode == int(LoopMode.WRAP):
        return "WRAP_LOOP"
    return "CLAMP_MOTION_OVER"


def source_label(env: BiomechanicsJoystickEnv, clip_id: int) -> str:
    source_path = Path(env._bvh_reference_source_paths[clip_id])
    start = int(env._bvh_reference_source_start_frames[clip_id])
    end = int(env._bvh_reference_source_end_frames[clip_id])
    return f"{source_path.stem}_f{start:04d}-{end:04d}"


def enable_body_floor_collision_debug(model: mujoco.MjModel) -> None:
    """Make failed PD playback easier to read by colliding all bodies with floor."""
    floor_id = model.geom("floor").id
    model.geom_contype[floor_id] = 1
    model.geom_conaffinity[floor_id] = 2
    for geom_id in range(model.ngeom):
        if geom_id == floor_id:
            continue
        model.geom_contype[geom_id] = 2
        model.geom_conaffinity[geom_id] = 1


def fail_reason(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> str:
    low = data.qpos[2] < env.MIN_STANDING_HEIGHT_RATIO * float(model.qpos0[2])
    tipped = torso_up(env, data) < 0.25
    if low:
        return "low_height"
    if tipped:
        return "tipped"
    return "none"


def host_foot_contacts(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
) -> tuple[bool, bool]:
    """Return host MuJoCo left/right foot-floor contact flags."""
    left_contact = False
    right_contact = False
    floor_id = env._floor_geom_id
    left_id = env._left_foot_sole_geom_id
    right_id = env._right_foot_sole_geom_id
    for contact_index in range(data.ncon):
        geom_a, geom_b = data.contact[contact_index].geom
        pair = {int(geom_a), int(geom_b)}
        if floor_id not in pair:
            continue
        left_contact = left_contact or left_id in pair
        right_contact = right_contact or right_id in pair
    return left_contact, right_contact


def actuator_force_stats(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[float, float]:
    """Return max force ratio and fraction of saturated actuators."""
    forces = np.abs(np.asarray(data.actuator_force, dtype=np.float64))
    ranges = np.asarray(model.actuator_forcerange, dtype=np.float64)
    limits = np.maximum(np.abs(ranges).max(axis=1), 1e-6)
    ratios = forces / limits
    return float(np.max(ratios)), float(np.mean(ratios > 0.98))


def actuator_state_trace(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
) -> dict[str, float]:
    """Return per-actuator torque and tracking state for CSV forensics."""
    actuator_forces = np.asarray(data.actuator_force, dtype=np.float64)
    actuator_limits = np.maximum(
        np.abs(np.asarray(model.actuator_forcerange, dtype=np.float64)).max(axis=1),
        1e-6,
    )
    actual_qpos = np.asarray(data.qpos[env._actuator_qpos_indices_np], dtype=np.float64)
    target_qpos = np.asarray(target_qpos, dtype=np.float64)

    row: dict[str, float] = {}
    for index, joint_name in enumerate(env._actuator_joint_names):
        row[f"pd_act_torque_{joint_name}"] = float(actuator_forces[index])
        row[f"pd_act_torque_ratio_{joint_name}"] = float(
            abs(actuator_forces[index]) / actuator_limits[index]
        )
        row[f"pd_act_target_{joint_name}"] = float(target_qpos[index])
        row[f"pd_act_actual_{joint_name}"] = float(actual_qpos[index])
        row[f"pd_act_error_{joint_name}"] = float(target_qpos[index] - actual_qpos[index])
    return row


def foot_snapshot(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
) -> dict[str, float]:
    """Compact host foot/root state for render traces."""
    foot_pos = np.asarray(data.geom_xpos[env._foot_geom_ids_np], dtype=np.float64)
    left_contact, right_contact = host_foot_contacts(env, data)
    return {
        "left_foot_x": float(foot_pos[0, 0]),
        "left_foot_y": float(foot_pos[0, 1]),
        "left_foot_z": float(foot_pos[0, 2]),
        "right_foot_x": float(foot_pos[1, 0]),
        "right_foot_y": float(foot_pos[1, 1]),
        "right_foot_z": float(foot_pos[1, 2]),
        "left_contact": float(left_contact),
        "right_contact": float(right_contact),
    }


def foot_speed_xy(
    current: dict[str, float],
    previous: dict[str, float] | None,
    dt: float,
    foot_name: str,
) -> float:
    """Approximate per-render-frame foot XY speed."""
    if previous is None or dt <= 0.0:
        return 0.0
    dx = current[f"{foot_name}_foot_x"] - previous[f"{foot_name}_foot_x"]
    dy = current[f"{foot_name}_foot_y"] - previous[f"{foot_name}_foot_y"]
    return float(np.hypot(dx, dy) / dt)


def body_rotation_matrix(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Return one body world rotation matrix."""
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)


def body_pitch_deg(data: mujoco.MjData, body_id: int) -> float:
    """Approximate world pitch angle in degrees from the body rotation matrix."""
    rotation = body_rotation_matrix(data, body_id)
    return float(np.degrees(rotation_pitch_rad(rotation)))


def rotation_pitch_rad(rotation: np.ndarray) -> float:
    """Approximate world pitch angle in radians from a 3x3 rotation matrix."""
    return float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert one MuJoCo/root quaternion in wxyz order to a 3x3 rotation matrix."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - (2.0 * (y * y + z * z)), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - (2.0 * (x * x + z * z)), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - (2.0 * (x * x + y * y))],
        ],
        dtype=np.float64,
    )


def center_of_mass(env: BiomechanicsJoystickEnv, data: mujoco.MjData) -> np.ndarray:
    """Host-side center of mass using the same weighted body set as DeepMimic."""
    body_positions = np.asarray(
        data.xpos[env._deepmimic_body_ids_np],
        dtype=np.float64,
    )
    return np.average(
        body_positions,
        axis=0,
        weights=np.asarray(env._deepmimic_body_weights_np, dtype=np.float64),
    )


def heading_frame_axes_from_rotation(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal heading-forward and heading-left axes from one rotation matrix."""
    heading_forward = rotation[:, 0].copy()
    heading_forward[2] = 0.0
    heading_norm = np.linalg.norm(heading_forward)
    if heading_norm < 1e-9:
        heading_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        heading_forward /= heading_norm
    heading_left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), heading_forward)
    return heading_forward, heading_left


def heading_frame_axes(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal heading-forward and heading-left axes from the anchor body."""
    rotation = body_rotation_matrix(data, env._reference_anchor_body_id)
    return heading_frame_axes_from_rotation(rotation)


def support_point(feet: dict[str, float]) -> np.ndarray:
    """Support point from contacting feet, or mid-feet when airborne."""
    left = np.array(
        [feet["left_foot_x"], feet["left_foot_y"], feet["left_foot_z"]],
        dtype=np.float64,
    )
    right = np.array(
        [feet["right_foot_x"], feet["right_foot_y"], feet["right_foot_z"]],
        dtype=np.float64,
    )
    left_contact = feet["left_contact"] > 0.5
    right_contact = feet["right_contact"] > 0.5
    if left_contact and right_contact:
        return 0.5 * (left + right)
    if left_contact:
        return left
    if right_contact:
        return right
    return 0.5 * (left + right)


def balance_snapshot(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
    feet: dict[str, float],
) -> dict[str, float]:
    """Compact balance diagnostics for backward-fall forensic work."""
    com = center_of_mass(env, data)
    support = support_point(feet)
    heading_forward, heading_left = heading_frame_axes(env, data)
    support_delta = com - support
    pelvis_pitch = body_pitch_deg(data, env._reference_anchor_body_id)
    torso_pitch = body_pitch_deg(data, env._torso_body_id)
    return {
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "com_z": float(com[2]),
        "support_x": float(support[0]),
        "support_y": float(support[1]),
        "support_z": float(support[2]),
        "com_support_sagittal": float(np.dot(support_delta, heading_forward)),
        "com_support_lateral": float(np.dot(support_delta, heading_left)),
        "pelvis_pitch_deg": pelvis_pitch,
        "torso_pitch_deg": torso_pitch,
    }


def emulate_policy_motor_targets(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    motion_time: float,
    desired_ctrl: np.ndarray,
    previous_action: np.ndarray,
    query_reference_at,
    *,
    control_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Emulate the training action map and return motor targets for PD playback."""
    desired_ctrl = np.asarray(desired_ctrl, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    action_size = int(env.action_size)
    zero_action = np.zeros(action_size, dtype=np.float64)
    action_scale = np.asarray(env._action_scale, dtype=np.float64)
    lower_limits = np.asarray(env._actuator_ctrl_lower_limits_np, dtype=np.float64)
    upper_limits = np.asarray(env._actuator_ctrl_upper_limits_np, dtype=np.float64)
    action_smoothing = float(getattr(env._config, "action_smoothing", 1.0))
    reference_action_mode = str(env._config.get("reference_action_mode", "residual"))

    if control_source == "direct_reference":
        return desired_ctrl, zero_action, zero_action, {
            "pd_policy_action_l2": 0.0,
            "pd_smoothed_action_l2": 0.0,
            "pd_policy_action_clip_fraction": 0.0,
            "pd_ctrl_target_rmse": 0.0,
        }

    if control_source == "zero_policy":
        policy_action = zero_action
    elif control_source == "encoded_reference":
        if reference_action_mode == "mimickit":
            center = np.asarray(env._mimickit_action_center, dtype=np.float64)
            half_range = np.maximum(
                np.asarray(env._mimickit_action_half_range, dtype=np.float64),
                1e-6,
            )
            policy_action = (desired_ctrl - center) / half_range
        else:
            replay_step = int(env._config.get("reference_replay_target_step", 0))
            replay_ref = query_reference_at(
                clip_id,
                motion_time + (replay_step * float(env.dt)),
            )
            reference_ctrl = np.asarray(replay_ref["qpos"], dtype=np.float64)
            residual_scale = max(
                float(env._config.get("reference_residual_scale", 1.0)),
                1e-6,
            )
            policy_action = (desired_ctrl - reference_ctrl) / (
                np.maximum(action_scale, 1e-6) * residual_scale
            )
    else:
        raise ValueError(f"Unknown --control-source {control_source}.")

    clipped_policy_action = np.clip(policy_action, -1.0, 1.0)
    smoothed_action = (
        action_smoothing * clipped_policy_action
        + (1.0 - action_smoothing) * previous_action
    )
    if reference_action_mode == "mimickit":
        center = np.asarray(env._mimickit_action_center, dtype=np.float64)
        half_range = np.asarray(env._mimickit_action_half_range, dtype=np.float64)
        motor_targets = center + smoothed_action * half_range
        mimickit_lower = np.maximum(
            np.asarray(env._mimickit_action_lower_limits, dtype=np.float64),
            lower_limits,
        )
        mimickit_upper = np.minimum(
            np.asarray(env._mimickit_action_upper_limits, dtype=np.float64),
            upper_limits,
        )
        motor_targets = np.clip(motor_targets, mimickit_lower, mimickit_upper)
    else:
        replay_step = int(env._config.get("reference_replay_target_step", 0))
        replay_ref = query_reference_at(
            clip_id,
            motion_time + (replay_step * float(env.dt)),
        )
        reference_ctrl = np.asarray(replay_ref["qpos"], dtype=np.float64)
        residual_scale = float(env._config.get("reference_residual_scale", 1.0))
        motor_targets = reference_ctrl + smoothed_action * action_scale * residual_scale
        motor_targets = np.clip(motor_targets, lower_limits, upper_limits)

    return motor_targets, clipped_policy_action, smoothed_action, {
        "pd_policy_action_l2": float(np.linalg.norm(clipped_policy_action)),
        "pd_smoothed_action_l2": float(np.linalg.norm(smoothed_action)),
        "pd_policy_action_clip_fraction": float(
            np.mean(np.abs(policy_action) > 1.0)
        ),
        "pd_ctrl_target_rmse": float(
            np.sqrt(np.mean(np.square(motor_targets - desired_ctrl)))
        ),
    }


def render_clip(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    model = env._mj_model
    base_gravity = np.asarray(model.opt.gravity, dtype=np.float64).copy()
    base_gainprm = np.asarray(model.actuator_gainprm, dtype=np.float64).copy()
    base_biasprm = np.asarray(model.actuator_biasprm, dtype=np.float64).copy()
    base_forcerange = np.asarray(model.actuator_forcerange, dtype=np.float64).copy()
    base_geom_friction = np.asarray(model.geom_friction, dtype=np.float64).copy()
    base_jnt_stiffness = np.asarray(model.jnt_stiffness, dtype=np.float64).copy()
    base_dof_damping = np.asarray(model.dof_damping, dtype=np.float64).copy()
    base_dof_frictionloss = np.asarray(model.dof_frictionloss, dtype=np.float64).copy()
    model.opt.gravity[:] = base_gravity * float(args.gravity_scale)
    model.actuator_gainprm[:] = base_gainprm
    model.actuator_biasprm[:] = base_biasprm
    model.actuator_forcerange[:] = base_forcerange
    model.geom_friction[:] = base_geom_friction
    model.jnt_stiffness[:] = base_jnt_stiffness
    model.dof_damping[:] = base_dof_damping
    model.dof_frictionloss[:] = base_dof_frictionloss
    apply_actuator_scales(
        env,
        model,
        kp_scale=float(args.actuator_kp_scale),
        force_scale=float(args.actuator_force_scale),
        trunk_kp_scale=float(args.trunk_kp_scale),
        pelvis_kp_scale=float(args.pelvis_kp_scale),
        ankle_kp_scale=float(args.ankle_kp_scale),
        hip_kp_scale=float(args.hip_kp_scale),
    )
    apply_passive_joint_scales(
        model,
        trunk_passive_scale=float(args.trunk_passive_scale),
        pelvis_passive_scale=float(args.pelvis_passive_scale),
        head_passive_scale=float(args.head_passive_scale),
        hip_passive_scale=float(args.hip_passive_scale),
    )
    apply_contact_friction_scale(
        env,
        model,
        friction_scale=float(args.contact_friction_scale),
    )
    if args.debug_body_floor_collision:
        enable_body_floor_collision_debug(model)
    data = mujoco.MjData(model)
    kinematic_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.distance = 4.2
    camera.azimuth = 125
    camera.elevation = -18

    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
    frame_count = int(np.asarray(env._bvh_reference_frame_counts)[clip_id])
    frame_time = float(np.asarray(env._bvh_reference_frame_times)[clip_id])
    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    if not 0.0 <= float(args.start_phase) <= 1.0:
        raise ValueError(f"--start-phase must be in [0, 1], got {args.start_phase}.")
    if float(args.reference_speed_scale) <= 0.0:
        raise ValueError(
            "--reference-speed-scale must be positive, got "
            f"{args.reference_speed_scale}."
        )
    motion_start_time = float(args.start_phase) * motion_length
    motion_window_seconds = motion_length - motion_start_time
    if args.segment_seconds is not None:
        motion_window_seconds = min(
            motion_window_seconds,
            max(0.0, float(args.segment_seconds)),
        )
    motion_window_seconds = max(motion_window_seconds, frame_time)
    segment_end_time = motion_start_time + motion_window_seconds
    slowed_window_seconds = motion_window_seconds / float(args.reference_speed_scale)
    if args.video_seconds is not None:
        video_seconds = float(args.video_seconds)
    else:
        video_seconds = max(
            slowed_window_seconds + args.end_hold_seconds,
            float(args.min_video_seconds),
        )
    frame_total = max(1, int(np.ceil(video_seconds * args.fps)))
    if float(args.trace_dt) <= 0.0:
        raise ValueError(f"--trace-dt must be positive, got {args.trace_dt}.")
    rendered: list[np.ndarray] = []
    pd_fail_step: int | None = None
    pd_fail_time: float | None = None
    pd_fail_reason = "none"
    trace_rows: list[dict[str, object]] = []
    detail_trace_rows: list[dict[str, object]] = []
    previous_kinematic_feet: dict[str, float] | None = None
    previous_pd_feet: dict[str, float] | None = None
    previous_detail_kinematic_feet: dict[str, float] | None = None
    previous_detail_pd_feet: dict[str, float] | None = None
    previous_detail_time: float | None = None
    previous_policy_action = np.zeros(int(env.action_size), dtype=np.float64)
    next_detail_trace_time = 0.0
    detail_trace_index = 0
    substep_dt = float(env.dt) / float(env.n_substeps)
    trace_kinematic_data = mujoco.MjData(model)

    def query_reference_at(query_clip_id: int, query_motion_time: float) -> dict[str, np.ndarray]:
        bounded_motion_time = query_motion_time
        if args.segment_seconds is not None or loop_mode == int(LoopMode.CLAMP):
            bounded_motion_time = min(bounded_motion_time, segment_end_time)
        return query_reference_np(env, query_clip_id, bounded_motion_time)

    def append_detail_trace_row(
        sample_elapsed: float,
        pd_elapsed_time: float,
        pd_motion_time: float,
    ) -> None:
        nonlocal previous_detail_kinematic_feet
        nonlocal previous_detail_pd_feet
        nonlocal previous_detail_time
        nonlocal detail_trace_index

        sample_motion_time = motion_start_time + (
            sample_elapsed * float(args.reference_speed_scale)
        )
        if args.segment_seconds is not None or loop_mode == int(LoopMode.CLAMP):
            sample_motion_time = min(sample_motion_time, segment_end_time)
        sample_dt = (
            float(args.trace_dt)
            if previous_detail_time is None
            else max(sample_elapsed - previous_detail_time, 1e-9)
        )
        row: dict[str, object] = {
            "clip_id": clip_id,
            "trace_index": detail_trace_index,
            "trace_time_s": sample_elapsed,
            "trace_motion_time_s": sample_motion_time,
            "gravity_scale": float(args.gravity_scale),
            "pin_root_to_reference": bool(args.pin_root_to_reference),
            "pin_root_position_to_reference": bool(args.pin_root_position_to_reference),
            "pin_root_xy_to_reference": bool(args.pin_root_xy_to_reference),
            "pin_root_z_to_reference": bool(args.pin_root_z_to_reference),
            "pin_root_rotation_to_reference": bool(args.pin_root_rotation_to_reference),
            "actuator_force_scale": float(args.actuator_force_scale),
            "actuator_kp_scale": float(args.actuator_kp_scale),
            "trunk_kp_scale": float(args.trunk_kp_scale),
            "pelvis_kp_scale": float(args.pelvis_kp_scale),
            "ankle_kp_scale": float(args.ankle_kp_scale),
            "hip_kp_scale": float(args.hip_kp_scale),
            "trunk_passive_scale": float(args.trunk_passive_scale),
            "pelvis_passive_scale": float(args.pelvis_passive_scale),
            "head_passive_scale": float(args.head_passive_scale),
            "hip_passive_scale": float(args.hip_passive_scale),
            "contact_friction_scale": float(args.contact_friction_scale),
            "root_z_assist_kp": float(args.root_z_assist_kp),
            "root_z_assist_kd": float(args.root_z_assist_kd),
            "root_z_assist_max_force": float(args.root_z_assist_max_force),
            "root_pitch_assist_kp": float(args.root_pitch_assist_kp),
            "root_pitch_assist_kd": float(args.root_pitch_assist_kd),
            "root_pitch_assist_max_torque": float(args.root_pitch_assist_max_torque),
            "trace_actuator_state": bool(args.trace_actuator_state),
            "control_source": args.control_source,
            "reference_action_mode": args.reference_action_mode,
            "reference_action_center": args.reference_action_center,
            "reference_action_range": args.reference_action_range,
            "reference_action_range_scale": float(args.reference_action_range_scale),
            "reference_residual_scale": float(args.reference_residual_scale),
            "reference_replay_target_step": int(args.reference_replay_target_step),
            "pd_state_time_s": pd_elapsed_time if args.mode in ("pd", "compare") else None,
            "pd_motion_time_s": pd_motion_time if args.mode in ("pd", "compare") else None,
        }
        if args.mode in ("kinematic", "compare"):
            ref = query_reference_at(clip_id, sample_motion_time)
            qpos, qvel = build_full_state(env, ref)
            trace_kinematic_data.qpos[:] = qpos
            trace_kinematic_data.qvel[:] = qvel
            trace_kinematic_data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
            mujoco.mj_forward(model, trace_kinematic_data)
            kin_feet = foot_snapshot(env, trace_kinematic_data)
            kin_balance = balance_snapshot(env, trace_kinematic_data, kin_feet)
            row.update(
                {
                    "kin_root_x": float(trace_kinematic_data.qpos[0]),
                    "kin_root_y": float(trace_kinematic_data.qpos[1]),
                    "kin_root_z": float(trace_kinematic_data.qpos[2]),
                    "kin_left_foot_z": kin_feet["left_foot_z"],
                    "kin_right_foot_z": kin_feet["right_foot_z"],
                    "kin_left_contact": kin_feet["left_contact"],
                    "kin_right_contact": kin_feet["right_contact"],
                    "kin_left_foot_speed_xy": foot_speed_xy(
                        kin_feet,
                        previous_detail_kinematic_feet,
                        sample_dt,
                        "left",
                    ),
                    "kin_right_foot_speed_xy": foot_speed_xy(
                        kin_feet,
                        previous_detail_kinematic_feet,
                        sample_dt,
                        "right",
                    ),
                    "kin_com_support_sagittal": kin_balance["com_support_sagittal"],
                    "kin_com_support_lateral": kin_balance["com_support_lateral"],
                    "kin_pelvis_pitch_deg": kin_balance["pelvis_pitch_deg"],
                    "kin_torso_pitch_deg": kin_balance["torso_pitch_deg"],
                }
            )
            previous_detail_kinematic_feet = kin_feet
        if args.mode in ("pd", "compare"):
            target_ref = query_reference_at(clip_id, sample_motion_time)
            pd_feet = foot_snapshot(env, data)
            pd_balance = balance_snapshot(env, data, pd_feet)
            target_qpos = np.asarray(target_ref["qpos"], dtype=np.float64)
            actual_qpos = data.qpos[env._actuator_qpos_indices_np]
            force_ratio, saturated_fraction = actuator_force_stats(model, data)
            _, clipped_policy_action, smoothed_action, control_diag = emulate_policy_motor_targets(
                env,
                clip_id,
                sample_motion_time,
                target_qpos,
                previous_policy_action,
                query_reference_at,
                control_source=args.control_source,
            )
            actuator_trace = (
                actuator_state_trace(env, model, data, target_qpos)
                if args.trace_actuator_state
                else {}
            )
            assist_force = root_z_assist_force(
                env,
                data,
                target_ref,
                kp=float(args.root_z_assist_kp),
                kd=float(args.root_z_assist_kd),
                max_force=float(args.root_z_assist_max_force),
            )
            _, pitch_diag = root_pitch_assist_torque(
                env,
                data,
                target_ref,
                kp=float(args.root_pitch_assist_kp),
                kd=float(args.root_pitch_assist_kd),
                max_torque=float(args.root_pitch_assist_max_torque),
            )
            row.update(
                {
                    "pd_root_x": float(data.qpos[0]),
                    "pd_root_y": float(data.qpos[1]),
                    "pd_root_z": float(data.qpos[2]),
                    "pd_torso_up": torso_up(env, data),
                    "pd_fail_reason": fail_reason(env, model, data),
                    "pd_pose_rmse": float(
                        np.sqrt(np.mean(np.square(actual_qpos - target_qpos)))
                    ),
                    "pd_left_foot_z": pd_feet["left_foot_z"],
                    "pd_right_foot_z": pd_feet["right_foot_z"],
                    "pd_left_contact": pd_feet["left_contact"],
                    "pd_right_contact": pd_feet["right_contact"],
                    "pd_left_foot_speed_xy": foot_speed_xy(
                        pd_feet,
                        previous_detail_pd_feet,
                        sample_dt,
                        "left",
                    ),
                    "pd_right_foot_speed_xy": foot_speed_xy(
                        pd_feet,
                        previous_detail_pd_feet,
                        sample_dt,
                        "right",
                    ),
                    "pd_com_x": pd_balance["com_x"],
                    "pd_com_y": pd_balance["com_y"],
                    "pd_com_z": pd_balance["com_z"],
                    "pd_support_x": pd_balance["support_x"],
                    "pd_support_y": pd_balance["support_y"],
                    "pd_support_z": pd_balance["support_z"],
                    "pd_com_support_sagittal": pd_balance["com_support_sagittal"],
                    "pd_com_support_lateral": pd_balance["com_support_lateral"],
                    "pd_pelvis_pitch_deg": pd_balance["pelvis_pitch_deg"],
                    "pd_torso_pitch_deg": pd_balance["torso_pitch_deg"],
                    "pd_root_z_assist_force": assist_force,
                    "pd_max_actuator_force_ratio": force_ratio,
                    "pd_saturated_actuator_fraction": saturated_fraction,
                    "pd_policy_action_l2": control_diag["pd_policy_action_l2"],
                    "pd_smoothed_action_l2": control_diag["pd_smoothed_action_l2"],
                    "pd_policy_action_clip_fraction": control_diag[
                        "pd_policy_action_clip_fraction"
                    ],
                    "pd_ctrl_target_rmse": control_diag["pd_ctrl_target_rmse"],
                    **pitch_diag,
                    **actuator_trace,
                }
            )
            previous_detail_pd_feet = pd_feet
        detail_trace_rows.append(row)
        previous_detail_time = sample_elapsed
        detail_trace_index += 1

    try:
        initial_ref = query_reference_at(clip_id, motion_start_time)
        qpos, qvel = build_full_state(env, initial_ref)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        initial_ctrl, previous_policy_action, _initial_smoothed, _initial_diag = (
            emulate_policy_motor_targets(
                env,
                clip_id,
                motion_start_time,
                np.asarray(initial_ref["qpos"], dtype=np.float64),
                np.zeros(int(env.action_size), dtype=np.float64),
                query_reference_at,
                control_source=args.control_source,
            )
        )
        data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(model, data)

        pd_motion_time = motion_start_time
        pd_elapsed_time = 0.0
        append_detail_trace_row(0.0, pd_elapsed_time, pd_motion_time)
        next_detail_trace_time = float(args.trace_dt)
        for video_frame in range(frame_total):
            render_elapsed = video_frame / float(args.fps)
            render_motion_time = motion_start_time + (
                render_elapsed * float(args.reference_speed_scale)
            )
            if args.segment_seconds is not None or loop_mode == int(LoopMode.CLAMP):
                render_motion_time = min(render_motion_time, segment_end_time)

            if args.mode in ("kinematic", "compare"):
                ref = query_reference_at(clip_id, render_motion_time)
                qpos, qvel = build_full_state(env, ref)
                kinematic_data.qpos[:] = qpos
                kinematic_data.qvel[:] = qvel
                kinematic_data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
                mujoco.mj_forward(model, kinematic_data)
            if args.mode in ("pd", "compare"):
                target_ref = query_reference_at(clip_id, pd_motion_time)
                while pd_elapsed_time + 1e-9 < render_elapsed:
                    target_time = pd_motion_time + (
                        float(env.dt) * float(args.reference_speed_scale)
                    )
                    if args.segment_seconds is not None or loop_mode == int(LoopMode.CLAMP):
                        target_time = min(target_time, segment_end_time)
                    target_ref = query_reference_at(clip_id, target_time)
                    data.xfrc_applied[:, :] = 0.0
                    if args.pin_root_to_reference or args.pin_root_position_to_reference:
                        data.qpos[:3] = np.asarray(target_ref["root_pos"], dtype=np.float64)
                        data.qvel[:3] = np.asarray(target_ref["root_vel"], dtype=np.float64)
                    elif args.pin_root_xy_to_reference:
                        root_pos = np.asarray(target_ref["root_pos"], dtype=np.float64)
                        root_vel = np.asarray(target_ref["root_vel"], dtype=np.float64)
                        data.qpos[0:2] = root_pos[0:2]
                        data.qvel[0:2] = root_vel[0:2]
                    elif args.pin_root_z_to_reference:
                        root_pos = np.asarray(target_ref["root_pos"], dtype=np.float64)
                        root_vel = np.asarray(target_ref["root_vel"], dtype=np.float64)
                        data.qpos[2] = root_pos[2]
                        data.qvel[2] = root_vel[2]
                    if args.pin_root_to_reference or args.pin_root_rotation_to_reference:
                        data.qpos[3:7] = np.asarray(target_ref["root_quat"], dtype=np.float64)
                        data.qvel[3:6] = np.asarray(
                            target_ref["root_angvel"],
                            dtype=np.float64,
                        )
                    assist_force = root_z_assist_force(
                        env,
                        data,
                        target_ref,
                        kp=float(args.root_z_assist_kp),
                        kd=float(args.root_z_assist_kd),
                        max_force=float(args.root_z_assist_max_force),
                    )
                    assist_torque, _ = root_pitch_assist_torque(
                        env,
                        data,
                        target_ref,
                        kp=float(args.root_pitch_assist_kp),
                        kd=float(args.root_pitch_assist_kd),
                        max_torque=float(args.root_pitch_assist_max_torque),
                    )
                    if assist_force != 0.0:
                        data.xfrc_applied[env._reference_anchor_body_id, 2] = assist_force
                    if np.any(np.abs(assist_torque) > 0.0):
                        data.xfrc_applied[env._reference_anchor_body_id, 3:6] = assist_torque
                    data.ctrl[:], previous_policy_action, _smoothed_action, _control_diag = (
                        emulate_policy_motor_targets(
                            env,
                            clip_id,
                            target_time,
                            np.asarray(target_ref["qpos"], dtype=np.float64),
                            previous_policy_action,
                            query_reference_at,
                            control_source=args.control_source,
                        )
                    )
                    for _ in range(env.n_substeps):
                        mujoco.mj_step(model, data)
                        pd_elapsed_time += substep_dt
                        while (
                            next_detail_trace_time <= pd_elapsed_time + 1e-9
                            and next_detail_trace_time <= video_seconds + 1e-9
                        ):
                            append_detail_trace_row(
                                next_detail_trace_time,
                                pd_elapsed_time,
                                pd_motion_time,
                            )
                            next_detail_trace_time += float(args.trace_dt)
                    pd_motion_time = target_time
                    reason = fail_reason(env, model, data)
                    if pd_fail_step is None and reason != "none":
                        pd_fail_step = int(round(pd_elapsed_time / float(env.dt)))
                        pd_fail_time = pd_elapsed_time
                        pd_fail_reason = reason
            elif args.mode == "kinematic":
                while (
                    next_detail_trace_time <= render_elapsed + 1e-9
                    and next_detail_trace_time <= video_seconds + 1e-9
                ):
                    append_detail_trace_row(
                        next_detail_trace_time,
                        pd_elapsed_time,
                        pd_motion_time,
                    )
                    next_detail_trace_time += float(args.trace_dt)

            row: dict[str, object] = {
                "clip_id": clip_id,
                "video_frame": video_frame,
                "render_time_s": render_elapsed,
                "render_motion_time_s": render_motion_time,
                "gravity_scale": float(args.gravity_scale),
                "pin_root_to_reference": bool(args.pin_root_to_reference),
                "pin_root_position_to_reference": bool(args.pin_root_position_to_reference),
                "pin_root_xy_to_reference": bool(args.pin_root_xy_to_reference),
                "pin_root_z_to_reference": bool(args.pin_root_z_to_reference),
                "pin_root_rotation_to_reference": bool(args.pin_root_rotation_to_reference),
                "actuator_force_scale": float(args.actuator_force_scale),
                "actuator_kp_scale": float(args.actuator_kp_scale),
                "trunk_kp_scale": float(args.trunk_kp_scale),
                "pelvis_kp_scale": float(args.pelvis_kp_scale),
                "ankle_kp_scale": float(args.ankle_kp_scale),
                "hip_kp_scale": float(args.hip_kp_scale),
                "trunk_passive_scale": float(args.trunk_passive_scale),
                "pelvis_passive_scale": float(args.pelvis_passive_scale),
                "head_passive_scale": float(args.head_passive_scale),
                "hip_passive_scale": float(args.hip_passive_scale),
                "contact_friction_scale": float(args.contact_friction_scale),
                "root_z_assist_kp": float(args.root_z_assist_kp),
                "root_z_assist_kd": float(args.root_z_assist_kd),
                "root_z_assist_max_force": float(args.root_z_assist_max_force),
                "root_pitch_assist_kp": float(args.root_pitch_assist_kp),
                "root_pitch_assist_kd": float(args.root_pitch_assist_kd),
                "root_pitch_assist_max_torque": float(args.root_pitch_assist_max_torque),
                "trace_actuator_state": bool(args.trace_actuator_state),
                "control_source": args.control_source,
                "reference_action_mode": args.reference_action_mode,
                "reference_action_center": args.reference_action_center,
                "reference_action_range": args.reference_action_range,
                "reference_action_range_scale": float(args.reference_action_range_scale),
                "reference_residual_scale": float(args.reference_residual_scale),
                "reference_replay_target_step": int(args.reference_replay_target_step),
                "pd_sim_time_s": pd_elapsed_time if args.mode in ("pd", "compare") else None,
                "pd_motion_time_s": pd_motion_time if args.mode in ("pd", "compare") else None,
            }
            if args.mode in ("kinematic", "compare"):
                kin_feet = foot_snapshot(env, kinematic_data)
                kin_balance = balance_snapshot(env, kinematic_data, kin_feet)
                row.update(
                    {
                        "kin_root_x": float(kinematic_data.qpos[0]),
                        "kin_root_y": float(kinematic_data.qpos[1]),
                        "kin_root_z": float(kinematic_data.qpos[2]),
                        "kin_left_foot_z": kin_feet["left_foot_z"],
                        "kin_right_foot_z": kin_feet["right_foot_z"],
                        "kin_left_contact": kin_feet["left_contact"],
                        "kin_right_contact": kin_feet["right_contact"],
                        "kin_left_foot_speed_xy": foot_speed_xy(
                            kin_feet,
                            previous_kinematic_feet,
                            1.0 / float(args.fps),
                            "left",
                        ),
                        "kin_right_foot_speed_xy": foot_speed_xy(
                            kin_feet,
                            previous_kinematic_feet,
                            1.0 / float(args.fps),
                            "right",
                        ),
                        "kin_com_support_sagittal": kin_balance["com_support_sagittal"],
                        "kin_com_support_lateral": kin_balance["com_support_lateral"],
                        "kin_pelvis_pitch_deg": kin_balance["pelvis_pitch_deg"],
                        "kin_torso_pitch_deg": kin_balance["torso_pitch_deg"],
                    }
                )
                previous_kinematic_feet = kin_feet
            if args.mode in ("pd", "compare"):
                pd_feet = foot_snapshot(env, data)
                pd_balance = balance_snapshot(env, data, pd_feet)
                target_qpos = np.asarray(target_ref["qpos"], dtype=np.float64)
                actual_qpos = data.qpos[env._actuator_qpos_indices_np]
                force_ratio, saturated_fraction = actuator_force_stats(model, data)
                _, clipped_policy_action, smoothed_action, control_diag = emulate_policy_motor_targets(
                    env,
                    clip_id,
                    pd_motion_time,
                    target_qpos,
                    previous_policy_action,
                    query_reference_at,
                    control_source=args.control_source,
                )
                actuator_trace = (
                    actuator_state_trace(env, model, data, target_qpos)
                    if args.trace_actuator_state
                    else {}
                )
                assist_force = root_z_assist_force(
                    env,
                    data,
                    target_ref,
                    kp=float(args.root_z_assist_kp),
                    kd=float(args.root_z_assist_kd),
                    max_force=float(args.root_z_assist_max_force),
                )
                _, pitch_diag = root_pitch_assist_torque(
                    env,
                    data,
                    target_ref,
                    kp=float(args.root_pitch_assist_kp),
                    kd=float(args.root_pitch_assist_kd),
                    max_torque=float(args.root_pitch_assist_max_torque),
                )
                row.update(
                    {
                        "pd_root_x": float(data.qpos[0]),
                        "pd_root_y": float(data.qpos[1]),
                        "pd_root_z": float(data.qpos[2]),
                        "pd_torso_up": torso_up(env, data),
                        "pd_fail_reason": fail_reason(env, model, data),
                        "pd_pose_rmse": float(
                            np.sqrt(np.mean(np.square(actual_qpos - target_qpos)))
                        ),
                        "pd_left_foot_z": pd_feet["left_foot_z"],
                        "pd_right_foot_z": pd_feet["right_foot_z"],
                        "pd_left_contact": pd_feet["left_contact"],
                        "pd_right_contact": pd_feet["right_contact"],
                        "pd_left_foot_speed_xy": foot_speed_xy(
                            pd_feet,
                            previous_pd_feet,
                            1.0 / float(args.fps),
                            "left",
                        ),
                        "pd_right_foot_speed_xy": foot_speed_xy(
                            pd_feet,
                            previous_pd_feet,
                            1.0 / float(args.fps),
                            "right",
                        ),
                        "pd_com_x": pd_balance["com_x"],
                        "pd_com_y": pd_balance["com_y"],
                        "pd_com_z": pd_balance["com_z"],
                        "pd_support_x": pd_balance["support_x"],
                        "pd_support_y": pd_balance["support_y"],
                        "pd_support_z": pd_balance["support_z"],
                        "pd_com_support_sagittal": pd_balance["com_support_sagittal"],
                        "pd_com_support_lateral": pd_balance["com_support_lateral"],
                        "pd_pelvis_pitch_deg": pd_balance["pelvis_pitch_deg"],
                        "pd_torso_pitch_deg": pd_balance["torso_pitch_deg"],
                        "pd_root_z_assist_force": assist_force,
                        "pd_max_actuator_force_ratio": force_ratio,
                        "pd_saturated_actuator_fraction": saturated_fraction,
                        "pd_policy_action_l2": control_diag["pd_policy_action_l2"],
                        "pd_smoothed_action_l2": control_diag["pd_smoothed_action_l2"],
                        "pd_policy_action_clip_fraction": control_diag[
                            "pd_policy_action_clip_fraction"
                        ],
                        "pd_ctrl_target_rmse": control_diag["pd_ctrl_target_rmse"],
                        **pitch_diag,
                        **actuator_trace,
                    }
                )
                previous_pd_feet = pd_feet
            trace_rows.append(row)

            if args.mode == "compare":
                camera.lookat[:] = np.asarray(kinematic_data.qpos[:3], dtype=np.float64)
                renderer.update_scene(kinematic_data, camera=camera)
                kinematic_frame = renderer.render()
                camera.lookat[:] = np.asarray(data.qpos[:3], dtype=np.float64)
                renderer.update_scene(data, camera=camera)
                pd_frame = renderer.render()
                rendered.append(
                    overlay_frame_title(
                        np.concatenate([kinematic_frame, pd_frame], axis=1),
                        args.overlay_title,
                        args.mode,
                    )
                )
            else:
                render_data = kinematic_data if args.mode == "kinematic" else data
                camera.lookat[:] = np.asarray(render_data.qpos[:3], dtype=np.float64)
                renderer.update_scene(render_data, camera=camera)
                rendered.append(
                    overlay_frame_title(
                        renderer.render(),
                        args.overlay_title,
                        args.mode,
                    )
                )
    finally:
        renderer.close()

    label = clip_label(loop_mode)
    stem = safe_name(source_label(env, clip_id))
    duration_label = f"{motion_length:.2f}".replace(".", "p")
    start_label = f"p{args.start_phase:.3f}".replace(".", "p")
    speed_label = f"x{args.reference_speed_scale:.2f}".replace(".", "p")
    file_name = (
        f"clip_{clip_id:03d}_{label}_{stem}_"
        f"frames{frame_count:03d}_dur{duration_label}s_"
        f"{start_label}_{speed_label}_{args.mode}.mp4"
    )
    output_path = args.out_dir / file_name
    write_video(output_path, rendered, args.fps)

    return output_path, {
        "clip_id": clip_id,
        "file": output_path.name,
        "mode": args.mode,
        "label": label,
        "loop_mode": LoopMode(loop_mode).name,
        "source_path": env._bvh_reference_source_paths[clip_id],
        "source_start_frame": int(env._bvh_reference_source_start_frames[clip_id]),
        "source_end_frame": int(env._bvh_reference_source_end_frames[clip_id]),
        "frame_count": frame_count,
        "frame_time": frame_time,
        "motion_length_s": motion_length,
        "motion_start_time_s": motion_start_time,
        "motion_window_s": motion_window_seconds,
        "reference_speed_scale": float(args.reference_speed_scale),
        "gravity_scale": float(args.gravity_scale),
        "pin_root_to_reference": bool(args.pin_root_to_reference),
        "pin_root_position_to_reference": bool(args.pin_root_position_to_reference),
        "pin_root_xy_to_reference": bool(args.pin_root_xy_to_reference),
        "pin_root_z_to_reference": bool(args.pin_root_z_to_reference),
        "pin_root_rotation_to_reference": bool(args.pin_root_rotation_to_reference),
        "actuator_force_scale": float(args.actuator_force_scale),
        "actuator_kp_scale": float(args.actuator_kp_scale),
        "trunk_kp_scale": float(args.trunk_kp_scale),
        "pelvis_kp_scale": float(args.pelvis_kp_scale),
        "ankle_kp_scale": float(args.ankle_kp_scale),
        "hip_kp_scale": float(args.hip_kp_scale),
        "trunk_passive_scale": float(args.trunk_passive_scale),
        "pelvis_passive_scale": float(args.pelvis_passive_scale),
        "head_passive_scale": float(args.head_passive_scale),
        "hip_passive_scale": float(args.hip_passive_scale),
        "contact_friction_scale": float(args.contact_friction_scale),
        "root_z_assist_kp": float(args.root_z_assist_kp),
        "root_z_assist_kd": float(args.root_z_assist_kd),
        "root_z_assist_max_force": float(args.root_z_assist_max_force),
        "root_pitch_assist_kp": float(args.root_pitch_assist_kp),
        "root_pitch_assist_kd": float(args.root_pitch_assist_kd),
        "root_pitch_assist_max_torque": float(args.root_pitch_assist_max_torque),
        "trace_actuator_state": bool(args.trace_actuator_state),
        "control_source": args.control_source,
        "reference_action_mode": args.reference_action_mode,
        "reference_action_center": args.reference_action_center,
        "reference_action_range": args.reference_action_range,
        "reference_action_range_scale": float(args.reference_action_range_scale),
        "reference_residual_scale": float(args.reference_residual_scale),
        "reference_replay_target_step": int(args.reference_replay_target_step),
        "trace_dt_s": float(args.trace_dt),
        "rendered_seconds": video_seconds,
        "pd_fail_step": pd_fail_step,
        "pd_fail_time_s": pd_fail_time,
        "pd_fail_reason": pd_fail_reason,
        "debug_body_floor_collision": bool(args.debug_body_floor_collision),
        "overlay_title": args.overlay_title or "",
    }, trace_rows, detail_trace_rows


def main() -> None:
    args = parse_args()
    jax.config.update("jax_platform_name", "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    clip_count = int(np.asarray(env._bvh_reference_clip_count))
    if args.clip_id is not None:
        if args.clip_id < 0 or args.clip_id >= clip_count:
            raise ValueError(
                f"--clip-id {args.clip_id} out of range for {clip_count} clips."
            )
        clip_ids = [int(args.clip_id)]
    elif args.max_clips is not None:
        clip_count = min(clip_count, int(args.max_clips))
        clip_ids = list(range(clip_count))
    else:
        clip_ids = list(range(clip_count))

    rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    detail_trace_rows: list[dict[str, object]] = []
    print(f"Rendering {len(clip_ids)} BVH clips to {args.out_dir}")
    for clip_id in clip_ids:
        path, row, clip_trace_rows, clip_detail_trace_rows = render_clip(
            env,
            clip_id,
            args,
        )
        rows.append(row)
        trace_rows.extend(clip_trace_rows)
        detail_trace_rows.extend(clip_detail_trace_rows)
        print(
            f"{clip_id:03d} {row['label']} {row['motion_length_s']:.3f}s "
            f"-> {path.name}",
            flush=True,
        )

    manifest_path = args.out_dir / f"manifest_{args.mode}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest: {manifest_path}")
    if trace_rows:
        trace_path = args.out_dir / f"trace_{args.mode}.csv"
        fieldnames = sorted({key for row in trace_rows for key in row})
        with trace_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_rows)
        print(f"Trace: {trace_path}")
    if detail_trace_rows:
        detail_trace_path = args.out_dir / f"trace_detail_{args.mode}.csv"
        fieldnames = sorted({key for row in detail_trace_rows for key in row})
        with detail_trace_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_trace_rows)
        print(f"Detail trace: {detail_trace_path}")


if __name__ == "__main__":
    main()
