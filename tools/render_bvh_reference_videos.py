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
        "reference_action_mode": "residual",
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


def render_clip(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    model = env._mj_model
    base_gravity = np.asarray(model.opt.gravity, dtype=np.float64).copy()
    model.opt.gravity[:] = base_gravity * float(args.gravity_scale)
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
    next_detail_trace_time = 0.0
    detail_trace_index = 0
    substep_dt = float(env.dt) / float(env.n_substeps)
    trace_kinematic_data = mujoco.MjData(model)

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
            "pd_state_time_s": pd_elapsed_time if args.mode in ("pd", "compare") else None,
            "pd_motion_time_s": pd_motion_time if args.mode in ("pd", "compare") else None,
        }
        if args.mode in ("kinematic", "compare"):
            ref = query_reference_np(env, clip_id, sample_motion_time)
            qpos, qvel = build_full_state(env, ref)
            trace_kinematic_data.qpos[:] = qpos
            trace_kinematic_data.qvel[:] = qvel
            trace_kinematic_data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
            mujoco.mj_forward(model, trace_kinematic_data)
            kin_feet = foot_snapshot(env, trace_kinematic_data)
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
                }
            )
            previous_detail_kinematic_feet = kin_feet
        if args.mode in ("pd", "compare"):
            target_ref = query_reference_np(env, clip_id, sample_motion_time)
            pd_feet = foot_snapshot(env, data)
            target_qpos = np.asarray(target_ref["qpos"], dtype=np.float64)
            actual_qpos = data.qpos[env._actuator_qpos_indices_np]
            force_ratio, saturated_fraction = actuator_force_stats(model, data)
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
                    "pd_max_actuator_force_ratio": force_ratio,
                    "pd_saturated_actuator_fraction": saturated_fraction,
                }
            )
            previous_detail_pd_feet = pd_feet
        detail_trace_rows.append(row)
        previous_detail_time = sample_elapsed
        detail_trace_index += 1

    try:
        initial_ref = query_reference_np(env, clip_id, motion_start_time)
        qpos, qvel = build_full_state(env, initial_ref)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = np.asarray(initial_ref["qpos"], dtype=np.float64)
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
                ref = query_reference_np(env, clip_id, render_motion_time)
                qpos, qvel = build_full_state(env, ref)
                kinematic_data.qpos[:] = qpos
                kinematic_data.qvel[:] = qvel
                kinematic_data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
                mujoco.mj_forward(model, kinematic_data)
            if args.mode in ("pd", "compare"):
                target_ref = query_reference_np(env, clip_id, pd_motion_time)
                while pd_elapsed_time + 1e-9 < render_elapsed:
                    target_time = pd_motion_time + (
                        float(env.dt) * float(args.reference_speed_scale)
                    )
                    if args.segment_seconds is not None or loop_mode == int(LoopMode.CLAMP):
                        target_time = min(target_time, segment_end_time)
                    target_ref = query_reference_np(env, clip_id, target_time)
                    data.ctrl[:] = np.asarray(target_ref["qpos"], dtype=np.float64)
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
                "pd_sim_time_s": pd_elapsed_time if args.mode in ("pd", "compare") else None,
                "pd_motion_time_s": pd_motion_time if args.mode in ("pd", "compare") else None,
            }
            if args.mode in ("kinematic", "compare"):
                kin_feet = foot_snapshot(env, kinematic_data)
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
                    }
                )
                previous_kinematic_feet = kin_feet
            if args.mode in ("pd", "compare"):
                pd_feet = foot_snapshot(env, data)
                target_qpos = np.asarray(target_ref["qpos"], dtype=np.float64)
                actual_qpos = data.qpos[env._actuator_qpos_indices_np]
                force_ratio, saturated_fraction = actuator_force_stats(model, data)
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
                        "pd_max_actuator_force_ratio": force_ratio,
                        "pd_saturated_actuator_fraction": saturated_fraction,
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
                rendered.append(np.concatenate([kinematic_frame, pd_frame], axis=1))
            else:
                render_data = kinematic_data if args.mode == "kinematic" else data
                camera.lookat[:] = np.asarray(render_data.qpos[:3], dtype=np.float64)
                renderer.update_scene(render_data, camera=camera)
                rendered.append(renderer.render())
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
        "trace_dt_s": float(args.trace_dt),
        "rendered_seconds": video_seconds,
        "pd_fail_step": pd_fail_step,
        "pd_fail_time_s": pd_fail_time,
        "pd_fail_reason": pd_fail_reason,
        "debug_body_floor_collision": bool(args.debug_body_floor_collision),
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
