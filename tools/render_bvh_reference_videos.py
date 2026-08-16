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
from debug_reference_playback import build_full_state, query_reference_np  # noqa: E402


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
    parser.add_argument("--mode", choices=["kinematic", "pd"], default="kinematic")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-video-seconds", type=float, default=1.2)
    parser.add_argument("--end-hold-seconds", type=float, default=0.35)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--training-filtered",
        action="store_true",
        help="Render only clips kept by the current training filter.",
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


def render_clip(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object]]:
    model = env._mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.distance = 4.2
    camera.azimuth = 125
    camera.elevation = -18

    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
    frame_count = int(np.asarray(env._bvh_reference_frame_counts)[clip_id])
    frame_time = float(np.asarray(env._bvh_reference_frame_times)[clip_id])
    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    video_seconds = max(
        motion_length + args.end_hold_seconds,
        float(args.min_video_seconds),
    )
    frame_total = max(1, int(np.ceil(video_seconds * args.fps)))
    rendered: list[np.ndarray] = []

    try:
        ref = query_reference_np(env, clip_id, 0.0)
        qpos, qvel = build_full_state(env, ref)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
        mujoco.mj_forward(model, data)

        for video_frame in range(frame_total):
            time_s = video_frame / float(args.fps)
            if loop_mode == int(LoopMode.CLAMP):
                time_s = min(time_s, motion_length)

            if args.mode == "kinematic":
                ref = query_reference_np(env, clip_id, time_s)
                qpos, qvel = build_full_state(env, ref)
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
                mujoco.mj_forward(model, data)
            else:
                ref = query_reference_np(env, clip_id, time_s + float(env.dt))
                data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
                for _ in range(env.n_substeps):
                    mujoco.mj_step(model, data)

            camera.lookat[:] = np.asarray(data.qpos[:3], dtype=np.float64)
            renderer.update_scene(data, camera=camera)
            rendered.append(renderer.render())
    finally:
        renderer.close()

    label = clip_label(loop_mode)
    stem = safe_name(source_label(env, clip_id))
    duration_label = f"{motion_length:.2f}".replace(".", "p")
    file_name = (
        f"clip_{clip_id:03d}_{label}_{stem}_"
        f"frames{frame_count:03d}_dur{duration_label}s_{args.mode}.mp4"
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
        "rendered_seconds": video_seconds,
    }


def main() -> None:
    args = parse_args()
    jax.config.update("jax_platform_name", "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    clip_count = int(np.asarray(env._bvh_reference_clip_count))
    if args.max_clips is not None:
        clip_count = min(clip_count, int(args.max_clips))

    rows: list[dict[str, object]] = []
    print(f"Rendering {clip_count} BVH clips to {args.out_dir}")
    for clip_id in range(clip_count):
        path, row = render_clip(env, clip_id, args)
        rows.append(row)
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


if __name__ == "__main__":
    main()
