"""Static reference-pose hold diagnostic for BVH/SMPL clips.

This is a stricter pre-training oracle than kinematic playback: it asks whether
one exact retargeted reference frame is physically supportable when held by the
MuJoCo position actuators with zero root velocity.  If many phases fall from a
static hold, PPO is not the first suspect; the retargeted pose, support foot, or
body geometry is already marginal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv  # noqa: E402
from bvh_reference import LoopMode  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_BVH_REFERENCE_LIST,
    DEFAULT_SMPL_REFERENCE_FILES,
    EnvConfig,
)
from debug_reference_playback import build_full_state, query_reference_np, torso_up  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gait", choices=["bvh", "smpl"], default="bvh")
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--clip-id", type=int, default=None)
    parser.add_argument(
        "--phases",
        default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875",
        help="Comma-separated normalized clip phases to test.",
    )
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--print-every", type=int, default=0)
    parser.add_argument(
        "--reference-loop-mode",
        choices=["auto", "wrap", "clamp"],
        default=EnvConfig.reference_loop_mode,
    )
    parser.add_argument(
        "--reference-root-xy-scale",
        type=float,
        default=EnvConfig.reference_root_xy_scale,
    )
    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=1.0,
        help=(
            "Scale MuJoCo gravity for hold diagnostics. 0.0 isolates actuator "
            "pose tracking from balance/contact loading."
        ),
    )
    return parser.parse_args()


def default_reference_files(reference_gait: str) -> list[str]:
    if reference_gait == "smpl":
        return [path.as_posix() for path in DEFAULT_SMPL_REFERENCE_FILES]
    return [DEFAULT_BVH_REFERENCE_LIST.as_posix()]


def parse_phases(raw_phases: str) -> list[float]:
    phases = []
    for item in raw_phases.split(","):
        item = item.strip()
        if not item:
            continue
        phase = float(item)
        if phase < 0.0 or phase > 1.0:
            raise ValueError(f"Phase must be in [0, 1], got {phase}.")
        phases.append(phase)
    if not phases:
        raise ValueError("At least one phase is required.")
    return phases


def fail_reason(low: bool, tipped: bool) -> str:
    if low:
        return "low_height"
    if tipped:
        return "tipped"
    return "none"


def center_of_mass(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = np.asarray(model.body_mass[1:], dtype=np.float64)
    positions = np.asarray(data.xpos[1:], dtype=np.float64)
    return np.average(positions, axis=0, weights=masses)


def support_margin_xy(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    """Approximate signed COM margin inside both foot-sole AABBs in world XY."""
    foot_ids = np.asarray(env._foot_geom_ids_np)
    centers = np.asarray(data.geom_xpos[foot_ids, :2], dtype=np.float64)
    sizes = np.asarray(model.geom_size[foot_ids, :2], dtype=np.float64)
    half_extents = np.maximum(sizes, 0.04)
    min_xy = np.min(centers - half_extents, axis=0)
    max_xy = np.max(centers + half_extents, axis=0)
    com_xy = center_of_mass(model, data)[:2]
    inside_margin = np.minimum(com_xy - min_xy, max_xy - com_xy)
    return float(np.min(inside_margin))


def run_hold(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    clip_id: int,
    phase: float,
    seconds: float,
    print_every: int,
) -> dict[str, float | int | str | bool]:
    data = mujoco.MjData(model)
    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    motion_time = phase * motion_length
    ref = query_reference_np(env, clip_id, motion_time)
    qpos, _qvel = build_full_state(env, ref)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = np.asarray(ref["qpos"], dtype=np.float64)
    mujoco.mj_forward(model, data)

    step_count = max(1, int(round(seconds / float(env.dt))))
    min_height = float(data.qpos[2])
    min_torso_up = torso_up(env, data)
    min_support_margin = support_margin_xy(env, model, data)
    max_pose_rmse = 0.0
    reason = "none"
    final_step = step_count - 1

    target_qpos = np.asarray(ref["qpos"], dtype=np.float64)
    for step in range(step_count):
        for _ in range(env.n_substeps):
            mujoco.mj_step(model, data)

        actual_qpos = data.qpos[np.asarray(env._actuator_qpos_indices_np)]
        pose_rmse = float(np.sqrt(np.mean(np.square(actual_qpos - target_qpos))))
        max_pose_rmse = max(max_pose_rmse, pose_rmse)
        min_height = min(min_height, float(data.qpos[2]))
        current_torso_up = torso_up(env, data)
        min_torso_up = min(min_torso_up, current_torso_up)
        min_support_margin = min(
            min_support_margin,
            support_margin_xy(env, model, data),
        )

        low = data.qpos[2] < env.MIN_STANDING_HEIGHT_RATIO * float(model.qpos0[2])
        tipped = current_torso_up < 0.25
        done = bool(low or tipped)
        reason = fail_reason(bool(low), bool(tipped))

        should_print = (
            print_every > 0
            and (step % print_every == 0 or done or step == step_count - 1)
        )
        if should_print:
            print(
                f"clip={clip_id:02d} phase={phase:.3f} step={step:04d} "
                f"done={done} reason={reason} h={float(data.qpos[2]):.3f} "
                f"torso_up={current_torso_up:.3f} pose_rmse={pose_rmse:.5f} "
                f"support_margin_xy={min_support_margin:.3f}"
            )

        if done:
            final_step = step
            break

    return {
        "clip_id": clip_id,
        "phase": phase,
        "survived": reason == "none",
        "final_step": final_step,
        "reason": reason,
        "min_height": min_height,
        "min_torso_up": min_torso_up,
        "min_support_margin_xy": min_support_margin,
        "max_pose_rmse": max_pose_rmse,
    }


def main() -> None:
    args = parse_args()
    jax.config.update("jax_platform_name", "cpu")

    reference_files = (
        [path.as_posix() for path in args.reference_gait_file]
        if args.reference_gait_file
        else default_reference_files(args.reference_gait)
    )
    env = BiomechanicsJoystickEnv(
        config_overrides={
            "impl": "jax",
            "physics_backend": "mjx_jax",
            "reference_gait": args.reference_gait,
            "reference_gait_file": reference_files,
            "reference_loop_mode": args.reference_loop_mode,
            "reference_root_xy_scale": args.reference_root_xy_scale,
            "enable_erfi": False,
        }
    )
    env._mj_model.opt.gravity[:] = (
        np.asarray(env._mj_model.opt.gravity, dtype=np.float64) * float(args.gravity_scale)
    )

    phases = parse_phases(args.phases)
    clip_count = int(np.asarray(env._bvh_reference_clip_count))
    clip_ids = [args.clip_id] if args.clip_id is not None else list(range(clip_count))
    print(
        f"reference_gait={args.reference_gait} clips={clip_count} "
        f"tested={clip_ids} phases={phases} hold_seconds={args.seconds:.2f} "
        f"dt={float(env.dt):.4f} substeps={env.n_substeps} "
        f"gravity_scale={float(args.gravity_scale):.3f}"
    )
    print(f"files={reference_files}")

    results = []
    for clip_id in clip_ids:
        loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
        motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
        print(
            f"clip_header clip={clip_id} loop={LoopMode(loop_mode).name.lower()} "
            f"length={motion_length:.3f}s"
        )
        for phase in phases:
            result = run_hold(
                env,
                env._mj_model,
                int(clip_id),
                float(phase),
                float(args.seconds),
                int(args.print_every),
            )
            results.append(result)
            print(
                "summary_hold "
                f"clip={result['clip_id']} phase={result['phase']:.3f} "
                f"survived={result['survived']} "
                f"final_step={result['final_step']} reason={result['reason']} "
                f"min_h={result['min_height']:.3f} "
                f"min_torso_up={result['min_torso_up']:.3f} "
                f"min_support_margin_xy={result['min_support_margin_xy']:.3f} "
                f"max_pose_rmse={result['max_pose_rmse']:.5f}"
            )

    survived = sum(bool(result["survived"]) for result in results)
    print(f"summary survived={survived}/{len(results)}")


if __name__ == "__main__":
    main()
