"""PD reference tracking oracle for BVH/SMPL walking clips.

This script answers a pre-training invariant: starting from an exact reference
state, can MuJoCo position actuators track the next reference targets without
falling or ending immediately?
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
from config import DEFAULT_BVH_REFERENCE_LIST, DEFAULT_SMPL_REFERENCE_FILES  # noqa: E402
from debug_reference_playback import build_full_state, query_reference_np, torso_up  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gait", choices=["bvh", "smpl"], default="bvh")
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--clip-id", type=int, default=None)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def default_reference_files(reference_gait: str) -> list[str]:
    if reference_gait == "smpl":
        return [path.as_posix() for path in DEFAULT_SMPL_REFERENCE_FILES]
    return [DEFAULT_BVH_REFERENCE_LIST.as_posix()]


def fail_reason(
    low: bool,
    tipped: bool,
    motion_over: bool,
) -> str:
    if low:
        return "low_height"
    if tipped:
        return "tipped"
    if motion_over:
        return "motion_over"
    return "none"


def run_clip(
    env: BiomechanicsJoystickEnv,
    model: mujoco.MjModel,
    clip_id: int,
    steps: int,
    print_every: int,
) -> dict[str, float | int | str | bool]:
    data = mujoco.MjData(model)
    ref = query_reference_np(env, clip_id, 0.0)
    qpos, qvel = build_full_state(env, ref)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = np.asarray(ref["qpos"])
    mujoco.mj_forward(model, data)

    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
    min_height = float(data.qpos[2])
    max_pose_rmse = 0.0
    final_step = steps - 1
    reason = "none"

    for step in range(steps):
        motion_time = step * float(env.dt)
        ref = query_reference_np(env, clip_id, motion_time + float(env.dt))
        target_qpos = np.asarray(ref["qpos"], dtype=np.float64)
        data.ctrl[:] = target_qpos

        for _ in range(env.n_substeps):
            mujoco.mj_step(model, data)

        actual_qpos = data.qpos[np.asarray(env._actuator_qpos_indices_np)]
        pose_rmse = float(np.sqrt(np.mean(np.square(actual_qpos - target_qpos))))
        max_pose_rmse = max(max_pose_rmse, pose_rmse)
        min_height = min(min_height, float(data.qpos[2]))

        low = data.qpos[2] < env.MIN_STANDING_HEIGHT_RATIO * float(model.qpos0[2])
        tipped = torso_up(env, data) < 0.25
        motion_over = loop_mode == int(LoopMode.CLAMP) and motion_time >= motion_length
        done = bool(low or tipped or motion_over)

        if step % print_every == 0 or done or step == steps - 1:
            reason = fail_reason(bool(low), bool(tipped), bool(motion_over))
            print(
                f"clip={clip_id:02d} step={step:04d} t={motion_time:.3f} "
                f"done={done} reason={reason} h={float(data.qpos[2]):.3f} "
                f"torso_up={torso_up(env, data):.3f} pose_rmse={pose_rmse:.5f}"
            )

        if done:
            final_step = step
            break

    survived = reason in {"none", "motion_over"}
    return {
        "clip_id": clip_id,
        "survived": survived,
        "final_step": final_step,
        "reason": reason,
        "min_height": min_height,
        "max_pose_rmse": max_pose_rmse,
        "motion_length": motion_length,
        "loop_mode": LoopMode(loop_mode).name.lower(),
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
            "enable_erfi": False,
        }
    )

    clip_count = int(np.asarray(env._bvh_reference_clip_count))
    clip_ids = [args.clip_id] if args.clip_id is not None else list(range(clip_count))
    print(
        f"reference_gait={args.reference_gait} clips={clip_count} "
        f"tested={clip_ids} dt={float(env.dt):.4f} substeps={env.n_substeps}"
    )
    print(f"files={reference_files}")

    results = [
        run_clip(env, env._mj_model, int(clip_id), args.steps, args.print_every)
        for clip_id in clip_ids
    ]
    survived = sum(bool(result["survived"]) for result in results)
    print(f"summary survived={survived}/{len(results)}")
    for result in results:
        print(
            "summary_clip "
            f"clip={result['clip_id']} survived={result['survived']} "
            f"final_step={result['final_step']} reason={result['reason']} "
            f"loop={result['loop_mode']} length={result['motion_length']:.3f}s "
            f"min_h={result['min_height']:.3f} "
            f"max_pose_rmse={result['max_pose_rmse']:.5f}"
        )


if __name__ == "__main__":
    main()
