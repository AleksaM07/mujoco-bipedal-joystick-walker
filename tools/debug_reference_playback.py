"""Reference playback oracle diagnostics for BVH/SMPL walking.

Modes:
- kinematic: directly places qpos/qvel at the reference state each control step.
- pd: commands the reference qpos as MuJoCo position-actuator targets.

The script loads the same reference pipeline as training, but uses host MuJoCo
for the playback loop so it stays lightweight and deterministic.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gait", choices=["bvh", "smpl"], default="bvh")
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--mode", choices=["kinematic", "pd"], default="kinematic")
    parser.add_argument("--clip-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    return parser.parse_args()


def default_reference_files(reference_gait: str) -> list[str]:
    if reference_gait == "smpl":
        return [path.as_posix() for path in DEFAULT_SMPL_REFERENCE_FILES]
    return [DEFAULT_BVH_REFERENCE_LIST.as_posix()]


def query_reference_np(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    motion_time: float,
) -> dict[str, np.ndarray | float | int]:
    """Query one interpolated reference state using NumPy arrays."""
    frame_count = int(np.asarray(env._bvh_reference_frame_counts)[clip_id])
    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])

    if loop_mode == int(LoopMode.WRAP):
        phase = motion_time / max(motion_length, 1e-6)
        phase = phase - np.floor(phase)
        loop_count = np.floor(motion_time / max(motion_length, 1e-6))
    else:
        phase = np.clip(motion_time / max(motion_length, 1e-6), 0.0, 1.0)
        loop_count = 0.0

    frame_float = phase * max(frame_count - 1, 0)
    frame0 = int(np.floor(frame_float))
    frame1 = min(frame0 + 1, frame_count - 1)
    alpha = float(frame_float - frame0)

    def lerp(values: np.ndarray) -> np.ndarray:
        return values[clip_id, frame0] + alpha * (
            values[clip_id, frame1] - values[clip_id, frame0]
        )

    wrap_delta = np.asarray(env._bvh_reference_wrap_deltas_np)[clip_id] * loop_count
    root_quat = lerp(np.asarray(env._bvh_reference_reset_root_quat_targets))
    root_quat = root_quat / max(float(np.linalg.norm(root_quat)), 1e-9)

    return {
        "frame0": frame0,
        "frame1": frame1,
        "alpha": alpha,
        "qpos": lerp(np.asarray(env._bvh_reference_qpos_targets)),
        "qvel": lerp(np.asarray(env._bvh_reference_qvel_targets)),
        "root_pos": lerp(np.asarray(env._bvh_reference_reset_root_pos_targets))
        + wrap_delta,
        "root_quat": root_quat,
        "root_vel": lerp(np.asarray(env._bvh_reference_reset_root_vel_targets)),
        "root_angvel": lerp(np.asarray(env._bvh_reference_reset_root_angvel_targets)),
    }


def build_full_state(
    env: BiomechanicsJoystickEnv,
    ref: dict[str, np.ndarray | float | int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build MuJoCo qpos/qvel from an interpolated reference state."""
    qpos = np.asarray(env._init_q_np, dtype=np.float64).copy()
    qvel = np.zeros(env._mj_model.nv, dtype=np.float64)
    qpos[:3] = np.asarray(ref["root_pos"], dtype=np.float64)
    qpos[3:7] = np.asarray(ref["root_quat"], dtype=np.float64)
    qpos[np.asarray(env._actuator_qpos_indices_np)] = np.asarray(ref["qpos"])
    qvel[:3] = np.asarray(ref["root_vel"])
    qvel[3:6] = np.asarray(ref["root_angvel"])
    qvel[np.asarray(env._actuator_dof_indices_np)] = np.asarray(ref["qvel"])
    return qpos, qvel


def torso_up(env: BiomechanicsJoystickEnv, data: mujoco.MjData) -> float:
    """Return torso up alignment in the world z direction."""
    xmat = data.xmat[env._torso_body_id].reshape(3, 3)
    return float(xmat[2, 1])


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

    model = env._mj_model
    data = mujoco.MjData(model)
    clip_id = int(args.clip_id)
    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
    ref = query_reference_np(env, clip_id, 0.0)
    qpos, qvel = build_full_state(env, ref)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = np.asarray(ref["qpos"])
    mujoco.mj_forward(model, data)

    print(
        f"mode={args.mode} reference_gait={args.reference_gait} clip_id={clip_id} "
        f"loop_mode={LoopMode(loop_mode).name.lower()} "
        f"motion_length={motion_length:.3f}s dt={float(env.dt):.4f} "
        f"substeps={env.n_substeps}"
    )
    print(f"files={reference_files}")

    for step in range(args.steps):
        motion_time = step * float(env.dt)
        if args.mode == "kinematic":
            ref = query_reference_np(env, clip_id, motion_time)
            target_qpos = np.asarray(ref["qpos"], dtype=np.float64)
            qpos, qvel = build_full_state(env, ref)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.ctrl[:] = target_qpos
            mujoco.mj_forward(model, data)
        else:
            ref = query_reference_np(env, clip_id, motion_time + float(env.dt))
            target_qpos = np.asarray(ref["qpos"], dtype=np.float64)
            data.ctrl[:] = target_qpos
            for _ in range(env.n_substeps):
                mujoco.mj_step(model, data)

        actual_qpos = data.qpos[np.asarray(env._actuator_qpos_indices_np)]
        pose_rmse = float(np.sqrt(np.mean(np.square(actual_qpos - target_qpos))))
        low = data.qpos[2] < env.MIN_STANDING_HEIGHT_RATIO * float(model.qpos0[2])
        tipped = torso_up(env, data) < 0.25
        motion_over = loop_mode == int(LoopMode.CLAMP) and motion_time >= motion_length
        done = bool(low or tipped or motion_over)

        if step % 10 == 0 or done or step == args.steps - 1:
            print(
                f"step={step:04d} t={motion_time:.3f} "
                f"frame={ref['frame0']}:{ref['frame1']} done={done} "
                f"low={bool(low)} tipped={bool(tipped)} "
                f"motion_over={bool(motion_over)} h={float(data.qpos[2]):.3f} "
                f"torso_up={torso_up(env, data):.3f} pose_rmse={pose_rmse:.5f}"
            )
        if done:
            break


if __name__ == "__main__":
    main()
