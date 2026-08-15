"""Check DeepMimic reward invariants for the active reference pipeline.

This script does not train. It places the MuJoCo/MJX state at a sampled
reference frame, applies controlled perturbations, and prints reward component
ordering. The exact reference state should be the maximum.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from biomechanics_env import BiomechanicsJoystickEnv
from config import DEFAULT_BVH_REFERENCE_LIST, DEFAULT_SMPL_REFERENCE_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gait", choices=["bvh", "smpl"], default="bvh")
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--clip-id", type=int, default=0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--joint-index", type=int, default=9)
    parser.add_argument("--assert-invariants", action="store_true")
    return parser.parse_args()


def default_reference_files(reference_gait: str) -> list[str]:
    if reference_gait == "smpl":
        return [path.as_posix() for path in DEFAULT_SMPL_REFERENCE_FILES]
    return [DEFAULT_BVH_REFERENCE_LIST.as_posix()]


def scalar_dict(values: dict[str, jax.Array]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if getattr(value, "shape", ()) == ()
    }


def build_info(env: BiomechanicsJoystickEnv, clip_id: int, motion_time: jax.Array, data):
    return {
        "motion_step": jnp.array(0, dtype=jnp.int32),
        "bvh_reference_time_offset": motion_time,
        "bvh_reference_clip_id": jnp.array(clip_id, dtype=jnp.int32),
        "reference_fallback_standing": jnp.array(False),
        "command": jnp.zeros(3),
        "last_action": jnp.zeros(env.action_size),
        "last_foot_xy": env._foot_xy(data),
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
            "reset_sample_attempts": 1,
            "reset_projection_levels": (1.0,),
        }
    )

    clip_id = int(args.clip_id)
    motion_length = float(env._bvh_reference_motion_lengths[clip_id])
    motion_time = jnp.array(args.phase * motion_length, dtype=jnp.float32)
    qpos, qvel, ctrl, _ = env._sample_initial_state_from_reference(
        jnp.array(clip_id, dtype=jnp.int32),
        motion_time,
        exact=True,
    )

    cases: list[tuple[str, float | None]] = [
        ("exact_reference", None),
        ("perturb_1_deg", np.deg2rad(1.0)),
        ("perturb_5_deg", np.deg2rad(5.0)),
        ("perturb_20_deg", np.deg2rad(20.0)),
        ("random_pose", "random"),  # type: ignore[list-item]
    ]

    print(f"reference_gait={args.reference_gait} files={reference_files}")
    print(f"clip_id={clip_id} phase={args.phase:.3f} motion_time={float(motion_time):.6f}")
    totals: dict[str, float] = {}
    for name, perturb in cases:
        case_qpos = qpos
        if perturb == "random":
            rng = np.random.default_rng(7)
            random_ctrl = rng.uniform(
                np.asarray(env._actuator_ctrl_lower_limits_np),
                np.asarray(env._actuator_ctrl_upper_limits_np),
            ).astype(np.float32)
            case_qpos = case_qpos.at[env._actuator_qpos_indices].set(random_ctrl)
        elif perturb is not None:
            case_qpos = case_qpos.at[
                env._actuator_qpos_indices[int(args.joint_index)]
            ].add(float(perturb))

        data = env._fresh_reset_data().replace(qpos=case_qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(env._mjx_model, data)
        info = build_info(env, clip_id, motion_time, data)
        rewards = scalar_dict(env._get_bvh_deepmimic_reward(data, info))
        done = bool(env._get_done(data, info))
        totals[name] = rewards["total"]
        print(
            name,
            "done=",
            done,
            "total=",
            round(rewards["total"], 6),
            "pose=",
            round(rewards["pose"], 6),
            "vel=",
            round(rewards["velocity"], 6),
            "root=",
            round(rewards["root_pose"], 6),
            "key=",
            round(rewards["key_position"], 6),
            "pose_err=",
            round(rewards["pose_error"], 6),
            "key_err=",
            round(rewards["key_pos_error"], 6),
        )

    if args.assert_invariants:
        exact_total = totals["exact_reference"]
        if exact_total < 0.99:
            raise AssertionError(
                f"Exact reference reward must be near 1.0, got {exact_total:.6f}"
            )
        if not (
            totals["exact_reference"]
            > totals["perturb_1_deg"]
            > totals["perturb_5_deg"]
            > totals["perturb_20_deg"]
            > totals["random_pose"]
        ):
            raise AssertionError(f"Reward ordering failed: {totals}")


if __name__ == "__main__":
    main()
