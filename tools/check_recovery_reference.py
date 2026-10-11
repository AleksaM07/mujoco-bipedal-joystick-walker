"""Check recovery timing on a real BVH, without training or changing checkpoints.

This is a phase-one integration check, not a certificate of physical gait
feasibility. Runtime derivative and loop metrics are deliberately reported even
when they expose remaining problems in the unchanged reference interpolator.
Run from the repository root with ``python -m tools.check_recovery_reference``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import PROJECT_ROOT, EnvConfig
from evaluate import make_environment
from reference_kinematics import differentiate_reference_positions


def reference_digest(env: BiomechanicsJoystickEnv) -> str:
    """Fingerprint processed motion arrays, including their time base."""
    digest = hashlib.sha256()
    for name in (
        "qpos",
        "qvel",
        "sim_root_pos",
        "sim_root_quat",
        "sim_root_vel",
        "sim_root_angvel",
        "frame_times",
        "frame_counts",
        "loop_modes",
    ):
        suffix = (
            "" if name in {"frame_times", "frame_counts", "loop_modes"} else "_targets"
        )
        values = np.ascontiguousarray(getattr(env, f"_bvh_reference_{name}{suffix}"))
        digest.update(name.encode())
        digest.update(str((values.shape, values.dtype.str)).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def inspect_runtime(
    env: BiomechanicsJoystickEnv, clip_id: int
) -> dict[str, int | float]:
    """Measure the actual JAX query, away from knots where slopes can jump."""
    count = int(env._bvh_reference_frame_counts[clip_id])
    interval = float(env._bvh_reference_frame_times[clip_id])
    if count < 2:
        raise ValueError("The diagnostic clip needs at least two frames.")
    indices = np.unique(np.linspace(0, count - 2, min(32, count - 1)).astype(int))
    times = jp.array((indices + 0.5) * interval, dtype=jp.float32)

    def sample(time):
        reference = env._query_bvh_reference_clip(jp.array(clip_id), time)

        def positions(at_time):
            queried = env._query_bvh_reference_clip(jp.array(clip_id), at_time)
            return queried["qpos"], queried["sim_root_pos"]

        _, derivatives = jax.jvp(positions, (time,), (jp.ones_like(time),))
        return derivatives[0] - reference["qvel"], derivatives[1] - reference[
            "sim_root_vel"
        ]

    joint_errors, root_errors = jax.jit(jax.vmap(sample))(times)
    qpos = np.asarray(env._bvh_reference_qpos_targets)[clip_id, :count]
    qvel = np.asarray(env._bvh_reference_qvel_targets)[clip_id, :count]
    return {
        "sample_count": len(indices),
        "motion_seconds": float(env._bvh_reference_motion_lengths[clip_id]),
        "loop_mode": int(env._bvh_reference_loop_modes[clip_id]),
        "runtime_dqdt_vs_target_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(joint_errors)))
        ),
        "runtime_root_derivative_vs_target_rmse_m_s": float(
            np.sqrt(np.mean(np.square(root_errors)))
        ),
        "endpoint_joint_gap_max_rad": float(np.max(np.abs(qpos[-1] - qpos[0]))),
        "endpoint_joint_velocity_gap_max_rad_s": float(
            np.max(np.abs(qvel[-1] - qvel[0]))
        ),
    }


def main() -> None:
    """Compare legacy, corrected full speed, and corrected quarter speed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=PROJECT_ROOT
        / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=PROJECT_ROOT
        / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh",
    )
    parser.add_argument("--clip-id", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mjx-smoke",
        action="store_true",
        help="Also compile and execute one CPU reset and step (no PPO).",
    )
    args = parser.parse_args()
    for path in (args.xml_path, args.reference_file):
        if not path.is_file():
            parser.error(f"Required local input does not exist: {path}")
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    config = EnvConfig(
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=str(args.xml_path),
        reference_gait="bvh",
        reference_gait_file=[str(args.reference_file)],
        reference_forced_clip_id=args.clip_id,
        deepmimic_reward_mode="pure",
        recovery_physics_version="consistent_v1",
        reference_motion_speed_scale=1.0,
    )
    print("Loading full-speed corrected reference...", flush=True)
    normal = make_environment(config)
    print("Loading quarter-speed corrected reference...", flush=True)
    slow = make_environment(replace(config, reference_motion_speed_scale=0.25))
    print("Loading project-v2 legacy reference...", flush=True)
    legacy = make_environment(replace(config, recovery_physics_version="legacy"))
    for suffix in ("qpos_targets", "sim_root_pos_targets", "sim_root_quat_targets"):
        name = f"_bvh_reference_{suffix}"
        np.testing.assert_array_equal(getattr(normal, name), getattr(slow, name))
        np.testing.assert_array_equal(getattr(normal, name), getattr(legacy, name))
    np.testing.assert_allclose(
        slow._bvh_reference_frame_times, normal._bvh_reference_frame_times * 4
    )
    np.testing.assert_allclose(
        slow._bvh_reference_motion_lengths, normal._bvh_reference_motion_lengths * 4
    )
    for suffix in ("qvel_targets", "sim_root_vel_targets", "sim_root_angvel_targets"):
        name = f"_bvh_reference_{suffix}"
        np.testing.assert_allclose(
            getattr(slow, name), getattr(normal, name) * 0.25, atol=2e-6
        )
    expected = differentiate_reference_positions(
        np.asarray(normal._bvh_reference_qpos_targets),
        np.asarray(normal._bvh_reference_frame_times),
        np.asarray(normal._bvh_reference_frame_counts),
    )
    np.testing.assert_allclose(normal._bvh_reference_qvel_targets, expected, atol=2e-6)
    # This source has IK targets: the historical path applied its hidden x0.25.
    np.testing.assert_allclose(
        legacy._bvh_reference_qvel_targets, expected * 0.25, atol=2e-6
    )
    output = {
        "scope": "Timing and measurement integration only; not a walking success gate.",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True
        ).splitlines(),
        "xml_sha256": hashlib.sha256(args.xml_path.read_bytes()).hexdigest(),
        "bvh_sha256": hashlib.sha256(args.reference_file.read_bytes()).hexdigest(),
        "config": asdict(config),
        "processed_reference_sha256": reference_digest(normal),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "biomechanics_env.py",
                "config.py",
                "evaluate.py",
                "train.py",
                "reference_kinematics.py",
                "bvh_reference.py",
                "biomechanics_model.py",
                "smpl_reference.py",
                "phase1_backends.py",
                "tools/check_recovery_reference.py",
            )
        },
        "timing_integration_assertions": "PASS",
        "clip_id": args.clip_id,
        "source_start_frame": int(
            normal._bvh_reference_source_start_frames[args.clip_id]
        ),
        "source_end_frame": int(normal._bvh_reference_source_end_frames[args.clip_id]),
        "runtime": {
            "legacy": inspect_runtime(legacy, args.clip_id),
            "consistent_v1": inspect_runtime(normal, args.clip_id),
        },
        "limitations": [
            "Nodal gradients need not equal the runtime interpolation derivative.",
            "Loop/contact/actuator feasibility and learned walking are not certified.",
            "Legacy compatibility covers project-v2, not arbitrary other branches.",
            "Geometry, reward coefficients, RSI projection and PPO are unchanged.",
        ],
    }
    if args.mjx_smoke:
        print("Compiling one CPU reset and step...", flush=True)
        state = jax.jit(normal.reset)(jax.random.PRNGKey(7))
        state = jax.jit(normal.step)(state, jp.zeros(normal.action_size))
        jax.block_until_ready(state)
        for leaf in jax.tree_util.tree_leaves(
            (state.obs, state.reward, state.data.qpos, state.data.qvel)
        ):
            if not np.all(np.isfinite(np.asarray(leaf))):
                raise AssertionError("Non-finite MJX smoke-test state")
        output["mjx_smoke"] = {
            "finite": True,
            "steps": 1,
            "done": float(state.done),
            "reward": float(state.reward),
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
