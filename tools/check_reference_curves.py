"""Audit prepared reference curves on real BVH data without training.

Numerical curve consistency and contact geometry are separate findings. A low
derivative error is not evidence of force feasibility or learned walking.
Run from the repository root with ``python -m tools.check_reference_curves``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import PROJECT_ROOT, EnvConfig
from evaluate import make_environment
from reference_interpolation import prepare_reference_curves
from tools.check_recovery_reference import inspect_runtime, reference_digest


def full_state(
    env: BiomechanicsJoystickEnv, reference: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble native coordinates without assuming actuator ordering."""
    qpos = np.array(env._init_q, dtype=np.float64)
    qvel = np.zeros(env._mj_model.nv)
    qpos[:3], qpos[3:7] = reference["sim_root_pos"], reference["sim_root_quat"]
    qpos[np.asarray(env._actuator_qpos_indices)] = reference["qpos"]
    qvel[:3], qvel[3:6] = reference["sim_root_vel"], reference["sim_root_angvel"]
    qvel[np.asarray(env._actuator_dof_indices)] = reference["qvel"]
    return qpos, qvel


def lowest_foot_point(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
    """Return a world-space lowest support point for a supported foot shape."""
    position = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    shape = model.geom_type[geom_id]
    if shape == mujoco.mjtGeom.mjGEOM_BOX:
        return position + rotation @ (-np.sign(rotation[2]) * size)
    if shape == mujoco.mjtGeom.mjGEOM_SPHERE:
        return position - np.array([0.0, 0.0, size[0]])
    if shape == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = rotation[:, 2]
        return (
            position - np.sign(axis[2]) * axis * size[1] - np.array([0.0, 0.0, size[0]])
        )
    raise ValueError(f"Unsupported foot geom type {shape}; do not guess penetration.")


def inspect_geometry(env: BiomechanicsJoystickEnv, clip_id: int) -> dict:
    """Compare runtime FK to native MuJoCo and report dense contact proxies."""
    count = int(env._bvh_reference_frame_counts[clip_id])
    duration = float(env._bvh_reference_motion_lengths[clip_id])
    times = jp.linspace(0.0, duration, (count - 1) * 4 + 1)
    query = jax.jit(
        jax.vmap(lambda time: env._query_bvh_reference_clip(jp.array(clip_id), time))
    )
    started = perf_counter()
    batch = jax.block_until_ready(query(times))
    compile_and_query_seconds = perf_counter() - started
    started = perf_counter()
    jax.block_until_ready(query(times))
    warm_query_seconds = perf_counter() - started
    batch = jax.tree.map(np.asarray, batch)
    if not all(np.isfinite(values).all() for values in batch.values()):
        raise AssertionError("Reference query produced a non-finite value.")
    model = env._mj_model
    data = mujoco.MjData(model)
    jacobian = np.zeros((3, model.nv))
    angular = np.zeros_like(jacobian)
    anchor_id = env._reference_anchor_body_id
    body_ids = np.asarray(env._deepmimic_key_body_ids)
    site_ids = np.asarray(env._deepmimic_key_site_ids)
    is_site = np.asarray(env._deepmimic_key_is_site)
    errors = {
        name: []
        for name in (
            "anchor_position_m",
            "anchor_velocity_m_s",
            "anchor_angular_velocity_rad_s",
            "key_position_m",
        )
    }
    foot_heights, near_floor_speeds = [], []
    for index in range(len(times)):
        reference = {name: values[index] for name, values in batch.items()}
        data.qpos[:], data.qvel[:] = full_state(env, reference)
        mujoco.mj_forward(model, data)
        mujoco.mj_jacBody(model, data, jacobian, angular, anchor_id)
        keys = np.where(is_site[:, None], data.site_xpos[site_ids], data.xpos[body_ids])
        for name, left, right in (
            ("anchor_position_m", reference["root_pos"], data.xpos[anchor_id]),
            ("anchor_velocity_m_s", reference["root_vel"], jacobian @ data.qvel),
            (
                "anchor_angular_velocity_rad_s",
                reference["root_angvel"],
                angular @ data.qvel,
            ),
            ("key_position_m", reference["key_pos"], keys),
        ):
            errors[name].append(float(np.max(np.abs(left - right))))
        floor_id = env._floor_geom_id
        floor_normal = data.geom_xmat[floor_id].reshape(3, 3)[:, 2]
        if model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(
            floor_normal, [0, 0, 1], atol=1e-8
        ):
            raise ValueError(
                "This contact diagnostic requires a horizontal floor plane."
            )
        for geom_id in env._foot_geom_ids_np:
            geom_id = int(geom_id)
            point = lowest_foot_point(model, data, geom_id)
            height = float(point[2] - data.geom_xpos[floor_id, 2])
            foot_heights.append(height)
            if height <= 0.01:
                mujoco.mj_jac(
                    model,
                    data,
                    jacobian,
                    angular,
                    point,
                    int(model.geom_bodyid[geom_id]),
                )
                near_floor_speeds.append(
                    float(np.linalg.norm((jacobian @ data.qvel)[:2]))
                )
    lower = np.maximum(
        env._actuator_ctrl_lower_limits_np, env._actuator_qpos_lower_limits_np
    )
    upper = np.minimum(
        env._actuator_ctrl_upper_limits_np, env._actuator_qpos_upper_limits_np
    )
    violations = np.maximum(lower - batch["qpos"], batch["qpos"] - upper)
    heights = np.asarray(foot_heights)
    return {
        "sample_count": len(times),
        "native_fk_max_errors": {name: max(values) for name, values in errors.items()},
        "joint_control_and_position_limit_excess_rad": max(
            0.0, float(violations.max())
        ),
        "minimum_foot_surface_height_m": float(heights.min()),
        "maximum_floor_penetration_m": max(0.0, float(-heights.min())),
        "foot_sample_fraction_penetrating_more_than_1mm": float(
            np.mean(heights < -0.001)
        ),
        "near_floor_point_speed_definition": (
            "Horizontal material-point velocity from queried qvel at a lowest foot "
            "support point with surface height <= 1 cm (includes penetration). "
            "Not force-labeled stance; v1 qvel differs from its curve derivative."
        ),
        "near_floor_point_sample_count": len(near_floor_speeds),
        "near_floor_point_speed_median_m_s": float(np.median(near_floor_speeds))
        if near_floor_speeds
        else None,
        "near_floor_point_speed_p95_m_s": float(np.percentile(near_floor_speeds, 95))
        if near_floor_speeds
        else None,
        "query_compile_and_execute_seconds": compile_and_query_seconds,
        "query_warm_batch_seconds": warm_query_seconds,
    }


def inspect_consistency(env: BiomechanicsJoystickEnv, clip_id: int) -> dict:
    """Check rotation derivatives, native finite differences, wrap and exact RSI."""
    clip = jp.array(clip_id)
    duration = float(env._bvh_reference_motion_lengths[clip_id])
    interval = float(env._bvh_reference_frame_times[clip_id])
    if int(env._bvh_reference_loop_modes[clip_id]) != 1 or duration <= 0.0:
        raise ValueError("This audit requires a nontrivial WRAP clip.")
    query = jax.jit(lambda time: env._query_bvh_reference_clip(clip, time))
    times = (np.arange(32) + 0.37) * duration / 32

    def rotation_error(time):
        reference = query(time)
        quat, derivative = jax.jvp(
            lambda at: query(at)["sim_root_quat"], (time,), (jp.ones_like(time),)
        )
        omega = 2 * (
            quat[0] * derivative[1:]
            - derivative[0] * quat[1:]
            - jp.cross(quat[1:], derivative[1:])
        )
        return omega - reference["sim_root_angvel"]

    rotation_errors = np.asarray(jax.jit(jax.vmap(rotation_error))(jp.array(times)))
    difference = np.zeros(env._mj_model.nv)
    fd_errors = []
    epsilon = min(1e-4, interval * 0.01)
    for time in times:
        _, velocity = full_state(env, query(jp.array(time)))
        before, _ = full_state(env, query(jp.array(time - epsilon)))
        after, _ = full_state(env, query(jp.array(time + epsilon)))
        mujoco.mj_differentiatePos(
            env._mj_model, difference, 2 * epsilon, before, after
        )
        fd_errors.append(difference - velocity)
    seam_epsilon = 1e-6
    before, after = (
        query(jp.array(duration - seam_epsilon)),
        query(jp.array(duration + seam_epsilon)),
    )
    seam_fields = (
        "qpos",
        "qvel",
        "sim_root_pos",
        "sim_root_quat",
        "sim_root_vel",
        "sim_root_angvel",
    )
    seam = {
        name: float(np.max(np.abs(before[name] - after[name]))) for name in seam_fields
    }
    base_time = duration * 0.37
    base = query(jp.array(base_time))
    loop_errors = []
    for cycle in (-3, -1, 1, 4):
        reference = query(jp.array(base_time + cycle * duration))
        for name in seam_fields:
            expected = base[name]
            if name == "sim_root_pos":
                expected = expected + cycle * env._bvh_reference_wrap_deltas[clip_id]
            loop_errors.append(float(np.max(np.abs(reference[name] - expected))))
    reset = jax.jit(
        lambda time: env._sample_initial_state_from_reference(clip, time, exact=True)
    )
    reset_qpos, reset_qvel, _, _ = reset(jp.array(base_time))
    expected_qpos, expected_qvel = full_state(env, base)
    np.testing.assert_allclose(reset_qpos, expected_qpos, atol=2e-6)
    np.testing.assert_allclose(reset_qvel, expected_qvel, atol=2e-6)
    metrics = inspect_runtime(env, clip_id)
    fd_errors = np.asarray(fd_errors)
    metrics.update(
        {
            "rotation_derivative_rmse_rad_s": float(
                np.sqrt(np.mean(rotation_errors**2))
            ),
            "native_pose_finite_difference_epsilon_s": epsilon,
            "native_pose_finite_difference_rmse_mixed_units": float(
                np.sqrt(np.mean(np.square(fd_errors)))
            ),
            "native_root_finite_difference_rmse_m_s": float(
                np.sqrt(np.mean(fd_errors[:, :3] ** 2))
            ),
            "native_rotation_finite_difference_rmse_rad_s": float(
                np.sqrt(np.mean(fd_errors[:, 3:6] ** 2))
            ),
            "native_joint_finite_difference_rmse_rad_s": float(
                np.sqrt(
                    np.mean(fd_errors[:, np.asarray(env._actuator_dof_indices)] ** 2)
                )
            ),
            "runtime_seam_epsilon_s": seam_epsilon,
            "runtime_seam_max_component_difference": seam,
            "multiple_cycle_max_error_mixed_units": max(loop_errors),
            "exact_reset_matches_runtime": True,
        }
    )
    for name in (
        "runtime_dqdt_vs_target_rmse_rad_s",
        "runtime_root_derivative_vs_target_rmse_m_s",
        "rotation_derivative_rmse_rad_s",
    ):
        if metrics[name] > 1e-4:
            raise AssertionError(f"Curve derivative mismatch: {name}={metrics[name]}")
    if (
        metrics["endpoint_joint_gap_max_rad"] > 1e-6
        or metrics["endpoint_joint_velocity_gap_max_rad_s"] > 1e-6
    ):
        raise AssertionError("Prepared joint seam is not closed.")
    if max(loop_errors) > 1e-3 or max(seam.values()) > 0.01:
        raise AssertionError("Runtime cycle/seam consistency failed.")
    if not np.isfinite(fd_errors).all() or np.max(np.abs(fd_errors)) > 0.02:
        raise AssertionError("Native pose finite differences disagree with velocity.")
    return metrics


def main() -> None:
    """Run the phase-two numerical gate and separate contact diagnostics."""
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
        "--mjx-smoke", action="store_true", help="One compiled CPU reset/step; no PPO."
    )
    args = parser.parse_args()
    for path in (args.xml_path, args.reference_file):
        if not path.is_file():
            parser.error(f"Missing local input: {path}")
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
        recovery_physics_version="consistent_v2",
        reference_loop_closure="bounded",
    )
    print("Loading v2 bounded, v1 unchanged, and v2 quarter speed...", flush=True)
    current = make_environment(config)
    previous = make_environment(
        replace(
            config,
            recovery_physics_version="consistent_v1",
            reference_loop_closure="strict",
        )
    )
    slow = make_environment(replace(config, reference_motion_speed_scale=0.25))
    for suffix in ("qpos_targets", "sim_root_pos_targets", "sim_root_quat_targets"):
        np.testing.assert_array_equal(
            getattr(current, f"_bvh_reference_{suffix}"),
            getattr(slow, f"_bvh_reference_{suffix}"),
        )
    for name in (
        "_reference_joint_tangents",
        "_reference_root_tangents",
        "_reference_quaternion_tangents",
    ):
        np.testing.assert_allclose(
            getattr(slow, name), getattr(current, name) * 0.25, atol=2e-6
        )
    np.testing.assert_allclose(
        slow._bvh_reference_motion_lengths, current._bvh_reference_motion_lengths * 4
    )
    try:
        prepare_reference_curves(
            previous._bvh_reference_qpos_targets,
            previous._bvh_reference_sim_root_pos_targets,
            previous._bvh_reference_sim_root_quat_targets,
            previous._bvh_reference_frame_times,
            previous._bvh_reference_frame_counts,
            previous._bvh_reference_loop_modes,
            previous._bvh_reference_wrap_deltas,
            loop_closure="strict",
        )
    except ValueError as exc:
        strict_result = str(exc)
    else:
        strict_result = "Accepted: source endpoints already satisfy strict closure."
    print(
        "Measuring runtime derivatives, resets and native contact geometry...",
        flush=True,
    )
    runtime = inspect_consistency(current, args.clip_id)
    geometry = {
        "consistent_v1": inspect_geometry(previous, args.clip_id),
        "consistent_v2": inspect_geometry(current, args.clip_id),
    }
    if geometry["consistent_v2"]["joint_control_and_position_limit_excess_rad"] > 1e-6:
        raise AssertionError("Prepared runtime targets exceed joint/control limits.")
    if max(geometry["consistent_v2"]["native_fk_max_errors"].values()) > 1e-4:
        raise AssertionError("Reference features disagree with native FK.")
    output = {
        "scope": "Phase 2 curve consistency; not a physical tracking certificate.",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT
        ).strip(),
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--short"], text=True, cwd=PROJECT_ROOT
        ).splitlines(),
        "xml_sha256": hashlib.sha256(args.xml_path.read_bytes()).hexdigest(),
        "bvh_sha256": hashlib.sha256(args.reference_file.read_bytes()).hexdigest(),
        "config": asdict(config),
        "clip_id": args.clip_id,
        "source_start_frame": int(
            current._bvh_reference_source_start_frames[args.clip_id]
        ),
        "source_end_frame": int(current._bvh_reference_source_end_frames[args.clip_id]),
        "processed_reference_sha256": reference_digest(current),
        "unchanged_v1_processed_reference_sha256": reference_digest(previous),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "biomechanics_env.py",
                "reference_interpolation.py",
                "reference_kinematics.py",
                "config.py",
                "evaluate.py",
                "train.py",
                "bvh_reference.py",
                "biomechanics_model.py",
                "phase1_backends.py",
                "tools/check_reference_curves.py",
            )
        },
        "strict_source_preparation": strict_result,
        "all_clip_closure_diagnostics": current._reference_closure_diagnostics,
        "runtime": {
            "consistent_v1": inspect_runtime(previous, args.clip_id),
            "consistent_v2": runtime,
        },
        "geometry": geometry,
        "numerical_curve_gate": "PASS",
        "quarter_speed_same_geometry_and_scaled_tangents": "PASS",
        "limitations": [
            "C1 curves do not guarantee continuous acceleration or feasible torque.",
            "Contact metrics are geometric proxies, not force-labeled stance/rollouts.",
            "Motor target limits do not certify actuator force limits.",
            "Exact RSI no longer height-projects v2; audit reset acceptance/fallback.",
            "Warp simulation and full PPO throughput are not tested by this CPU audit.",
            "Reward, physical tracking, learned gait and joystick remain unvalidated.",
        ],
    }
    if args.mjx_smoke:
        print(
            json.dumps({"runtime": output["runtime"], "geometry": geometry}, indent=2),
            flush=True,
        )
        print("Compiling one CPU MJX reset and step; no training...", flush=True)
        state = jax.jit(current.reset)(jax.random.PRNGKey(7))
        reset_diagnostics = {
            name: float(value)
            for name, value in state.metrics.items()
            if name.startswith("init_")
        }
        reset_diagnostics["reference_fallback_standing"] = bool(
            state.info["reference_fallback_standing"]
        )
        state = jax.jit(current.step)(state, jp.zeros(current.action_size))
        jax.block_until_ready(state)
        for leaf in jax.tree.leaves(
            (state.obs, state.reward, state.data.qpos, state.data.qvel)
        ):
            if not np.isfinite(np.asarray(leaf)).all():
                raise AssertionError("Non-finite smoke-test state.")
        output["mjx_smoke"] = {
            "finite": True,
            "steps": 1,
            "done": float(state.done),
            "reward": float(state.reward),
            "reset_diagnostics": reset_diagnostics,
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
