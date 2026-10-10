"""Validate one retargeted motion before spending compute on RL training.

The checks in this file are intentionally kinematic. Dynamic feasibility and
free-root tracking remain separate gates handled by ``audit_reference_dynamics``
and ``debug_reference_pd_tracking``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv  # noqa: E402

DEFAULT_REFERENCE = Path("marina_bvh_parse/walking_forward_bvh/02_01.bvh")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        default=DEFAULT_REFERENCE,
    )
    parser.add_argument("--clip-id", type=int, default=1)
    parser.add_argument(
        "--reference-loop-mode",
        choices=["auto", "wrap", "clamp"],
        default="wrap",
    )
    parser.add_argument("--max-velocity-error", type=float, default=1e-3)
    parser.add_argument("--max-seam-joint-error", type=float, default=0.20)
    parser.add_argument("--max-seam-root-angle", type=float, default=0.20)
    parser.add_argument("--max-seam-velocity-error", type=float, default=1.0)
    parser.add_argument("--max-foot-penetration", type=float, default=0.015)
    parser.add_argument("--max-stance-slip-speed", type=float, default=0.15)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/reference_preflight/invariants.json"),
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> BiomechanicsJoystickEnv:
    """Build the same single-clip reference pipeline used by the tracker."""
    return BiomechanicsJoystickEnv(
        config_overrides={
            "impl": "jax",
            "physics_backend": "mjx_jax",
            "reference_gait": "bvh",
            "reference_gait_file": [args.reference_gait_file.as_posix()],
            "reference_loop_mode": args.reference_loop_mode,
            "reference_min_motion_length": 0.0,
            "reference_target_observation": False,
            "reference_action_mode": "residual",
            "deepmimic_reward_mode": "pure",
            "pose_termination": False,
            "enable_erfi": False,
            "arm_actuators": False,
        }
    )


def quaternion_distance(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Return the shortest rotation angle between two wxyz quaternions."""
    quat_a = quat_a / max(float(np.linalg.norm(quat_a)), 1e-12)
    quat_b = quat_b / max(float(np.linalg.norm(quat_b)), 1e-12)
    cosine = float(np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


def differentiate_reference(
    values: np.ndarray,
    frame_time: float,
    wrap_delta: np.ndarray | None = None,
) -> np.ndarray:
    """Differentiate a finite clip, optionally using a cyclic seam derivative."""
    derivative = np.gradient(values, frame_time, axis=0)
    if wrap_delta is None or len(values) <= 2:
        return derivative
    previous_value = values[-2] - wrap_delta
    seam_derivative = (values[1] - previous_value) / (2.0 * frame_time)
    derivative[0] = seam_derivative
    derivative[-1] = seam_derivative
    return derivative


def foot_metrics(
    env: BiomechanicsJoystickEnv,
    qpos_targets: np.ndarray,
    root_pos_targets: np.ndarray,
    root_quat_targets: np.ndarray,
    frame_time: float,
) -> tuple[float, float, float, int]:
    """Measure maximum sole penetration and stance-foot horizontal slip."""
    data = mujoco.MjData(env._mj_model)
    foot_xy: list[np.ndarray] = []
    foot_min_z: list[np.ndarray] = []

    for joint_qpos, root_pos, root_quat in zip(
        qpos_targets,
        root_pos_targets,
        root_quat_targets,
        strict=True,
    ):
        full_qpos = np.asarray(env._init_q_np, dtype=np.float64).copy()
        full_qpos[:3] = root_pos
        full_qpos[3:7] = root_quat
        full_qpos[env._actuator_qpos_indices_np] = joint_qpos
        data.qpos[:] = full_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(env._mj_model, data)
        foot_xy.append(
            np.asarray(data.geom_xpos[env._foot_geom_ids_np, :2], dtype=np.float64)
            .copy()
        )
        foot_min_z.append(
            np.asarray(
                [
                    env._geom_min_z(data, int(geom_id))
                    for geom_id in env._foot_geom_ids_np
                ],
                dtype=np.float64,
            )
        )

    foot_xy_array = np.asarray(foot_xy)
    foot_z_array = np.asarray(foot_min_z)
    max_penetration = float(np.maximum(-foot_z_array, 0.0).max(initial=0.0))
    if len(foot_xy_array) <= 1:
        return max_penetration, 0.0, 0.0, 0

    stance_height = float(env._config.get("reference_foot_lock_height", 0.035))
    active_foot: int | None = None
    stance_slip_values: list[float] = []
    for frame_id in range(len(foot_xy_array)):
        contact_feet = foot_z_array[frame_id] <= stance_height
        if not np.any(contact_feet):
            active_foot = None
            continue
        if active_foot is None or not bool(contact_feet[active_foot]):
            active_foot = int(
                np.argmin(np.where(contact_feet, foot_z_array[frame_id], np.inf))
            )
            continue
        displacement = (
            foot_xy_array[frame_id, active_foot]
            - foot_xy_array[frame_id - 1, active_foot]
        )
        stance_slip_values.append(
            float(np.linalg.norm(displacement) / max(frame_time, 1e-9))
        )

    stance_slip = np.asarray(stance_slip_values, dtype=np.float64)
    max_stance_slip = float(stance_slip.max(initial=0.0))
    stance_slip_p95 = (
        float(np.percentile(stance_slip, 95.0)) if stance_slip.size else 0.0
    )
    return max_penetration, max_stance_slip, stance_slip_p95, int(stance_slip.size)


def audit_clip(args: argparse.Namespace) -> dict[str, Any]:
    """Compute invariant metrics and threshold results for one reference clip."""
    env = build_env(args)
    clip_count = int(env._bvh_reference_clip_count)
    if args.clip_id < 0 or args.clip_id >= clip_count:
        raise ValueError(
            f"--clip-id {args.clip_id} is outside [0, {clip_count - 1}]."
        )

    clip_id = int(args.clip_id)
    frame_count = int(np.asarray(env._bvh_reference_frame_counts)[clip_id])
    frame_time = float(np.asarray(env._bvh_reference_frame_times)[clip_id])
    if frame_count < 2:
        raise ValueError(f"Clip {clip_id} has only {frame_count} frame(s).")

    qpos = np.asarray(env._bvh_reference_qpos_targets)[clip_id, :frame_count]
    qvel = np.asarray(env._bvh_reference_qvel_targets)[clip_id, :frame_count]
    root_pos = np.asarray(env._bvh_reference_sim_root_pos_targets_np)[
        clip_id, :frame_count
    ]
    root_quat = np.asarray(env._bvh_reference_sim_root_quat_targets_np)[
        clip_id, :frame_count
    ]
    root_vel = np.asarray(env._bvh_reference_sim_root_vel_targets_np)[
        clip_id, :frame_count
    ]
    root_angvel = np.asarray(env._bvh_reference_sim_root_angvel_targets_np)[
        clip_id, :frame_count
    ]

    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
    is_wrap = loop_mode == 1
    root_wrap_delta = root_pos[-1] - root_pos[0]
    root_wrap_delta[2] = 0.0
    implied_qvel = differentiate_reference(
        qpos,
        frame_time,
        np.zeros(qpos.shape[1], dtype=qpos.dtype) if is_wrap else None,
    )
    implied_root_vel = differentiate_reference(
        root_pos,
        frame_time,
        root_wrap_delta if is_wrap else None,
    )
    joint_velocity_error = np.abs(qvel - implied_qvel)
    root_velocity_error = np.abs(root_vel - implied_root_vel)

    lower = np.asarray(env._actuator_qpos_lower_limits_np)
    upper = np.asarray(env._actuator_qpos_upper_limits_np)
    actuator_limit_violation = np.maximum(
        np.maximum(lower - qpos, qpos - upper),
        0.0,
    )

    (
        max_penetration,
        max_stance_slip,
        stance_slip_p95,
        stance_sample_count,
    ) = foot_metrics(
        env,
        qpos,
        root_pos,
        root_quat,
        frame_time,
    )
    seam_joint_error = float(np.max(np.abs(qpos[-1] - qpos[0])))
    seam_root_angle = quaternion_distance(root_quat[-1], root_quat[0])
    seam_velocity_error = float(
        max(
            np.max(np.abs(qvel[-1] - qvel[0])),
            np.max(np.abs(root_vel[-1] - root_vel[0])),
            np.max(np.abs(root_angvel[-1] - root_angvel[0])),
        )
    )

    metrics = {
        "joint_velocity_max_abs_error": float(joint_velocity_error.max()),
        "joint_velocity_mean_abs_error": float(joint_velocity_error.mean()),
        "root_velocity_max_abs_error": float(root_velocity_error.max()),
        "root_velocity_mean_abs_error": float(root_velocity_error.mean()),
        "actuator_position_limit_max_violation": float(
            actuator_limit_violation.max(initial=0.0)
        ),
        "seam_joint_max_abs_error": seam_joint_error,
        "seam_root_angle_error": seam_root_angle,
        "seam_velocity_max_abs_error": seam_velocity_error,
        "foot_max_penetration": max_penetration,
        "stance_foot_max_slip_speed": max_stance_slip,
        "stance_foot_p95_slip_speed": stance_slip_p95,
        "stance_foot_sample_count": stance_sample_count,
    }
    checks = {
        "joint_velocity_consistent": (
            metrics["joint_velocity_max_abs_error"] <= args.max_velocity_error
        ),
        "root_velocity_consistent": (
            metrics["root_velocity_max_abs_error"] <= args.max_velocity_error
        ),
        "actuator_positions_within_limits": (
            metrics["actuator_position_limit_max_violation"] <= 1e-6
        ),
        "loop_joint_seam": (
            not is_wrap or seam_joint_error <= args.max_seam_joint_error
        ),
        "loop_root_orientation_seam": (
            not is_wrap or seam_root_angle <= args.max_seam_root_angle
        ),
        "loop_velocity_seam": (
            not is_wrap or seam_velocity_error <= args.max_seam_velocity_error
        ),
        "foot_penetration": max_penetration <= args.max_foot_penetration,
        "stance_foot_slip": max_stance_slip <= args.max_stance_slip_speed,
    }
    return {
        "passed": all(checks.values()),
        "source": env._bvh_reference_source_paths[clip_id],
        "clip_id": clip_id,
        "frame_count": frame_count,
        "frame_time": frame_time,
        "motion_length": frame_time * (frame_count - 1),
        "checks": checks,
        "metrics": metrics,
    }


def main() -> None:
    """Run the audit, persist JSON, and fail the process when a gate fails."""
    args = parse_args()
    jax.config.update("jax_platform_name", "cpu")
    result = audit_clip(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    status = "PASS" if result["passed"] else "FAIL"
    print(f"reference invariant audit: {status}")
    for name, passed in result["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"report: {args.out}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
