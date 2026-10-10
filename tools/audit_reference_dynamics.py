"""Audit whether one reference clip is dynamically plausible for this MuJoCo model.

This script does not run PD playback. Instead, it places the model directly on the
reference kinematic trajectory, estimates qacc with finite differences, and asks
MuJoCo inverse dynamics how much generalized force would be required to realize
that motion exactly.

That gives us a cleaner answer to questions like:
- Is the reference demanding a huge free-root wrench?
- Are actuated joint torques regularly above actuator force limits?
- Is the problem "PD is bad", or "this motion is dynamically incompatible as-is"?
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import jax
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_BVH_REFERENCE_LIST,
    EnvConfig,
    expand_reference_gait_files,
)
from tools.debug_reference_playback import build_full_state, query_reference_np  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--reference-gait-list", type=Path, action="append")
    parser.add_argument("--clip-id", type=int, default=0)
    parser.add_argument("--start-phase", type=float, default=0.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.35,
        help="Motion-window duration in reference seconds before any wrap/clamp handling.",
    )
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.01,
        help="Audit sampling spacing in motion seconds.",
    )
    parser.add_argument(
        "--reference-loop-mode",
        choices=["auto", "wrap", "clamp"],
        default="auto",
    )
    parser.add_argument(
        "--reference-motion-speed-scale",
        type=float,
        default=EnvConfig.reference_motion_speed_scale,
        help="Reference clock multiplier; values below one slow every trajectory consistently.",
    )
    parser.add_argument(
        "--arms-on",
        dest="arm_actuators",
        action="store_true",
        help="Use the arm-actuated model variant.",
    )
    parser.add_argument(
        "--arms-off",
        dest="arm_actuators",
        action="store_false",
        help="Use the no-arms model variant.",
    )
    parser.set_defaults(arm_actuators=False)
    parser.add_argument(
        "--reference-lock-stance-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reference-root-xy-scale",
        type=float,
        default=EnvConfig.reference_root_xy_scale,
    )
    parser.add_argument("--reference-root-x-scale", type=float, default=None)
    parser.add_argument("--reference-root-y-scale", type=float, default=None)
    parser.add_argument(
        "--reference-stability-sagittal-alpha",
        type=float,
        default=EnvConfig.reference_stability_sagittal_alpha,
    )
    parser.add_argument(
        "--reference-stability-other-alpha",
        type=float,
        default=EnvConfig.reference_stability_other_alpha,
    )
    parser.add_argument(
        "--reference-stability-before-ik",
        action=argparse.BooleanOptionalAction,
        default=EnvConfig.reference_stability_before_ik,
        help="Apply the stability prior as the IK seed so fitted feet stay fitted.",
    )
    parser.add_argument(
        "--reference-ik-correction-alpha",
        type=float,
        default=EnvConfig.reference_ik_correction_alpha,
        help="Fraction of the fitted IK correction retained after a pre-IK prior.",
    )
    parser.add_argument("--reference-ik-com-weight", type=float, default=0.0)
    parser.add_argument(
        "--reference-ik-support-height-scale",
        type=float,
        default=EnvConfig.reference_ik_support_height_scale,
    )
    parser.add_argument(
        "--reference-ik-pose-regularization",
        type=float,
        default=EnvConfig.reference_ik_pose_regularization,
    )
    parser.add_argument(
        "--reference-ik-smooth-regularization",
        type=float,
        default=EnvConfig.reference_ik_smooth_regularization,
    )
    parser.add_argument(
        "--reference-ik-root-smooth-regularization",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--reference-ik-optimize-root-xy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reference-ik-root-regularization",
        type=float,
        default=EnvConfig.reference_ik_root_regularization,
    )
    parser.add_argument(
        "--reference-ik-swing-foot-weight-scale",
        type=float,
        default=EnvConfig.reference_ik_swing_foot_weight_scale,
    )
    parser.add_argument(
        "--support-contact-height",
        type=float,
        default=0.01,
        help="Maximum sole-bottom height treated as reference support contact.",
    )
    parser.add_argument(
        "--training-filtered",
        action="store_true",
        help="Keep the current training motion-length filter.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("videos/reference_dynamics_audit"),
    )
    return parser.parse_args()


def default_reference_files(args: argparse.Namespace) -> list[str]:
    """Resolve reference files using the same defaults as the repo tools."""
    files = expand_reference_gait_files(
        args.reference_gait_file,
        args.reference_gait_list,
    )
    if files is None:
        files = expand_reference_gait_files(reference_gait_lists=[DEFAULT_BVH_REFERENCE_LIST])
    return [Path(path).as_posix() for path in files]


def make_env(args: argparse.Namespace) -> BiomechanicsJoystickEnv:
    """Build the host-side env using the same reference pipeline as training."""
    config_overrides = {
        "impl": "jax",
        "physics_backend": "mjx_jax",
        "reference_gait": "bvh",
        "arm_actuators": args.arm_actuators,
        "reference_gait_file": default_reference_files(args),
        "reference_loop_mode": args.reference_loop_mode,
        "reference_motion_speed_scale": args.reference_motion_speed_scale,
        "reference_target_observation": False,
        "reference_action_mode": "residual",
        "deepmimic_reward_mode": "pure",
        "pose_termination": False,
        "enable_erfi": False,
        "reference_lock_stance_feet": args.reference_lock_stance_feet,
        "reference_root_xy_scale": args.reference_root_xy_scale,
        "reference_root_x_scale": args.reference_root_x_scale,
        "reference_root_y_scale": args.reference_root_y_scale,
        "reference_stability_sagittal_alpha": (
            args.reference_stability_sagittal_alpha
        ),
        "reference_stability_other_alpha": args.reference_stability_other_alpha,
        "reference_stability_before_ik": args.reference_stability_before_ik,
        "reference_ik_correction_alpha": args.reference_ik_correction_alpha,
        "reference_ik_com_weight": args.reference_ik_com_weight,
        "reference_ik_support_height_scale": args.reference_ik_support_height_scale,
        "reference_ik_pose_regularization": args.reference_ik_pose_regularization,
        "reference_ik_smooth_regularization": args.reference_ik_smooth_regularization,
        "reference_ik_root_smooth_regularization": (
            args.reference_ik_root_smooth_regularization
        ),
        "reference_ik_optimize_root_xy": args.reference_ik_optimize_root_xy,
        "reference_ik_root_regularization": args.reference_ik_root_regularization,
        "reference_ik_swing_foot_weight_scale": (
            args.reference_ik_swing_foot_weight_scale
        ),
    }
    if not args.training_filtered:
        config_overrides["reference_min_motion_length"] = 0.0
    return BiomechanicsJoystickEnv(config_overrides=config_overrides)


def clip_time(
    motion_time: float,
    motion_length: float,
    loop_mode: int,
) -> float:
    """Wrap or clamp one motion time to a valid reference query time."""
    if loop_mode == 0:
        return float(np.clip(motion_time, 0.0, motion_length))
    if loop_mode == 1:
        if motion_length <= 1e-9:
            return 0.0
        return float(motion_time % motion_length)
    return float(np.clip(motion_time, 0.0, motion_length))


def summarize_series(values: list[float]) -> tuple[float, float, float]:
    """Return mean, max abs, and max for one numeric series."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    return (
        float(statistics.mean(values)),
        float(max(abs(value) for value in values)),
        float(max(values)),
    )


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2-D convex hull without extra dependencies."""
    unique_points = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique_points) <= 1:
        return np.asarray(unique_points, dtype=np.float64)

    def cross(
        origin: tuple[float, float],
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def signed_support_margin(point_xy: np.ndarray, support_points: np.ndarray) -> float:
    """Return positive distance inside a convex support polygon, negative outside."""
    hull = convex_hull_xy(support_points)
    if len(hull) < 3:
        return float("nan")
    margins = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 1e-12:
            continue
        relative = point_xy - start
        cross_value = edge[0] * relative[1] - edge[1] * relative[0]
        margins.append(float(cross_value / edge_length))
    return min(margins, default=float("nan"))


def support_polygon_points(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
    contact_height: float,
) -> tuple[np.ndarray, int]:
    """Build the ground-plane support polygon from contact-ready sole geoms."""
    points: list[np.ndarray] = []
    contact_count = 0
    for geom_id in env._foot_geom_ids_np:
        if env._geom_min_z(data, int(geom_id)) > contact_height:
            continue
        contact_count += 1
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        size = np.asarray(env._mj_model.geom_size[geom_id], dtype=np.float64)
        geom_type = int(env._mj_model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            for sign_x in (-1.0, 1.0):
                for sign_y in (-1.0, 1.0):
                    local_corner = np.array(
                        [sign_x * size[0], sign_y * size[1], 0.0],
                        dtype=np.float64,
                    )
                    points.append((center + rotation @ local_corner)[:2])
        else:
            radius = float(max(size[0], size[1]))
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                points.append(
                    center[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
                )
    if not points:
        return np.empty((0, 2), dtype=np.float64), 0
    return np.asarray(points, dtype=np.float64), contact_count


def whole_body_com(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mass-weighted world COM position and velocity."""
    body_masses = np.asarray(model.body_mass[1:], dtype=np.float64)
    total_mass = max(float(np.sum(body_masses)), 1e-9)
    com_position = np.sum(
        np.asarray(data.xipos[1:], dtype=np.float64) * body_masses[:, None],
        axis=0,
    ) / total_mass
    com_velocity = np.sum(
        np.asarray(data.cvel[1:, 3:6], dtype=np.float64) * body_masses[:, None],
        axis=0,
    ) / total_mass
    return com_position, com_velocity


def main() -> None:
    """Run the inverse-dynamics audit and write per-sample CSV output."""
    args = parse_args()
    if not 0.0 <= float(args.start_phase) <= 1.0:
        raise ValueError(f"--start-phase must be in [0, 1], got {args.start_phase}.")
    if float(args.duration_seconds) <= 0.0:
        raise ValueError(
            f"--duration-seconds must be positive, got {args.duration_seconds}."
        )
    if float(args.sample_dt) <= 0.0:
        raise ValueError(f"--sample-dt must be positive, got {args.sample_dt}.")

    jax.config.update("jax_platform_name", "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    model = env._mj_model
    data = mujoco.MjData(model)

    clip_id = int(args.clip_id)
    clip_count = int(np.asarray(env._bvh_reference_clip_count))
    if clip_id < 0 or clip_id >= clip_count:
        raise ValueError(f"--clip-id {clip_id} out of range for {clip_count} clips.")

    motion_length = float(np.asarray(env._bvh_reference_motion_lengths)[clip_id])
    frame_time = float(np.asarray(env._bvh_reference_frame_times)[clip_id])
    loop_mode = int(np.asarray(env._bvh_reference_loop_modes)[clip_id])
    motion_start_time = float(args.start_phase) * motion_length
    motion_end_time = min(motion_start_time + float(args.duration_seconds), motion_length)
    sample_times = np.arange(
        motion_start_time,
        motion_end_time + (0.5 * float(args.sample_dt)),
        float(args.sample_dt),
        dtype=np.float64,
    )

    actuator_limits = np.maximum(
        np.abs(np.asarray(model.actuator_forcerange, dtype=np.float64)).max(axis=1),
        1e-6,
    )

    rows: list[dict[str, object]] = []
    root_force_norms: list[float] = []
    root_torque_norms: list[float] = []
    actuator_ratio_maxes: list[float] = []
    actuator_ratio_means: list[float] = []
    over_limit_counts: list[float] = []
    max_actuator_speeds: list[float] = []
    max_actuator_accelerations: list[float] = []
    com_support_margins: list[float] = []
    capture_support_margins: list[float] = []

    for sample_index, motion_time in enumerate(sample_times):
        prev_time = clip_time(motion_time - float(args.sample_dt), motion_length, loop_mode)
        next_time = clip_time(motion_time + float(args.sample_dt), motion_length, loop_mode)
        ref_prev = query_reference_np(env, clip_id, prev_time)
        ref_now = query_reference_np(env, clip_id, float(motion_time))
        ref_next = query_reference_np(env, clip_id, next_time)

        qpos_now, qvel_now = build_full_state(env, ref_now)
        _, qvel_prev = build_full_state(env, ref_prev)
        _, qvel_next = build_full_state(env, ref_next)
        qacc_now = (qvel_next - qvel_prev) / max(2.0 * float(args.sample_dt), 1e-9)

        data.qpos[:] = qpos_now
        data.qvel[:] = qvel_now
        data.qacc[:] = qacc_now
        data.ctrl[:] = np.asarray(ref_now["qpos"], dtype=np.float64)
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:, :] = 0.0
        mujoco.mj_inverse(model, data)

        com_position, com_velocity = whole_body_com(model, data)
        foot_bottom_heights = [
            env._geom_min_z(data, int(geom_id))
            for geom_id in env._foot_geom_ids_np
        ]
        foot_centers = np.asarray(
            data.geom_xpos[env._foot_geom_ids_np],
            dtype=np.float64,
        )
        support_points, support_contact_count = support_polygon_points(
            env,
            data,
            float(args.support_contact_height),
        )
        com_support_margin = signed_support_margin(
            com_position[:2],
            support_points,
        )
        natural_frequency = np.sqrt(9.81 / max(float(com_position[2]), 0.1))
        capture_point = com_position[:2] + com_velocity[:2] / natural_frequency
        capture_support_margin = signed_support_margin(capture_point, support_points)

        root_force = np.asarray(data.qfrc_inverse[:3], dtype=np.float64)
        root_torque = np.asarray(data.qfrc_inverse[3:6], dtype=np.float64)
        actuator_torques = np.asarray(
            data.qfrc_inverse[np.asarray(env._actuator_dof_indices_np)],
            dtype=np.float64,
        )
        actuator_velocities = qvel_now[np.asarray(env._actuator_dof_indices_np)]
        actuator_accelerations = qacc_now[
            np.asarray(env._actuator_dof_indices_np)
        ]
        actuator_ratios = np.abs(actuator_torques) / actuator_limits
        over_limit = actuator_ratios > 1.0

        root_force_norm = float(np.linalg.norm(root_force))
        root_torque_norm = float(np.linalg.norm(root_torque))
        actuator_ratio_max = float(np.max(actuator_ratios))
        actuator_ratio_mean = float(np.mean(actuator_ratios))
        over_limit_count = float(np.sum(over_limit))

        root_force_norms.append(root_force_norm)
        root_torque_norms.append(root_torque_norm)
        actuator_ratio_maxes.append(actuator_ratio_max)
        actuator_ratio_means.append(actuator_ratio_mean)
        over_limit_counts.append(over_limit_count)
        max_actuator_speeds.append(float(np.max(np.abs(actuator_velocities))))
        max_actuator_accelerations.append(
            float(np.max(np.abs(actuator_accelerations)))
        )
        if np.isfinite(com_support_margin):
            com_support_margins.append(com_support_margin)
        if np.isfinite(capture_support_margin):
            capture_support_margins.append(capture_support_margin)

        rows.append(
            {
                "sample_index": sample_index,
                "motion_time_s": float(motion_time),
                "frame0": int(ref_now["frame0"]),
                "frame1": int(ref_now["frame1"]),
                "alpha": float(ref_now["alpha"]),
                "frame_time_s": frame_time,
                "root_x": float(qpos_now[0]),
                "root_y": float(qpos_now[1]),
                "root_z": float(qpos_now[2]),
                "com_x": float(com_position[0]),
                "com_y": float(com_position[1]),
                "com_z": float(com_position[2]),
                "com_velocity_x": float(com_velocity[0]),
                "com_velocity_y": float(com_velocity[1]),
                "capture_point_x": float(capture_point[0]),
                "capture_point_y": float(capture_point[1]),
                "support_contact_count": support_contact_count,
                "left_foot_bottom_height_m": float(foot_bottom_heights[0]),
                "right_foot_bottom_height_m": float(foot_bottom_heights[1]),
                "left_foot_x": float(foot_centers[0, 0]),
                "left_foot_y": float(foot_centers[0, 1]),
                "right_foot_x": float(foot_centers[1, 0]),
                "right_foot_y": float(foot_centers[1, 1]),
                "com_support_margin_m": com_support_margin,
                "capture_support_margin_m": capture_support_margin,
                "root_force_x": float(root_force[0]),
                "root_force_y": float(root_force[1]),
                "root_force_z": float(root_force[2]),
                "root_force_norm": root_force_norm,
                "root_torque_x": float(root_torque[0]),
                "root_torque_y": float(root_torque[1]),
                "root_torque_z": float(root_torque[2]),
                "root_torque_norm": root_torque_norm,
                "max_actuator_ratio": actuator_ratio_max,
                "mean_actuator_ratio": actuator_ratio_mean,
                "over_limit_actuator_count": over_limit_count,
                "max_abs_actuator_speed_rad_s": float(
                    np.max(np.abs(actuator_velocities))
                ),
                "max_abs_actuator_acceleration_rad_s2": float(
                    np.max(np.abs(actuator_accelerations))
                ),
                "worst_acceleration_joint": env._actuator_joint_names[
                    int(np.argmax(np.abs(actuator_accelerations)))
                ],
                "worst_actuator_name": env._actuator_joint_names[int(np.argmax(actuator_ratios))],
                "worst_actuator_ratio": actuator_ratio_max,
            }
        )

    csv_path = args.out_dir / f"clip_{clip_id:03d}_dynamics_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / f"clip_{clip_id:03d}_dynamics_summary.csv"
    root_force_mean, root_force_maxabs, root_force_max = summarize_series(root_force_norms)
    root_torque_mean, root_torque_maxabs, root_torque_max = summarize_series(root_torque_norms)
    ratio_mean, ratio_maxabs, ratio_max = summarize_series(actuator_ratio_maxes)
    ratio_avg_mean, ratio_avg_maxabs, ratio_avg_max = summarize_series(actuator_ratio_means)
    over_limit_mean, over_limit_maxabs, over_limit_max = summarize_series(over_limit_counts)
    com_margin_mean, _, com_margin_max = summarize_series(com_support_margins)
    capture_margin_mean, _, capture_margin_max = summarize_series(
        capture_support_margins
    )
    speed_mean, _, speed_max = summarize_series(max_actuator_speeds)
    acceleration_mean, _, acceleration_max = summarize_series(
        max_actuator_accelerations
    )
    summary_row = {
        "clip_id": clip_id,
        "motion_start_time_s": motion_start_time,
        "motion_end_time_s": motion_end_time,
        "sample_dt_s": float(args.sample_dt),
        "samples": len(rows),
        "root_force_norm_mean": root_force_mean,
        "root_force_norm_max": root_force_max,
        "root_torque_norm_mean": root_torque_mean,
        "root_torque_norm_max": root_torque_max,
        "max_actuator_ratio_mean": ratio_mean,
        "max_actuator_ratio_max": ratio_max,
        "mean_actuator_ratio_mean": ratio_avg_mean,
        "mean_actuator_ratio_max": ratio_avg_max,
        "over_limit_actuator_count_mean": over_limit_mean,
        "over_limit_actuator_count_max": over_limit_max,
        "max_abs_actuator_speed_mean_rad_s": speed_mean,
        "max_abs_actuator_speed_max_rad_s": speed_max,
        "max_abs_actuator_acceleration_mean_rad_s2": acceleration_mean,
        "max_abs_actuator_acceleration_max_rad_s2": acceleration_max,
        "com_support_margin_mean_m": com_margin_mean,
        "com_support_margin_min_m": min(com_support_margins, default=float("nan")),
        "com_support_margin_max_m": com_margin_max,
        "com_outside_support_fraction": (
            sum(margin < 0.0 for margin in com_support_margins)
            / max(len(com_support_margins), 1)
        ),
        "capture_support_margin_mean_m": capture_margin_mean,
        "capture_support_margin_min_m": min(
            capture_support_margins,
            default=float("nan"),
        ),
        "capture_support_margin_max_m": capture_margin_max,
        "capture_outside_support_fraction": (
            sum(margin < 0.0 for margin in capture_support_margins)
            / max(len(capture_support_margins), 1)
        ),
    }
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Audit CSV: {csv_path}")
    print(f"Summary CSV: {summary_path}")
    print(
        "root_force_norm mean={:.3f} max={:.3f} | "
        "root_torque_norm mean={:.3f} max={:.3f}".format(
            root_force_mean,
            root_force_max,
            root_torque_mean,
            root_torque_max,
        )
    )
    print(
        "max_actuator_ratio mean={:.3f} max={:.3f} | "
        "over_limit_actuator_count mean={:.3f} max={:.0f}".format(
            ratio_mean,
            ratio_max,
            over_limit_mean,
            over_limit_max,
        )
    )
    print(
        "support margins | COM mean={:.3f}m min={:.3f}m outside={:.1%} | "
        "capture mean={:.3f}m min={:.3f}m outside={:.1%}".format(
            com_margin_mean,
            min(com_support_margins, default=float("nan")),
            summary_row["com_outside_support_fraction"],
            capture_margin_mean,
            min(capture_support_margins, default=float("nan")),
            summary_row["capture_outside_support_fraction"],
        )
    )


if __name__ == "__main__":
    main()
