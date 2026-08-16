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
from config import DEFAULT_BVH_REFERENCE_LIST, expand_reference_gait_files  # noqa: E402
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

        root_force = np.asarray(data.qfrc_inverse[:3], dtype=np.float64)
        root_torque = np.asarray(data.qfrc_inverse[3:6], dtype=np.float64)
        actuator_torques = np.asarray(
            data.qfrc_inverse[np.asarray(env._actuator_dof_indices_np)],
            dtype=np.float64,
        )
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


if __name__ == "__main__":
    main()
