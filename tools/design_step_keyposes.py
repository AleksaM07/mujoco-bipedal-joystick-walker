"""Solve static left/right swing keyposes for the v21 MuJoCo model."""

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


MARKER_NAMES = (
    "metatarsal_midpoint_right",
    "metatarsal_midpoint_left",
)


def parse_args() -> argparse.Namespace:
    """Parse static IK design options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance", type=float, default=0.04)
    parser.add_argument("--com-weight", type=float, default=20.0)
    parser.add_argument("--pose-regularization", type=float, default=0.05)
    parser.add_argument("--support-height-scale", type=float, default=0.01)
    return parser.parse_args()


def solve_keypose(
    env: BiomechanicsJoystickEnv,
    swing_marker_index: int,
    clearance: float,
) -> np.ndarray:
    """Fit one neutral-root keypose with one raised metatarsal target."""
    data = mujoco.MjData(env._mj_model)
    data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(env._mj_model, data)
    marker_specs = env._resolve_host_ik_markers(MARKER_NAMES)
    marker_targets = np.array(
        [env._host_marker_position(data, marker) for marker in marker_specs],
        dtype=np.float32,
    )
    marker_targets[swing_marker_index, 2] += clearance

    fitted, _ = env._retarget_reference_qpos_to_ik_targets(
        qpos_targets=np.asarray(env._default_ctrl, dtype=np.float32)[None, None, :],
        ik_target_positions=marker_targets[None, None, :, :],
        ik_target_names=MARKER_NAMES,
        root_pos_targets=np.asarray(env._init_q_np[:3], dtype=np.float32)[
            None, None, :
        ],
        root_quat_targets=np.asarray(env._init_q_np[3:7], dtype=np.float32)[
            None, None, :
        ],
        frame_counts=np.array([1], dtype=np.int32),
    )
    return fitted[0, 0].astype(np.float64)


def summarize_keypose(
    env: BiomechanicsJoystickEnv,
    label: str,
    keypose: np.ndarray,
) -> None:
    """Print actuator offsets and achieved marker/COM displacement."""
    data = mujoco.MjData(env._mj_model)
    data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    data.qpos[env._actuator_qpos_indices_np] = keypose
    data.qvel[:] = 0.0
    mujoco.mj_forward(env._mj_model, data)

    neutral_data = mujoco.MjData(env._mj_model)
    neutral_data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    mujoco.mj_forward(env._mj_model, neutral_data)
    marker_specs = env._resolve_host_ik_markers(MARKER_NAMES)
    foot_delta = np.array(
        [
            env._host_marker_position(data, marker)
            - env._host_marker_position(neutral_data, marker)
            for marker in marker_specs
        ]
    )
    com_delta = env._host_whole_body_com(data) - env._host_whole_body_com(
        neutral_data
    )
    offsets = keypose - np.asarray(env._default_ctrl, dtype=np.float64)
    named_offsets = {
        name: round(float(offset), 5)
        for name, offset in zip(env._actuator_joint_names, offsets, strict=True)
        if abs(float(offset)) >= 1e-4
    }
    print(f"{label} offsets={named_offsets}")
    print(
        f"{label} right_foot_delta={foot_delta[0].round(5).tolist()} "
        f"left_foot_delta={foot_delta[1].round(5).tolist()} "
        f"com_delta={com_delta.round(5).tolist()}"
    )


def search_sagittal_swing_keypose(
    env: BiomechanicsJoystickEnv,
    side: str,
    clearance: float,
) -> np.ndarray:
    """Find a simple hip/knee pair that raises one foot with little XY drift."""
    data = mujoco.MjData(env._mj_model)
    neutral_qpos = np.asarray(env._init_q_np, dtype=np.float64)
    default_ctrl = np.asarray(env._default_ctrl, dtype=np.float64)
    data.qpos[:] = neutral_qpos
    mujoco.mj_forward(env._mj_model, data)
    marker_name = f"metatarsal_midpoint_{side}"
    marker = env._resolve_host_ik_markers((marker_name,))[0]
    neutral_marker = env._host_marker_position(data, marker).copy()
    hip_index = env._actuator_joint_names.index(f"{side}_hip_z")
    knee_index = env._actuator_joint_names.index(f"{side}_knee_z")
    best_score = np.inf
    best_target = default_ctrl.copy()

    for hip_offset in np.linspace(-0.30, 0.30, 61):
        for knee_offset in np.linspace(-0.50, 0.0, 51):
            target = default_ctrl.copy()
            target[hip_index] += hip_offset
            target[knee_index] += knee_offset
            target = np.clip(
                target,
                env._actuator_ctrl_lower_limits_np,
                env._actuator_ctrl_upper_limits_np,
            )
            data.qpos[:] = neutral_qpos
            data.qpos[env._actuator_qpos_indices_np] = target
            mujoco.mj_forward(env._mj_model, data)
            marker_delta = env._host_marker_position(data, marker) - neutral_marker
            height_error = marker_delta[2] - clearance
            score = (
                100.0 * height_error**2
                + 0.20 * marker_delta[0] ** 2
                + 5.0 * marker_delta[1] ** 2
                + 0.002 * (hip_offset**2 + knee_offset**2)
            )
            if score < best_score:
                best_score = float(score)
                best_target = target.copy()
    return best_target


def main() -> None:
    """Solve and print both alternating keyposes."""
    args = parse_args()
    if args.clearance <= 0.0:
        raise ValueError("--clearance must be positive")
    jax.config.update("jax_platform_name", "cpu")
    env = BiomechanicsJoystickEnv(
        config_overrides={
            "impl": "jax",
            "physics_backend": "mjx_jax",
            "reference_gait": "none",
            "arm_actuators": False,
            "enable_erfi": False,
            "reference_ik_com_weight": args.com_weight,
            "reference_ik_support_height_scale": args.support_height_scale,
            "reference_ik_pose_regularization": args.pose_regularization,
            "reference_ik_smooth_regularization": 0.0,
        }
    )
    neutral_data = mujoco.MjData(env._mj_model)
    neutral_data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    mujoco.mj_forward(env._mj_model, neutral_data)
    neutral_markers = env._resolve_host_ik_markers(MARKER_NAMES)
    print(
        "neutral "
        f"right_foot={env._host_marker_position(neutral_data, neutral_markers[0]).round(5).tolist()} "
        f"left_foot={env._host_marker_position(neutral_data, neutral_markers[1]).round(5).tolist()} "
        f"com={env._host_whole_body_com(neutral_data).round(5).tolist()}"
    )
    for hip_x_magnitude in (0.18, 0.12, 0.06, 0.0, -0.06):
        stance_data = mujoco.MjData(env._mj_model)
        stance_data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
        stance_targets = np.asarray(env._default_ctrl, dtype=np.float64).copy()
        stance_targets[env._actuator_joint_names.index("left_hip_x")] = (
            hip_x_magnitude
        )
        stance_targets[env._actuator_joint_names.index("right_hip_x")] = (
            -hip_x_magnitude
        )
        stance_data.qpos[env._actuator_qpos_indices_np] = stance_targets
        mujoco.mj_forward(env._mj_model, stance_data)
        right_y = env._host_marker_position(stance_data, neutral_markers[0])[1]
        left_y = env._host_marker_position(stance_data, neutral_markers[1])[1]
        print(
            f"stance hip_x=+/-{hip_x_magnitude:.2f} "
            f"width={abs(left_y - right_y):.5f} "
            f"right_y={right_y:.5f} left_y={left_y:.5f}"
        )
    right_swing = solve_keypose(env, 0, args.clearance)
    left_swing = solve_keypose(env, 1, args.clearance)
    summarize_keypose(env, "right_swing", right_swing)
    summarize_keypose(env, "left_swing", left_swing)
    simple_right_swing = search_sagittal_swing_keypose(
        env,
        "right",
        args.clearance,
    )
    simple_left_swing = search_sagittal_swing_keypose(
        env,
        "left",
        args.clearance,
    )
    summarize_keypose(env, "simple_right_swing", simple_right_swing)
    summarize_keypose(env, "simple_left_swing", simple_left_swing)


if __name__ == "__main__":
    main()
