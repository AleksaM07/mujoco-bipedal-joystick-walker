"""Check whether narrow-stance lateral COM targets are kinematically reachable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv  # noqa: I001


FOOT_MARKERS = (
    "metatarsal_midpoint_right",
    "metatarsal_midpoint_left",
)


def parse_args() -> argparse.Namespace:
    """Parse static balance-pose options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--foot-weight", type=float, default=200.0)
    parser.add_argument("--com-weight", type=float, default=80.0)
    parser.add_argument("--pose-weight", type=float, default=0.05)
    parser.add_argument("--root-weight", type=float, default=0.05)
    return parser.parse_args()


def solve_transfer_pose(
    env: BiomechanicsJoystickEnv,
    stance_index: int,
    target_fraction: float,
    iterations: int,
    foot_weight: float,
    com_weight: float,
    pose_weight: float,
    root_weight: float,
) -> tuple[np.ndarray, float]:
    """Solve root-Y and actuator pose while keeping both feet fixed."""
    model = env._mj_model
    data = mujoco.MjData(model)
    neutral_qpos = np.asarray(env._init_q_np, dtype=np.float64)
    neutral_ctrl = np.asarray(env._default_ctrl, dtype=np.float64)
    marker_specs = env._resolve_host_ik_markers(FOOT_MARKERS)

    data.qpos[:] = neutral_qpos
    mujoco.mj_forward(model, data)
    foot_targets = np.array(
        [env._host_marker_position(data, marker) for marker in marker_specs]
    )
    midpoint_y = float(np.mean(foot_targets[:, 1]))
    stance_y = float(foot_targets[stance_index, 1])
    target_com_y = midpoint_y + target_fraction * (stance_y - midpoint_y)

    root_y = float(neutral_qpos[1])
    actuator_qpos = neutral_ctrl.copy()
    root_y_start = root_y

    for _ in range(iterations):
        data.qpos[:] = neutral_qpos
        data.qpos[1] = root_y
        data.qpos[env._actuator_qpos_indices_np] = actuator_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        residuals: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        sqrt_foot_weight = np.sqrt(foot_weight)
        for marker, target in zip(marker_specs, foot_targets, strict=True):
            current = env._host_marker_position(data, marker)
            marker_jacobian = env._host_marker_jacobian(data, marker)
            root_y_jacobian = np.array([[0.0], [1.0], [0.0]])
            residuals.append(sqrt_foot_weight * (target - current))
            jacobians.append(
                sqrt_foot_weight
                * np.hstack((root_y_jacobian, marker_jacobian))
            )

        current_com = env._host_whole_body_com(data)
        com_jacobian = env._host_whole_body_com_jacobian(data)
        sqrt_com_weight = np.sqrt(com_weight)
        residuals.append(
            np.array([sqrt_com_weight * (target_com_y - current_com[1])])
        )
        jacobians.append(
            sqrt_com_weight
            * np.hstack((np.ones((1, 1)), com_jacobian[1:2]))
        )

        sqrt_pose_weight = np.sqrt(pose_weight)
        residuals.append(sqrt_pose_weight * (neutral_ctrl - actuator_qpos))
        jacobians.append(
            sqrt_pose_weight
            * np.hstack(
                (
                    np.zeros((env.action_size, 1)),
                    np.eye(env.action_size),
                )
            )
        )
        residuals.append(
            np.array([np.sqrt(root_weight) * (root_y_start - root_y)])
        )
        jacobians.append(
            np.sqrt(root_weight)
            * np.hstack((np.ones((1, 1)), np.zeros((1, env.action_size))))
        )

        update, *_ = np.linalg.lstsq(
            np.vstack(jacobians),
            np.concatenate(residuals),
            rcond=None,
        )
        root_y += float(np.clip(update[0], -0.02, 0.02))
        actuator_qpos += np.clip(update[1:], -0.08, 0.08)
        actuator_qpos = np.clip(
            actuator_qpos,
            env._actuator_ctrl_lower_limits_np,
            env._actuator_ctrl_upper_limits_np,
        )

    return actuator_qpos, root_y


def summarize_pose(
    env: BiomechanicsJoystickEnv,
    label: str,
    actuator_qpos: np.ndarray,
    root_y: float,
    target_fraction: float,
) -> None:
    """Print achieved COM shift, foot drift, and actuator offsets."""
    model = env._mj_model
    neutral_data = mujoco.MjData(model)
    neutral_data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    mujoco.mj_forward(model, neutral_data)

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    data.qpos[1] = root_y
    data.qpos[env._actuator_qpos_indices_np] = actuator_qpos
    mujoco.mj_forward(model, data)

    marker_specs = env._resolve_host_ik_markers(FOOT_MARKERS)
    neutral_feet = np.array(
        [env._host_marker_position(neutral_data, marker) for marker in marker_specs]
    )
    feet = np.array(
        [env._host_marker_position(data, marker) for marker in marker_specs]
    )
    neutral_com = env._host_whole_body_com(neutral_data)
    com = env._host_whole_body_com(data)
    offsets = actuator_qpos - np.asarray(env._default_ctrl, dtype=np.float64)
    named_offsets = {
        name: round(float(offset), 5)
        for name, offset in zip(env._actuator_joint_names, offsets, strict=True)
        if abs(float(offset)) >= 1e-3
    }
    target_shift = target_fraction * abs(neutral_feet[1, 1] - neutral_feet[0, 1]) / 2.0
    print(
        f"{label} target_shift={target_shift:.5f} "
        f"com_y_shift={com[1] - neutral_com[1]:.5f} "
        f"max_foot_drift={np.max(np.linalg.norm(feet - neutral_feet, axis=1)):.5f} "
        f"root_y_shift={root_y - neutral_data.qpos[1]:.5f}"
    )
    print(f"{label} offsets={named_offsets}")


def main() -> None:
    """Solve both left- and right-stance transfer poses."""
    args = parse_args()
    if not 0.0 <= args.target_fraction <= 1.0:
        raise ValueError("--target-fraction must be between 0 and 1")
    jax.config.update("jax_platform_name", "cpu")
    env = BiomechanicsJoystickEnv(
        config_overrides={
            "impl": "jax",
            "physics_backend": "mjx_jax",
            "reference_gait": "none",
            "reference_gait_file": None,
            "neutral_stance_hip_x": 0.0,
            "arm_actuators": False,
            "enable_erfi": False,
        }
    )
    for stance_index, label in ((0, "right_stance"), (1, "left_stance")):
        actuator_qpos, root_y = solve_transfer_pose(
            env,
            stance_index,
            args.target_fraction,
            args.iterations,
            args.foot_weight,
            args.com_weight,
            args.pose_weight,
            args.root_weight,
        )
        summarize_pose(
            env,
            label,
            actuator_qpos,
            root_y,
            args.target_fraction,
        )


if __name__ == "__main__":
    main()
