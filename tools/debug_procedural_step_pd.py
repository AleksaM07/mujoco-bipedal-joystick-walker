"""Test a phase-conditioned procedural stepping target with host MuJoCo.

The diagnostic deliberately bypasses PPO.  It answers whether the existing
sinusoidal leg target can create alternating foot clearance from the neutral
standing pose before we use it as an action prior for an RL curriculum.
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
from tools.debug_reference_playback import torso_up  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse diagnostic options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=float,
        action="append",
        help="Prior amplitude to test. May be repeated.",
    )
    parser.add_argument("--period-steps", type=int, default=100)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args()


def foot_lowest_heights(
    env: BiomechanicsJoystickEnv,
    data: mujoco.MjData,
) -> np.ndarray:
    """Return the lowest world-Z point of both sole geoms."""
    return np.array(
        [
            env._geom_min_z(data, int(geom_id))
            for geom_id in env._foot_geom_ids_np
        ],
        dtype=np.float64,
    )


def procedural_offset(
    env: BiomechanicsJoystickEnv,
    phase: float,
) -> np.ndarray:
    """Reproduce the environment's existing sinusoidal gait target."""
    sin_phase = np.sin(phase)
    return (
        np.asarray(env._reference_gait_sin_offsets) * sin_phase
        + np.asarray(env._reference_gait_pos_sin_offsets) * max(sin_phase, 0.0)
        + np.asarray(env._reference_gait_neg_sin_offsets) * max(-sin_phase, 0.0)
    )


def run_scale(
    env: BiomechanicsJoystickEnv,
    scale: float,
    period_steps: int,
    settle_steps: int,
    steps: int,
    print_every: int,
) -> dict[str, float | int | str]:
    """Run one open-loop PD trial and return foot-clearance diagnostics."""
    model = env._mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(env._init_q_np, dtype=np.float64)
    data.qvel[:] = 0.0
    default_ctrl = np.asarray(env._default_ctrl, dtype=np.float64)
    data.ctrl[:] = default_ctrl
    mujoco.mj_forward(model, data)

    initial_root_xy = data.qpos[:2].copy()
    baseline_foot_z = foot_lowest_heights(env, data)
    max_relative_clearance = np.zeros(2, dtype=np.float64)
    min_torso_up = torso_up(env, data)
    min_root_height = float(data.qpos[2])
    final_step = steps - 1
    reason = "none"

    for step in range(settle_steps + steps):
        gait_step = max(step - settle_steps, 0)
        phase = 2.0 * np.pi * gait_step / max(period_steps, 1)
        target = default_ctrl
        if step >= settle_steps:
            target = default_ctrl + scale * procedural_offset(env, phase)
        data.ctrl[:] = np.clip(
            target,
            env._actuator_ctrl_lower_limits_np,
            env._actuator_ctrl_upper_limits_np,
        )

        for _ in range(env.n_substeps):
            mujoco.mj_step(model, data)

        foot_z = foot_lowest_heights(env, data)
        relative_clearance = foot_z - np.min(foot_z)
        max_relative_clearance = np.maximum(
            max_relative_clearance,
            relative_clearance,
        )
        current_torso_up = torso_up(env, data)
        min_torso_up = min(min_torso_up, current_torso_up)
        min_root_height = min(min_root_height, float(data.qpos[2]))

        low = data.qpos[2] < env.MIN_STANDING_HEIGHT_RATIO * model.qpos0[2]
        tipped = current_torso_up < 0.25
        invalid = not (
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
        if invalid:
            reason = "invalid"
        elif low:
            reason = "low_height"
        elif tipped:
            reason = "tipped"

        should_print = (
            step == settle_steps - 1
            or step == settle_steps
            or (step >= settle_steps and gait_step % print_every == 0)
            or reason != "none"
            or step == settle_steps + steps - 1
        )
        if should_print:
            print(
                f"scale={scale:.3f} step={step:04d} gait_step={gait_step:04d} "
                f"phase={phase:.3f} reason={reason} root_z={data.qpos[2]:.3f} "
                f"torso_up={current_torso_up:.3f} "
                f"foot_low_z=({foot_z[0]:.3f},{foot_z[1]:.3f}) "
                f"rel_clear=({relative_clearance[0]:.3f},"
                f"{relative_clearance[1]:.3f})"
            )

        if reason != "none":
            final_step = step - settle_steps
            break

    root_travel = float(np.linalg.norm(data.qpos[:2] - initial_root_xy))
    return {
        "scale": scale,
        "survived_gait_steps": max(final_step, 0),
        "reason": reason,
        "left_max_relative_clearance_m": float(max_relative_clearance[0]),
        "right_max_relative_clearance_m": float(max_relative_clearance[1]),
        "min_root_height_m": min_root_height,
        "min_torso_up": min_torso_up,
        "root_xy_travel_m": root_travel,
        "baseline_left_foot_z_m": float(baseline_foot_z[0]),
        "baseline_right_foot_z_m": float(baseline_foot_z[1]),
    }


def main() -> None:
    """Run every requested scale from the same neutral initial state."""
    args = parse_args()
    if args.period_steps <= 0:
        raise ValueError("--period-steps must be positive")
    if args.steps <= 0 or args.settle_steps < 0:
        raise ValueError("--steps must be positive and --settle-steps non-negative")

    jax.config.update("jax_platform_name", "cpu")
    env = BiomechanicsJoystickEnv(
        config_overrides={
            "impl": "jax",
            "physics_backend": "mjx_jax",
            "reference_gait": "none",
            "arm_actuators": False,
            "enable_erfi": False,
        }
    )
    scales = args.scale or [0.0, 0.10, 0.25, 0.50, 1.0]
    print(
        f"period_steps={args.period_steps} settle_steps={args.settle_steps} "
        f"steps={args.steps} dt={float(env.dt):.4f} scales={scales}"
    )
    print(f"actuators={env._actuator_joint_names}")

    summaries = [
        run_scale(
            env,
            scale,
            args.period_steps,
            args.settle_steps,
            args.steps,
            args.print_every,
        )
        for scale in scales
    ]
    print("SUMMARY")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
