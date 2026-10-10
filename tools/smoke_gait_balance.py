"""Run one CPU environment step and report lateral gait-balance metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv


def main() -> None:
    """Verify that the COM-transfer reward executes in the real MJX env."""
    env = BiomechanicsJoystickEnv(
        config_overrides={
            "impl": "jax",
            "physics_backend": "mjx_jax",
            "reference_gait": "none",
            "reference_gait_file": None,
            "command_profile": "step_in_place",
            "neutral_stance_hip_x": 0.0,
            "gait_active_without_command": True,
            "gait_shaping_scale": 0.0,
            "gait_balance_reward_scale": 1.0,
        }
    )
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, jnp.zeros(env.action_size))
    print(
        "gait balance smoke | "
        f"reward={float(state.metrics['gait_balance_reward']):.6f} | "
        f"com_error={float(state.metrics['gait_com_error']):.6f} | "
        f"com_height={float(state.metrics['com_height']):.6f}"
    )


if __name__ == "__main__":
    main()
