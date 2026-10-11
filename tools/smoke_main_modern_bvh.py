"""Compile and exercise the main-based environment with the modern BVH prior."""

import jax
import jax.numpy as jp
import numpy as np

from config import EnvConfig
from tools.audit_main_modern_bvh import DEFAULT_REFERENCE
from train import make_environment


def main() -> None:
    """Check physics, rewards and reference sampling before a longer run."""
    env = make_environment(
        EnvConfig(
            reference_gait="bvh",
            reference_pipeline="modern",
            reference_gait_file=DEFAULT_REFERENCE,
            reference_target_observation=True,
        ),
        enable_erfi=False,
    )
    print("Compiling reset...", flush=True)
    state = jax.jit(env.reset)(jax.random.PRNGKey(7))
    jax.block_until_ready(state)
    print("Compiling step...", flush=True)
    step = jax.jit(env.step)
    for _ in range(12):
        state = step(state, jp.zeros(env.action_size))
    jax.block_until_ready(state)
    for leaf in jax.tree.leaves((state.obs, state.reward, state.data.qpos)):
        if not np.isfinite(np.asarray(leaf)).all():
            raise RuntimeError("Non-finite observation, reward or physics state")
    assert int(state.info["motion_step"]) == 12
    print(
        f"PASS: xml={env._xml_path.name}, actions={env.action_size}, "
        f"observation={state.obs['state'].shape}, reward={float(state.reward):.4f}, "
        f"done={float(state.done):.0f}, height={float(state.data.qpos[2]):.4f}",
        flush=True,
    )
    print("Smoke test only; zero actions do not establish walking.", flush=True)


if __name__ == "__main__":
    main()
