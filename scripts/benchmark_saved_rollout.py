"""Compare batched physics throughput with a fixed saved policy, without PPO.

Run from the project root in WSL. Compilation is timed separately. The model,
reference, policy and episode limit stay fixed while the number of simultaneous
worlds changes. Reported rates exclude optimizer updates and training callbacks.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np
from brax.training import acting
from mujoco_playground._src import mjx_env

from evaluate import (
    find_run_config,
    load_ppo_policy,
    restore_saved_env_config,
    validate_xml_manifest,
)
from train import choose_device, make_environment, wrap_biomechanics_training


def measure_batch(
    env: mjx_env.MjxEnv,
    policy: Callable,
    count: int,
    blocks: int,
    steps: int,
) -> dict[str, Any]:
    """Measure synchronized warm blocks including production auto-reset."""
    print(f"Preparing {count} environments", flush=True)
    # Keep each world's initial RNG independent of the requested batch size.
    keys = jax.vmap(lambda index: jax.random.fold_in(jax.random.PRNGKey(7), index))(
        np.arange(count, dtype=np.int32)
    )
    started = time.perf_counter()
    state = jax.block_until_ready(jax.jit(env.reset)(keys))
    reset_seconds = time.perf_counter() - started
    print(f"envs={count}: reset finished in {reset_seconds:.2f}s", flush=True)

    @jax.jit
    def rollout(state, key):
        return acting.generate_unroll(env, state, policy, key, steps)[0]

    started = time.perf_counter()
    state = jax.block_until_ready(rollout(state, jax.random.PRNGKey(11)))
    cold_seconds = time.perf_counter() - started
    print(
        f"envs={count}: reset={reset_seconds:.2f}s, cold block={cold_seconds:.2f}s",
        flush=True,
    )
    times = []
    for block in range(blocks):
        started = time.perf_counter()
        state = jax.block_until_ready(rollout(state, jax.random.PRNGKey(12 + block)))
        seconds = time.perf_counter() - started
        times.append(seconds)
        print(
            f"envs={count} block={block + 1}: {count * steps / seconds:.2f} steps/s",
            flush=True,
        )
    memory = jax.devices("gpu")[0].memory_stats() or {}
    result = {
        "num_envs": count,
        "reset_seconds": reset_seconds,
        "cold_block_seconds": cold_seconds,
        "block_seconds": times,
        # The first repeat can specialize again on the post-rollout state.
        # Preserve its timing, but exclude it from steady-state throughput.
        "settling_block_seconds": times[0],
        "warm_block_seconds": times[1:],
        "steps_per_second": count * steps * (blocks - 1) / sum(times[1:]),
        "gpu_bytes_in_use": memory.get("bytes_in_use"),
        "gpu_peak_bytes_in_use": memory.get("peak_bytes_in_use"),
    }
    # Release compiled shapes before testing a larger batch on a 6GB GPU.
    del state, rollout
    jax.clear_caches()
    gc.collect()
    return result


def main() -> None:
    """Profile only; never update the policy or restart a training process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if min(*args.envs, args.blocks, args.steps) < 1:
        parser.error("Counts must be positive")
    if args.blocks < 2:
        parser.error("Use at least two repeats: one settling and one measured")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    config = find_run_config(args.checkpoint)
    if config is None:
        parser.error("Checkpoint has no owning run config")
    device = choose_device("gpu", False)
    jax.config.update("jax_default_device", device)
    env_config = restore_saved_env_config(config)
    if env_config.physics_backend != "mjx_jax":
        parser.error("This benchmark is validated only for mjx_jax")
    env = make_environment(env_config, enable_erfi=False)
    validate_xml_manifest(args.checkpoint, env)
    env = wrap_biomechanics_training(
        env, config["brax_ppo"]["episode_length"], config["brax_ppo"]["action_repeat"]
    )
    policy = load_ppo_policy(args.checkpoint, deterministic=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        report = {
            "checkpoint": str(args.checkpoint),
            "jax_version": jax.__version__,
            "steps_per_block": args.steps,
            "results": [],
        }
        for count in args.envs:
            report["results"].append(
                measure_batch(env, policy, count, args.blocks, args.steps)
            )
            # Preserve completed measurements if a later batch exhausts RAM.
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False)
            output.truncate()
            output.flush()
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
