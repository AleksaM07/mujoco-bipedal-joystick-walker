"""Time a saved policy using the original Brax evaluation RNG sequence.

Example (from the project root, Linux/WSL):
    .venv/bin/python scripts/benchmark_saved_evaluation.py \
        --checkpoint runs/<run>/checkpoints/000002001920 \
        --eval-index 1 --output artifacts/eval_timing.json

Index 0 is the pre-training evaluation. Index 1 is the first saved checkpoint
evaluation. Use --legacy to measure the previous full-horizon physics path.
This measures evaluation only, never updates weights or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np
from brax.training import acting, checkpoint
from brax.training.agents.ppo import networks

from evaluate import (
    find_run_config,
    load_ppo_networks,
    restore_saved_env_config,
    validate_xml_manifest,
)
from train import (
    choose_device,
    make_environment,
    wrap_biomechanics_training,
    wrap_biomechanics_evaluation,
)


def main() -> None:
    """Compare repeatable evaluation scores and separate cold/warm timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--eval-index", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.eval_index < 0:
        parser.error("--eval-index must be nonnegative")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to separate startup from reuse")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    run_config = find_run_config(args.checkpoint)
    if run_config is None:
        parser.error("Checkpoint has no owning run config")
    config = run_config["brax_ppo"]
    device = choose_device(args.device, False)
    jax.config.update("jax_default_device", device)
    params = checkpoint.load(str(args.checkpoint.resolve()))
    leaves = jax.tree.leaves(params)
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise ValueError("Checkpoint contains nonfinite parameters")
    print(f"Restored {len(leaves)} finite parameter leaves on {device}", flush=True)
    policy_factory = networks.make_inference_fn(load_ppo_networks(args.checkpoint))
    env_config = restore_saved_env_config(run_config)
    if env_config.physics_backend != "mjx_jax":
        parser.error("This benchmark is validated only for mjx_jax")
    env = make_environment(env_config, enable_erfi=False)
    validate_xml_manifest(args.checkpoint, env)
    wrap = wrap_biomechanics_training if args.legacy else wrap_biomechanics_evaluation
    env = wrap(
        env, config["episode_length"], config["action_repeat"]
    )

    # Match installed Brax PPO's single-process evaluation key construction.
    _, local_key = jax.random.split(jax.random.PRNGKey(run_config["train"]["seed"]))
    local_key = jax.random.fold_in(local_key, 0)
    _, _, eval_key = jax.random.split(local_key, 3)
    for _ in range(args.eval_index):
        eval_key, _ = jax.random.split(eval_key)

    print("Initializing evaluator", flush=True)
    started = time.perf_counter()
    evaluator = acting.Evaluator(
        env,
        functools.partial(
            policy_factory, deterministic=config.get("deterministic_eval", False)
        ),
        num_eval_envs=config["num_eval_envs"],
        episode_length=config["episode_length"],
        action_repeat=config["action_repeat"],
        key=eval_key,
    )
    # These private fields belong to the installed Brax Evaluator; replaying the
    # same key isolates timing from changes in initial phases and policy noise.
    jax.block_until_ready(evaluator._eval_state_to_donate)
    initialization_seconds = time.perf_counter() - started
    print(f"Evaluator initialization: {initialization_seconds:.3f}s", flush=True)
    repeats = []
    for index in range(args.repeats):
        evaluator._key = eval_key
        print(
            f"Evaluation {index + 1}/{args.repeats} (early calls may compile)",
            flush=True,
        )
        metrics = evaluator.run_evaluation(params, {})
        state = evaluator._eval_state_to_donate
        episode_steps = np.asarray(state.info["eval_metrics"].episode_steps).tolist()
        result = {
            "metrics": {key: float(value) for key, value in metrics.items()},
            "episode_steps": episode_steps,
        }
        repeats.append(result)
        print(
            f"seconds={metrics['eval/epoch_eval_time']:.3f} "
            f"reward={metrics['eval/episode_reward']:.6f} "
            f"episode_steps={episode_steps}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "eval_index": args.eval_index,
                "legacy": args.legacy,
                "jax_version": jax.__version__,
                "device": str(device),
                "initialization_seconds": initialization_seconds,
                "repeats": repeats,
                "score_repeat_tolerance": {"rtol": 1e-6, "atol": 2e-5},
            },
            output,
            indent=2,
            allow_nan=False,
        )
    print(f"Saved: {args.output}", flush=True)
    for repeated in repeats[1:]:
        np.testing.assert_array_equal(
            repeats[0]["episode_steps"], repeated["episode_steps"]
        )
        for name in ("eval/episode_reward", "eval/avg_episode_length"):
            np.testing.assert_allclose(
                repeated["metrics"][name], repeats[0]["metrics"][name],
                rtol=1e-6, atol=2e-5, err_msg=name,
            )
        # Auxiliary diagnostics, especially standard deviations, can amplify
        # GPU roundoff. Report their differences instead of claiming bitwise
        # trajectory equivalence from this performance measurement.
        differences = {
            name: abs(repeated["metrics"][name] - value)
            for name, value in repeats[0]["metrics"].items()
            if name.startswith("eval/episode_")
        }
        largest = max(differences, key=differences.get)
        print(f"Largest auxiliary metric delta: {largest}={differences[largest]:.8g}")


if __name__ == "__main__":
    main()
