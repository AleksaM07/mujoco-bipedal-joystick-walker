"""Measure retained walking and command response with fixed seeds and commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from evaluate import (
    find_run_config,
    load_ppo_policy,
    make_environment,
    restore_saved_env_config,
    set_command,
    validate_xml_manifest,
)


def measure_rollouts(env, policy, commands: list[list[float]], seeds: list[int],
                     steps: int = 500) -> list[dict]:
    """Return first-episode metrics; never count frames after a physical fall."""
    reset = jax.jit(env.reset)

    @jax.jit
    def rollout(state, rng, command):
        def advance(carry, unused):
            state, key, active = carry
            key, action_key = jax.random.split(key)
            state = set_command(state, command,
                                pure_imitation=env._is_pure_reference_imitation())
            action, _ = policy(state.obs, action_key)
            stepped = jax.lax.cond(active, lambda s: env.step(s, action), lambda s: s, state)
            measured = env._measured_command(stepped.data)
            errors = measured - command
            metrics = jp.concatenate((
                jp.array([active, stepped.reward, jp.linalg.norm(errors[:2]),
                          jp.abs(errors[2]), stepped.metrics["deepmimic_pose"]]),
                measured,
            ))
            metrics = jp.where(active, metrics, 0.0)
            return (stepped, key, active & ~stepped.done.astype(bool)), metrics

        final, values = jax.lax.scan(advance, (state, rng, jp.array(True)), None,
                                    length=steps)
        return jp.sum(values, axis=0), final[2]

    records = []
    for command in commands:
        for seed in seeds:
            print(f"probe | command={command} seed={seed} steps={steps}", flush=True)
            rng = jax.random.PRNGKey(seed)
            initial = reset(rng)
            totals, survived = jax.device_get(rollout(initial, rng, jp.array(command)))
            length = int(totals[0])
            record = {
                "seed": seed, "command": command, "length": length,
                "survived": bool(survived), "reward": float(totals[1]),
                "linear_mae": float(totals[2] / max(length, 1)),
                "yaw_mae": float(totals[3] / max(length, 1)),
                "pose": float(totals[4] / max(length, 1)),
                "measured": (totals[5:] / max(length, 1)).tolist(),
            }
            if not np.isfinite(totals).all():
                raise RuntimeError("Non-finite joystick probe results.")
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


def main() -> None:
    """Probe the frozen source or a command-safe transfer without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--command-mode", action="store_true")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    jax.config.update("jax_default_device", jax.devices(args.device)[0])
    config = restore_saved_env_config(find_run_config(args.checkpoint))
    config.physics_backend = "mjx_jax"
    config.playground_impl = "jax"
    if args.command_mode:
        config.deepmimic_reward_mode = "command"
        config.command_profile = "joystick_curriculum"
        config.joystick_speed_center = args.speed
    env = make_environment(config)
    validate_xml_manifest(args.checkpoint, env)
    policy = load_ppo_policy(args.checkpoint, deterministic=True)
    commands = [[args.speed, 0.0, 0.0]] if args.command_mode else [[0.0, 0.0, 0.0]]
    records = measure_rollouts(env, policy, commands, args.seeds, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
