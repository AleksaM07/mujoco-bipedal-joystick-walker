"""Audit whether an arm-transfer policy keeps walking and bends its elbows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import fmean

import jax
import jax.numpy as jp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (  # noqa: E402
    find_run_config,
    load_ppo_policy,
    make_environment,
    restore_saved_env_config,
    set_command,
    validate_xml_manifest,
)


def evaluate_promotion_gate(records: list[dict[str, object]]) -> dict[str, object]:
    """Require retained locomotion and non-frozen, reference-following elbows."""
    if not records:
        raise ValueError("At least one rollout record is required.")

    checks = {
        "no_physical_termination": all(
            not bool(record["physical_termination"]) for record in records
        ),
        "full_length": all(int(record["length"]) == 500 for record in records),
        "reward": fmean(float(record["reward_per_step"]) for record in records)
        >= 2.60,
        "pose": min(float(record["deepmimic_pose"]) for record in records) >= 0.90,
        "arm_pose": min(float(record["deepmimic_arm_pose"]) for record in records)
        >= 0.70,
        "left_elbow_motion": min(
            float(record["left_elbow_amplitude_ratio"]) for record in records
        )
        >= 0.45,
        "right_elbow_motion": min(
            float(record["right_elbow_amplitude_ratio"]) for record in records
        )
        >= 0.45,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "steps": 500,
            "reward_per_step": 2.60,
            "deepmimic_pose": 0.90,
            "deepmimic_arm_pose": 0.70,
            "elbow_amplitude_ratio": 0.45,
        },
    }


def measure_elbow_rollouts(
    env,
    policy,
    seeds: list[int],
    steps: int = 500,
) -> list[dict[str, object]]:
    """Measure physical survival and simulated/reference elbow amplitudes."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    elbow_indices = jp.array(
        [
            env._actuator_joint_names.index("left_elbow_z"),
            env._actuator_joint_names.index("right_elbow_z"),
        ]
    )
    reset = jax.jit(env.reset)
    zero_command = jp.zeros(3)

    @jax.jit
    def rollout(initial_state, rng):
        def advance(carry, unused):
            state, key, active, physical_termination = carry
            key, action_key = jax.random.split(key)
            state = set_command(state, zero_command, pure_imitation=True)
            action, _ = policy(state.obs, action_key)
            stepped = jax.lax.cond(
                active,
                lambda current: env.step(current, action),
                lambda current: current,
                state,
            )
            reference = env._query_bvh_reference(stepped.info, 0)
            actual_elbows = stepped.data.qpos[env._actuator_qpos_indices][
                elbow_indices
            ]
            reference_elbows = reference["qpos"][elbow_indices]
            terminated = stepped.metrics["terminated"] > 0.5
            physical_termination = physical_termination | (active & terminated)
            values = jp.concatenate(
                (
                    jp.array(
                        [
                            active,
                            stepped.reward,
                            stepped.metrics["deepmimic_pose"],
                            stepped.metrics["deepmimic_arm_pose"],
                            stepped.metrics["deepmimic_arm_velocity"],
                        ]
                    ),
                    actual_elbows,
                    reference_elbows,
                )
            )
            values = jp.where(active, values, jp.nan)
            next_active = active & ~stepped.done.astype(bool)
            return (
                stepped,
                key,
                next_active,
                physical_termination,
            ), values

        final, values = jax.lax.scan(
            advance,
            (initial_state, rng, jp.array(True), jp.array(False)),
            None,
            length=steps,
        )
        return values, final[3]

    records: list[dict[str, object]] = []
    for seed in seeds:
        print(f"elbow probe | seed={seed} steps={steps}", flush=True)
        rng = jax.random.PRNGKey(seed)
        initial_state = reset(rng)
        values, physical_termination = jax.device_get(rollout(initial_state, rng))
        active_values = np.asarray(values)[np.isfinite(np.asarray(values)[:, 0])]
        if active_values.size == 0 or not np.isfinite(active_values).all():
            raise RuntimeError(f"Non-finite elbow audit for seed {seed}.")

        length = int(active_values.shape[0])
        actual_amplitude = np.ptp(active_values[:, 5:7], axis=0)
        reference_amplitude = np.ptp(active_values[:, 7:9], axis=0)
        amplitude_ratio = actual_amplitude / np.maximum(reference_amplitude, 1e-6)
        record: dict[str, object] = {
            "seed": seed,
            "length": length,
            "physical_termination": bool(physical_termination),
            "reward_per_step": float(np.mean(active_values[:, 1])),
            "deepmimic_pose": float(np.mean(active_values[:, 2])),
            "deepmimic_arm_pose": float(np.mean(active_values[:, 3])),
            "deepmimic_arm_velocity": float(np.mean(active_values[:, 4])),
            "left_elbow_range_deg": float(np.degrees(actual_amplitude[0])),
            "right_elbow_range_deg": float(np.degrees(actual_amplitude[1])),
            "reference_left_elbow_range_deg": float(
                np.degrees(reference_amplitude[0])
            ),
            "reference_right_elbow_range_deg": float(
                np.degrees(reference_amplitude[1])
            ),
            "left_elbow_amplitude_ratio": float(amplitude_ratio[0]),
            "right_elbow_amplitude_ratio": float(amplitude_ratio[1]),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in record.items()
            if key not in {"seed", "length", "physical_termination"}
        ):
            raise RuntimeError(f"Non-finite elbow audit summary for seed {seed}.")
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def main() -> None:
    """Audit one checkpoint and write a machine-readable promotion report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44])
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    jax.config.update("jax_default_device", jax.devices(args.device)[0])
    run_config = find_run_config(args.checkpoint)
    if run_config is None:
        raise FileNotFoundError(f"No config.json found for {args.checkpoint}")
    env_config = restore_saved_env_config(run_config)
    if env_config.deepmimic_reward_mode != "pure":
        raise ValueError("Elbow audit requires a pure-reference checkpoint.")
    env = make_environment(env_config)
    validate_xml_manifest(args.checkpoint, env)
    policy = load_ppo_policy(args.checkpoint, deterministic=True)
    records = measure_elbow_rollouts(env, policy, args.seeds, args.steps)
    report = {
        "checkpoint": str(args.checkpoint),
        "reference_elbow_flexion_alpha": env_config.reference_elbow_flexion_alpha,
        "records": records,
        "gate": evaluate_promotion_gate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
