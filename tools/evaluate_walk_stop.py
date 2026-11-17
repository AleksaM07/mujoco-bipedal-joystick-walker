"""Evaluate walk-to-stop checkpoints with a deterministic physical rollout."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
from brax.training import checkpoint as brax_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (
    choose_device,
    file_sha256,
    find_run_config,
    find_run_dir,
    infer_xml_path,
    make_environment,
    restore_saved_env_config,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
)
from skills.stand_metrics import required_dwell_steps
from skills.walk_stop import WalkStopEnv
from tools.evaluate_stand import make_rollout_step

GATE_STEPS = 500
DEFAULT_SEEDS = (11, 22, 33, 44)


def load_walk_stop_config(checkpoint: Path, walking_baseline: bool) -> Any:
    """Restore the saved walker settings and opt into only the new skill logic."""
    saved = find_run_config(checkpoint)
    if saved is None:
        raise ValueError("Checkpoint needs a parent config.json containing 'env'.")
    skill = saved["env"].get("locomotion_skill", "walk")
    if skill != "walk_stop" and not walking_baseline:
        raise ValueError("A walking checkpoint requires --walking-baseline.")
    if saved["env"].get("physics_backend", "mjx_jax") != "mjx_jax":
        raise ValueError("The walk-stop gate does not migrate physics backends.")
    run_dir = find_run_dir(checkpoint)
    raw_xml = saved["env"].get("xml_path")
    local_xml = run_dir / raw_xml if run_dir is not None and raw_xml else None
    xml_path = (
        str(local_xml.resolve())
        if local_xml is not None and local_xml.is_file()
        else infer_xml_path(checkpoint, saved)
    )
    if xml_path is None:
        raise ValueError("Cannot resolve the checkpoint's saved XML.")
    return restore_saved_env_config(
        saved,
        locomotion_skill="walk_stop",
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=xml_path,
    )


def rollout(
    env: WalkStopEnv,
    checkpoint: Path,
    seed: int,
    reset: Any,
    advance: Any,
) -> dict[str, Any]:
    """Run one episode without autoreset and judge the final stand state."""
    params = brax_checkpoint.load(str(checkpoint.resolve()))
    key = jax.random.PRNGKey(seed)
    state = reset(key)
    validate_observation_compatibility(checkpoint, state.obs)
    samples = []
    for _ in range(GATE_STEPS):
        key, action_key = jax.random.split(key)
        state, sample = advance(state, action_key, params)
        host = {name: float(value) for name, value in jax.device_get(sample).items()}
        host["alpha"] = float(state.metrics["transition_alpha"])
        samples.append(host)
        if host["physical_failure"] or host["done"] or not host["finite"]:
            break
    final = samples[-1]
    stop_samples = [sample for sample in samples if sample["alpha"] >= 1.0]
    required = required_dwell_steps(env.dt, env.thresholds)
    passed = bool(
        len(samples) == GATE_STEPS
        and not any(sample["physical_failure"] or sample["done"] for sample in samples)
        and stop_samples
        and final["success"] == 1.0
        and final["dwell_steps"] >= required
    )
    return {
        "seed": seed,
        "steps": len(samples),
        "transition_completed": bool(stop_samples),
        "transition_stop_step": int(state.info["transition_stop_step"]),
        "stand_steps": len(stop_samples),
        "final_success": bool(final["success"]),
        "final_dwell_seconds": final["dwell_steps"] * env.dt,
        "final_drift_m": final["distance"],
        "final_speed_m_s": final["speed"],
        "max_stand_drift_m": max(
            (sample["distance"] for sample in stop_samples), default=None
        ),
        "max_stand_speed_m_s": max(
            (sample["speed"] for sample in stop_samples), default=None
        ),
        "physical_failure": any(sample["physical_failure"] for sample in samples),
        "env_done": any(sample["done"] for sample in samples),
        "passed": passed,
    }


def evaluate(
    checkpoint: Path,
    output: Path,
    *,
    walking_baseline: bool,
    device: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Evaluate one checkpoint and save a machine-readable gate report."""
    checkpoint = checkpoint.resolve()
    config = load_walk_stop_config(checkpoint, walking_baseline)
    with jax.default_device(choose_device(device)):
        env = make_environment(config)
        if not isinstance(env, WalkStopEnv):
            raise TypeError("Expected WalkStopEnv after config restoration.")
        validate_xml_manifest(checkpoint, env)
        validate_action_compatibility(checkpoint, env.action_size)
        reset = jax.jit(env.reset)
        advance = make_rollout_step(env, checkpoint)
        rollouts = [
            rollout(env, checkpoint, seed, reset, advance) for seed in seeds
        ]
    report = {
        "checkpoint": str(checkpoint),
        "walking_baseline": walking_baseline,
        "environment": asdict(config),
        "xml_sha256": file_sha256(env.xml_path),
        "steps_per_seed": GATE_STEPS,
        "rollouts": rollouts,
        "passed": all(item["passed"] for item in rollouts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--walking-baseline", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    try:
        report = evaluate(
            args.checkpoint,
            args.output,
            walking_baseline=args.walking_baseline,
            device=args.device,
            seeds=tuple(args.seeds),
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"walk_stop gate passed={report['passed']} | {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
