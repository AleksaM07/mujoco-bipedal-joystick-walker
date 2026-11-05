"""Evaluate a stand checkpoint with unwrapped, deterministic physical rollouts."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from evaluate import (
    choose_device,
    file_sha256,
    find_run_config,
    find_run_dir,
    infer_xml_path,
    load_ppo_networks,
    make_environment,
    read_checkpoint_network_config,
    render_host_replay_frames,
    restore_saved_env_config,
    state_qpos_qvel_host,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
    write_video,
)
from skills.stand import StandEnv
from skills.stand_metrics import required_dwell_steps

DEFAULT_SEEDS = (11, 22, 33, 44)
GATE_STEPS = 500


def load_stand_config(checkpoint: Path, walking_baseline: bool = False) -> EnvConfig:
    """Restore saved settings; converting a walking actor requires an opt-in."""
    saved = find_run_config(checkpoint)
    if saved is None:
        raise ValueError("Checkpoint needs a parent config.json containing 'env'.")
    if saved["env"].get("locomotion_skill", "walk") != "stand" and not walking_baseline:
        raise ValueError("Walking checkpoint requires --walking-baseline.")
    if saved["env"].get("physics_backend", "mjx_jax") != "mjx_jax":
        raise ValueError(
            "Stand gate supports saved mjx_jax physics; backend migration is unsupported."
        )
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
        locomotion_skill="stand",
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=xml_path,
        reference_gait_file=(),
        reference_clip_ids=None,
        reference_forced_clip_id=0,
    )


def summarize_rollout(
    samples: list[dict[str, float]], *, seed: int, dt: float, dwell_steps: int
) -> dict[str, Any]:
    """Judge complete rollouts from instantaneous values, never episode sums."""
    if not samples:
        raise ValueError("A rollout must contain at least one physics step.")
    final = samples[-1]
    finite = all(
        bool(sample["finite"]) and all(np.isfinite(value) for value in sample.values())
        for sample in samples
    )
    terminated = any(bool(sample["physical_failure"]) for sample in samples)
    done = any(bool(sample["done"]) for sample in samples)
    success = bool(final["success"] == 1 and final["dwell_steps"] >= dwell_steps)

    def aggregate(name: str, operation: Any) -> float | None:
        values = np.asarray([sample[name] for sample in samples])
        return float(operation(values)) if np.isfinite(values).all() else None

    return {
        "seed": seed,
        "steps": len(samples),
        "requested_steps": GATE_STEPS,
        "seconds": len(samples) * dt,
        "finite": finite,
        "terminated": terminated,
        "env_done": done,
        "final_success": success,
        "final_dwell_seconds": (
            float(final["dwell_steps"] * dt)
            if np.isfinite(final["dwell_steps"])
            else None
        ),
        "max_dwell_seconds": (
            float(max(sample["dwell_steps"] for sample in samples) * dt)
            if all(np.isfinite(sample["dwell_steps"]) for sample in samples)
            else None
        ),
        "final_drift": float(final["distance"])
        if np.isfinite(final["distance"])
        else None,
        "max_drift": aggregate("distance", np.max),
        "max_speed": aggregate("speed", np.max),
        "mean_speed": aggregate("speed", np.mean),
        "min_torso_up": aggregate("torso_up", np.min),
        "min_height": aggregate("height", np.min),
        "min_height_ratio": aggregate("height_ratio", np.min),
        "eligible_fraction": aggregate("eligible", np.mean),
        "reward": aggregate("reward", np.sum),
        "passed": bool(
            len(samples) == GATE_STEPS
            and finite
            and not terminated
            and not done
            and success
        ),
    }


def make_rollout_step(env: StandEnv, checkpoint: Path) -> Any:
    """Compile physics and inference with dynamic parameters for checkpoint reuse."""
    make_policy = ppo_networks.make_inference_fn(load_ppo_networks(checkpoint))

    @jax.jit
    def advance(state, key, params):
        policy = make_policy(params, deterministic=True)
        action, _ = policy(state.obs, key)
        state = env.step(state, action)
        low, tipped, invalid = env._get_done_reasons(state.data)
        finite = (
            jp.all(jp.isfinite(state.data.qpos))
            & jp.all(jp.isfinite(state.data.qvel))
            & jp.all(jp.isfinite(action))
            & jp.isfinite(state.reward)
        )
        for observation in jax.tree_util.tree_leaves(state.obs):
            finite = finite & jp.all(jp.isfinite(observation))
        sample = {
            "finite": finite,
            "physical_failure": low | tipped | invalid | ~finite,
            "done": state.done,
            "success": state.metrics["stand_success"],
            "dwell_steps": state.info["stand_dwell_steps"],
            "distance": state.metrics["stand_distance"],
            "speed": state.metrics["stand_speed"],
            "torso_up": env._torso_up(state.data),
            "height": state.data.qpos[2],
            "height_ratio": state.data.qpos[2] / env._init_q[2],
            "eligible": state.metrics["stand_eligible"],
            "reward": state.reward,
        }
        return state, sample

    return advance


def rollout_checkpoint(
    env: StandEnv,
    checkpoint: Path,
    seeds: tuple[int, ...],
    video_dir: Path | None,
    *,
    reset: Any,
    advance: Any,
) -> list[dict[str, Any]]:
    """Run exactly 500 steps per seed unless failure occurs; never auto-reset."""
    params = brax_checkpoint.load(str(checkpoint.resolve()))
    results = []
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        state = reset(key)
        validate_observation_compatibility(checkpoint, state.obs)
        samples = []
        snapshots = []
        print(
            f"Stand gate seed={seed}: compiling/running 500 physics steps.", flush=True
        )
        for _ in range(GATE_STEPS):
            key, action_key = jax.random.split(key)
            state, device_sample = advance(state, action_key, params)
            sample = {
                name: float(value)
                for name, value in jax.device_get(device_sample).items()
            }
            samples.append(sample)
            if video_dir is not None and sample["finite"]:
                snapshots.append(state_qpos_qvel_host(env.mj_model, state))
            if sample["physical_failure"] or sample["done"] or not sample["finite"]:
                break
        result = summarize_rollout(
            samples,
            seed=seed,
            dt=env.dt,
            dwell_steps=required_dwell_steps(env.dt, env.thresholds),
        )
        print(json.dumps(result, allow_nan=False), flush=True)
        if video_dir is not None and snapshots:
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"stand_seed_{seed}.mp4"
            frames = render_host_replay_frames(
                env.mj_model, snapshots, 960, 720, "quad"
            )
            write_video(video_path, frames, fps=round(1.0 / env.dt))
            result["video"] = str(video_path)
        results.append(result)
    return results


def validate_seeds(seeds: tuple[int, ...]) -> None:
    """Require the four fixed baseline seeds, optionally followed by extra seeds."""
    if len(set(seeds)) != len(seeds) or any(
        seed < 0 or seed >= 2**32 for seed in seeds
    ):
        raise ValueError("Seeds must be unique uint32 values.")
    if not set(DEFAULT_SEEDS).issubset(seeds):
        raise ValueError("The gate requires at least seeds 11 22 33 44.")


def validate_output_path(path: Path) -> None:
    """Keep generated gate evidence outside the frozen natural-walk archive."""
    archive = (PROJECT_ROOT / "prezentacija/5th_success_final_natural_walk").resolve()
    if path.resolve().is_relative_to(archive):
        raise ValueError(
            "Gate outputs must be outside the protected natural-walk archive."
        )


def evaluate_checkpoints(
    checkpoints: list[Path],
    output_dir: Path,
    *,
    device: str = "cpu",
    walking_baseline: bool = False,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    video_dir: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate checkpoints from one run, reusing the compiled environment/policy.

    Each checkpoint writes ``output_dir / (checkpoint.name + '.json')``. Different
    environment or network settings must be evaluated in separate calls.
    """
    validate_seeds(seeds)
    validate_output_path(output_dir)
    if output_path is not None:
        validate_output_path(output_path)
        if len(checkpoints) != 1:
            raise ValueError("A single output file requires exactly one checkpoint.")
    if video_dir is not None:
        validate_output_path(video_dir)
    if not checkpoints or len({path.name for path in checkpoints}) != len(checkpoints):
        raise ValueError("Provide checkpoints with distinct names from one run.")
    checkpoints = [path.resolve() for path in checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    config = load_stand_config(checkpoints[0], walking_baseline)
    network_config = read_checkpoint_network_config(checkpoints[0])
    for checkpoint in checkpoints[1:]:
        if load_stand_config(checkpoint, walking_baseline) != config:
            raise ValueError(
                "Checkpoint environment settings differ; evaluate separately."
            )
        if read_checkpoint_network_config(checkpoint) != network_config:
            raise ValueError("Checkpoint network settings differ; evaluate separately.")
    reports = []
    with jax.default_device(choose_device(device)):
        env = make_environment(config)
        reset = jax.jit(env.reset)
        advance = make_rollout_step(env, checkpoints[0])
        for checkpoint in checkpoints:
            validate_xml_manifest(checkpoint, env)
            validate_action_compatibility(checkpoint, env.action_size)
            results = rollout_checkpoint(
                env,
                checkpoint,
                seeds,
                video_dir / checkpoint.name if video_dir is not None else None,
                reset=reset,
                advance=advance,
            )
            report = {
                "checkpoint": str(checkpoint),
                "walking_baseline": walking_baseline,
                "deterministic_policy": True,
                "auto_reset": False,
                "environment": asdict(config),
                "xml_sha256": file_sha256(env.xml_path),
                "thresholds": asdict(env.thresholds),
                "control_dt": env.dt,
                "simulation_dt": float(env.mj_model.opt.timestep),
                "seeds": list(seeds),
                "requested_steps_per_seed": GATE_STEPS,
                "rollouts": results,
                "passed": all(result["passed"] for result in results),
                "limitation": (
                    "Neutral resets do not vary physical initial conditions across "
                    "seeds. These rollouts establish reproducibility, not "
                    "perturbation or transition robustness."
                ),
            }
            report_path = output_path or output_dir / f"{checkpoint.name}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            reports.append(report)
    return reports


def main() -> int:
    """Write a physical acceptance report and exit nonzero when any seed fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", type=Path)
    inputs.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--walking-baseline", action="store_true")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--output-dir", type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if args.checkpoints_dir is not None and args.output is not None:
        parser.error("--checkpoints-dir requires --output-dir")
    if args.checkpoints_dir is not None and not args.checkpoints_dir.is_dir():
        parser.error("--checkpoints-dir must name an existing directory")
    checkpoints = (
        [args.checkpoint]
        if args.checkpoint is not None
        else sorted(
            path
            for path in args.checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    )
    output_dir = args.output_dir if args.output_dir is not None else args.output.parent
    try:
        validate_seeds(seeds)
        if args.output is not None:
            validate_output_path(args.output)
        reports = evaluate_checkpoints(
            checkpoints,
            output_dir,
            device=args.device,
            walking_baseline=args.walking_baseline,
            seeds=seeds,
            video_dir=args.video_dir,
            output_path=args.output,
        )
    except ValueError as exc:
        parser.error(str(exc))
    passed = all(report["passed"] for report in reports)
    print(
        f"Stand gate all_checkpoints_passed={passed} | reports={output_dir}", flush=True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
