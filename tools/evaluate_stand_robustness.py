"""Validate frozen stand policies over 60 seconds and controlled torso pushes.

No training, auto-reset, reward changes or altered model physics occur here.
Forces are world-frame Cartesian forces at the thorax centre of mass; the
remaining xfrc_applied entries, including torques, are explicitly zero.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from brax.training import checkpoint as brax_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (
    choose_device,
    file_sha256,
    make_environment,
    read_checkpoint_network_config,
    state_qpos_qvel_host,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
)
from scripts.run_stand_ablation import write_report
from skills.stand import StandEnv
from skills.stand_metrics import required_dwell_steps
from tools.evaluate_stand import (
    load_stand_config,
    make_rollout_step,
    summarize_rollout,
    validate_output_path,
)

SEED = 11
RECOVERY_DEADLINE_SECONDS = 5.0


@dataclass(frozen=True)
class StandProbe:
    """A fixed physical probe with a half-open force application interval."""

    name: str
    seconds: float
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_weight_fraction: float = 0.0
    push_start_seconds: float = 5.0
    push_duration_seconds: float = 0.2


PROBES = (
    StandProbe("hold_60s", 60.0),
    StandProbe("push_x_pos", 20.0, (1.0, 0.0, 0.0), 0.05),
    StandProbe("push_x_neg", 20.0, (-1.0, 0.0, 0.0), 0.05),
    StandProbe("push_y_pos", 20.0, (0.0, 1.0, 0.0), 0.05),
    StandProbe("push_y_neg", 20.0, (0.0, -1.0, 0.0), 0.05),
)


def force_schedule(
    probe: StandProbe, *, mass: float, gravity: float, dt: float
) -> np.ndarray:
    """Return one world force vector per control step, never a velocity edit."""
    if not all(np.isfinite(value) and value > 0 for value in (mass, gravity, dt)):
        raise ValueError("Mass, gravity and dt must be finite and positive.")
    if not np.isfinite(probe.seconds) or probe.seconds < 10.0:
        raise ValueError("A probe must run for at least ten seconds.")
    if not np.isfinite(probe.body_weight_fraction) or probe.body_weight_fraction < 0:
        raise ValueError("Force fraction must be finite and nonnegative.")
    steps = round(probe.seconds / dt)
    start = round(probe.push_start_seconds / dt)
    duration = round(probe.push_duration_seconds / dt)
    forces = np.zeros((steps, 3), dtype=np.float32)
    if probe.body_weight_fraction:
        if not np.isfinite(probe.direction).all() or not np.isclose(
            np.linalg.norm(probe.direction), 1.0
        ):
            raise ValueError("Push direction must be a finite unit vector.")
        if start < 0 or duration < 1 or start + duration >= steps:
            raise ValueError("Push must fit inside the rollout, with recovery time.")
        forces[start : start + duration] = (
            np.asarray(probe.direction) * probe.body_weight_fraction * mass * gravity
        )
    return forces


def recovery_seconds(
    samples: list[dict[str, float]], *, release_step: int, dt: float, dwell_steps: int
) -> float | None:
    """Measure a fresh full eligible dwell after release, excluding prior stance."""
    consecutive = 0
    for elapsed, sample in enumerate(samples[release_step:], start=1):
        if not sample["finite"] or sample["physical_failure"] or sample["done"]:
            return None
        consecutive = consecutive + 1 if sample["eligible"] == 1 else 0
        if consecutive >= dwell_steps:
            return elapsed * dt
    return None


def summarize_probe(
    samples: list[dict[str, float]], probe: StandProbe, *, dt: float, dwell_steps: int
) -> dict[str, Any]:
    """Require continuous nominal stance or timely post-push recovery plus final hold."""
    result = summarize_rollout(
        samples,
        seed=SEED,
        dt=dt,
        dwell_steps=dwell_steps,
        requested_steps=round(probe.seconds / dt),
    )
    result["probe"] = asdict(probe)
    if probe.body_weight_fraction:
        release = round(probe.push_start_seconds / dt) + round(
            probe.push_duration_seconds / dt
        )
        recovered = recovery_seconds(
            samples, release_step=release, dt=dt, dwell_steps=dwell_steps
        )
        result["recovery_seconds"] = recovered
        result["passed"] = bool(
            result["passed"]
            and recovered is not None
            and recovered <= RECOVERY_DEADLINE_SECONDS
        )
    else:
        result["recovery_seconds"] = None
        result["passed"] = bool(
            result["passed"] and all(row["eligible"] == 1 for row in samples)
        )
    return result


def run_probe(
    env: StandEnv,
    checkpoint: Path,
    probe: StandProbe,
    output: Path,
    reset: Any,
    advance: Any,
    params: Any,
) -> dict[str, Any]:
    """Collect scalar traces and sparse replay states without spending time rendering."""
    key = jax.random.PRNGKey(SEED)
    state = reset(key)
    validate_observation_compatibility(checkpoint, state.obs)
    mass = float(np.sum(env.mj_model.body_mass))
    gravity = float(np.linalg.norm(env.mj_model.opt.gravity))
    forces = force_schedule(probe, mass=mass, gravity=gravity, dt=env.dt)
    body_id = env.mj_model.body("thorax").id
    zero = jp.zeros_like(state.data.xfrc_applied)
    force_vector = (
        jp.asarray(probe.direction) * probe.body_weight_fraction * mass * gravity
    )
    pushed = zero.at[body_id, :3].set(force_vector)
    samples = []
    snapshots = [state_qpos_qvel_host(env.mj_model, state)]
    snapshot_steps = [0]
    print(f"Probe {checkpoint.name}/{probe.name}: {len(forces)} steps", flush=True)
    for index, force in enumerate(forces):
        key, action_key = jax.random.split(key)
        state, device_sample = advance(
            state, action_key, params, pushed if np.any(force) else zero
        )
        sample = {
            name: float(value) for name, value in jax.device_get(device_sample).items()
        }
        samples.append(sample)
        if sample["finite"] and (
            (index + 1) % 5 == 0 or index == len(forces) - 1 or sample["done"]
        ):
            snapshots.append(state_qpos_qvel_host(env.mj_model, state))
            snapshot_steps.append(index + 1)
        if sample["physical_failure"] or sample["done"] or not sample["finite"]:
            break
    folder = output / checkpoint.name
    folder.mkdir(parents=True, exist_ok=True)
    trace_path = folder / f"{probe.name}.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step",
                "seconds",
                "force_x",
                "force_y",
                "force_z",
                *samples[0],
            ],
        )
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow(
                {
                    "step": index + 1,
                    "seconds": (index + 1) * env.dt,
                    "force_x": float(forces[index, 0]),
                    "force_y": float(forces[index, 1]),
                    "force_z": float(forces[index, 2]),
                    **sample,
                }
            )
    replay_path = folder / f"{probe.name}_replay.npz"
    np.savez_compressed(
        replay_path,
        qpos=np.stack([row[0] for row in snapshots]),
        qvel=np.stack([row[1] for row in snapshots]),
        steps=snapshot_steps,
        dt=env.dt,
    )
    result = summarize_probe(
        samples,
        probe,
        dt=env.dt,
        dwell_steps=required_dwell_steps(env.dt, env.thresholds),
    )
    result.update(
        checkpoint=str(checkpoint),
        trace=str(trace_path),
        replay=str(replay_path),
        body="thorax",
        mass_kg=mass,
        force_newtons=np.asarray(force_vector).tolist(),
        applied_impulse_ns=(forces[: len(samples)].sum(axis=0) * env.dt).tolist(),
    )
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def main() -> int:
    """Run the fixed probe suite on matching frozen checkpoints, serially on one GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=[probe.name for probe in PROBES],
        help="Optional diagnostic subset; only the full suite can certify robustness.",
    )
    parser.add_argument("--control-trace", action="store_true")
    args = parser.parse_args()
    names = args.probes or [probe.name for probe in PROBES]
    if len(set(names)) != len(names):
        parser.error("Probe names must be unique.")
    probes = [next(probe for probe in PROBES if probe.name == name) for name in names]
    full_suite = len(probes) == len(PROBES)
    checkpoints = [path.resolve() for path in args.checkpoints]
    output = args.output_dir.resolve()
    validate_output_path(output)
    if not output.is_relative_to((PROJECT_ROOT / "runs").resolve()) or output.exists():
        parser.error("Use a fresh output directory under runs/.")
    if len({path.name for path in checkpoints}) != len(checkpoints):
        parser.error("Checkpoint names must be distinct.")
    config = load_stand_config(checkpoints[0])
    network = read_checkpoint_network_config(checkpoints[0])
    for checkpoint in checkpoints[1:]:
        if (
            load_stand_config(checkpoint) != config
            or read_checkpoint_network_config(checkpoint) != network
        ):
            parser.error(
                "Checkpoints must use identical environment and network settings."
            )
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "protocol": "stand_robustness_v1",
        "full_suite": full_suite,
        "requested_probes": names,
        "control_trace": args.control_trace,
        "status": "running",
        "seed": SEED,
        "environment": asdict(config),
        "results": [],
        "recovery_deadline_seconds": RECOVERY_DEADLINE_SECONDS,
        "limitation": "One fixed neutral start per case; deterministic world-frame pushes are not broad stochastic robustness or a walk-to-stand transition.",
    }
    write_report(output / "report.json", report)
    with jax.default_device(choose_device(args.device)):
        env = make_environment(config)
        report.update(
            xml_sha256=file_sha256(env.xml_path),
            control_dt=env.dt,
            thresholds=asdict(env.thresholds),
        )
        reset = jax.jit(env.reset)
        advance = make_rollout_step(
            env, checkpoints[0], control_trace=args.control_trace
        )
        if args.control_trace:
            reference = np.asarray(env._init_q)[np.asarray(env._actuator_qpos_indices)]
            scales = np.asarray(env._action_scale) * config.reference_residual_scale
            limits = np.asarray(env.mj_model.actuator_ctrlrange)
            report["control_layout"] = [
                {
                    "joint": name,
                    "reference_radians": float(reference[index]),
                    "residual_half_range_radians": float(scales[index]),
                    "target_min_radians": float(
                        max(reference[index] - scales[index], limits[index, 0])
                    ),
                    "target_max_radians": float(
                        min(reference[index] + scales[index], limits[index, 1])
                    ),
                    "xml_ctrl_range": limits[index].tolist(),
                    "xml_force_range": env.mj_model.actuator_forcerange[index].tolist(),
                }
                for index, name in enumerate(env._actuator_joint_names)
            ]
        for checkpoint in checkpoints:
            validate_xml_manifest(checkpoint, env)
            validate_action_compatibility(checkpoint, env.action_size)
            params = brax_checkpoint.load(str(checkpoint))
            for probe in probes:
                report["results"].append(
                    run_probe(env, checkpoint, probe, output, reset, advance, params)
                )
                write_report(output / "report.json", report)
    cases_passed = all(row["passed"] for row in report["results"])
    report.update(
        status="complete",
        passed=full_suite and cases_passed,
        requested_cases_passed=cases_passed,
    )
    write_report(output / "report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
