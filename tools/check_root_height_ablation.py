"""Measure final-pass floor-height smoothing without changing training behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import jax
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import PROJECT_ROOT, EnvConfig
from tools.check_recovery_reference import reference_digest
from tools.check_reference_collisions import inspect_collision_geometry
from tools.check_reference_contacts import inspect_contacts, sample_reference_states
from tools.check_reference_curves import inspect_consistency
from tools.check_retarget_stages import capture_retarget_stages
from tools.check_root_lock_transitions import trace_lock_call
from tools.recovery_dynamics import inspect_dynamics
from tools.recovery_replay import replay_reference


def smooth_height_raise(heights: np.ndarray, epsilon: float) -> float:
    """Return the smooth-maximum bias of two required floor-clearance heights.

    Only their difference matters, so the physical foot heights can be supplied
    directly. This raises the hard maximum by at most epsilon/2 BEFORE the
    historical height limiter and curve closure, which can change clearance.
    """
    heights = np.asarray(heights, dtype=float)
    if heights.shape != (2,) or not np.isfinite(heights).all():
        raise ValueError("Smoothing requires exactly two finite foot heights.")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Smoothing epsilon must be finite and nonnegative.")
    difference = abs(float(heights[0] - heights[1]))
    return float((np.hypot(difference, epsilon) - difference) / 2)


@contextmanager
def final_height_ablation(epsilon: float) -> Iterator[None]:
    """Modify only the final floor-height envelope during environment creation.

    The real aligner still applies its minimum root height, speed limiter, XY
    lock and velocity calculation. Real foot heights are used inside the lock
    and everywhere outside final alignment. No method stays patched afterward.
    This process-local diagnostic is not safe for concurrent constructors.
    """
    smooth_height_raise(np.zeros(2), epsilon)
    original_align = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor
    original_height = BiomechanicsJoystickEnv._geom_min_z
    original_lock = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet
    states: dict[int, dict[str, Any]] = {}

    def align(env, *args, **kwargs):
        state = states.setdefault(
            id(env), {"passes": 0, "active": -1, "in_lock": False}
        )
        if state["passes"] >= 2 or state["active"] != -1:
            raise ValueError("Ablation expects two non-nested alignment passes.")
        state["active"] = state["passes"]
        state["passes"] += 1
        try:
            return original_align(env, *args, **kwargs)
        finally:
            state["active"] = -1

    def height(env, data, geom_id):
        real_height = original_height(env, data, geom_id)
        state = states.get(id(env))
        if (
            epsilon == 0
            or state is None
            or state["active"] != 1
            or state["in_lock"]
            or geom_id not in env._foot_geom_ids_np
        ):
            return real_height
        heights = np.array(
            [original_height(env, data, int(foot)) for foot in env._foot_geom_ids_np],
            dtype=np.float32,
        )
        return real_height - smooth_height_raise(heights, epsilon)

    def lock(env, *args, **kwargs):
        state = states.get(id(env))
        if state is None or state["active"] == -1:
            return original_lock(env, *args, **kwargs)
        was_in_lock = state["in_lock"]
        state["in_lock"] = True
        try:
            return original_lock(env, *args, **kwargs)
        finally:
            state["in_lock"] = was_in_lock

    with (
        patch.object(
            BiomechanicsJoystickEnv, "_align_reference_roots_to_sim_floor", align
        ),
        patch.object(BiomechanicsJoystickEnv, "_geom_min_z", height),
        patch.object(
            BiomechanicsJoystickEnv, "_lock_clip_root_xy_to_stance_feet", lock
        ),
    ):
        yield


def summarize_clearance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report geometric flight, not force-based contact or balance feasibility."""
    samples: dict[int, list[float]] = {}
    for row in rows:
        samples.setdefault(row["sample"], []).append(row["surface_height_m"])
    if not samples or any(len(heights) != 2 for heights in samples.values()):
        raise ValueError("Clearance audit requires two feet per nonempty sample.")
    heights = np.array(list(samples.values()), dtype=float)
    if not np.isfinite(heights).all():
        raise ValueError("Clearance samples must be finite.")
    lowest = heights.min(axis=1)
    return {
        "samples": len(lowest),
        "maximum_lowest_sole_height_m": float(lowest.max()),
        "both_soles_above_floor_fraction": float(np.mean(lowest > 0)),
        "both_soles_above_0p1mm_fraction": float(np.mean(lowest > 1e-4)),
    }


def measure_variant(
    config: EnvConfig, epsilon: float, run_replay: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild complete curves and measure real geometry after patches restore."""
    original_lock = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet
    calls = []

    def observed(env, joints, roots, quaternions, count, **kwargs):
        result = original_lock(env, joints, roots, quaternions, count, **kwargs)
        calls.append(
            trace_lock_call(
                env,
                joints,
                roots,
                quaternions,
                count,
                result,
                float(env._bvh_reference_frame_times[0]),
                contact_history=bool(kwargs.get("contact_history", False)),
            )
        )
        return result

    with (
        patch.object(
            BiomechanicsJoystickEnv, "_lock_clip_root_xy_to_stance_feet", observed
        ),
        final_height_ablation(epsilon),
    ):
        env, metadata, stages = capture_retarget_stages(config)
    result: dict[str, Any] = {
        "experimental_override": {"final_height_envelope_epsilon_m": epsilon},
        "root_lock_passes": metadata["root_lock_passes"],
        "support_traces": calls,
    }
    if "curve_preparation_error" in metadata:
        result["preparation_rejected"] = metadata["curve_preparation_error"]
        return result, stages
    count = int(env._bvh_reference_frame_counts[0])
    interval = float(env._bvh_reference_frame_times[0])
    duration = float(env._bvh_reference_motion_lengths[0])
    times = np.linspace(0, duration, (count - 1) * 4 + 1)
    qpos, qvel, _ = sample_reference_states(env, 0, times)
    contacts, rows = inspect_contacts(env, times, qpos, qvel, 4)
    off_times = (np.arange((count - 1) * 4) + 0.5) * interval / 4
    states = sample_reference_states(env, 0, off_times)
    _, off_rows = inspect_contacts(env, off_times, *states[:2], 4)
    peak = int(np.argmax(np.linalg.norm(states[2][:, :3], axis=1)))
    result.update(
        reference_digest=reference_digest(env),
        closure=env._reference_closure_diagnostics,
        ik_warmstart=env._reference_ik_warmstart_diagnostics,
        numerical_curve_checks=inspect_consistency(env, 0),
        contacts=contacts,
        clearance=summarize_clearance(rows),
        off_knot_clearance=summarize_clearance(off_rows),
        off_knot_maximum_penetration_m=max(
            0.0, -min(row["surface_height_m"] for row in off_rows)
        ),
        collisions=inspect_collision_geometry(env._mj_model, qpos, qvel),
        off_knot_collisions=inspect_collision_geometry(env._mj_model, *states[:2]),
        dynamics=inspect_dynamics(env._mj_model, *states),
        root_acceleration_peak={
            "fractional_frame": float(off_times[peak] / interval),
            "xyz_m_s2": states[2][peak, :3].tolist(),
        },
        sampled_root_z_speed_abs_max_m_s=float(
            max(np.max(np.abs(qvel[:, 2])), np.max(np.abs(states[1][:, 2])))
        ),
    )
    if run_replay:
        result["replays"] = {
            "physics_rate": replay_reference(env, 0),
            "control_rate_current": replay_reference(env, 0, control_dt=env.dt),
            "control_rate_lookahead": replay_reference(
                env, 0, control_dt=env.dt, target_lookahead_seconds=env.dt
            ),
        }
    return result, stages


def main() -> None:
    """Compare unchanged v6 with two isolated final-height smoothing candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon-mm", nargs="+", type=float, default=[0, 5, 10])
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase8_20261013/height_ablation.json",
    )
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < 0 for value in args.epsilon_mm):
        parser.error("--epsilon-mm values must be finite and nonnegative.")
    if args.epsilon_mm[0] != 0 or len(set(args.epsilon_mm)) != len(args.epsilon_mm):
        parser.error("Start with the zero baseline and do not repeat epsilon values.")
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    config = EnvConfig(
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=str(
            PROJECT_ROOT
            / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml"
        ),
        reference_gait="bvh",
        reference_gait_file=[
            str(
                PROJECT_ROOT / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
            )
        ],
        reference_clip_ids=[1],
        reference_forced_clip_id=0,
        recovery_physics_version="consistent_v6",
        reference_loop_closure="bounded",
        deepmimic_reward_mode="pure",
    )
    sources = (
        "biomechanics_env.py",
        "config.py",
        "evaluate.py",
        "bvh_reference.py",
        "reference_ik.py",
        "reference_root.py",
        "reference_interpolation.py",
        "tools/check_root_height_ablation.py",
        "tools/check_root_lock_transitions.py",
        "tools/check_retarget_stages.py",
        "tools/check_reference_contacts.py",
        "tools/check_reference_curves.py",
        "tools/check_reference_collisions.py",
        "tools/check_recovery_reference.py",
        "tools/recovery_dynamics.py",
        "tools/recovery_replay.py",
    )
    report = {
        "scope": "Construction-only diagnostic; not a saved training configuration.",
        "completed": False,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": asdict(config),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        "input_sha256": {
            str(Path(name).relative_to(PROJECT_ROOT)): hashlib.sha256(
                Path(name).read_bytes()
            ).hexdigest()
            for name in (config.xml_path, config.reference_gait_file[0])
        },
        "limitations": [
            "Smoothing precedes the unchanged Z speed limiter, XY lock and closure.",
            "Root lifting can change support labels and contact forces.",
            "Sampled clearance is not a continuous-time nonpenetration guarantee.",
            "Geometric floor proximity is not force-based support.",
            "Exact inverse dynamics uses existing soft contacts, "
            "not contact allocation.",
            "Native open-loop replay is not a learned feedback controller.",
            "Never resume a policy with these unversioned experimental overrides.",
        ],
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_stages = {}
    for epsilon_mm in args.epsilon_mm:
        name = f"epsilon_{epsilon_mm:g}mm"
        print(f"Measuring {name}...", flush=True)
        result, stages = measure_variant(
            config, epsilon_mm / 1000, not args.skip_replay
        )
        if epsilon_mm == 0 and "preparation_rejected" in result:
            report["variants"][name] = result
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            raise ValueError("The control reference was rejected; stop the ablation.")
        if epsilon_mm == 0:
            baseline_stages = {stage["name"]: stage for stage in stages}
        result["stage_differences_from_baseline"] = {
            stage["name"]: {
                field: float(
                    np.max(np.abs(stage[field] - baseline_stages[stage["name"]][field]))
                )
                for field in ("joints", "roots", "quaternions", "targets")
            }
            for stage in stages
        }
        report["variants"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if "dynamics" in result:
            root = result["dynamics"]["free_joint_residuals"][0]
            print(
                json.dumps(
                    {
                        "variant": name,
                        "root_acceleration_max_m_s2": root[
                            "linear_acceleration_norm_m_s2"
                        ]["max"],
                        "root_force_max_N": root["force_norm_N"]["max"],
                        "penetration_m": result["contacts"]["maximum_penetration_m"],
                        "clearance": result["clearance"],
                    }
                ),
                flush=True,
            )
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
