"""Test a limit-preserving BVH knee bind offset on the complete v6 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, replace
from numbers import Real
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import jax
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import (
    BvhReferenceBatch,
    _parse_bvh,
    _resolve_loop_mode,
    _target_velocities,
)
from config import PROJECT_ROOT, EnvConfig
from tools import check_root_height_ablation as height_audit
from tools.check_retarget_stages import inspect_stage
from tools.check_stability_prior_attribution import (
    inspect_knee_mapping,
    inspect_pose_limits,
    nodal_second_difference,
)


def fit_constant_offset(
    raw: np.ndarray, preferred: float, lower: float, upper: float
) -> float:
    """Project only the constant bind offset; reject amplitudes that cannot fit."""
    raw = np.asarray(raw)
    if (
        raw.ndim != 1
        or not raw.size
        or raw.dtype.kind not in "fiu"
        or not np.isfinite(raw).all()
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in (preferred, lower, upper)
        )
        or lower > upper
    ):
        raise ValueError("Need finite real knee angles, offset and ordered bounds.")
    feasible_lower = float(lower) - float(raw.min())
    feasible_upper = float(upper) - float(raw.max())
    if feasible_lower > feasible_upper:
        raise ValueError(
            "Knee amplitude cannot fit the unchanged joint/control bounds."
        )
    return float(np.clip(preferred, feasible_lower, feasible_upper))


def modify_selected_knees(
    env: BiomechanicsJoystickEnv, references: BvhReferenceBatch
) -> tuple[BvhReferenceBatch, dict[str, Any]]:
    """Replace knee seeds after clip selection but before all pose preprocessing.

    The selected source cohort remains fixed. Recompute the original loop
    heuristic and reject a mode change instead of silently changing the problem.
    Non-knee poses, physical limits and original Cartesian targets are untouched.
    """
    if len(references.frame_counts) != 1:
        raise ValueError("This diagnostic requires exactly one selected source clip.")
    count = int(references.frame_counts[0])
    if count < 3:
        raise ValueError("Need at least three knee-seed frames.")
    metadata = {
        "source_start_frames": references.source_start_frames,
        "source_end_frames": references.source_end_frames,
    }
    mapping = inspect_knee_mapping(
        env,
        Path(references.source_paths[0]),
        metadata,
        references.qpos_targets[0, :count],
    )
    source_interval = _parse_bvh(Path(references.source_paths[0])).frame_time
    joints = np.array(references.qpos_targets, copy=True)
    velocities = np.array(references.qvel_targets, copy=True)
    record: dict[str, Any] = {"original_mapping": mapping, "knees": {}}
    for name, knee in mapping["knees"].items():
        index = env._actuator_joint_names.index(name)
        raw = np.array(knee["raw_hinge_rad"], dtype=float)
        lower = float(
            max(
                env._actuator_ctrl_lower_limits_np[index],
                env._actuator_qpos_lower_limits_np[index],
            )
        )
        upper = float(
            min(
                env._actuator_ctrl_upper_limits_np[index],
                env._actuator_qpos_upper_limits_np[index],
            )
        )
        preferred = knee["bind_offset_rad"]
        offset = fit_constant_offset(raw, preferred, lower, upper)
        # Leave already valid knees bit-identical, including their derivatives.
        if offset != preferred:
            corrected = (raw + offset).astype(np.float32)
            if np.any(corrected < lower - 1e-6) or np.any(corrected > upper + 1e-6):
                raise ValueError("Float32 knee output exceeds unchanged bounds.")
            joints[0, :count, index] = corrected
            joints[0, count:, index] = corrected[-1]
            velocities[0, :count, index] = _target_velocities(
                corrected, source_interval
            )
            velocities[0, count:, index] = 0
        record["knees"][name] = {
            "preferred_offset_rad": preferred,
            "fitted_offset_rad": offset,
            "feasible_offset_interval_rad": [
                lower - float(raw.min()),
                upper - float(raw.max()),
            ],
            "unchanged_bounds_rad": [lower, upper],
            "changed": offset != preferred,
            "new_seed_rad": joints[0, :count, index].tolist(),
            "new_seed_velocity_rad_s": velocities[0, :count, index].tolist(),
            "new_seed_second_difference_rad_s2": nodal_second_difference(
                joints[0, :count, index, None], float(references.frame_times[0])
            )[:, 0].tolist(),
        }
    new_mode = _resolve_loop_mode(
        qpos_targets=joints[0, :count],
        qvel_targets=velocities[0, :count],
        root_pos_targets=references.root_pos_targets[0, :count],
        root_quat_targets=references.root_quat_targets[0, :count],
        root_vel_targets=references.root_vel_targets[0, :count],
        root_angvel_targets=references.root_angvel_targets[0, :count],
    )
    record["original_loop_mode"] = int(references.loop_modes[0])
    record["recomputed_loop_mode"] = int(new_mode)
    if int(new_mode) != int(references.loop_modes[0]):
        raise ValueError("Knee correction changes the selected clip's loop heuristic.")
    return replace(references, qpos_targets=joints, qvel_targets=velocities), record


@contextmanager
def knee_offset_ablation(enabled: bool) -> Iterator[list[dict[str, Any]]]:
    """Scope the experimental seed change to a single-threaded constructor."""
    original = BiomechanicsJoystickEnv._filter_reference_clips
    records: list[dict[str, Any]] = []
    if not enabled:
        yield records
        return

    def filtered(env, references):
        selected = original(env, references)
        corrected, record = modify_selected_knees(env, selected)
        records.append(record)
        return corrected

    with patch.object(BiomechanicsJoystickEnv, "_filter_reference_clips", filtered):
        yield records


def measure_knee_variant(
    config: EnvConfig, enabled: bool, run_replay: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse the complete physics audit and add marker/seed-stage evidence."""
    original_capture = height_audit.capture_retarget_stages
    captured = []

    def observe(configuration):
        result = original_capture(configuration)
        captured.append(result)
        return result

    with (
        knee_offset_ablation(enabled) as records,
        patch.object(height_audit, "capture_retarget_stages", observe),
    ):
        result, stages = height_audit.measure_variant(config, 0, run_replay)
    if len(captured) != 1:
        raise ValueError("Expected exactly one captured environment per condition.")
    env, metadata, _ = captured[0]
    count = int(metadata["counts"][0])
    interval = float(env._bvh_reference_frame_times[0])
    index = env._actuator_joint_names.index("left_knee_z")
    result["knee_offset_records"] = records
    result["experimental_override"]["bounded_knee_offset"] = enabled
    result["left_knee_second_difference_frame114_rad_s2"] = {
        stage["name"]: float(
            nodal_second_difference(stage["joints"][0, :count], interval)[113, index]
        )
        for stage in stages
    }
    result["stage_geometry"] = {
        stage["name"]: {
            "markers_and_soles": inspect_stage(env, stage, metadata["names"], 0, count),
            "pose_limits": inspect_pose_limits(env, stage["joints"][0, :count]),
        }
        for stage in stages
        if stage["name"] in ("ik_solution", "prepared_curve_nodes")
    }
    return result, stages


def main() -> None:
    """Compare unchanged v6 and one offset-only candidate with the same diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase10_20261013/knee_offset.json",
    )
    args = parser.parse_args()
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
        "tools/check_knee_offset_ablation.py",
        "tools/check_root_height_ablation.py",
        "tools/check_stability_prior_attribution.py",
        "tools/check_retarget_stages.py",
        "tools/check_root_lock_transitions.py",
        "tools/check_reference_contacts.py",
        "tools/check_reference_curves.py",
        "tools/check_reference_collisions.py",
        "tools/check_recovery_reference.py",
        "tools/recovery_dynamics.py",
        "tools/recovery_replay.py",
    )
    report: dict[str, Any] = {
        "scope": "Construction-only knee-seed ablation; not a training configuration.",
        "completed": False,
        "config": asdict(config),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
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
            "Selected source cohort is fixed; no automatic source reranking.",
            "Changed seed enters before initial alignment; "
            "downstream geometry may change.",
            "No height smoothing, prior, force-limit or reward changes.",
            "Same curve gates, inverse soft contacts and unassisted open-loop replay.",
            "Sampled diagnostics do not certify continuous feasibility "
            "or RL performance.",
            "Experimental overrides are not serialized training behavior; "
            "never resume a policy with them.",
        ],
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_stages = {}
    for name, enabled in (("baseline", False), ("bounded_knee_offset", True)):
        print(f"Measuring {name}...", flush=True)
        result, stages = measure_knee_variant(config, enabled, not args.skip_replay)
        if not enabled:
            baseline_stages = {stage["name"]: stage for stage in stages}
        result["stage_differences_from_baseline"] = {
            stage["name"]: {
                field: float(
                    np.max(np.abs(stage[field] - baseline_stages[stage["name"]][field]))
                )
                for field in ("joints", "roots", "quaternions", "targets")
            }
            for stage in stages
            if stage["name"] in baseline_stages
        }
        report["variants"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if "preparation_rejected" in result:
            print(
                f"Curve preparation rejected: {result['preparation_rejected']}",
                flush=True,
            )
            if not enabled:
                raise ValueError("Control reference rejected; stop the comparison.")
        else:
            root = result["dynamics"]["free_joint_residuals"][0]
            print(
                json.dumps(
                    {
                        "variant": name,
                        "peak_root_acceleration_m_s2": root[
                            "linear_acceleration_norm_m_s2"
                        ]["max"],
                        "peak_root_force_N": root["force_norm_N"]["max"],
                        "penetration_m": result["contacts"]["maximum_penetration_m"],
                    }
                ),
                flush=True,
            )
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
