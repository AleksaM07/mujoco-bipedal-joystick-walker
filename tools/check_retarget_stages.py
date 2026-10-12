"""Locate foot overlap through the real retargeting pipeline, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import BVH_IK_TARGET_MAP, _parse_bvh, _resolve_bvh_ik_joint_names
from config import PROJECT_ROOT, EnvConfig
from evaluate import make_environment
from reference_interpolation import prepare_reference_curves
from tools.check_recovery_reference import reference_digest
from tools.check_reference_collisions import inspect_collision_geometry
from tools.check_reference_contacts import sample_reference_states


class CurvePreparationRejected(ValueError):
    """The observed production curve gate rejected this diagnostic reference."""


def capture_retarget_stages(
    config: EnvConfig,
) -> tuple[BiomechanicsJoystickEnv, dict[str, Any], list[dict[str, Any]]]:
    """Observe real method inputs/outputs; return copies, never mutate the pipeline."""
    metadata: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    environments: list[BiomechanicsJoystickEnv] = []
    originals = {
        name: getattr(BiomechanicsJoystickEnv, name)
        for name in (
            "_filter_reference_clips",
            "_align_reference_roots_to_sim_floor",
            "_retarget_reference_qpos_to_ik_targets",
            "_apply_reference_stability_prior",
            "_lock_clip_root_xy_to_stance_feet",
        )
    }

    def record(name, joints, roots, quaternions, targets):
        stages.append(
            {
                "name": name,
                "joints": np.array(joints, copy=True),
                "roots": np.array(roots, copy=True),
                "quaternions": np.array(quaternions, copy=True),
                "targets": np.array(targets, copy=True),
            }
        )

    def filtered(env, references):
        environments.append(env)
        result = originals["_filter_reference_clips"](env, references)
        if result.ik_target_positions is None:
            raise ValueError("The stage audit requires Cartesian IK targets.")
        metadata.update(
            names=result.ik_target_names,
            counts=np.array(result.frame_counts, copy=True),
            source_start_frames=np.array(result.source_start_frames, copy=True),
            source_end_frames=np.array(result.source_end_frames, copy=True),
            source_clip_ids=np.array(
                getattr(
                    env,
                    "_bvh_reference_source_clip_ids_np",
                    np.arange(len(result.frame_counts)),
                ),
                copy=True,
            ),
        )
        record(
            "raw_mapped_pose",
            result.qpos_targets,
            result.root_pos_targets,
            result.root_quat_targets,
            result.ik_target_positions,
        )
        return result

    def aligned(env, joints, roots, quaternions, *args):
        result = originals["_align_reference_roots_to_sim_floor"](
            env, joints, roots, quaternions, *args
        )
        previous = stages[-1]
        shifted_targets = (
            previous["targets"] + (result[0] - previous["roots"])[:, :, None, :]
        )
        record(
            "initial_floor_alignment" if len(stages) == 1 else "final_floor_alignment",
            joints,
            result[0],
            quaternions,
            shifted_targets,
        )
        return result

    def fitted(env, joints, targets, names, roots, quaternions, counts):
        record("normalized_ik_input", joints, roots, quaternions, targets)
        result = originals["_retarget_reference_qpos_to_ik_targets"](
            env, joints, targets, names, roots, quaternions, counts
        )
        record("ik_solution", result, roots, quaternions, targets)
        return result

    def prior(env, joints):
        result = originals["_apply_reference_stability_prior"](env, joints)
        previous = stages[-1]
        record(
            "stability_prior",
            result,
            previous["roots"],
            previous["quaternions"],
            previous["targets"],
        )
        return result

    def locked(env, joints, roots, quaternions, count):
        result = originals["_lock_clip_root_xy_to_stance_feet"](
            env, joints, roots, quaternions, count
        )
        records = metadata.setdefault("root_lock_passes", [])
        clip = len(records) % len(metadata["counts"])
        interval = float(env._bvh_reference_frame_times[clip])
        records.append(
            {
                "pass_index": len(records) // len(metadata["counts"]),
                "runtime_clip_id": clip,
                "source_clip_id": int(metadata["source_clip_ids"][clip]),
                "before": inspect_root_node_motion(roots[:count], interval),
                "after": inspect_root_node_motion(result[:count], interval),
            }
        )
        return result

    def prepared(*args, **kwargs):
        metadata["per_clip_curve_gates"] = []
        for clip in range(len(args[0])):
            try:
                curves = prepare_reference_curves(
                    *(np.asarray(array)[clip : clip + 1] for array in args), **kwargs
                )
            except ValueError as exc:
                gate = {"clip_id": clip, "accepted": False, "error": str(exc)}
            else:
                gate = {
                    "clip_id": clip,
                    "accepted": True,
                    "closure_diagnostics": curves["closure_diagnostics"],
                }
            metadata["per_clip_curve_gates"].append(gate)
        try:
            return prepare_reference_curves(*args, **kwargs)
        except ValueError as exc:
            metadata["curve_preparation_error"] = str(exc)
            raise CurvePreparationRejected(str(exc)) from exc

    with ExitStack() as stack:
        for name, wrapper in (
            ("_filter_reference_clips", filtered),
            ("_align_reference_roots_to_sim_floor", aligned),
            ("_retarget_reference_qpos_to_ik_targets", fitted),
            ("_apply_reference_stability_prior", prior),
            ("_lock_clip_root_xy_to_stance_feet", locked),
        ):
            stack.enter_context(patch.object(BiomechanicsJoystickEnv, name, wrapper))
        stack.enter_context(
            patch("biomechanics_env.prepare_reference_curves", prepared)
        )
        try:
            env = make_environment(config)
        except CurvePreparationRejected:
            # Only the known curve gate is captured. Missing source/host markers
            # and unrelated constructor errors still fail normally. This partial
            # object is usable for host FK only, never for stepping or querying.
            return environments[0], metadata, stages
    previous = stages[-1]
    roots = env._bvh_reference_sim_root_pos_targets_np
    record(
        "prepared_curve_nodes",
        env._bvh_reference_qpos_targets,
        roots,
        env._bvh_reference_sim_root_quat_targets_np,
        previous["targets"] + (roots - previous["roots"])[:, :, None, :],
    )
    return env, metadata, stages


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize a nonempty finite series in its original units."""
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Measurements must be finite and nonempty.")
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def inspect_root_node_motion(positions: np.ndarray, interval: float) -> dict[str, Any]:
    """Report nodal XY roughness, not the interpolant's exact acceleration.

    Second differences expose abrupt root translations already present before
    curve preparation. They are diagnostics, not continuous-curve derivatives.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 3:
        raise ValueError("Root node audit requires at least three XYZ positions.")
    if not np.isfinite(positions).all() or not np.isfinite(interval) or interval <= 0:
        raise ValueError("Root positions and a positive interval must be finite.")
    velocities = np.diff(positions[:, :2], axis=0) / interval
    acceleration = np.diff(velocities, axis=0) / interval
    norms = np.linalg.norm(acceleration, axis=1)
    worst_frame = int(norms.argmax()) + 1
    start = max(0, worst_frame - 2)
    end = min(len(positions), worst_frame + 3)
    return {
        "frames": len(positions),
        "interval_seconds": interval,
        "xy_nodal_speed_m_s": scalar_summary(np.linalg.norm(velocities, axis=1)),
        "xy_second_difference_m_s2": scalar_summary(norms),
        "worst_center_frame": worst_frame,
        "worst_window_start_frame": start,
        "worst_window_root_xyz_m": positions[start:end].tolist(),
    }


def inspect_stage(
    env: BiomechanicsJoystickEnv,
    stage: dict[str, Any],
    names: tuple[str, ...],
    clip_id: int,
    count: int,
) -> dict[str, Any]:
    """Measure actual sole collisions and exact named-marker Cartesian residuals."""
    model = env._mj_model
    data = mujoco.MjData(model)
    left = model.geom("left_foot_sole").id
    right = model.geom("right_foot_sole").id
    markers = env._resolve_host_ik_markers(names)
    if tuple(marker["name"] for marker in markers) != tuple(names):
        raise ValueError("Cannot audit unresolved or re-ordered IK marker names.")
    data.qpos[:] = env._init_q_np
    mujoco.mj_forward(model, data)
    lateral = (data.geom_xpos[left] - data.geom_xpos[right]).copy()
    lateral[2] = 0.0
    lateral /= np.linalg.norm(lateral)
    centers, positions, depths = [], [], []
    for frame in range(count):
        data.qpos[:] = env._init_q_np
        data.qpos[:3] = stage["roots"][clip_id, frame]
        data.qpos[3:7] = stage["quaternions"][clip_id, frame]
        data.qpos[env._actuator_qpos_indices_np] = stage["joints"][clip_id, frame]
        mujoco.mj_forward(model, data)
        centers.append((data.geom_xpos[left] - data.geom_xpos[right]).copy())
        positions.append(
            [env._host_marker_position(data, marker).copy() for marker in markers]
        )
        depths.append(
            min(
                (
                    float(contact.dist)
                    for contact in data.contact
                    if {int(contact.geom1), int(contact.geom2)} == {left, right}
                ),
                default=0.0,
            )
        )
    centers = np.asarray(centers)
    positions = np.asarray(positions)
    depths = np.asarray(depths)
    targets = stage["targets"][clip_id, :count]
    left_target = names.index("metatarsal_midpoint_left")
    right_target = names.index("metatarsal_midpoint_right")
    target_delta = targets[:, left_target] - targets[:, right_target]
    errors = positions - targets
    return {
        "stage": stage["name"],
        "frames": count,
        "root_node_motion": inspect_root_node_motion(
            stage["roots"][clip_id, :count],
            float(env._bvh_reference_frame_times[clip_id]),
        ),
        "net_root_translation_m": (
            stage["roots"][clip_id, count - 1] - stage["roots"][clip_id, 0]
        ).tolist(),
        "penetrating_frames": int(np.count_nonzero(depths < -1e-6)),
        "sole_contact_min_distance_m": float(depths.min()),
        "worst_frame": int(depths.argmin()),
        "sole_center_left_minus_right_world_xyz_m": {
            axis: scalar_summary(centers[:, i]) for i, axis in enumerate("xyz")
        },
        "standing_lateral_axis_world": lateral.tolist(),
        "sole_signed_lateral_spacing_m": scalar_summary(centers @ lateral),
        "target_signed_lateral_spacing_m": scalar_summary(target_delta @ lateral),
        "target_pair_distance_m": scalar_summary(np.linalg.norm(target_delta, axis=1)),
        "marker_cartesian_residuals": {
            name: {
                "distance_m": scalar_summary(np.linalg.norm(errors[:, i], axis=1)),
                "xyz_rmse_m": np.sqrt(np.mean(errors[:, i] ** 2, axis=0)).tolist(),
            }
            for i, name in enumerate(names)
        },
        "per_frame_penetration_m": depths.tolist(),
        "per_frame_sole_lateral_spacing_m": (centers @ lateral).tolist(),
    }


def main() -> None:
    """Save stage evidence for the fixed v21 / CMU02_01 experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-clip-ids",
        nargs="+",
        type=int,
        help="Loader clip indices to retain before preparation.",
    )
    parser.add_argument(
        "--version",
        default="consistent_v3",
        choices=["consistent_v2", "consistent_v3", "consistent_v4"],
    )
    parser.add_argument(
        "--loop-mode",
        choices=["auto", "wrap", "clamp"],
        default="auto",
        help="CLAMP is a finite-clip diagnostic, not certification of a WRAP loop.",
    )
    parser.add_argument(
        "--ik-fit",
        action="store_true",
        help="Run a selected-frame solver ablation without changing references.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Also inspect off-knot collisions if initialization succeeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs/recovery_phase4_20261012/stages.json",
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
        reference_forced_clip_id=0 if args.reference_clip_ids is not None else 1,
        reference_clip_ids=args.reference_clip_ids,
        deepmimic_reward_mode="pure",
        recovery_physics_version=args.version,
        reference_loop_closure="bounded",
        reference_loop_mode=args.loop_mode,
    )
    env, metadata, stages = capture_retarget_stages(config)
    summaries = []
    for clip, count in enumerate(metadata["counts"]):
        measured = [
            inspect_stage(env, stage, metadata["names"], clip, int(count))
            for stage in stages
        ]
        clip_summary = {
            "clip_id": clip,
            "source_clip_id": int(metadata["source_clip_ids"][clip]),
            "source_start_frame": int(metadata["source_start_frames"][clip]),
            "source_end_frame": int(metadata["source_end_frames"][clip]),
            "stages": measured,
        }
        if args.dense and "curve_preparation_error" not in metadata:
            interval = float(env._bvh_reference_frame_times[clip])
            times = (np.arange((count - 1) * 4) + 0.5) * interval / 4
            qpos, qvel, _ = sample_reference_states(env, clip, times)
            clip_summary["off_knot_collisions"] = inspect_collision_geometry(
                env._mj_model, qpos, qvel
            )
        summaries.append(clip_summary)
        print(
            json.dumps(
                {
                    "clip": clip,
                    "stages": [
                        {
                            key: row[key]
                            for key in (
                                "stage",
                                "penetrating_frames",
                                "sole_contact_min_distance_m",
                                "sole_signed_lateral_spacing_m",
                                "target_signed_lateral_spacing_m",
                            )
                        }
                        for row in measured
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
    source = Path(config.reference_gait_file[0])
    parsed = _parse_bvh(source)
    output = {
        "config": asdict(config),
        "initialization_complete": "curve_preparation_error" not in metadata,
        "curve_preparation_error": metadata.get("curve_preparation_error"),
        "per_clip_curve_gates": metadata["per_clip_curve_gates"],
        "root_lock_passes": metadata.get("root_lock_passes", []),
        "bvh_ik_mapping": {
            "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_file_sha256": hashlib.sha256(
                Path(config.xml_path).read_bytes()
            ).hexdigest(),
            "legacy_expected_names": [name for _, name in BVH_IK_TARGET_MAP],
            "missing_legacy_names": [
                name for _, name in BVH_IK_TARGET_MAP if name not in parsed.joints
            ],
            "strict_resolved_names": _resolve_bvh_ik_joint_names(parsed),
            "marker_names": metadata["names"],
        },
        "reference_digest": (
            reference_digest(env) if "curve_preparation_error" not in metadata else None
        ),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "biomechanics_env.py",
                "bvh_reference.py",
                "reference_interpolation.py",
                "tools/check_retarget_stages.py",
            )
        },
        "limitations": [
            "Kinematic nodes, not a policy rollout or dynamic feasibility proof.",
            "Signed spacing uses the neutral stance lateral axis, "
            "not a rotating heading frame.",
            "Targets after root alignment follow common root shifts; after curve "
            "closure they are diagnostic comparators, not an additional IK solve.",
            "No sole contact is recorded as zero depth; "
            "this is not positive clearance.",
            "CLAMP mode diagnoses a finite motion only; "
            "it does not repair or certify WRAP seams.",
        ],
        "clips": summaries,
    }
    if args.ik_fit:
        from tools.audit_ik_fit import inspect_ik_fit

        output["ik_fit"] = inspect_ik_fit(env, metadata, stages)
        name = "tools/audit_ik_fit.py"
        output["source_sha256"][name] = hashlib.sha256(
            (PROJECT_ROOT / name).read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)
    if "curve_preparation_error" in metadata:
        print(f"Production curve gate REJECTED: {metadata['curve_preparation_error']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
