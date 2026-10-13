"""Attribute post-IK Cartesian changes to joint groups without rebuilding motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import (
    MotionSegment,
    _joint_hinge_angles,
    _parse_bvh,
    _standing_like_frame_index,
)
from config import PROJECT_ROOT, EnvConfig
from tools.check_recovery_reference import reference_digest
from tools.check_retarget_stages import (
    capture_retarget_stages,
    inspect_stage,
    scalar_summary,
)

RESTORE_GROUPS = {
    "torso_pelvis": tuple(
        f"{body}_{axis}" for body in ("abdomen", "pelvis") for axis in "xyz"
    ),
    "hip_y": ("left_hip_y", "right_hip_y"),
    "ankle_z": ("left_ankle_z", "right_ankle_z"),
}
WINDOWS = ((64, 80), (108, 120))


def restore_joint_group(
    prior: np.ndarray,
    ik: np.ndarray,
    names: tuple[str, ...],
    group: tuple[str, ...],
) -> np.ndarray:
    """Copy a prior pose sequence, replacing named columns with raw IK values."""
    prior, ik = np.asarray(prior), np.asarray(ik)
    if (
        prior.ndim != 2
        or not all(prior.shape)
        or ik.shape != prior.shape
        or len(names) != prior.shape[1]
        or len(set(names)) != len(names)
        or len(set(group)) != len(group)
        or not set(group).issubset(names)
    ):
        raise ValueError("Need matching nonempty pose arrays and unique known names.")
    for values in (prior, ik):
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("Pose arrays must contain finite real values.")
    result = prior.astype(np.result_type(prior.dtype, ik.dtype), copy=True)
    indices = [names.index(name) for name in group]
    result[:, indices] = ik[:, indices]
    return result


def nodal_second_difference(values: np.ndarray, interval: float) -> np.ndarray:
    """Return interior-node second differences, not interpolant accelerations."""
    values = np.asarray(values)
    if (
        values.ndim < 2
        or values.shape[0] < 3
        or not values.size
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or not np.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("Need at least three finite nodes and a positive interval.")
    return np.diff(values.astype(float), n=2, axis=0) / interval**2


def inspect_local_feet(
    env: BiomechanicsJoystickEnv,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    interval: float,
) -> dict[str, Any]:
    """Observe capsule centers relative to root translation, in world axes."""
    joints = np.asarray(joints)
    roots, quaternions = np.asarray(roots), np.asarray(quaternions)
    nodal_second_difference(joints, interval)
    count = len(joints)
    if (
        joints.shape != (count, len(env._actuator_qpos_indices_np))
        or roots.shape != (count, 3)
        or quaternions.shape != (count, 4)
        or roots.dtype.kind not in "fiu"
        or quaternions.dtype.kind not in "fiu"
        or not np.isfinite(roots).all()
        or not np.isfinite(quaternions).all()
        or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid pose/root arrays or nonunit root quaternions.")
    model = env._mj_model
    feet = np.asarray(env._foot_geom_ids_np, dtype=int)
    if np.any(model.geom_type[feet] != mujoco.mjtGeom.mjGEOM_CAPSULE):
        raise ValueError("This attribution audit expects capsule foot geometry.")
    data = mujoco.MjData(model)
    relative = np.zeros((count, len(feet), 3))
    axes = np.zeros((count, len(feet)))
    for frame in range(count):
        data.qpos[:] = env._init_q_np
        data.qpos[:3] = roots[frame]
        data.qpos[3:7] = quaternions[frame]
        data.qpos[env._actuator_qpos_indices_np] = joints[frame]
        mujoco.mj_kinematics(model, data)
        relative[frame] = data.geom_xpos[feet] - roots[frame]
        axes[frame] = data.geom_xmat[feet].reshape(-1, 3, 3)[:, 2, 2]
    result = {}
    for index, foot in enumerate(feet):
        acceleration = nodal_second_difference(relative[:, index, :2], interval)
        norms = np.linalg.norm(acceleration, axis=1)
        result[model.geom(int(foot)).name] = {
            "root_relative_xyz_m": relative[:, index].tolist(),
            "capsule_axis_z": axes[:, index].tolist(),
            "xy_second_difference_m_s2": acceleration.tolist(),
            "xy_second_difference_norm_summary_m_s2": scalar_summary(norms),
            "worst_center_frame": int(norms.argmax()) + 1,
        }
    return result


def inspect_pose_limits(
    env: BiomechanicsJoystickEnv, joints: np.ndarray
) -> dict[str, Any]:
    """Measure nodal target bounds; do not infer required actuator force/control."""
    result = {}
    for label, lower, upper in (
        (
            "physical_joint",
            env._actuator_qpos_lower_limits_np,
            env._actuator_qpos_upper_limits_np,
        ),
        (
            "position_control_target",
            env._actuator_ctrl_lower_limits_np,
            env._actuator_ctrl_upper_limits_np,
        ),
    ):
        excess = np.maximum(np.maximum(lower - joints, joints - upper), 0)
        result[label] = {
            "max_excess_rad": float(excess.max()),
            "violating_coordinates": int(np.count_nonzero(excess > 1e-6)),
            "active_bound_coordinates": int(
                np.count_nonzero(
                    (np.abs(joints - lower) <= 1e-6) | (np.abs(joints - upper) <= 1e-6)
                )
            ),
        }
    return result


def inspect_windows(
    feet: dict[str, Any], joints: np.ndarray, interval: float, names: tuple[str, ...]
) -> dict[str, Any]:
    """Keep explicit center-frame indexing in the two previously identified windows."""
    joint_differences = nodal_second_difference(joints, interval)
    result = {}
    for start, end in WINDOWS:
        if end >= len(joints) - 1:
            raise ValueError("Selected reference is too short for diagnostic windows.")
        result[f"{start}:{end}"] = {
            "center_frames": list(range(start, end + 1)),
            "joint_second_difference_rad_s2": {
                name: joint_differences[start - 1 : end, index].tolist()
                for index, name in enumerate(names)
            },
            "feet": {
                name: {
                    "xy_second_difference_m_s2": foot["xy_second_difference_m_s2"][
                        start - 1 : end
                    ],
                    "capsule_axis_z_with_neighbor_nodes": foot["capsule_axis_z"][
                        start - 1 : end + 2
                    ],
                }
                for name, foot in feet.items()
            },
        }
    return result


def inspect_knee_mapping(
    env: BiomechanicsJoystickEnv,
    source: Path,
    metadata: dict[str, Any],
    seed: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct the loader's knee seed before/after clipping; never alter it."""
    segment = MotionSegment(
        int(metadata["source_start_frames"][0]),
        int(metadata["source_end_frames"][0]),
    )
    parsed = _parse_bvh(source).slice(segment)
    seed = np.asarray(seed)
    if (
        seed.shape != (parsed.frames, len(env._actuator_joint_names))
        or seed.dtype.kind not in "fiu"
        or not np.isfinite(seed).all()
    ):
        raise ValueError("Captured seed must match the segment and be finite and real.")
    angles = {
        side: _joint_hinge_angles(
            parsed, (f"{side.title()}Leg", f"{side.title()}Knee"), allow_missing=True
        )[:, 1]
        for side in ("left", "right")
    }
    bind = _standing_like_frame_index(
        parsed, left_knee_flex=angles["left"], right_knee_flex=angles["right"]
    )
    result: dict[str, Any] = {"bind_frame": int(bind), "knees": {}}
    for side, flexion in angles.items():
        name = f"{side}_knee_z"
        index = env._actuator_joint_names.index(name)
        raw = -flexion
        offset = float(np.asarray(env._default_ctrl)[index] - raw[bind])
        mapped = raw + offset
        lower = np.float32(env._actuator_ctrl_lower_limits_np[index])
        upper = np.float32(env._actuator_ctrl_upper_limits_np[index])
        clipped = np.clip(mapped, lower, upper).astype(np.float32)
        difference = float(np.max(np.abs(clipped - seed[:, index])))
        if difference > 1e-6:
            raise ValueError(f"Reconstructed {name} differs from captured loader seed.")
        result["knees"][name] = {
            "raw_hinge_rad": raw.tolist(),
            "bind_offset_rad": offset,
            "before_clip_rad": mapped.tolist(),
            "after_clip_rad": clipped.tolist(),
            "lower_limit_rad": float(lower),
            "upper_limit_rad": float(upper),
            "upper_clipped_frames": np.flatnonzero(mapped > upper).tolist(),
            "lower_clipped_frames": np.flatnonzero(mapped < lower).tolist(),
            "captured_seed_difference_max_rad": difference,
        }
    return result


def main() -> None:
    """Capture once; measure global joint-group counterfactuals on frozen geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase9_20261013/attribution.json",
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
    env, metadata, captured = capture_retarget_stages(config)
    if "curve_preparation_error" in metadata:
        raise ValueError("Production v6 control failed curve preparation.")
    stages = {stage["name"]: stage for stage in captured}
    count = int(metadata["counts"][0])
    interval = float(env._bvh_reference_frame_times[0])
    names = tuple(env._actuator_joint_names)
    frozen = stages["normalized_ik_input"]
    prior = stages["stability_prior"]["joints"][0, :count]
    ik = stages["ik_solution"]["joints"][0, :count]
    for stage in (stages["stability_prior"], stages["ik_solution"]):
        for field in ("roots", "quaternions", "targets"):
            if not np.array_equal(stage[field], frozen[field]):
                raise ValueError(f"Frozen {field} differ before the stability prior.")
    variants = {
        "baseline_prior": restore_joint_group(prior, ik, names, ()),
        "raw_ik": restore_joint_group(prior, ik, names, names),
        **{
            f"restore_{group}": restore_joint_group(prior, ik, names, members)
            for group, members in RESTORE_GROUPS.items()
        },
    }
    group_names = set().union(*RESTORE_GROUPS.values())
    prior_changes = np.max(np.abs(prior.astype(float) - ik.astype(float)), axis=0)
    # Alpha=1 still evaluates neutral + (q-neutral) in float32, so bitwise
    # differences alone would misclassify rounding as intentional damping.
    changed = [name for i, name in enumerate(names) if prior_changes[i] > 1e-6]
    if not set(changed).issubset(group_names):
        raise ValueError("Selected reference has prior changes outside audited groups.")
    digest_before = reference_digest(env)
    sources = (
        "biomechanics_env.py",
        "config.py",
        "evaluate.py",
        "bvh_reference.py",
        "reference_ik.py",
        "reference_root.py",
        "reference_interpolation.py",
        "tools/check_stability_prior_attribution.py",
        "tools/check_retarget_stages.py",
        "tools/check_recovery_reference.py",
    )
    report: dict[str, Any] = {
        "scope": "Native FK attribution only, not new prepared references or replay.",
        "completed": False,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": asdict(config),
        "control_reference_digest": digest_before,
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
        "source_clip_id": int(metadata["source_clip_ids"][0]),
        "source_frame_range": [
            int(metadata[key][0])
            for key in ("source_start_frames", "source_end_frames")
        ],
        "interval_seconds": interval,
        "frames": count,
        "joint_names": names,
        "restore_groups": RESTORE_GROUPS,
        "prior_changed_joints": changed,
        "prior_change_classification_tolerance_rad": 1e-6,
        "prior_change_max_rad_by_joint": dict(zip(names, prior_changes.tolist())),
        "frozen_geometry_stage": "normalized_ik_input",
        "frozen_roots_m": frozen["roots"][0, :count].tolist(),
        "frozen_quaternions_wxyz": frozen["quaternions"][0, :count].tolist(),
        "marker_names": metadata["names"],
        "frozen_normalized_targets_m": frozen["targets"][0, :count].tolist(),
        "seed_joint_positions_rad": frozen["joints"][0, :count].tolist(),
        "loader_knee_mapping": inspect_knee_mapping(
            env,
            Path(config.reference_gait_file[0]),
            metadata,
            frozen["joints"][0, :count],
        ),
        "seed_joint_second_difference_rad_s2": nodal_second_difference(
            frozen["joints"][0, :count], interval
        ).tolist(),
        "root_relative_target_second_difference_m_s2": nodal_second_difference(
            frozen["targets"][0, :count].astype(float)
            - frozen["roots"][0, :count, None, :],
            interval,
        ).tolist(),
        "limitations": [
            "Same frozen pre-final-alignment geometry; no root refit or curve closure.",
            "Interior nodal second differences, not exact interpolant acceleration.",
            "Raw endpoint mismatch is not wrapped when taking differences.",
            "World-axis root-relative feet, not a heading-rotated coordinate frame.",
            "Only sole-sole contact masks are checked; not complete self-collision.",
            "Absent sole contacts have distance zero, not certified clearance.",
            "Position-target ranges do not certify dynamic actuator feasibility.",
            "Restoring a group is a nonlinear conditional effect, "
            "not additive attribution.",
        ],
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, joints in variants.items():
        stage = dict(frozen, name=name, joints=joints[None])
        feet = inspect_local_feet(
            env,
            joints,
            frozen["roots"][0, :count],
            frozen["quaternions"][0, :count],
            interval,
        )
        result = {
            "joint_positions_rad": joints.tolist(),
            "geometry": inspect_stage(env, stage, metadata["names"], 0, count),
            "feet": feet,
            "limits": inspect_pose_limits(env, joints),
            "windows": inspect_windows(feet, joints, interval, names),
        }
        report["variants"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "variant": name,
                    "right_foot_frame69_m_s2": float(
                        np.linalg.norm(
                            feet["right_foot_sole"]["xy_second_difference_m_s2"][68]
                        )
                    ),
                    "sole_penetration_frames": result["geometry"]["penetrating_frames"],
                }
            ),
            flush=True,
        )
    if reference_digest(env) != digest_before:
        raise ValueError("Attribution unexpectedly changed the runtime reference.")
    report["completed"] = True
    report["control_reference_unchanged"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
