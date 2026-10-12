"""Controlled phase-three contact/dynamics audit; no training or model edits.

Compare root-XY scale and foot-center locking without changing the reward or
actuator settings. Contact-height labels are geometric proxies, not true stance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import PROJECT_ROOT, EnvConfig
from evaluate import make_environment
from tools.check_recovery_reference import reference_digest
from tools.check_reference_curves import lowest_foot_point


def make_audited_environment(config: EnvConfig) -> tuple[BiomechanicsJoystickEnv, list]:
    """Observe normalization outputs without modifying the selected algorithm."""
    normalize = BiomechanicsJoystickEnv._normalize_ik_foot_target_heights
    diagnostics = []

    def record(env, targets, names, counts):
        result = normalize(env, targets, names, counts)
        feet = [
            i for i, name in enumerate(names) if "metatarsal" in name or "foot" in name
        ]
        if feet:
            expected = env._standing_marker_height(names[feet[0]])
            for clip, count in enumerate(counts):
                if count < 1:
                    continue
                heights = result[clip, :count][:, feet, 2]
                minima = heights.min(axis=1)
                error = minima - expected
                diagnostics.append(
                    {
                        "clip_id": clip,
                        "frame_count": int(count),
                        "standing_foot_marker_height_m": expected,
                        "per_frame_lowest_target_height_min_m": float(minima.min()),
                        "per_frame_lowest_target_height_max_m": float(minima.max()),
                        "per_frame_lowest_target_height_error_rmse_m": float(
                            np.sqrt(np.mean(error**2))
                        ),
                        "per_frame_lowest_target_height_error_max_m": float(
                            np.max(np.abs(error))
                        ),
                        "frames_exceeding_1micrometer_error": int(
                            np.count_nonzero(np.abs(error) > 1e-6)
                        ),
                    }
                )
        return result

    with patch.object(
        BiomechanicsJoystickEnv, "_normalize_ik_foot_target_heights", record
    ):
        env = make_environment(config)
    return env, diagnostics


def sample_reference_states(
    env: BiomechanicsJoystickEnv, clip_id: int, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full native coordinates and analytic qvel derivative at given times.

    Acceleration at C1 knots is one-sided and must not be used as a unique physical
    value; the caller uses off-knot samples for inverse dynamics.
    """
    if not np.isfinite(times).all() or np.asarray(times).ndim != 1:
        raise ValueError("Sample times must be a finite one-dimensional array.")
    if not env._continuous_reference:
        raise ValueError(
            "Analytic acceleration requires consistent_v2 or later curves."
        )

    def state(time):
        reference = env._query_bvh_reference_clip(jp.array(clip_id), time)
        qpos = env._init_q.at[:3].set(reference["sim_root_pos"])
        qpos = qpos.at[3:7].set(reference["sim_root_quat"])
        qpos = qpos.at[env._actuator_qpos_indices].set(reference["qpos"])
        qvel = jp.zeros(env._mj_model.nv)
        qvel = qvel.at[:3].set(reference["sim_root_vel"])
        qvel = qvel.at[3:6].set(reference["sim_root_angvel"])
        qvel = qvel.at[env._actuator_dof_indices].set(reference["qvel"])
        return qpos, qvel

    def sample(time):
        qpos, qvel = state(time)
        _, acceleration = jax.jvp(
            lambda at: state(at)[1], (time,), (jp.ones_like(time),)
        )
        return qpos, qvel, acceleration

    sampled = jax.jit(jax.vmap(sample))(jp.asarray(times, dtype=jp.float32))
    result = tuple(np.asarray(values, dtype=np.float64) for values in sampled)
    if not all(np.isfinite(values).all() for values in result):
        raise ValueError("Reference produced non-finite coordinates/derivatives.")
    return result


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize a finite scalar series; empty groups have explicit null values."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Cannot summarize non-finite measurements.")
    return {
        "count": int(values.size),
        "median": float(np.median(values)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
    }


def inspect_contacts(
    env: BiomechanicsJoystickEnv,
    times: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    subdivisions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure each foot center and lowest material point against the same floor."""
    model = env._mj_model
    data = mujoco.MjData(model)
    foot_ids = env._foot_geom_ids_np
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError("State batch shape does not match model/time count.")
    if subdivisions < 1 or len(times) < 2:
        raise ValueError("Need positive subdivisions and at least two sample times.")
    if not all(np.isfinite(values).all() for values in (times, qpos, qvel)):
        raise ValueError("Contact samples must be finite.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Contact sample times must strictly increase.")
    jacobian = np.zeros((3, model.nv))
    angular = np.zeros_like(jacobian)
    heights = np.zeros((len(times), len(foot_ids)))
    center_speeds = np.zeros_like(heights)
    point_speeds = np.zeros_like(heights)
    centers = np.zeros((len(times), len(foot_ids), 3))
    rows: list[dict[str, Any]] = []
    preload = float(env.FOOT_CONTACT_PRELOAD)
    for sample_index, time in enumerate(times):
        data.qpos[:], data.qvel[:] = qpos[sample_index], qvel[sample_index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        floor = env._floor_geom_id
        if model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(
            data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8
        ):
            raise ValueError("Contact audit requires a horizontal floor plane.")
        for foot, geom_id in enumerate(foot_ids):
            geom_id = int(geom_id)
            center = data.geom_xpos[geom_id].copy()
            point = lowest_foot_point(model, data, geom_id)
            centers[sample_index, foot] = center
            heights[sample_index, foot] = point[2] - data.geom_xpos[floor, 2]
            body = int(model.geom_bodyid[geom_id])
            mujoco.mj_jac(model, data, jacobian, angular, center, body)
            center_speeds[sample_index, foot] = np.linalg.norm(
                (jacobian @ data.qvel)[:2]
            )
            mujoco.mj_jac(model, data, jacobian, angular, point, body)
            point_speeds[sample_index, foot] = np.linalg.norm(
                (jacobian @ data.qvel)[:2]
            )
            rows.append(
                {
                    "sample": sample_index,
                    "time_s": float(time),
                    "node": sample_index % subdivisions == 0,
                    "foot": model.geom(geom_id).name,
                    "surface_height_m": heights[sample_index, foot],
                    "center_speed_xy_m_s": center_speeds[sample_index, foot],
                    "support_material_speed_xy_m_s": point_speeds[sample_index, foot],
                    "center_x_m": center[0],
                    "center_y_m": center[1],
                }
            )
    summary: dict[str, Any] = {
        "samples": len(times),
        "subdivisions": subdivisions,
        "intentional_preload_m": preload,
        "maximum_penetration_m": max(0.0, float(-heights.min())),
        "maximum_penetration_beyond_preload_m": max(
            0.0, float(-heights.min() - preload)
        ),
        "net_free_root_translation_m": (qpos[-1, :3] - qpos[0, :3]).tolist(),
        "root_xy_mean_speed_m_s": float(
            np.linalg.norm(qpos[-1, :2] - qpos[0, :2]) / (times[-1] - times[0])
        ),
        "node_maximum_penetration_m": max(0.0, float(-heights[::subdivisions].min())),
        "feet": {},
    }
    between = np.arange(len(times)) % subdivisions != 0
    summary["between_node_maximum_penetration_m"] = (
        max(0.0, float(-heights[between].min())) if between.any() else None
    )
    for threshold in (0.01, float(env._config.reference_foot_lock_height)):
        near = heights <= threshold
        summary[f"both_feet_below_{threshold:g}m_fraction"] = float(
            np.mean(np.all(near, axis=1))
        )
        summary[f"no_feet_below_{threshold:g}m_fraction"] = float(
            np.mean(~np.any(near, axis=1))
        )
        for foot, geom_id in enumerate(foot_ids):
            name = model.geom(int(geom_id)).name
            foot_summary = summary["feet"].setdefault(name, {})
            mask = near[:, foot]
            # Consecutive NODAL near-floor samples avoid interpreting the
            # derivative of a cubic as the actual displacement of its nodes.
            node_mask = mask[::subdivisions]
            node_centers = centers[::subdivisions, foot, :2]
            consecutive = node_mask[:-1] & node_mask[1:]
            displacements = np.linalg.norm(np.diff(node_centers, axis=0), axis=1)
            foot_summary[f"height_threshold_{threshold:g}m"] = {
                "near_floor_fraction": float(np.mean(mask)),
                "center_speed_m_s": summarize(center_speeds[mask, foot]),
                "material_point_speed_m_s": summarize(point_speeds[mask, foot]),
                "consecutive_near_floor_node_center_displacement_m": summarize(
                    displacements[consecutive]
                ),
                "threshold_crossings": int(np.count_nonzero(mask[1:] != mask[:-1])),
            }
    return summary, rows


def main() -> None:
    """Run controlled variants, optional inverse dynamics and free-root replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=PROJECT_ROOT
        / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=PROJECT_ROOT
        / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh",
    )
    parser.add_argument("--clip-id", type=int, default=1)
    parser.add_argument(
        "--reference-clip-ids",
        nargs="+",
        type=int,
        help="Loader clip indices to retain before preprocessing; "
        "--clip-id is the resulting runtime index.",
    )
    parser.add_argument(
        "--recovery-physics-version",
        choices=("consistent_v2", "consistent_v3", "consistent_v4", "consistent_v5"),
        default="consistent_v2",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Measure only root XY scale .35 with foot locking enabled.",
    )
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs/recovery_phase3_20261012",
    )
    args = parser.parse_args()
    if args.subdivisions < 1:
        parser.error("--subdivisions must be positive.")
    for path in (args.xml_path, args.reference_file):
        if not path.is_file():
            parser.error(f"Missing local input: {path}")
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    config = EnvConfig(
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=str(args.xml_path),
        reference_gait="bvh",
        reference_gait_file=[str(args.reference_file)],
        reference_forced_clip_id=args.clip_id,
        reference_clip_ids=args.reference_clip_ids,
        recovery_physics_version=args.recovery_physics_version,
        reference_loop_closure="bounded",
        deepmimic_reward_mode="pure",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "scope": "Contact/dynamics diagnosis, no model edits, policy or training.",
        "completed": False,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "base_config": asdict(config),
        "clip_id": args.clip_id,
        "xml_sha256": hashlib.sha256(args.xml_path.read_bytes()).hexdigest(),
        "bvh_sha256": hashlib.sha256(args.reference_file.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "biomechanics_env.py",
                "bvh_reference.py",
                "reference_interpolation.py",
                "reference_ik.py",
                "tools/check_reference_contacts.py",
                "tools/check_reference_curves.py",
            )
        },
        "variants": {},
        "limitations": [
            "Height labels are not true stance; 5mm preload is intentional.",
            "Center lock and material contact-point velocity are different quantities.",
            "Changing lock/XY settings can change the whole retargeting pipeline.",
            "Inverse dynamics uses exact accelerations and the model's soft contacts; "
            "it does not optimize contact-force allocation.",
            "Open-loop replay is not proof an RL policy cannot learn.",
        ],
    }
    report_path = args.output_dir / "summary.json"
    variants = (
        ((0.35, True),)
        if args.baseline_only
        else ((0.35, True), (1.0, True), (0.35, False), (1.0, False))
    )
    for scale, lock in variants:
        label = f"xy{scale:g}_lock{int(lock)}"
        print(f"Preparing {label}...", flush=True)
        variant_config = replace(
            config, reference_root_xy_scale=scale, reference_lock_stance_feet=lock
        )
        try:
            env, normalization = make_audited_environment(variant_config)
        except ValueError as exc:
            output["variants"][label] = {"preparation_rejected": str(exc)}
            print(f"Rejected without relaxing guards: {exc}", flush=True)
            report_path.write_text(
                json.dumps(output, indent=2) + "\n", encoding="utf-8"
            )
            continue
        count = int(env._bvh_reference_frame_counts[args.clip_id])
        duration = float(env._bvh_reference_motion_lengths[args.clip_id])
        interval = float(env._bvh_reference_frame_times[args.clip_id])
        if count < 2 or int(env._bvh_reference_loop_modes[args.clip_id]) != 1:
            raise ValueError("This controlled audit requires a nontrivial WRAP clip.")
        times = np.linspace(0.0, duration, (count - 1) * args.subdivisions + 1)
        qpos, qvel, _ = sample_reference_states(env, args.clip_id, times)
        contact_summary, rows = inspect_contacts(
            env, times, qpos, qvel, args.subdivisions
        )
        with (args.output_dir / f"{label}_contacts.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        variant: dict[str, Any] = {
            "config": asdict(variant_config),
            "ik_target_normalization": normalization,
            "contacts": contact_summary,
            "processed_reference_sha256": reference_digest(env),
            "closure": env._reference_closure_diagnostics,
            "ik_warmstart": getattr(env, "_reference_ik_warmstart_diagnostics", []),
            "source_clip_ids": env._bvh_reference_source_clip_ids_np.tolist(),
            "source_frame_range": [
                int(env._bvh_reference_source_start_frames[args.clip_id]),
                int(env._bvh_reference_source_end_frames[args.clip_id]),
            ],
        }
        from tools.check_reference_collisions import inspect_collision_geometry
        from tools.check_reference_curves import inspect_consistency

        variant["numerical_curve_checks"] = inspect_consistency(env, args.clip_id)
        variant["collisions"] = inspect_collision_geometry(env._mj_model, qpos, qvel)
        if not args.skip_dynamics:
            from tools.recovery_dynamics import inspect_dynamics

            print(f"Inverse dynamics: {label}...", flush=True)
            dynamic_times = (
                (np.arange((count - 1) * args.subdivisions) + 0.5)
                * interval
                / args.subdivisions
            )
            state_batch = sample_reference_states(env, args.clip_id, dynamic_times)
            variant["dynamics"] = inspect_dynamics(env._mj_model, *state_batch)
            variant["dynamics"]["sampling"] = (
                "Off-knot midpoints; analytic qacc, not averaged knot discontinuities."
            )
        if not args.skip_replay and label == "xy0.35_lock1":
            from tools.recovery_replay import replay_reference

            print(
                "Baseline free-root native replay at 3 controller timings...",
                flush=True,
            )
            variant["replays"] = {
                "physics_rate": replay_reference(env, args.clip_id),
                "control_rate_current": replay_reference(
                    env, args.clip_id, control_dt=env.dt
                ),
                "control_rate_lookahead": replay_reference(
                    env,
                    args.clip_id,
                    control_dt=env.dt,
                    target_lookahead_seconds=env.dt,
                ),
            }
        output["variants"][label] = variant
        print(
            json.dumps({"variant": label, "contacts": contact_summary}, indent=2),
            flush=True,
        )
        report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for name in ("tools/recovery_dynamics.py", "tools/recovery_replay.py"):
        if (PROJECT_ROOT / name).is_file():
            output["source_sha256"][name] = hashlib.sha256(
                (PROJECT_ROOT / name).read_bytes()
            ).hexdigest()
    output["completed"] = True
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
