"""Measure the frozen v6 imitation objective on controlled native snapshots.

No rollout, policy optimization, reference mutation or reward change. Perturbed
velocities are instantaneous probes, not claims of realizable trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import LoopMode
from config import PROJECT_ROOT, EnvConfig
from evaluate import make_environment
from tools.check_recovery_reference import reference_digest
from tools.check_reference_curves import full_state, lowest_foot_point


def build_cases(
    env: BiomechanicsJoystickEnv,
    reference: dict[str, np.ndarray],
    quarter: dict[str, np.ndarray],
    half: dict[str, np.ndarray],
    frozen_qpos: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create independent perturbations without modifying source arrays.

    Only sagittal hip/knee/ankle channels are exchanged, by joint name. This
    is not an anatomical reflection, and no candidate is clamped to limits.
    """
    qpos, qvel = full_state(env, reference)
    cases = {
        "exact_reference": (qpos.copy(), qvel.copy()),
        "standing_default": (np.array(env._init_q), np.zeros_like(qvel)),
        "frozen_phase_zero": (frozen_qpos.copy(), np.zeros_like(qvel)),
        "stopped_reference_pose": (qpos.copy(), np.zeros_like(qvel)),
        "wrong_phase_quarter": full_state(env, quarter),
        "wrong_phase_half": full_state(env, half),
    }
    swapped_qpos, swapped_qvel = qpos.copy(), qvel.copy()
    positions = np.asarray(env._actuator_qpos_indices)
    velocities = np.asarray(env._actuator_dof_indices)
    names = list(env._actuator_joint_names)
    for joint in ("hip_z", "knee_z", "ankle_z"):
        left, right = (names.index(f"{side}_{joint}") for side in ("left", "right"))
        swapped_qpos[positions[[left, right]]] = qpos[positions[[right, left]]]
        swapped_qvel[velocities[[left, right]]] = qvel[velocities[[right, left]]]
    cases["swapped_sagittal_legs"] = swapped_qpos, swapped_qvel
    translated = qpos.copy()
    translated[:2] += [1.0, -0.5]
    cases["translated_xy_control"] = translated, qvel.copy()
    drifting = qvel.copy()
    drifting[0] += 0.5
    cases["root_velocity_plus_0p5_world_x"] = qpos.copy(), drifting
    for side in ("left", "right"):
        index = velocities[names.index(f"{side}_knee_z")]
        for label, delta in (("plus", 2.0), ("minus", -2.0)):
            altered = qvel.copy()
            altered[index] += delta
            cases[f"{side}_knee_velocity_{label}_2"] = qpos.copy(), altered
    return cases


def snapshot(data: mujoco.MjData) -> dict[str, np.ndarray]:
    """Copy native kinematic fields consumed by the actual pure reward path."""
    return {
        name: np.array(getattr(data, name), dtype=np.float32, copy=True)
        for name in (
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "xmat",
            "site_xpos",
            "cvel",
            "subtree_com",
        )
    }


def foot_geometry(
    env: BiomechanicsJoystickEnv, data: mujoco.MjData
) -> list[dict[str, Any]]:
    """Measure lowest-sole material-point velocity; not force-labeled stance."""
    model = env._mj_model
    result = []
    jacobian = np.zeros((3, model.nv))
    angular = np.zeros_like(jacobian)
    floor = env._floor_geom_id
    if model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(
        data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8
    ):
        raise ValueError("Reward geometry diagnostic requires a horizontal plane.")
    for geom in env._foot_geom_ids_np:
        geom = int(geom)
        point = lowest_foot_point(model, data, geom)
        mujoco.mj_jac(
            model, data, jacobian, angular, point, int(model.geom_bodyid[geom])
        )
        velocity = jacobian @ data.qvel
        height = float(point[2] - data.geom_xpos[floor, 2])
        result.append(
            {
                "geom": model.geom(geom).name,
                "lowest_height_m": height,
                "near_floor_1cm": height <= 0.01,
                "point_velocity_xy_m_s": velocity[:2].tolist(),
                "point_speed_xy_m_s": float(np.linalg.norm(velocity[:2])),
            }
        )
    return result


def summarize(values: np.ndarray) -> dict[str, float]:
    """Summarize finite scalar samples without silently ignoring invalid data."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite scalar samples.")
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def audit_reward(env: BiomechanicsJoystickEnv, sample_count: int) -> dict[str, Any]:
    """Evaluate actual reward and termination with a fixed independent clock."""
    if sample_count < 4 or sample_count % 4:
        raise ValueError("Sample count must be a positive multiple of four.")
    if env._config.deepmimic_reward_mode != "pure":
        raise ValueError("Only pure imitation is covered by this snapshot adapter.")
    if env._config.recovery_physics_version != "consistent_v6":
        raise ValueError("This diagnostic pins the consistent_v6 control.")
    if len(env._bvh_reference_frame_counts) != 1 or int(
        env._bvh_reference_loop_modes[0]
    ) != int(LoopMode.WRAP):
        raise ValueError("Phase-index wrapping requires one selected WRAP clip.")
    digest = reference_digest(env)
    duration = float(env._bvh_reference_motion_lengths[0])
    times = (np.arange(sample_count, dtype=np.float32) + 0.5) * duration / sample_count
    query = jax.jit(
        jax.vmap(lambda time: env._query_bvh_reference_clip(jp.array(0), time))
    )
    batch = jax.tree.map(np.asarray, query(jp.array(times)))
    zero = jax.tree.map(np.asarray, query(jp.zeros(1)))
    frozen_qpos, _ = full_state(env, {key: value[0] for key, value in zero.items()})
    data = mujoco.MjData(env._mj_model)
    states, rows = [], []
    lower = np.maximum(
        env._actuator_ctrl_lower_limits_np, env._actuator_qpos_lower_limits_np
    )
    upper = np.minimum(
        env._actuator_ctrl_upper_limits_np, env._actuator_qpos_upper_limits_np
    )
    for index, time in enumerate(times):
        references = [
            {
                key: value[(index + offset) % sample_count]
                for key, value in batch.items()
            }
            for offset in (0, sample_count // 4, sample_count // 2)
        ]
        for name, (qpos, qvel) in build_cases(env, *references, frozen_qpos).items():
            data.qpos[:], data.qvel[:] = qpos, qvel
            mujoco.mj_forward(env._mj_model, data)
            states.append(snapshot(data))
            joints = qpos[np.asarray(env._actuator_qpos_indices)]
            rows.append(
                {
                    "case": name,
                    "phase": float(time / duration),
                    "time_s": float(time),
                    "joint_limit_excess_rad": float(
                        max(0, np.max(lower - joints), np.max(joints - upper))
                    ),
                    "feet": foot_geometry(env, data),
                }
            )

    def score(fields: dict[str, jax.Array], time: jax.Array) -> dict[str, jax.Array]:
        state = SimpleNamespace(**fields)
        info = {
            "motion_step": jp.array(0, dtype=jp.int32),
            "bvh_reference_time_offset": time,
            "bvh_reference_clip_id": jp.array(0, dtype=jp.int32),
            "reference_fallback_standing": jp.array(False),
            "command": jp.zeros(3),
        }
        components = env._get_bvh_deepmimic_reward(state, info)
        action = jp.zeros(env.action_size)
        return {
            **components,
            "pure_step_reward": env._get_reward(state, action, action, info),
            "done": env._get_done(state, info),
            "physical_fail": env._get_physical_fail_done(state),
        }

    fields = {
        key: jp.array(np.stack([state[key] for state in states])) for key in states[0]
    }
    scores = jax.tree.map(
        np.asarray,
        jax.jit(jax.vmap(score))(
            fields,
            jp.array([row["time_s"] for row in rows]),
        ),
    )
    constants = {
        key: float(getattr(env, key))
        for key in (
            "DEEPMIMIC_POSE_SCALE",
            "DEEPMIMIC_VELOCITY_SCALE",
            "DEEPMIMIC_ROOT_POSE_SCALE",
            "DEEPMIMIC_ROOT_VELOCITY_SCALE",
            "DEEPMIMIC_KEY_POSITION_SCALE",
            "DEEPMIMIC_POSE_WEIGHT",
            "DEEPMIMIC_VELOCITY_WEIGHT",
            "DEEPMIMIC_ROOT_POSE_WEIGHT",
            "DEEPMIMIC_ROOT_VELOCITY_WEIGHT",
            "DEEPMIMIC_KEY_POSITION_WEIGHT",
            "REWARD_MAX",
            "FALL_REWARD",
        )
    }
    weights = [
        env.DEEPMIMIC_POSE_WEIGHT,
        env.DEEPMIMIC_VELOCITY_WEIGHT,
        env.DEEPMIMIC_ROOT_POSE_WEIGHT,
        env.DEEPMIMIC_ROOT_VELOCITY_WEIGHT
        * env._config.deepmimic_root_velocity_weight_scale,
        env.DEEPMIMIC_KEY_POSITION_WEIGHT,
    ]
    for index, row in enumerate(rows):
        row["reward"] = {key: float(value[index]) for key, value in scores.items()}
        reward = row["reward"]
        # Algebra-only sensitivity comparison, NOT a MimicKit replication or preset.
        pose_sum = np.exp(
            -env.DEEPMIMIC_POSE_SCALE * env.action_size * reward["pose_error"]
        )
        velocity_sum = np.exp(
            -env.DEEPMIMIC_VELOCITY_SCALE * env.action_size * reward["velocity_error"]
        )
        row["offline_sum_hinge_total"] = float(
            np.dot(
                weights,
                [
                    pose_sum,
                    velocity_sum,
                    reward["root_pose"],
                    reward["root_velocity"],
                    reward["key_position"],
                ],
            )
            / max(sum(weights), 1e-6)
        )
    summaries = {}
    for name in dict.fromkeys(row["case"] for row in rows):
        selected = [row for row in rows if row["case"] == name]
        summaries[name] = {
            "reward": {
                key: summarize([row["reward"][key] for row in selected])
                for key in scores
            },
            "offline_sum_hinge_total": summarize(
                [row["offline_sum_hinge_total"] for row in selected]
            ),
            "maximum_joint_limit_excess_rad": max(
                row["joint_limit_excess_rad"] for row in selected
            ),
        }
    if summaries["exact_reference"]["reward"]["total"]["min"] < 0.9999:
        raise AssertionError(
            "Exact reference failed the native reward consistency check."
        )
    if reference_digest(env) != digest:
        raise AssertionError("Reward audit changed the processed reference.")
    exact = [row for row in rows if row["case"] == "exact_reference"]
    translated = [row for row in rows if row["case"] == "translated_xy_control"]
    translation_error = max(
        abs(left["reward"]["total"] - right["reward"]["total"])
        for left, right in zip(exact, translated, strict=True)
    )
    if translation_error > 1e-6:
        raise AssertionError("Rigid root XY translation changed imitation reward.")
    return {
        "reference_digest": digest,
        "duration_s": duration,
        "phase_sample_count": sample_count,
        "snapshot_count": len(rows),
        "runtime_loop_mode": "WRAP",
        "translated_xy_total_max_error": translation_error,
        "actuator_joint_names": list(env._actuator_joint_names),
        "constants": constants,
        "summaries": summaries,
        "rows": rows,
    }


def main() -> None:
    """Write reproducible, phase-resolved reward evidence without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase11_20261013/reward_discrimination.json",
    )
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 4:
        parser.error("--samples must be a positive multiple of four")
    jax.config.update("jax_platform_name", "cpu")
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
        "reference_kinematics.py",
        "tools/check_reward_discrimination.py",
        "tools/check_recovery_reference.py",
        "tools/check_reference_curves.py",
    )
    paths = [PROJECT_ROOT / name for name in sources] + [
        Path(config.xml_path),
        Path(config.reference_gait_file[0]),
    ]
    report = {
        "completed": False,
        "config": asdict(config),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
        "limitations": [
            "Native snapshots scored with production JAX reward and termination; "
            "no policy or integration rollout.",
            "Exact score validates processed-pose FK, not original BVH marker fit "
            "or physical feasibility.",
            "All cases retain the original clock and disable standing fallback.",
            "Velocity-only and stopped-pose probes are instantaneous, "
            "not coherent trajectories.",
            "Near-floor material-point speed is a geometric proxy, "
            "not force-labeled stance; penetration is included.",
            "Offline sum of hinge errors is a sensitivity counterfactual, "
            "not an adopted reward or equivalent quaternion-based method.",
            "High static reward does not by itself establish a learned-policy optimum.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Preparing frozen v6 and scoring native snapshots...", flush=True)
    env = make_environment(config)
    report.update(audit_reward(env, args.samples))
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, summary in report["summaries"].items():
        print(
            name,
            "mean imitation",
            round(summary["reward"]["total"]["mean"], 6),
            "mean pure",
            round(summary["reward"]["pure_step_reward"]["mean"], 6),
            flush=True,
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
