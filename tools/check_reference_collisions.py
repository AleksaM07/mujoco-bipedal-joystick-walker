"""Inspect reference self-collision and a clearly labeled diagnostic ablation.

The optional ablation removes only foot-foot collision on a COPY of the native
model. It is not a production fix, a training configuration or a gait certificate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np

from config import PROJECT_ROOT, EnvConfig
from evaluate import make_environment
from tools.check_reference_contacts import sample_reference_states
from tools.check_reference_curves import inspect_consistency
from tools.recovery_dynamics import inspect_dynamics


def inspect_collision_geometry(
    model: mujoco.MjModel, positions: np.ndarray, velocities: np.ndarray
) -> dict[str, Any]:
    """Measure active collision geometry, counting a pair once per sample."""
    if positions.ndim != 2 or positions.shape[1] != model.nq:
        raise ValueError("Positions must have shape (samples, nq).")
    if velocities.shape != (len(positions), model.nv) or not len(positions):
        raise ValueError("Velocity shape must match a nonempty position batch.")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Collision states must be finite.")
    data = mujoco.MjData(model)
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for sample, (qpos, qvel) in enumerate(zip(positions, velocities, strict=True)):
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        observed = set()
        for contact in data.contact:
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if pair not in pairs:
                pairs[pair] = {
                    "geoms": [model.geom(i).name or f"geom{i}" for i in pair],
                    "bodies": [
                        model.body(int(model.geom_bodyid[i])).name for i in pair
                    ],
                    "penetrating_samples": 0,
                    "min_distance_m": 0.0,
                    "worst_sample_index": None,
                }
            item = pairs[pair]
            if contact.dist < item["min_distance_m"]:
                item["min_distance_m"] = float(contact.dist)
                item["worst_sample_index"] = sample
            if contact.dist < -1e-6 and pair not in observed:
                item["penetrating_samples"] += 1
                observed.add(pair)
    for item in pairs.values():
        item["penetrating_sample_fraction"] = item["penetrating_samples"] / len(
            positions
        )
    return {
        "samples": len(positions),
        "penetration_threshold_m": 1e-6,
        "pairs": sorted(pairs.values(), key=lambda item: item["min_distance_m"]),
    }


def copy_without_foot_pair_collision(
    model: mujoco.MjModel, foot_ids: np.ndarray, floor_id: int
) -> mujoco.MjModel:
    """Copy a foot/floor-only collision model and remove only its foot-foot pair.

    Refuse explicit contact pairs or any other collision-enabled geom. This
    ensures the bit-mask change cannot silently remove unrelated interactions.
    """
    feet = set(map(int, foot_ids))
    enabled = set(np.flatnonzero(model.geom_contype | model.geom_conaffinity))
    if len(feet) != 2 or enabled != feet | {floor_id} or model.npair:
        raise ValueError(
            "Ablation requires exactly two soles plus floor; no explicit pairs."
        )
    copied = copy.copy(model)
    copied.geom_contype[list(feet)] = 2
    copied.geom_conaffinity[list(feet)] = 1
    copied.geom_contype[floor_id] = 1
    copied.geom_conaffinity[floor_id] = 2
    return copied


def main() -> None:
    """Compare v2/v3 collision geometry without modifying either experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase3_20261012/collisions.json",
    )
    args = parser.parse_args()
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    output: dict[str, Any] = {
        "scope": "Native geometric self-collision and inverse-dynamics ablation only.",
        "production_collision_masks_changed": False,
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "biomechanics_env.py",
                "reference_interpolation.py",
                "tools/check_reference_collisions.py",
                "tools/check_reference_contacts.py",
                "tools/recovery_dynamics.py",
            )
        },
        "versions": {},
    }
    for version in ("consistent_v2", "consistent_v3"):
        print(f"Inspecting {version} self-collisions...", flush=True)
        config = EnvConfig(
            playground_impl="jax",
            physics_backend="mjx_jax",
            warp_num_worlds=1,
            xml_path=str(
                PROJECT_ROOT
                / "generated_models"
                / "human_male_180cm_75kg_standard_trainfast_v21_noarms.xml"
            ),
            reference_gait="bvh",
            reference_gait_file=[
                str(
                    PROJECT_ROOT
                    / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
                )
            ],
            reference_forced_clip_id=1,
            deepmimic_reward_mode="pure",
            recovery_physics_version=version,
            reference_loop_closure="bounded",
        )
        env = make_environment(config)
        count = int(env._bvh_reference_frame_counts[1])
        interval = float(env._bvh_reference_frame_times[1])
        times = (np.arange((count - 1) * 4) + 0.5) * interval / 4
        state = sample_reference_states(env, 1, times)
        geometry = inspect_collision_geometry(env._mj_model, state[0], state[1])
        original_masks = (
            env._mj_model.geom_contype.copy(),
            env._mj_model.geom_conaffinity.copy(),
        )
        diagnostic_model = copy_without_foot_pair_collision(
            env._mj_model, env._foot_geom_ids_np, env._floor_geom_id
        )
        ablated_geometry = inspect_collision_geometry(
            diagnostic_model, state[0], state[1]
        )
        dynamics = inspect_dynamics(diagnostic_model, *state)
        np.testing.assert_array_equal(env._mj_model.geom_contype, original_masks[0])
        np.testing.assert_array_equal(env._mj_model.geom_conaffinity, original_masks[1])
        result = {
            "config": asdict(config),
            "geometry": geometry,
            "standing_geometry": inspect_collision_geometry(
                env._mj_model,
                np.asarray(env._init_q)[None],
                np.zeros((1, env._mj_model.nv)),
            ),
            "curve_numerical_checks": inspect_consistency(env, 1),
            "counterfactual_no_sole_sole_collision": {
                "geometry": ablated_geometry,
                "dynamics": dynamics,
                "warning": (
                    "Diagnostic copy only. Removing collisions hides overlapping "
                    "feet; it is NOT a retargeting fix."
                ),
            },
        }
        output["versions"][version] = result
        print(
            json.dumps({"version": version, "geometry": geometry}, indent=2), flush=True
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
