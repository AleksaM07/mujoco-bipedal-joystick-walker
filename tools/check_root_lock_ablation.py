"""Isolate pre-IK locking and contact-history handoffs on the v5 reference."""

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
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import PROJECT_ROOT, EnvConfig
from tools.check_recovery_reference import reference_digest
from tools.check_reference_collisions import inspect_collision_geometry
from tools.check_reference_contacts import inspect_contacts, sample_reference_states
from tools.check_reference_curves import inspect_consistency
from tools.check_retarget_stages import capture_retarget_stages
from tools.recovery_dynamics import inspect_dynamics
from tools.recovery_replay import replay_reference


def lock_with_contact_history(
    env: BiomechanicsJoystickEnv,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    count: int,
) -> np.ndarray:
    """Anchor incoming support to its previous position when already in contact.

    A sole which was below the height threshold in both frames has a previous
    world position even if another sole was selected as support. Reuse it at a
    handoff instead of inheriting an unrelated input-root increment. Genuine
    landings and flight keep the original input displacement. This constrains
    foot centers only; it is not a material-point no-slip or dynamics guarantee.
    """
    locked = np.asarray(roots, dtype=np.float32).copy()
    if count <= 1:
        return locked
    data = mujoco.MjData(env._mj_model)
    feet = np.asarray(env._foot_geom_ids_np, dtype=int)
    threshold = float(env._config.get("reference_foot_lock_height", 0.035))
    active: int | None = None
    previous_contact = np.zeros(len(feet), dtype=bool)
    previous_local = np.zeros((len(feet), 2))
    for frame in range(count):
        data.qpos[:] = env._init_q_np
        data.qpos[:3] = roots[frame]
        data.qpos[3:7] = quaternions[frame]
        data.qpos[env._actuator_qpos_indices_np] = joints[frame]
        mujoco.mj_forward(env._mj_model, data)
        heights = np.array([env._geom_min_z(data, int(foot)) for foot in feet])
        contact = heights <= threshold
        local = data.geom_xpos[feet, :2] - roots[frame, :2]
        if not np.any(contact):
            active = None
        elif active is None or not contact[active]:
            active = int(np.argmin(np.where(contact, heights, np.inf)))
        if frame:
            displacement = roots[frame, :2] - roots[frame - 1, :2]
            if active is not None and previous_contact[active]:
                displacement = previous_local[active] - local[active]
            locked[frame, :2] = locked[frame - 1, :2] + displacement
        previous_contact = contact
        previous_local = local.copy()
    locked[count:] = locked[count - 1]
    return locked


@contextmanager
def root_lock_ablation(
    skip_initial: bool = False, handoff: bool = False
) -> Iterator[None]:
    """Scope experimental changes to construction; restore methods on any exit."""
    original_align = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor
    original_lock = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet
    states: dict[int, dict[str, int]] = {}

    def align(env, *args, **kwargs):
        state = states.setdefault(id(env), {"passes": 0, "active": -1})
        if state["passes"] >= 2 or state["active"] != -1:
            raise ValueError("Ablation expects two non-nested alignment passes.")
        state["active"] = state["passes"]
        state["passes"] += 1
        try:
            return original_align(env, *args, **kwargs)
        finally:
            state["active"] = -1

    def lock(env, joints, roots, quaternions, count):
        state = states.get(id(env))
        if state is None or state["active"] not in (0, 1):
            raise ValueError("Root lock called outside the observed alignment pass.")
        if state["active"] == 0 and skip_initial:
            return np.asarray(roots, dtype=np.float32).copy()
        if state["active"] == 1 and handoff:
            return lock_with_contact_history(env, joints, roots, quaternions, count)
        return original_lock(env, joints, roots, quaternions, count)

    with (
        patch.object(
            BiomechanicsJoystickEnv, "_align_reference_roots_to_sim_floor", align
        ),
        patch.object(
            BiomechanicsJoystickEnv, "_lock_clip_root_xy_to_stance_feet", lock
        ),
    ):
        yield


def measure_variant(
    config: EnvConfig, skip_initial: bool, handoff: bool, run_replay: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure the complete constructed reference, including closure geometry."""
    with root_lock_ablation(skip_initial=skip_initial, handoff=handoff):
        env, metadata, stages = capture_retarget_stages(config)
    if "curve_preparation_error" in metadata:
        return {"preparation_rejected": metadata["curve_preparation_error"]}, stages
    count = int(env._bvh_reference_frame_counts[0])
    interval = float(env._bvh_reference_frame_times[0])
    duration = float(env._bvh_reference_motion_lengths[0])
    times = np.linspace(0, duration, (count - 1) * 4 + 1)
    qpos, qvel, _ = sample_reference_states(env, 0, times)
    contacts, _ = inspect_contacts(env, times, qpos, qvel, 4)
    dynamic_times = (np.arange((count - 1) * 4) + 0.5) * interval / 4
    states = sample_reference_states(env, 0, dynamic_times)
    result = {
        "experimental_overrides": {
            "skip_initial_lock": skip_initial,
            "handoff": handoff,
        },
        "reference_digest": reference_digest(env),
        "root_lock_passes": metadata["root_lock_passes"],
        "closure": env._reference_closure_diagnostics,
        "ik_warmstart": env._reference_ik_warmstart_diagnostics,
        "numerical_curve_checks": inspect_consistency(env, 0),
        "contacts": contacts,
        "collisions": inspect_collision_geometry(env._mj_model, qpos, qvel),
        "off_knot_collisions": inspect_collision_geometry(env._mj_model, *states[:2]),
        "dynamics": inspect_dynamics(env._mj_model, *states),
    }
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
    """Run a fixed-source 2x2 diagnostic experiment with unchanged v5 settings."""
    choices = {
        "baseline": (False, False),
        "skip_initial": (True, False),
        "handoff": (False, True),
        "skip_initial_handoff": (True, True),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=choices, default=list(choices))
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase7_20261012/ablation.json",
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
        recovery_physics_version="consistent_v5",
        reference_loop_closure="bounded",
        deepmimic_reward_mode="pure",
    )
    sources = (
        "biomechanics_env.py",
        "bvh_reference.py",
        "reference_ik.py",
        "reference_interpolation.py",
        "tools/check_root_lock_ablation.py",
        "tools/check_retarget_stages.py",
        "tools/check_reference_contacts.py",
        "tools/check_reference_curves.py",
        "tools/check_reference_collisions.py",
        "tools/recovery_dynamics.py",
        "tools/recovery_replay.py",
    )
    report = {
        "scope": "Diagnostic construction overrides; not a training configuration.",
        "config": asdict(config),
        "completed": False,
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
            "Support labels are height proxies; foot centers can rotate and slide.",
            "Exact inverse dynamics does not optimize contact forces.",
            "Native open-loop replay has no learned balance feedback.",
            "Same v5 config with overrides is diagnostic only; never resume a policy with it.",
        ],
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_stages = None
    for name in args.variants:
        print(f"Measuring {name}...", flush=True)
        result, stages = measure_variant(config, *choices[name], not args.skip_replay)
        if name == "baseline":
            baseline_stages = {stage["name"]: stage for stage in stages}
        if baseline_stages is not None:
            result["joint_pose_difference_from_baseline_max_rad"] = {
                stage["name"]: float(
                    np.max(
                        np.abs(
                            stage["joints"] - baseline_stages[stage["name"]]["joints"]
                        )
                    )
                )
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
                    }
                ),
                flush=True,
            )
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
