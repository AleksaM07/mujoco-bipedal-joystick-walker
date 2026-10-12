"""Observe stance-lock transitions and upstream pose motion without physics edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import PROJECT_ROOT, EnvConfig
from tools.check_recovery_reference import reference_digest
from tools.check_retarget_stages import capture_retarget_stages


def trace_lock_call(
    env: BiomechanicsJoystickEnv,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    count: int,
    locked: np.ndarray,
    interval: float,
    contact_history: bool = False,
) -> dict[str, Any]:
    """Reconstruct support choices from geometry without reimplementing root updates.

    Root XY translation cannot change the horizontal-floor contact-height test.
    The unchanged output is compared against the stationary-center recurrence.
    No contact force or ground-truth stance claim is made from these labels.
    """
    if count < 3 or interval <= 0 or not np.isfinite(interval):
        raise ValueError("Need at least three frames and a finite positive interval.")
    model = env._mj_model
    feet = np.asarray(env._foot_geom_ids_np, dtype=int)
    data = mujoco.MjData(model)
    threshold = float(env._config.get("reference_foot_lock_height", 0.035))
    rows = []
    active: int | None = None
    local_feet = []
    previous_contact = np.zeros(len(feet), dtype=bool)
    for frame in range(count):
        data.qpos[:] = env._init_q_np
        data.qpos[:3] = roots[frame]
        data.qpos[3:7] = quaternions[frame]
        data.qpos[env._actuator_qpos_indices_np] = joints[frame]
        mujoco.mj_forward(model, data)
        heights = np.array(
            [env._geom_min_z(data, int(foot)) for foot in feet], dtype=np.float32
        )
        contact = heights <= threshold
        previous = active
        if not np.any(contact):
            active = None
            event = "no_contact"
        elif active is None or not contact[active]:
            active = int(np.argmin(np.where(contact, heights, np.inf)))
            event = "initialize" if previous is None else "switch"
        else:
            event = "continue"
        relative = data.geom_xpos[feet, :2] - roots[frame, :2]
        local_feet.append(relative.copy())
        rows.append(
            {
                "frame": frame,
                "event": event,
                "active_foot": None
                if active is None
                else model.geom(int(feet[active])).name,
                "active_foot_index": active,
                "selected_foot_previously_in_contact": (
                    active is not None and bool(previous_contact[active])
                ),
                "foot_lowest_z_m": heights.tolist(),
                "foot_xy_relative_to_root_m": relative.tolist(),
                "input_root_xyz_m": roots[frame].tolist(),
                "output_root_xyz_m": locked[frame].tolist(),
                "joint_change_from_previous_max_rad": (
                    None
                    if frame == 0
                    else float(np.max(np.abs(joints[frame] - joints[frame - 1])))
                ),
            }
        )
        previous_contact = contact
    local_feet = np.asarray(local_feet)
    input_velocity = np.diff(roots[:count, :2], axis=0) / interval
    output_velocity = np.diff(locked[:count, :2], axis=0) / interval
    acceleration = np.diff(output_velocity, axis=0) / interval
    recurrence_errors = []
    for frame, row in enumerate(rows):
        active = row["active_foot_index"]
        row["output_xy_nodal_velocity_m_s"] = (
            None if frame == 0 else output_velocity[frame - 1].tolist()
        )
        row["input_xy_nodal_velocity_m_s"] = (
            None if frame == 0 else input_velocity[frame - 1].tolist()
        )
        if 0 < frame < count - 1:
            row["output_xy_second_difference_m_s2"] = acceleration[frame - 1].tolist()
        if frame and (
            row["event"] == "continue"
            or (contact_history and row["selected_foot_previously_in_contact"])
        ):
            expected = (
                -(local_feet[frame, active] - local_feet[frame - 1, active]) / interval
            )
            error = np.linalg.norm(output_velocity[frame - 1] - expected)
            recurrence_errors.append(error)
            row["stationary_center_velocity_m_s"] = expected.tolist()
            row["stationary_center_recurrence_error_m_s"] = float(error)
    worst = int(np.argmax(np.linalg.norm(acceleration, axis=1))) + 1
    return {
        "frames": count,
        "contact_history": contact_history,
        "interval_seconds": interval,
        "contact_height_threshold_m": threshold,
        "max_stationary_center_recurrence_error_m_s": max(
            recurrence_errors, default=0.0
        ),
        "support_transitions": [
            row["frame"] for row in rows if row["event"] != "continue"
        ],
        "worst_xy_second_difference_center_frame": worst,
        "worst_xy_second_difference_m_s2": float(
            np.linalg.norm(acceleration[worst - 1])
        ),
        "rows": rows,
    }


def main() -> None:
    """Save both actual locking passes of the fixed selected reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=["consistent_v4", "consistent_v5", "consistent_v6"],
        default="consistent_v4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "analysis_outputs/recovery_phase6_20261012/root_lock_trace.json",
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
        recovery_physics_version=args.version,
        reference_loop_closure="bounded",
        deepmimic_reward_mode="pure",
    )
    original = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet
    calls = []

    def observed(env, joints, roots, quaternions, count, **kwargs):
        inputs = tuple(
            np.array(value, copy=True) for value in (joints, roots, quaternions)
        )
        result = original(env, joints, roots, quaternions, count, **kwargs)
        calls.append(
            trace_lock_call(
                env,
                *inputs,
                count,
                result,
                float(env._bvh_reference_frame_times[0]),
                contact_history=bool(kwargs.get("contact_history", False)),
            )
        )
        return result

    with patch.object(
        BiomechanicsJoystickEnv, "_lock_clip_root_xy_to_stance_feet", observed
    ):
        env, _, _ = capture_retarget_stages(config)
    output = {
        "scope": "Observation only; no support/IK/trajectory/controller changes.",
        "config": asdict(config),
        "reference_digest": reference_digest(env),
        "ik_warmstart": getattr(env, "_reference_ik_warmstart_diagnostics", []),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "biomechanics_env.py",
                "bvh_reference.py",
                "reference_ik.py",
                "reference_root.py",
                "tools/check_retarget_stages.py",
                "tools/check_root_lock_transitions.py",
            )
        },
        "passes": calls,
        "limitations": [
            "Lowest-height labels are not force-based stance.",
            "Nodal second differences are not analytic interpolant accelerations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {key: value for key, value in call.items() if key != "rows"}
                for call in calls
            ],
            indent=2,
        )
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
