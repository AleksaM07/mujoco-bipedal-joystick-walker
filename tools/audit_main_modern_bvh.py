"""Audit or preview the modern BVH prior on main's robot (not a policy rollout).

Run from the repository root: python -m tools.audit_main_modern_bvh --preview
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bvh_retargeting import BvhReferenceRetargeter
from config import EnvConfig
from train import make_environment

DEFAULT_REFERENCE = "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"


def audit_reference(env: Any) -> dict[str, Any]:
    """Measure timing, limits, seams, penetration, slip and action reachability."""
    references = env._bvh_reference_batch
    retargeter = BvhReferenceRetargeter(
        env.mj_model, np.asarray(env._init_q), env._config
    )
    data = mujoco.MjData(env.mj_model)
    default_ctrl = np.asarray(env._default_ctrl)
    action_scale = np.asarray(env._action_scale)
    reachable_lower = default_ctrl - action_scale
    reachable_upper = default_ctrl + action_scale
    clips = []
    for clip_id, frame_count in enumerate(references.frame_counts):
        frame_count = int(frame_count)
        targets = references.qpos_targets[clip_id, :frame_count]
        velocities = references.qvel_targets[clip_id, :frame_count]
        dt = float(references.frame_times[clip_id])
        derivative = np.gradient(targets, dt, axis=0)
        derivative_error = float(np.max(np.abs(derivative[1:-1] - velocities[1:-1])))
        reachability = (targets >= reachable_lower - 1e-6) & (
            targets <= reachable_upper + 1e-6
        )
        limits = (targets >= retargeter._actuator_ctrl_lower_limits_np - 1e-6) & (
            targets <= retargeter._actuator_ctrl_upper_limits_np + 1e-6
        )
        foot_positions = []
        foot_heights = []
        for frame_id in range(frame_count):
            data.qpos[:] = np.asarray(env._init_q)
            data.qpos[:3] = references.root_pos_targets[clip_id, frame_id]
            data.qpos[3:7] = references.root_quat_targets[clip_id, frame_id]
            data.qpos[retargeter._actuator_qpos_indices_np] = targets[frame_id]
            mujoco.mj_forward(env.mj_model, data)
            foot_positions.append(
                data.geom_xpos[retargeter._foot_geom_ids_np, :2].copy()
            )
            foot_heights.append(
                [
                    retargeter._geom_min_z(data, int(geom_id))
                    for geom_id in retargeter._foot_geom_ids_np
                ]
            )
        heights = np.asarray(foot_heights)
        stance_pairs = (heights[:-1] <= 0.035) & (heights[1:] <= 0.035)
        foot_speeds = (
            np.linalg.norm(np.diff(np.asarray(foot_positions), axis=0), axis=-1) / dt
        )
        stance_speeds = foot_speeds[stance_pairs]
        unreachable_joints = [
            name
            for index, name in enumerate(env._actuator_joint_names)
            if not np.all(reachability[:, index])
        ]
        clips.append(
            {
                "clip_id": clip_id,
                "source": str(references.source_paths[clip_id]),
                "source_frames": [
                    int(references.source_start_frames[clip_id]),
                    int(references.source_end_frames[clip_id]),
                ],
                "frames": frame_count,
                "duration_seconds": (frame_count - 1) * dt,
                "loop_mode": int(references.loop_modes[clip_id]),
                "interior_dqdt_qvel_max_error": derivative_error,
                "endpoint_pose_gap_radians": float(
                    np.max(np.abs(targets[-1] - targets[0]))
                ),
                "control_limit_violations": int(np.count_nonzero(~limits)),
                "unreachable_target_fraction": float(np.mean(~reachability)),
                "unreachable_joints": unreachable_joints,
                "minimum_foot_height_m": float(np.min(heights)),
                "stance_slip_p95_mps": (
                    float(np.percentile(stance_speeds, 95))
                    if stance_speeds.size
                    else None
                ),
                "stance_slip_max_mps": (
                    float(np.max(stance_speeds)) if stance_speeds.size else None
                ),
            }
        )
    return {
        "main_source": "33eaa1f",
        "reference_source": "cd6557c",
        "xml": str(env._xml_path),
        "reward_mask": [
            name
            for name, enabled in zip(
                env._actuator_joint_names, np.asarray(env._reference_gait_mask)
            )
            if enabled
        ],
        "notes": [
            "This is a kinematic audit, not evidence of dynamically stable walking.",
            "Main's reset, action scales and reward mask are intentionally unchanged.",
            "Floor alignment allows the inherited 5 mm contact preload.",
        ],
        "clips": clips,
    }


def preview_reference(env: Any, clip_id: int) -> None:
    """Show kinematic reference playback in the native MuJoCo viewer."""
    import mujoco.viewer

    references = env._bvh_reference_batch
    if not 0 <= clip_id < len(references.frame_counts):
        raise ValueError("clip-id is outside the loaded reference batch")
    data = mujoco.MjData(env.mj_model)
    qpos_indices = np.asarray(env._actuator_qpos_indices)
    frame_count = int(references.frame_counts[clip_id])
    frame_time = float(references.frame_times[clip_id])
    frame_id = 0
    print("Kinematic preview only: poses are assigned, not tracked by a policy.")
    with mujoco.viewer.launch_passive(env.mj_model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        while viewer.is_running():
            start = time.perf_counter()
            data.qpos[:] = np.asarray(env._init_q)
            data.qpos[:3] = references.root_pos_targets[clip_id, frame_id]
            data.qpos[3:7] = references.root_quat_targets[clip_id, frame_id]
            data.qpos[qpos_indices] = references.qpos_targets[clip_id, frame_id]
            mujoco.mj_forward(env.mj_model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            frame_id = (frame_id + 1) % frame_count
            time.sleep(max(0.0, frame_time - (time.perf_counter() - start)))


def main() -> None:
    """Audit the exact modern reference consumed by the main-based env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", default=DEFAULT_REFERENCE)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--clip-id", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/main_modern_bvh_audit.json")
    )
    args = parser.parse_args()
    env = make_environment(
        EnvConfig(
            reference_gait="bvh",
            reference_pipeline="modern",
            reference_gait_file=args.reference_file,
            reference_motion_speed_scale=args.speed_scale,
            reference_target_observation=True,
        ),
        enable_erfi=False,
    )
    report = audit_reference(env)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.preview:
        preview_reference(env, args.clip_id)


if __name__ == "__main__":
    main()
