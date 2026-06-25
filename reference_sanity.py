import argparse
from pathlib import Path

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import expand_reference_gait_files


def summarize_range(name: str, values: np.ndarray) -> None:
    """Print compact min/max/span diagnostics for reference targets."""
    flat = values.reshape((-1, values.shape[-1]))
    mins = np.min(flat, axis=0)
    maxs = np.max(flat, axis=0)
    spans = maxs - mins
    print(
        f"{name}: min={np.round(mins, 4)} "
        f"max={np.round(maxs, 4)} span={np.round(spans, 4)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast diagnostics for BVH reference targets."
    )
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        action="append",
        default=None,
        help="BVH file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reference-gait-list",
        type=Path,
        action="append",
        default=None,
        help="Text file with one BVH path per line.",
    )
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument(
        "--fast-physics",
        action="store_true",
        help="Use sim_dt=0.01 instead of the accurate 0.005 setup.",
    )
    args = parser.parse_args()

    reference_files = expand_reference_gait_files(
        args.reference_gait_file,
        args.reference_gait_list,
    )
    if not reference_files:
        raise SystemExit(
            "Pass at least one --reference-gait-file or --reference-gait-list."
        )

    config_overrides = {
        "impl": "jax",
        "enable_erfi": False,
        "command_profile": "forward",
        "reference_gait": "bvh",
        "reference_gait_file": reference_files,
        "reference_target_observation": True,
    }
    if args.xml_path is not None:
        config_overrides["xml_path"] = str(args.xml_path)
    if not args.fast_physics:
        config_overrides["sim_dt"] = 0.005

    env = BiomechanicsJoystickEnv(config_overrides=config_overrides)
    qpos_targets = np.asarray(env._bvh_reference_qpos_targets)
    qvel_targets = np.asarray(env._bvh_reference_qvel_targets)
    frame_counts = np.asarray(env._bvh_reference_frame_counts)
    frame_times = np.asarray(env._bvh_reference_frame_times)
    active_mask = getattr(env, "_reference_gait_mask", None)
    if active_mask is None:
        active_mask = np.ones(qpos_targets.shape[-1], dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)

    print(f"xml={env.xml_path}")
    print(f"clips={qpos_targets.shape[0]} max_frames={qpos_targets.shape[1]}")
    print(f"frame_counts={frame_counts.tolist()}")
    print(f"frame_times={np.round(frame_times, 5).tolist()}")
    print(
        "active_reference_joints="
        f"{np.asarray(env._actuator_joint_names, dtype=object)[active_mask].tolist()}"
    )
    summarize_range("active_qpos", qpos_targets[:, :, active_mask])
    summarize_range("active_qvel", qvel_targets[:, :, active_mask])
    foot_targets = getattr(env, "_bvh_reference_foot_pos_targets", None)
    if foot_targets is not None:
        foot_targets = np.asarray(foot_targets)
        summarize_range("left_foot_local", foot_targets[:, :, 0, :])
        summarize_range("right_foot_local", foot_targets[:, :, 1, :])

        left_span = np.ptp(foot_targets[:, :, 0, :], axis=(0, 1))
        right_span = np.ptp(foot_targets[:, :, 1, :], axis=(0, 1))
        smallest_span = float(min(np.max(left_span), np.max(right_span)))
        if smallest_span < 0.02:
            print(
                "warning: foot target span is very small; this reference may be "
                "too weak or the BVH retargeting may be nearly static."
            )


if __name__ == "__main__":
    main()
