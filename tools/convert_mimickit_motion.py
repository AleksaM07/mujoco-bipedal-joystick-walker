"""Convert exported reference knots and verify them through upstream MotionLib."""

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.mimickit_adapter import (
    MIMICKIT_COMMIT,
    build_hinge_kinematics,
    interpolate_quaternions,
    load_mimickit,
)
from tools.check_mimickit_model import DEFAULT_XML, compare_poses


def main() -> None:
    """Write a fresh motion package and verify sampled source-frame coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--correct-small-angle-slerp", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    upstream = load_mimickit(args.mimickit)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    adapter = build_hinge_kinematics(model, upstream)
    with np.load(args.reference, allow_pickle=False) as source:
        poses = source["qpos"]
        metadata = json.loads(str(source["metadata"]))
    if metadata["xml_sha256"] != hashlib.sha256(args.xml.read_bytes()).hexdigest():
        raise ValueError("Reference and physical XML hashes differ")
    frame_dt = float(metadata["frame_dt"])
    if not np.isfinite(frame_dt) or frame_dt <= 0 or len(poses) < 2:
        raise ValueError("Reference needs positive frame_dt and at least two poses")
    root_pos, root_rot, angles = adapter.encode_qpos(poses)
    torch_util = import_module("util.torch_util")
    if args.correct_small_angle_slerp:
        torch_util.slerp = interpolate_quaternions
    frames = torch.cat((root_pos, torch_util.quat_to_exp_map(root_rot), angles), dim=-1)
    motion_module = import_module("anim.motion")
    motion = motion_module.Motion(
        loop_mode=motion_module.LoopMode(metadata["loop_mode"]),
        fps=1.0 / frame_dt,
        frames=frames.numpy(),
    )
    args.output_dir.mkdir(parents=True)
    motion_path = args.output_dir / "natural_walk.pkl"
    motion.save(str(motion_path))
    library = import_module("anim.motion_lib").MotionLib(
        str(motion_path.resolve()), adapter.skeleton, "cpu"
    )
    # A wrapping clip's final knot is the beginning of the next cycle; sample
    # interior source frames here, leaving seam/interpolation validation separate.
    times = torch.arange(len(poses) - 1, dtype=torch.float32) * frame_dt
    sampled = library.calc_motion_frame(
        torch.zeros_like(times, dtype=torch.int64), times
    )
    reconstructed = poses[:-1].copy()
    reconstructed[:, :3] = sampled[0].numpy()
    reconstructed[:, 3:7] = sampled[1].numpy()[:, [3, 0, 1, 2]]
    reconstructed[:, adapter.qpos_indices] = (
        library.joint_rot_to_dof(sampled[4]).numpy() + adapter.reference_angles
    )
    expected_pos, expected_rot = adapter.forward(poses[:-1])
    actual_pos, actual_rot = adapter.forward(reconstructed)
    position_error = float(np.linalg.norm(actual_pos - expected_pos, axis=-1).max())
    rotation_error = float(
        np.minimum(
            np.linalg.norm(actual_rot - expected_rot, axis=-1),
            np.linalg.norm(actual_rot + expected_rot, axis=-1),
        ).max()
    )
    fk_result = compare_poses(model, adapter, poses)
    velocities_finite = all(
        torch.isfinite(sampled[index]).all().item() for index in (2, 3, 5)
    )
    report = {
        "mimickit_commit": MIMICKIT_COMMIT,
        "small_angle_slerp_correction": args.correct_small_angle_slerp,
        "versions": {"mujoco": mujoco.__version__, "torch": torch.__version__},
        "source": metadata,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "reference_fk": fk_result,
        "sampled_frames": len(times),
        "motionlib_max_position_error_m": position_error,
        "motionlib_max_quaternion_chord_error": rotation_error,
        "motionlib_velocities_finite": velocities_finite,
        "passed": bool(
            fk_result["passed"]
            and position_error < 1e-5
            and rotation_error < 1e-5
            and velocities_finite
        ),
        "scope": "source_knots_only_not_velocity_parity_or_physical_tracking",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
