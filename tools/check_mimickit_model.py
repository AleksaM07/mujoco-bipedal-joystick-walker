"""Compare actual MimicKit forward kinematics with native MuJoCo, without RL."""

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.mimickit_adapter import (
    MIMICKIT_COMMIT,
    HingeKinematics,
    build_hinge_kinematics,
    load_mimickit,
)

DEFAULT_XML = (
    PROJECT_ROOT
    / "prezentacija/5th_success_final_natural_walk/generated_models"
    / "human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
)


def sample_poses(model: mujoco.MjModel, count: int, seed: int) -> np.ndarray:
    """Sample all hinges, including passive ones, with varied world root poses."""
    if count < 1:
        raise ValueError("At least one random pose is required")
    random = np.random.default_rng(seed)
    poses = np.tile(model.qpos0, (count + 1, 1))
    poses[1:, :3] += random.uniform(-2.0, 2.0, (count, 3))
    rotations = random.normal(size=(count, 4))
    poses[1:, 3:7] = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    for joint_id in range(1, model.njnt):
        low, high = (
            model.jnt_range[joint_id]
            if model.jnt_limited[joint_id]
            else (-np.pi, np.pi)
        )
        poses[1:, model.jnt_qposadr[joint_id]] = random.uniform(low, high, count)
    return poses


def compare_poses(
    model: mujoco.MjModel, adapter: HingeKinematics, poses: np.ndarray
) -> dict[str, Any]:
    """Use independent native MuJoCo FK as the oracle for every physical body."""
    positions, rotations = adapter.forward(poses)
    data = mujoco.MjData(model)
    expected_positions = []
    expected_rotations = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_normalizeQuat(model, data.qpos)
        mujoco.mj_forward(model, data)
        expected_positions.append(data.xpos[1:].copy())
        expected_rotations.append(data.xquat[1:, [1, 2, 3, 0]].copy())
    expected_positions = np.asarray(expected_positions)
    expected_rotations = np.asarray(expected_rotations)
    position_error = np.linalg.norm(positions - expected_positions, axis=-1)
    rotations = rotations.astype(np.float64)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    # q and -q describe the same orientation. Chord distance is stable near zero.
    chord = np.minimum(
        np.linalg.norm(rotations - expected_rotations, axis=-1),
        np.linalg.norm(rotations + expected_rotations, axis=-1),
    )
    angle_error = 4 * np.arcsin(np.clip(chord / 2, 0, 1))
    return {
        "pose_count": len(poses),
        "max_position_error_m": float(position_error.max()),
        "max_rotation_error_rad": float(angle_error.max()),
        "passed": bool(position_error.max() < 1e-5 and angle_error.max() < 1e-5),
    }


def actuator_mapping(model: mujoco.MjModel, adapter: HingeKinematics) -> list[dict]:
    """Record original actuator order separately from all kinematic hinges."""
    result = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Probe expects joint actuators")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        match = np.flatnonzero(adapter.joint_ids == joint_id)
        if len(match) != 1:
            raise ValueError("Actuator joint has no unique kinematic coordinate")
        result.append(
            {
                "actuator": model.actuator(actuator_id).name,
                "joint": model.joint(joint_id).name,
                "qpos_index": int(model.jnt_qposadr[joint_id]),
                "kinematic_dof_index": int(match[0]),
                "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
                "gear": model.actuator_gear[actuator_id].tolist(),
            }
        )
    return result


def main() -> None:
    """Write a reproducible report; fail if any kinematics comparison fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh report path: {args.output}")
    upstream = load_mimickit(args.mimickit)
    import torch

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    adapter = build_hinge_kinematics(model, upstream)
    original = import_module("anim.mjcf_char_model").MJCFCharModel(device="cpu")
    try:
        original.load(str(args.xml.resolve()))
        original_import = {"passed": True, "error": None}
    except AssertionError as error:
        original_import = {"passed": False, "error": str(error)}

    cases = {
        "neutral_and_random": compare_poses(
            model, adapter, sample_poses(model, args.samples, args.seed)
        )
    }
    for path in args.replay:
        with np.load(path, allow_pickle=False) as replay:
            poses = np.asarray(replay["qpos"], dtype=np.float64)
        case = compare_poses(model, adapter, poses)
        case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases[str(path)] = case

    report = {
        "scope": "kinematics_only_not_physics_or_policy_compatibility",
        "mimickit_commit": MIMICKIT_COMMIT,
        "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        "versions": {
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "seed": args.seed,
        "physical_bodies": model.nbody - 1,
        "kinematic_nodes": adapter.skeleton.get_num_joints(),
        "hinge_coordinates": adapter.skeleton.get_dof_size(),
        "actuators": model.nu,
        "mass_kg": float(model.body_mass.sum()),
        "unmodified_upstream_import": original_import,
        "actuator_mapping": actuator_mapping(model, adapter),
        "cases": cases,
        "passed": all(case["passed"] for case in cases.values()),
        "remaining_gates": [
            "simulator and PD/action mapping",
            "walking motion conversion",
            "commanded policy training and physical evaluation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"passed": report["passed"], "cases": cases, "output": str(args.output)},
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
