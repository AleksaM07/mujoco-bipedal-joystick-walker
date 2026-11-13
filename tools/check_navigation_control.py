"""Audit native MuJoCo controls before integrating a new simulation backend."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.mujoco_control import hinge_actuator_forces, validate_hinge_actuators
from tools.check_mimickit_model import DEFAULT_XML, sample_poses


def force_probe(model: mujoco.MjModel, count: int, seed: int) -> dict[str, Any]:
    """Compare explicit control with MuJoCo's independent force computation."""
    random = np.random.default_rng(seed)
    poses = sample_poses(model, count, seed)
    data = mujoco.MjData(model)
    active_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    passive_dofs = np.setdiff1d(model.jnt_dofadr[1:], active_dofs)
    errors = []
    passive_peak = 0.0
    saturated = 0
    for pose in poses:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = pose
        data.qvel[:] = random.uniform(-3, 3, model.nv)
        center = model.actuator_ctrlrange.mean(axis=1)
        half = np.diff(model.actuator_ctrlrange, axis=1)[:, 0] / 2
        data.ctrl[:] = center + random.uniform(-2, 2, model.nu) * half
        mujoco.mj_forward(model, data)
        if not all(
            np.isfinite(values).all()
            for values in (data.actuator_force, data.qfrc_actuator, data.qfrc_passive)
        ):
            raise ValueError("Non-finite native force computation")
        force, torque = hinge_actuator_forces(model, data.qpos, data.qvel, data.ctrl)
        errors.append(
            max(
                np.max(abs(force - data.actuator_force)),
                np.max(abs(torque - data.qfrc_actuator)),
            )
        )
        passive_peak = max(
            passive_peak, float(abs(data.qfrc_passive[passive_dofs]).max(initial=0))
        )
        saturated += int(
            np.count_nonzero(
                model.actuator_forcelimited
                & (
                    np.isclose(data.actuator_force, model.actuator_forcerange[:, 0])
                    | np.isclose(data.actuator_force, model.actuator_forcerange[:, 1])
                )
            )
        )
        if np.any(torque[passive_dofs] != 0):
            raise AssertionError("Actuator bridge powered a passive joint")
    return {
        "states": len(poses),
        "max_force_or_torque_error": float(max(errors)),
        "passive_joint_force_peak_nm": passive_peak,
        "saturated_actuator_samples": saturated,
        "passed": bool(max(errors) < 1e-9),
    }


def rollout_probe(
    model: mujoco.MjModel, initial_pose: np.ndarray, steps: int
) -> dict[str, Any]:
    """Compare native servos with torques recomputed at every physics substep.

    All models share masses, contacts, constraints and passive forces. A third
    variant holds torque for four substeps to detect an incorrect update rate.
    These are open-loop control traces, not locomotion-success evaluations.
    """
    if np.any(model.actuator_biasprm[:, 2]):
        raise ValueError("Trajectory probe requires zero actuator velocity bias")
    torque_model = copy.copy(model)
    torque_model.actuator_gainprm[:] = 0
    torque_model.actuator_biasprm[:] = 0
    original = mujoco.MjData(model)
    bridged = mujoco.MjData(torque_model)
    held = mujoco.MjData(torque_model)
    for data in (original, bridged, held):
        data.qpos[:] = initial_pose
        mujoco.mj_normalizeQuat(model, data.qpos)
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    centers = np.clip(initial_pose[indices], *model.actuator_ctrlrange.T)
    phase = np.arange(model.nu) * 0.37
    max_position_error = 0.0
    max_velocity_error = 0.0
    max_held_position_error = 0.0
    contact_steps = 0
    for step in range(steps):
        if step % 4 == 0:
            targets = centers + 0.04 * np.sin(phase + step * model.opt.timestep * 2)
            held.qfrc_applied[:] = hinge_actuator_forces(
                model, held.qpos, held.qvel, targets
            )[1]
        original.ctrl[:] = targets
        bridged.qfrc_applied[:] = hinge_actuator_forces(
            model, bridged.qpos, bridged.qvel, targets
        )[1]
        mujoco.mj_step(model, original)
        mujoco.mj_step(torque_model, bridged)
        mujoco.mj_step(torque_model, held)
        for data in (original, bridged, held):
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError("Non-finite rollout")
            if abs(data.time - (step + 1) * model.opt.timestep) > 1e-8:
                raise ValueError("Simulator reset during rollout")
        max_position_error = max(
            max_position_error, float(abs(original.qpos - bridged.qpos).max())
        )
        max_velocity_error = max(
            max_velocity_error, float(abs(original.qvel - bridged.qvel).max())
        )
        max_held_position_error = max(
            max_held_position_error, float(abs(original.qpos - held.qpos).max())
        )
        contact_steps += int(original.ncon > 0)
    return {
        "steps": steps,
        "seconds": steps * model.opt.timestep,
        "steps_with_contacts": contact_steps,
        "max_qpos_component_error": max_position_error,
        "max_qvel_component_error": max_velocity_error,
        "held_torque_max_qpos_component_error": max_held_position_error,
        "passed": bool(max_position_error < 1e-6 and max_velocity_error < 1e-6),
    }


def main() -> None:
    """Save actuator/passive contracts and measured native force/rollout parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 4:
        raise ValueError("At least four physics steps are required")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    validate_hinge_actuators(model)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(
        model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE
    ):
        raise ValueError("Probe requires a free root followed only by hinge joints")
    xml_hash = hashlib.sha256(args.xml.read_bytes()).hexdigest()
    poses = {"neutral": model.qpos0.copy()}
    reference_hash = None
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            if json.loads(str(reference["metadata"]))["xml_sha256"] != xml_hash:
                raise ValueError("Reference XML differs from control XML")
            poses["walk_start"] = reference["qpos"][0].copy()
            poses["walk_middle"] = reference["qpos"][len(reference["qpos"]) // 2].copy()
        reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    active_joints = model.actuator_trnid[:, 0]
    passive_joints = np.setdiff1d(np.arange(1, model.njnt), active_joints)
    actuators = []
    for index, joint in enumerate(active_joints):
        actuators.append(
            {
                "actuator": model.actuator(index).name,
                "joint": model.joint(joint).name,
                "qpos_index": int(model.jnt_qposadr[joint]),
                "dof_index": int(model.jnt_dofadr[joint]),
                "gear": float(model.actuator_gear[index, 0]),
                "gain": float(model.actuator_gainprm[index, 0]),
                "bias": model.actuator_biasprm[index, :3].tolist(),
                "ctrlrange": model.actuator_ctrlrange[index].tolist(),
                "forcerange": model.actuator_forcerange[index].tolist(),
                "passive_stiffness": float(model.jnt_stiffness[joint]),
                "passive_damping": float(model.dof_damping[model.jnt_dofadr[joint]]),
            }
        )
    forces = force_probe(model, args.samples, args.seed)
    rollouts = {
        name: rollout_probe(model, pose, args.steps) for name, pose in poses.items()
    }
    report = {
        "scope": "native_mujoco_force_and_substep_parity_not_newton_or_policy_transfer",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "skills/mujoco_control.py",
            )
        },
        "mujoco_version": mujoco.__version__,
        "xml_sha256": xml_hash,
        "reference_sha256": reference_hash,
        "seed": args.seed,
        "timestep": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "actuators": actuators,
        "passive_joints": [
            {
                "joint": model.joint(joint).name,
                "stiffness": float(model.jnt_stiffness[joint]),
                "spring_reference": float(model.qpos_spring[model.jnt_qposadr[joint]]),
                "damping": float(model.dof_damping[model.jnt_dofadr[joint]]),
            }
            for joint in passive_joints
        ],
        "active_joints_with_zero_passive_stiffness": int(
            np.count_nonzero(model.jnt_stiffness[active_joints] == 0)
        ),
        "force_probe": forces,
        "rollout_probes": rollouts,
        "passed": bool(
            forces["passed"] and all(case["passed"] for case in rollouts.values())
        ),
        "remaining": [
            "engine import/body/hinge mapping",
            "new engine force and integration parity",
            "normalized policy action mapping",
            "reference interpolation/seam/velocity validation",
            "commanded training and independent task evaluation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "force_probe",
                    "rollout_probes",
                    "active_joints_with_zero_passive_stiffness",
                )
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
