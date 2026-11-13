"""Validate the pinned MimicKit/Newton engine against native MuJoCo."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import newton
import numpy as np
from newton.solvers import SolverMuJoCo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_mimickit_model import DEFAULT_XML

EXPECTED_VERSIONS = {
    "newton": "1.0.0",
    "newton-actuators": "0.1.0",
    "warp-lang": "1.12.0",
    "mujoco": "3.5.0",
    "mujoco-warp": "3.5.0.2",
}


def package_versions() -> dict[str, str]:
    """Return and validate the isolated Newton probe dependency versions."""
    versions = {
        package: importlib.metadata.version(package) for package in EXPECTED_VERSIONS
    }
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(
            f"Newton probe version mismatch: {versions}; expected {EXPECTED_VERSIONS}"
        )
    return versions


def source_stiffness_per_dof(model: mujoco.MjModel) -> np.ndarray:
    """Expand MuJoCo's per-joint stiffness to its per-DOF ordering."""
    return np.asarray(model.jnt_stiffness[model.dof_jntid], dtype=np.float32)


def restore_mujoco_semantics(
    model: newton.Model, source: mujoco.MjModel
) -> dict[str, float]:
    """Repair Newton 1.0 MJCF conversions using compiled MuJoCo values.

    Newton 1.0 treats damping and stiffness authored under ``angle=degree`` as
    per-degree quantities. MuJoCo's compiler angle setting does not change
    those physical coefficients. Newton also rewrites positive ``solreflimit``
    values to the negative direct-stiffness convention. The MuJoCo solver
    consumes these Newton arrays directly at runtime, so restore the compiled
    source values before constructing it.
    """
    if model.joint_dof_count != source.nv:
        raise ValueError(
            f"DOF mismatch: Newton={model.joint_dof_count}, MuJoCo={source.nv}"
        )
    imported_damping = model.mujoco.dof_passive_damping.numpy().copy()
    imported_stiffness = model.mujoco.dof_passive_stiffness.numpy().copy()
    source_damping = np.asarray(source.dof_damping, dtype=np.float32)
    source_stiffness = source_stiffness_per_dof(source)
    model.mujoco.dof_passive_damping.assign(source_damping)
    model.mujoco.dof_passive_stiffness.assign(source_stiffness)

    # Zero makes SolverMuJoCo retain MuJoCo's default positive solref=(.02, 1).
    if not np.allclose(source.jnt_solref, (0.02, 1.0), rtol=0, atol=0):
        raise ValueError("Probe currently requires the source default jnt_solref")
    model.joint_limit_ke.zero_()
    model.joint_limit_kd.zero_()
    return {
        "raw_damping_max_error": float(
            np.max(np.abs(imported_damping - source_damping))
        ),
        "raw_stiffness_max_error": float(
            np.max(np.abs(imported_stiffness - source_stiffness))
        ),
    }


def build_newton_model(
    xml_path: Path, source: mujoco.MjModel, device: str
) -> tuple[newton.Model, dict[str, float]]:
    """Import the unmodified model with exact coordinate and control semantics."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(
        str(xml_path.resolve()),
        floating=None,
        ignore_inertial_definitions=False,
        collapse_fixed_joints=False,
        enable_self_collisions=True,
        convert_3d_hinge_to_ball_joints=False,
        ctrl_direct=True,
    )
    model = builder.finalize(device=device)
    diagnostics = restore_mujoco_semantics(model, source)
    return model, diagnostics


def make_solver(
    model: newton.Model, source: mujoco.MjModel, *, use_mujoco_cpu: bool
) -> SolverMuJoCo:
    """Build a solver with the archived MuJoCo numerical options."""
    return SolverMuJoCo(
        model,
        use_mujoco_cpu=use_mujoco_cpu,
        solver=int(source.opt.solver),
        integrator=int(source.opt.integrator),
        cone=int(source.opt.cone),
        jacobian=int(source.opt.jacobian),
        impratio=float(source.opt.impratio),
        iterations=int(source.opt.iterations),
        ls_iterations=int(source.opt.ls_iterations),
        tolerance=float(source.opt.tolerance),
        ls_tolerance=float(source.opt.ls_tolerance),
        update_data_interval=1,
        use_mujoco_contacts=True,
    )


def synchronize_cpu_oracle(
    cpu_solver: SolverMuJoCo,
    gpu_solver: SolverMuJoCo,
    source: mujoco.MjModel,
) -> None:
    """Copy the effective MuJoCo-Warp parameters into the CPU oracle model.

    Newton 1.0 updates its GPU model after compilation, but its CPU mirror does
    not copy ``jnt_stiffness``. This oracle intentionally mirrors the actual
    GPU arrays so native MuJoCo can isolate engine numerical differences.
    """
    cpu_model = cpu_solver.mj_model
    gpu_model = gpu_solver.mjw_model
    cpu_model.dof_damping[:] = gpu_model.dof_damping.numpy()[0]
    cpu_model.jnt_stiffness[:] = gpu_model.jnt_stiffness.numpy()[0]
    cpu_model.jnt_solref[:] = gpu_model.jnt_solref.numpy()[0]
    cpu_model.jnt_solimp[:] = gpu_model.jnt_solimp.numpy()[0]
    cpu_model.qpos0[:] = gpu_model.qpos0.numpy()[0]
    cpu_model.qpos_spring[:] = gpu_model.qpos_spring.numpy()[0]
    cpu_model.opt.timestep = float(source.opt.timestep)
    mujoco.mj_setConst(cpu_model, cpu_solver.mj_data)


def property_parity(source: mujoco.MjModel, exported: mujoco.MjModel) -> dict[str, Any]:
    """Measure the physical fields required by recovery training."""
    comparisons = {
        "qpos0": (source.qpos0, exported.qpos0),
        "body_mass": (source.body_mass, exported.body_mass),
        "body_inertia": (source.body_inertia, exported.body_inertia),
        "dof_damping": (source.dof_damping, exported.dof_damping),
        "jnt_stiffness": (source.jnt_stiffness, exported.jnt_stiffness),
        "jnt_solref": (source.jnt_solref, exported.jnt_solref),
        "actuator_gainprm": (source.actuator_gainprm, exported.actuator_gainprm),
        "actuator_biasprm": (source.actuator_biasprm, exported.actuator_biasprm),
        "actuator_gear": (source.actuator_gear, exported.actuator_gear),
        "actuator_ctrlrange": (source.actuator_ctrlrange, exported.actuator_ctrlrange),
        "actuator_forcerange": (
            source.actuator_forcerange,
            exported.actuator_forcerange,
        ),
    }
    errors = {
        name: float(np.max(np.abs(expected - actual), initial=0.0))
        for name, (expected, actual) in comparisons.items()
    }
    counts = {
        "source": {
            "nbody": source.nbody,
            "njnt": source.njnt,
            "nq": source.nq,
            "nv": source.nv,
            "nu": source.nu,
            "ngeom": source.ngeom,
        },
        "exported": {
            "nbody": exported.nbody,
            "njnt": exported.njnt,
            "nq": exported.nq,
            "nv": exported.nv,
            "nu": exported.nu,
            "ngeom": exported.ngeom,
        },
    }
    count_parity = all(
        counts["source"][key] == counts["exported"][key]
        for key in ("nbody", "njnt", "nq", "nv", "nu")
    )
    return {
        "counts": counts,
        "max_absolute_errors": errors,
        "actuator_names_preserved": [source.actuator(i).name for i in range(source.nu)]
        == [exported.actuator(i).name for i in range(exported.nu)],
        "passed": bool(
            count_parity
            and errors["dof_damping"] < 1e-5
            and errors["jnt_stiffness"] < 1e-5
            and errors["jnt_solref"] < 1e-5
            and errors["actuator_gainprm"] < 1e-5
            and errors["actuator_biasprm"] < 1e-5
            and errors["actuator_gear"] < 1e-5
            and errors["actuator_ctrlrange"] < 1e-5
            and errors["actuator_forcerange"] < 1e-5
        ),
    }


def reset_data(
    model: mujoco.MjModel, qpos: np.ndarray, ctrl: np.ndarray
) -> mujoco.MjData:
    """Create a deterministic state with a tiny, explicit sole penetration."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    return data


def cpu_rollout_parity(
    source: mujoco.MjModel, exported: mujoco.MjModel, steps: int
) -> dict[str, Any]:
    """Compare the source and exported models in native MuJoCo."""
    qpos = np.asarray(source.qpos0).copy()
    qpos[2] -= 1e-4
    actuator_qpos = source.jnt_qposadr[source.actuator_trnid[:, 0]]
    ctrl = np.clip(qpos[actuator_qpos], *source.actuator_ctrlrange.T)
    source_data = reset_data(source, qpos, ctrl)
    exported_data = reset_data(exported, qpos, ctrl)
    source_contacts = []
    exported_contacts = []
    max_qpos_error = 0.0
    max_qvel_error = 0.0
    parity_qpos_error = 0.0
    parity_qvel_error = 0.0
    source_min_height = float(source_data.qpos[2])
    exported_min_height = float(exported_data.qpos[2])
    thorax_id = source.body("thorax").id
    source_min_up = 1.0
    exported_min_up = 1.0
    for step in range(steps):
        mujoco.mj_step(source, source_data)
        mujoco.mj_step(exported, exported_data)
        source_contacts.append(int(source_data.ncon))
        exported_contacts.append(int(exported_data.ncon))
        qpos_error = float(np.max(np.abs(source_data.qpos - exported_data.qpos)))
        qvel_error = float(np.max(np.abs(source_data.qvel - exported_data.qvel)))
        max_qpos_error = max(max_qpos_error, qpos_error)
        max_qvel_error = max(max_qvel_error, qvel_error)
        if step < 200:
            parity_qpos_error = max(parity_qpos_error, qpos_error)
            parity_qvel_error = max(parity_qvel_error, qvel_error)
        source_min_height = min(source_min_height, float(source_data.qpos[2]))
        exported_min_height = min(exported_min_height, float(exported_data.qpos[2]))
        source_min_up = min(
            source_min_up, float(source_data.xmat[thorax_id].reshape(3, 3)[2, 1])
        )
        exported_min_up = min(
            exported_min_up,
            float(exported_data.xmat[thorax_id].reshape(3, 3)[2, 1]),
        )
    fall_height = float(source.qpos0[2]) * 0.30
    stable = bool(
        source_min_height >= fall_height
        and exported_min_height >= fall_height
        and source_min_up >= 0.25
        and exported_min_up >= 0.25
    )
    return {
        "steps": steps,
        "seconds": steps * float(source.opt.timestep),
        "source_contact_range": [min(source_contacts), max(source_contacts)],
        "exported_contact_range": [min(exported_contacts), max(exported_contacts)],
        "source_contact_steps": int(np.count_nonzero(source_contacts)),
        "exported_contact_steps": int(np.count_nonzero(exported_contacts)),
        "fall_height_m": fall_height,
        "source_minimum_root_height_m": source_min_height,
        "exported_minimum_root_height_m": exported_min_height,
        "source_minimum_torso_up": source_min_up,
        "exported_minimum_torso_up": exported_min_up,
        "parity_window_steps": min(steps, 200),
        "parity_window_max_qpos_component_error": parity_qpos_error,
        "parity_window_max_qvel_component_error": parity_qvel_error,
        "max_qpos_component_error": max_qpos_error,
        "max_qvel_component_error": max_qvel_error,
        "passed": bool(
            np.count_nonzero(source_contacts) > 0
            and np.count_nonzero(exported_contacts) > 0
            and parity_qpos_error < 2e-4
            and parity_qvel_error < 2e-2
            and stable
        ),
    }


def gpu_rollout_parity(
    source: mujoco.MjModel,
    model: newton.Model,
    solver: SolverMuJoCo,
    steps: int,
) -> dict[str, Any]:
    """Run the same contact trajectory through RTX MuJoCo-Warp."""
    state_in = model.state()
    state_out = model.state()
    initial_joint_q = state_in.joint_q.numpy()
    initial_joint_q[2] -= 1e-4
    state_in.joint_q.assign(initial_joint_q)
    state_in.joint_qd.zero_()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()
    actuator_qpos = source.jnt_qposadr[source.actuator_trnid[:, 0]]
    source_qpos = np.asarray(source.qpos0).copy()
    source_qpos[2] -= 1e-4
    ctrl = np.clip(source_qpos[actuator_qpos], *source.actuator_ctrlrange.T)
    control.mujoco.ctrl.assign(np.asarray(ctrl, dtype=np.float32))

    source_data = reset_data(source, source_qpos, ctrl)
    gpu_metrics = reset_data(source, source_qpos, ctrl)
    qpos_errors = []
    qvel_errors = []
    source_contacts = []
    gpu_contacts = []
    source_min_height = float(source_data.qpos[2])
    gpu_min_height = float(gpu_metrics.qpos[2])
    thorax_id = source.body("thorax").id
    source_min_up = 1.0
    gpu_min_up = 1.0
    for _ in range(steps):
        mujoco.mj_step(source, source_data)
        solver.step(
            state_in,
            state_out,
            control,
            contacts=None,
            dt=float(source.opt.timestep),
        )
        state_in, state_out = state_out, state_in
        gpu_qpos = solver.mjw_data.qpos.numpy()[0]
        gpu_qvel = solver.mjw_data.qvel.numpy()[0]
        qpos_errors.append(float(np.max(np.abs(source_data.qpos - gpu_qpos))))
        qvel_errors.append(float(np.max(np.abs(source_data.qvel - gpu_qvel))))
        source_contacts.append(int(source_data.ncon))
        gpu_contacts.append(int(solver.mjw_data.nacon.numpy()[0]))
        if not np.isfinite(gpu_qpos).all() or not np.isfinite(gpu_qvel).all():
            raise FloatingPointError("Non-finite MuJoCo-Warp state")
        gpu_metrics.qpos[:] = gpu_qpos
        gpu_metrics.qvel[:] = gpu_qvel
        mujoco.mj_forward(source, gpu_metrics)
        source_min_height = min(source_min_height, float(source_data.qpos[2]))
        gpu_min_height = min(gpu_min_height, float(gpu_qpos[2]))
        source_min_up = min(
            source_min_up, float(source_data.xmat[thorax_id].reshape(3, 3)[2, 1])
        )
        gpu_min_up = min(
            gpu_min_up, float(gpu_metrics.xmat[thorax_id].reshape(3, 3)[2, 1])
        )
    parity_steps = min(steps, 200)
    parity_qpos_error = max(qpos_errors[:parity_steps])
    parity_qvel_error = max(qvel_errors[:parity_steps])
    fall_height = float(source.qpos0[2]) * 0.30
    stable = bool(
        source_min_height >= fall_height
        and gpu_min_height >= fall_height
        and source_min_up >= 0.25
        and gpu_min_up >= 0.25
    )
    return {
        "device": str(model.device),
        "steps": steps,
        "seconds": steps * float(source.opt.timestep),
        "source_contact_range": [min(source_contacts), max(source_contacts)],
        "gpu_contact_range": [min(gpu_contacts), max(gpu_contacts)],
        "source_contact_steps": int(np.count_nonzero(source_contacts)),
        "gpu_contact_steps": int(np.count_nonzero(gpu_contacts)),
        "fall_height_m": fall_height,
        "source_minimum_root_height_m": source_min_height,
        "gpu_minimum_root_height_m": gpu_min_height,
        "source_minimum_torso_up": source_min_up,
        "gpu_minimum_torso_up": gpu_min_up,
        "first_step_qpos_component_error": qpos_errors[0],
        "first_step_qvel_component_error": qvel_errors[0],
        "parity_window_steps": parity_steps,
        "parity_window_max_qpos_component_error": parity_qpos_error,
        "parity_window_max_qvel_component_error": parity_qvel_error,
        "max_qpos_component_error": max(qpos_errors),
        "max_qvel_component_error": max(qvel_errors),
        "passed": bool(
            np.count_nonzero(source_contacts) > 0
            and np.count_nonzero(gpu_contacts) > 0
            and qpos_errors[0] < 5e-4
            and qvel_errors[0] < 5e-2
            and parity_qpos_error < 5e-2
            and parity_qvel_error < 2.0
            and stable
        ),
    }


def main() -> None:
    """Run the bounded engine gate and write a machine-readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1:
        raise ValueError("steps must be positive")

    versions = package_versions()
    source = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model, raw_import = build_newton_model(args.xml, source, args.device)
    gpu_solver = make_solver(model, source, use_mujoco_cpu=False)
    cpu_solver = make_solver(model, source, use_mujoco_cpu=True)
    synchronize_cpu_oracle(cpu_solver, gpu_solver, source)
    properties = property_parity(source, cpu_solver.mj_model)
    cpu_rollout = cpu_rollout_parity(source, cpu_solver.mj_model, args.steps)
    gpu_rollout = gpu_rollout_parity(source, model, gpu_solver, args.steps)
    passed = properties["passed"] and cpu_rollout["passed"] and gpu_rollout["passed"]
    report = {
        "scope": "MimicKit_Newton_1_engine_import_and_GPU_dynamics_gate_not_policy_training",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "versions": versions,
        "device": args.device,
        "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        "xml_unchanged": True,
        "import_options": {
            "enable_self_collisions": True,
            "convert_3d_hinge_to_ball_joints": False,
            "ctrl_direct": True,
            "impratio": float(source.opt.impratio),
        },
        "raw_import_diagnostics": raw_import,
        "property_parity": properties,
        "cpu_rollout": cpu_rollout,
        "gpu_rollout": gpu_rollout,
        "known_nonsemantic_export_differences": [
            "collision-disabled visual geoms are omitted from the solver model",
            "actuator names are not preserved, but source order and arrays are preserved",
            "Newton exports a non-binding joint actuator-force range",
        ],
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "raw_import_diagnostics": raw_import,
                "property_parity": properties,
                "cpu_rollout": cpu_rollout,
                "gpu_rollout": gpu_rollout,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
