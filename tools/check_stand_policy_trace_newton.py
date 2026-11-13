"""Replay saved stand-policy targets through native MuJoCo and Newton GPU."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_mimickit_model import DEFAULT_XML
from tools.check_mimickit_newton_engine import (
    build_newton_model,
    make_solver,
    package_versions,
)


def posture(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """Return root height and the environment's anatomical thorax-up metric."""
    thorax_id = model.body("thorax").id
    return (
        float(data.qpos[2]),
        float(data.xmat[thorax_id].reshape(3, 3)[2, 1]),
    )


def load_trace(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a policy trace without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: archive[name].copy()
            for name in (
                "initial_qpos",
                "initial_qvel",
                "controls",
                "source_qpos",
                "source_qvel",
            )
        }
    return metadata, arrays


def main() -> None:
    """Run the open-loop policy-control bridge gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    metadata, trace = load_trace(args.trace)
    xml_hash = hashlib.sha256(args.xml.read_bytes()).hexdigest()
    if metadata["xml_sha256"] != xml_hash:
        raise ValueError("Policy trace and replay XML hashes differ")
    source = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    if trace["controls"].shape[1] != source.nu:
        raise ValueError("Policy trace actuator count differs from replay model")
    substeps = round(metadata["control_dt"] / source.opt.timestep)
    if not np.isclose(substeps * source.opt.timestep, metadata["control_dt"]):
        raise ValueError("Control interval is not an integer physics-step multiple")

    model, raw_import = build_newton_model(args.xml, source, args.device)
    solver = make_solver(model, source, use_mujoco_cpu=False)
    source_data = mujoco.MjData(source)
    source_data.qpos[:] = trace["initial_qpos"]
    source_data.qvel[:] = trace["initial_qvel"]
    source_data.ctrl[:] = trace["controls"][0]
    mujoco.mj_forward(source, source_data)

    state_in = model.state()
    state_out = model.state()
    solver._update_newton_state(model, state_in, source_data, state_prev=state_in)
    control = model.control()
    gpu_metrics = mujoco.MjData(source)
    qpos_errors = []
    qvel_errors = []
    mjx_qpos_errors = []
    source_heights = []
    source_ups = []
    gpu_heights = []
    gpu_ups = []
    source_contact_steps = 0
    gpu_contact_steps = 0
    for control_index, targets in enumerate(trace["controls"]):
        source_data.ctrl[:] = targets
        control.mujoco.ctrl.assign(np.asarray(targets, dtype=np.float32))
        for _ in range(substeps):
            mujoco.mj_step(source, source_data)
            solver.step(
                state_in,
                state_out,
                control,
                contacts=None,
                dt=float(source.opt.timestep),
            )
            state_in, state_out = state_out, state_in
            source_contact_steps += int(source_data.ncon > 0)
            gpu_contact_steps += int(solver.mjw_data.nacon.numpy()[0] > 0)
        gpu_qpos = solver.mjw_data.qpos.numpy()[0]
        gpu_qvel = solver.mjw_data.qvel.numpy()[0]
        if not all(
            np.isfinite(values).all()
            for values in (source_data.qpos, source_data.qvel, gpu_qpos, gpu_qvel)
        ):
            raise FloatingPointError("Non-finite policy-trace replay")
        qpos_errors.append(float(np.max(np.abs(source_data.qpos - gpu_qpos))))
        qvel_errors.append(float(np.max(np.abs(source_data.qvel - gpu_qvel))))
        mjx_qpos_errors.append(
            float(
                np.max(np.abs(source_data.qpos - trace["source_qpos"][control_index]))
            )
        )
        mujoco.mj_forward(source, source_data)
        source_height, source_up = posture(source, source_data)
        source_heights.append(source_height)
        source_ups.append(source_up)
        gpu_metrics.qpos[:] = gpu_qpos
        gpu_metrics.qvel[:] = gpu_qvel
        mujoco.mj_forward(source, gpu_metrics)
        gpu_height, gpu_up = posture(source, gpu_metrics)
        gpu_heights.append(gpu_height)
        gpu_ups.append(gpu_up)

    parity_steps = min(len(qpos_errors), round(1.0 / metadata["control_dt"]))
    fall_height = float(source.qpos0[2]) * 0.30
    source_stable = min(source_heights) >= fall_height and min(source_ups) >= 0.25
    gpu_stable = min(gpu_heights) >= fall_height and min(gpu_ups) >= 0.25
    report = {
        "scope": "open_loop_saved_stand_policy_controls_native_MuJoCo_vs_Newton_GPU",
        "versions": package_versions(),
        "trace": str(args.trace),
        "trace_metadata": metadata,
        "xml_sha256": xml_hash,
        "raw_import_diagnostics": raw_import,
        "control_steps": len(qpos_errors),
        "physics_substeps_per_control": substeps,
        "seconds": len(qpos_errors) * metadata["control_dt"],
        "source_contact_physics_steps": source_contact_steps,
        "gpu_contact_physics_steps": gpu_contact_steps,
        "fall_height_m": fall_height,
        "source_minimum_root_height_m": min(source_heights),
        "gpu_minimum_root_height_m": min(gpu_heights),
        "source_minimum_torso_up": min(source_ups),
        "gpu_minimum_torso_up": min(gpu_ups),
        "source_open_loop_stable": source_stable,
        "gpu_open_loop_stable": gpu_stable,
        "parity_window_seconds": parity_steps * metadata["control_dt"],
        "parity_window_max_qpos_component_error": max(qpos_errors[:parity_steps]),
        "parity_window_max_qvel_component_error": max(qvel_errors[:parity_steps]),
        "full_horizon_max_qpos_component_error": max(qpos_errors),
        "full_horizon_max_qvel_component_error": max(qvel_errors),
        "native_MuJoCo_vs_source_MJX_max_qpos_component_error": max(mjx_qpos_errors),
        "limitation": metadata["limitation"],
    }
    report["passed"] = bool(
        metadata["passed_source_policy_rollout"]
        and source_stable
        and gpu_stable
        and report["parity_window_max_qpos_component_error"] < 0.05
        and report["parity_window_max_qvel_component_error"] < 2.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
