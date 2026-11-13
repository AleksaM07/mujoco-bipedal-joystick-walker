"""Evaluate the saved JAX stand actor with Newton or native MuJoCo physics."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_mimickit_model import DEFAULT_XML

MAX_SPEED = 0.15
MAX_DISTANCE = 0.25
MIN_TORSO_UP = 0.9
MIN_HEIGHT_RATIO = 0.85
DWELL_SECONDS = 1.0
RECOVERY_DEADLINE_SECONDS = 5.0


def read_protocol(stream: TextIO, prefix: str) -> dict[str, Any]:
    """Read through dependency logs until the requested protocol record arrives."""
    for line in stream:
        line = line.strip()
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise RuntimeError(f"Controller exited before sending {prefix.strip()}")


def send_state(
    process: subprocess.Popen[str],
    qpos: np.ndarray,
    qvel: np.ndarray,
    body_xpos: np.ndarray,
    body_xquat: np.ndarray,
    body_xmat: np.ndarray,
    body_cvel: np.ndarray,
) -> dict[str, Any]:
    """Request one control target from the persistent JAX process."""
    assert process.stdin is not None and process.stdout is not None
    request = {
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "body_xpos": body_xpos.tolist(),
        "body_xquat": body_xquat.tolist(),
        "body_xmat": body_xmat.tolist(),
        "body_cvel": body_cvel.tolist(),
    }
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return read_protocol(process.stdout, "CONTROL ")


def both_feet_touching(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Match the stand gate's geometric floor/sole contact requirement."""
    floor = model.geom("floor").id
    feet = {model.geom("left_foot_sole").id, model.geom("right_foot_sole").id}
    touching = set()
    for contact in data.contact[: data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor in pair and contact.dist <= 0:
            touching.update(pair & feet)
    return touching == feet


def evaluate_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    anchor_xy: np.ndarray,
    standing_height: float,
) -> dict[str, float | bool]:
    """Compute the existing physical stand gates from a Newton state."""
    thorax = model.body("thorax").id
    torso_up = float(data.xmat[thorax].reshape(3, 3)[2, 1])
    speed = float(np.linalg.norm(data.qvel[:2]))
    distance = float(np.linalg.norm(data.qpos[:2] - anchor_xy))
    height = float(data.qpos[2])
    height_ratio = height / standing_height
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    physical_failure = bool(
        not finite or height < 0.30 * standing_height or torso_up < 0.25
    )
    eligible = bool(
        finite
        and not physical_failure
        and speed < MAX_SPEED
        and distance < MAX_DISTANCE
        and torso_up >= MIN_TORSO_UP
        and height_ratio >= MIN_HEIGHT_RATIO
        and both_feet_touching(model, data)
    )
    return {
        "finite": finite,
        "physical_failure": physical_failure,
        "eligible": eligible,
        "speed": speed,
        "distance": distance,
        "torso_up": torso_up,
        "height": height,
        "height_ratio": height_ratio,
    }


def controller_command(args: argparse.Namespace) -> list[str]:
    """Build the isolated JAX controller command without shell interpolation."""
    return [
        str(args.controller_python.resolve()),
        str((PROJECT_ROOT / "tools/stand_policy_controller.py").resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--seed",
        str(args.seed),
        "--device",
        args.controller_device,
        "--serve",
    ]


def main() -> None:
    """Run neutral or backward-push closed-loop stand evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--physics-engine",
        choices=("newton", "native"),
        default="newton",
        help="Use Newton/MuJoCo-Warp or native MuJoCo as the physics backend.",
    )
    parser.add_argument(
        "--controller-python",
        type=Path,
        default=PROJECT_ROOT / ".venv-win/Scripts/python.exe",
    )
    parser.add_argument("--controller-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--push-x-fraction", type=float, default=0.0)
    parser.add_argument(
        "--push-x-direction",
        choices=("negative", "positive"),
        default="negative",
    )
    parser.add_argument("--push-start-seconds", type=float, default=5.0)
    parser.add_argument("--push-duration-seconds", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seconds < 1.0 or args.push_x_fraction < 0:
        raise ValueError("Invalid rollout seconds or push fraction")

    source = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model = None
    solver = None
    raw_import = None
    if args.physics_engine == "newton":
        from tools.check_mimickit_newton_engine import (
            build_newton_model,
            make_solver,
            package_versions,
        )

        model, raw_import = build_newton_model(args.xml, source, args.device)
        solver = make_solver(model, source, use_mujoco_cpu=False)
        versions = package_versions()
    else:
        versions = {"mujoco": mujoco.__version__}
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        controller_command(args),
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    try:
        assert process.stdout is not None
        ready = read_protocol(process.stdout, "READY ")
        if ready["action_size"] != source.nu:
            raise ValueError("Controller and Newton actuator counts differ")
        substeps = round(ready["control_dt"] / source.opt.timestep)
        if not np.isclose(substeps * source.opt.timestep, ready["control_dt"]):
            raise ValueError("Controller interval is not a physics-step multiple")
        total_steps = round(args.seconds / ready["control_dt"])
        push_start = round(args.push_start_seconds / ready["control_dt"])
        push_steps = round(args.push_duration_seconds / ready["control_dt"])
        if (
            args.push_x_fraction
            and not 0 <= push_start < push_start + push_steps < total_steps
        ):
            raise ValueError("Push interval must fit inside the rollout")

        initial_qpos = np.asarray(ready["initial_qpos"], dtype=np.float64)
        initial_qvel = np.asarray(ready["initial_qvel"], dtype=np.float64)
        initial_data = mujoco.MjData(source)
        initial_data.qpos[:] = initial_qpos
        initial_data.qvel[:] = initial_qvel
        mujoco.mj_forward(source, initial_data)
        state_in = None
        state_out = None
        control = None
        native_data = None
        if args.physics_engine == "newton":
            assert model is not None and solver is not None
            state_in = model.state()
            state_out = model.state()
            solver._update_newton_state(
                model, state_in, initial_data, state_prev=state_in
            )
            control = model.control()
        else:
            native_data = initial_data
        metrics_data = mujoco.MjData(source)
        anchor_xy = initial_qpos[:2].copy()
        standing_height = float(initial_qpos[2])
        mass = float(np.sum(source.body_mass))
        gravity = float(np.linalg.norm(source.opt.gravity))
        push_sign = -1.0 if args.push_x_direction == "negative" else 1.0
        push_force = push_sign * args.push_x_fraction * mass * gravity
        thorax_mj = source.body("thorax").id
        thorax_newton = None
        body_force = None
        if args.physics_engine == "newton":
            assert model is not None and solver is not None
            thorax_newton = int(solver.mjc_body_to_newton.numpy()[0, thorax_mj])
            body_force = np.zeros((model.body_count, 6), dtype=np.float32)
        rows = []
        dwell_steps = 0
        max_dwell_steps = 0
        recovered_seconds = None
        release_step = push_start + push_steps
        recovery_dwell = 0
        qpos = initial_qpos
        qvel = initial_qvel
        body_xpos = initial_data.xpos.copy()
        body_xquat = initial_data.xquat.copy()
        body_xmat = initial_data.xmat.reshape(source.nbody, 3, 3).copy()
        body_cvel = initial_data.cvel.copy()
        for step in range(total_steps):
            response = send_state(
                process,
                qpos,
                qvel,
                body_xpos,
                body_xquat,
                body_xmat,
                body_cvel,
            )
            targets = np.asarray(response["targets"], dtype=np.float32)
            pushed = bool(args.push_x_fraction and push_start <= step < release_step)
            if args.physics_engine == "newton":
                assert (
                    body_force is not None
                    and control is not None
                    and solver is not None
                    and state_in is not None
                    and state_out is not None
                    and thorax_newton is not None
                )
                control.mujoco.ctrl.assign(targets)
                body_force.fill(0.0)
                if pushed:
                    body_force[thorax_newton, 0] = push_force
                for _ in range(substeps):
                    state_in.body_f.assign(body_force)
                    solver.step(
                        state_in,
                        state_out,
                        control,
                        contacts=None,
                        dt=float(source.opt.timestep),
                    )
                    state_in, state_out = state_out, state_in
                qpos = solver.mjw_data.qpos.numpy()[0].astype(np.float64)
                qvel = solver.mjw_data.qvel.numpy()[0].astype(np.float64)
                body_xpos = solver.mjw_data.xpos.numpy()[0].astype(np.float32)
                body_xquat = solver.mjw_data.xquat.numpy()[0].astype(np.float32)
                body_xmat = (
                    solver.mjw_data.xmat.numpy()[0]
                    .astype(np.float32)
                    .reshape(source.nbody, 3, 3)
                )
                body_cvel = solver.mjw_data.cvel.numpy()[0].astype(np.float32)
            else:
                assert native_data is not None
                native_data.ctrl[:] = targets
                native_data.xfrc_applied[:] = 0.0
                if pushed:
                    native_data.xfrc_applied[thorax_mj, 0] = push_force
                for _ in range(substeps):
                    mujoco.mj_step(source, native_data)
                qpos = native_data.qpos.copy()
                qvel = native_data.qvel.copy()
                body_xpos = native_data.xpos.copy()
                body_xquat = native_data.xquat.copy()
                body_xmat = native_data.xmat.reshape(source.nbody, 3, 3).copy()
                body_cvel = native_data.cvel.copy()
            metrics_data.qpos[:] = qpos
            metrics_data.qvel[:] = qvel
            mujoco.mj_forward(source, metrics_data)
            sample = evaluate_metrics(source, metrics_data, anchor_xy, standing_height)
            dwell_steps = dwell_steps + 1 if sample["eligible"] else 0
            max_dwell_steps = max(max_dwell_steps, dwell_steps)
            if args.push_x_fraction and step >= release_step:
                recovery_dwell = recovery_dwell + 1 if sample["eligible"] else 0
                required = round(DWELL_SECONDS / ready["control_dt"])
                if recovery_dwell >= required and recovered_seconds is None:
                    recovered_seconds = (step - release_step + 1) * ready["control_dt"]
            rows.append(
                {
                    "step": step + 1,
                    "seconds": (step + 1) * ready["control_dt"],
                    "push_force_x": push_force if pushed else 0.0,
                    "action_l2": response["action_l2"],
                    "action_clip_fraction": response["action_clip_fraction"],
                    **sample,
                }
            )
            if sample["physical_failure"]:
                break
    finally:
        if process.poll() is None:
            assert process.stdin is not None
            process.stdin.write(json.dumps({"command": "close"}) + "\n")
            process.stdin.flush()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)

    required_dwell = round(DWELL_SECONDS / ready["control_dt"])
    completed = len(rows) == total_steps and not any(
        row["physical_failure"] for row in rows
    )
    final_success = bool(dwell_steps >= required_dwell)
    if args.push_x_fraction:
        passed = bool(
            completed
            and final_success
            and recovered_seconds is not None
            and recovered_seconds <= RECOVERY_DEADLINE_SECONDS
        )
    else:
        passed = bool(
            completed and final_success and all(row["eligible"] for row in rows)
        )
    report = {
        "scope": f"live_closed_loop_JAX_actor_with_{args.physics_engine}_physics",
        "physics_engine": args.physics_engine,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": hashlib.sha256(
            (args.checkpoint / "_CHECKPOINT_METADATA").read_bytes()
        ).hexdigest(),
        "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        "versions": versions,
        "controller": ready,
        "raw_import_diagnostics": raw_import,
        "requested_seconds": args.seconds,
        "completed_steps": len(rows),
        "requested_steps": total_steps,
        "physics_substeps_per_control": substeps,
        "push_x_body_weight_fraction": args.push_x_fraction,
        "push_x_direction": args.push_x_direction,
        "push_force_x_n": push_force,
        "push_start_seconds": args.push_start_seconds,
        "push_duration_seconds": args.push_duration_seconds,
        "recovery_seconds": recovered_seconds if args.push_x_fraction else None,
        "max_dwell_seconds": max_dwell_steps * ready["control_dt"],
        "final_dwell_seconds": dwell_steps * ready["control_dt"],
        "minimum_root_height_m": min(row["height"] for row in rows),
        "minimum_height_ratio": min(row["height_ratio"] for row in rows),
        "minimum_torso_up": min(row["torso_up"] for row in rows),
        "maximum_speed_m_s": max(row["speed"] for row in rows),
        "maximum_drift_m": max(row["distance"] for row in rows),
        "maximum_action_l2": max(row["action_l2"] for row in rows),
        "maximum_action_clip_fraction": max(
            row["action_clip_fraction"] for row in rows
        ),
        "eligible_fraction": float(np.mean([row["eligible"] for row in rows])),
        "completed_without_physical_failure": completed,
        "final_success": final_success,
        "passed": passed,
        "trace": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "trace"},
            indent=2,
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
