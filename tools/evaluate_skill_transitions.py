"""Probe two frozen stand/walk actors in one continuous MuJoCo rollout.

This is a diagnostic, not navigation training.  It preserves one physical
state while each actor keeps its own observation normalizer, reference clock
and action smoothing state.  Incoming motor targets are linearly blended for
the scheduled transition window; qpos and qvel are never edited at a switch.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (
    choose_device,
    file_sha256,
    find_run_config,
    find_run_dir,
    infer_xml_path,
    load_ppo_policy,
    make_environment,
    render_host_replay_frames,
    restore_saved_env_config,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
    write_video,
)
from skills.policy_registry import FINAL_STANDING, FINAL_WALKING
from skills.transitions import (
    BRAKE_HANDOFF_PHASES,
    DEFAULT_TRANSITION_PHASES,
    WALK_STOP_PROBE_PHASES,
    SkillName,
    build_transition_directives,
    transition_boundaries,
)
from tools.evaluate_stand import load_stand_config

DEFAULT_WALK_CHECKPOINT = FINAL_WALKING.checkpoint
DEFAULT_STAND_CHECKPOINT = FINAL_STANDING.checkpoint
STAND_SPEED_LIMIT = 0.15
STAND_DISTANCE_LIMIT = 0.25
STAND_TORSO_UP_LIMIT = 0.90
STAND_HEIGHT_RATIO_LIMIT = 0.85


def physical_match_score(
    current_qpos: np.ndarray,
    current_qvel: np.ndarray,
    candidate_qpos: np.ndarray,
    candidate_qvel: np.ndarray,
    actuator_qpos_indices: np.ndarray,
    actuator_dof_indices: np.ndarray,
) -> dict[str, float]:
    """Score how safely a reference reset can inherit the current state.

    World XY is deliberately ignored because every skill is rebased at
    activation.  The score favors a similar articulated pose and root
    orientation, then penalizes velocity discontinuities that a motor-target
    blend cannot physically create in one control step.
    """
    current_quat = current_qpos[3:7] / np.linalg.norm(current_qpos[3:7])
    candidate_quat = candidate_qpos[3:7] / np.linalg.norm(candidate_qpos[3:7])
    pose_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    candidate_qpos[actuator_qpos_indices]
                    - current_qpos[actuator_qpos_indices]
                )
            )
        )
    )
    joint_velocity_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    candidate_qvel[actuator_dof_indices]
                    - current_qvel[actuator_dof_indices]
                )
            )
        )
    )
    root_velocity_rmse = float(
        np.sqrt(np.mean(np.square(candidate_qvel[:6] - current_qvel[:6])))
    )
    orientation_error = float(1.0 - abs(np.dot(current_quat, candidate_quat)))
    height_error = float(abs(candidate_qpos[2] - current_qpos[2]))
    total = (
        pose_rmse
        + 0.10 * joint_velocity_rmse
        + 0.50 * root_velocity_rmse
        + 2.0 * orientation_error
        + height_error
    )
    return {
        "total": total,
        "pose_rmse": pose_rmse,
        "joint_velocity_rmse": joint_velocity_rmse,
        "root_velocity_rmse": root_velocity_rmse,
        "orientation_error": orientation_error,
        "height_error": height_error,
    }


def load_walk_config(checkpoint: Path):
    """Restore the archived walking environment without mutating the archive."""
    saved = find_run_config(checkpoint)
    if saved is None or saved["env"].get("locomotion_skill", "walk") != "walk":
        raise ValueError("Walking checkpoint needs a parent walk config.json")
    if saved["env"].get("physics_backend", "mjx_jax") != "mjx_jax":
        raise ValueError("Transition probe requires a saved mjx_jax checkpoint")
    run_dir = find_run_dir(checkpoint)
    raw_xml = saved["env"].get("xml_path")
    local_xml = run_dir / raw_xml if run_dir is not None and raw_xml else None
    xml_path = (
        str(local_xml.resolve())
        if local_xml is not None and local_xml.is_file()
        else infer_xml_path(checkpoint, saved)
    )
    if xml_path is None:
        raise ValueError("Cannot resolve the walking checkpoint XML")
    return restore_saved_env_config(
        saved,
        locomotion_skill="walk",
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=xml_path,
    )


def load_brake_config(checkpoint: Path):
    """Restore one trained BRAKE_TRANSITION actor without changing semantics."""
    saved = find_run_config(checkpoint)
    if saved is None or saved["env"].get("locomotion_skill") != "brake_transition":
        raise ValueError("Bridge checkpoint needs locomotion_skill=brake_transition")
    if saved["env"].get("physics_backend", "mjx_jax") != "mjx_jax":
        raise ValueError("Transition probe requires a saved mjx_jax checkpoint")
    xml_path = infer_xml_path(checkpoint, saved)
    if xml_path is None:
        raise ValueError("Cannot resolve the bridge checkpoint XML")
    return restore_saved_env_config(
        saved,
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=xml_path,
    )


class FrozenSkillController:
    """Evaluate one frozen actor against an externally advanced physical state."""

    def __init__(
        self,
        checkpoint: Path,
        skill: SkillName,
        *,
        seed: int,
        device: str,
    ) -> None:
        self.checkpoint = checkpoint.resolve(strict=True)
        self.skill = skill
        self.seed = seed
        self.device = choose_device(device)
        if skill == "stand":
            config = load_stand_config(self.checkpoint)
        elif skill == "walk":
            config = load_walk_config(self.checkpoint)
        else:
            config = load_brake_config(self.checkpoint)
        with jax.default_device(self.device):
            self.env = make_environment(config)
            validate_xml_manifest(self.checkpoint, self.env)
            validate_action_compatibility(self.checkpoint, self.env.action_size)
            self.policy = load_ppo_policy(self.checkpoint, deterministic=True)
            self._reset = jax.jit(self.env.reset)
            state = self._reset(jax.random.PRNGKey(seed))
            validate_observation_compatibility(self.checkpoint, state.obs)
            self.data_template = state.data
            self._control = jax.jit(self._controller_step)
        self.info: dict[str, Any] | None = None
        self.key: jax.Array | None = None
        self.origin_xy = np.zeros(2, dtype=np.float32)
        self.activation_state = state
        self.activations = 0
        self.activation_records: list[dict[str, Any]] = []

    def activate(
        self,
        root_xy: np.ndarray,
        *,
        physical_qpos: np.ndarray | None = None,
        physical_qvel: np.ndarray | None = None,
        candidate_count: int = 1,
    ) -> Any:
        """Reset only controller/reference state and rebase its local world frame."""
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if (physical_qpos is None) != (physical_qvel is None):
            raise ValueError(
                "physical_qpos and physical_qvel must be provided together"
            )
        activation_seed = self.seed + 1009 * self.activations
        best_state = None
        best_seed = activation_seed
        best_score: dict[str, float] | None = None
        for candidate_index in range(candidate_count):
            candidate_seed = activation_seed + candidate_index
            with jax.default_device(self.device):
                candidate = self._reset(jax.random.PRNGKey(candidate_seed))
            if physical_qpos is None:
                best_state = candidate
                break
            score = physical_match_score(
                physical_qpos,
                physical_qvel,
                np.asarray(jax.device_get(candidate.data.qpos)),
                np.asarray(jax.device_get(candidate.data.qvel)),
                self.env._actuator_qpos_indices_np,
                self.env._actuator_dof_indices_np,
            )
            if best_score is None or score["total"] < best_score["total"]:
                best_state = candidate
                best_seed = candidate_seed
                best_score = score
        if best_state is None:
            raise RuntimeError("No controller activation candidate was produced")
        state = best_state
        self.info = state.info
        self.key = jax.random.PRNGKey(best_seed + 100_000)
        self.origin_xy = np.asarray(root_xy, dtype=np.float32).copy()
        self.activation_state = state
        record: dict[str, Any] = {
            "activation": self.activations,
            "candidate_count": candidate_count,
            "selected_seed": best_seed,
            "reference_time": float(
                jax.device_get(state.info["bvh_reference_time_offset"])
            ),
        }
        if best_score is not None:
            record["physical_match"] = best_score
        self.activation_records.append(record)
        self.activations += 1
        return state

    def _controller_step(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        body_xpos: jax.Array,
        body_xquat: jax.Array,
        body_xmat: jax.Array,
        body_cvel: jax.Array,
        origin_xy: jax.Array,
        info: dict[str, Any],
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, Any], jax.Array]:
        """Build the actor observation and map its action to motor targets."""
        local_qpos = qpos.at[:2].add(-origin_xy)
        local_xpos = body_xpos.at[:, :2].add(-origin_xy)
        data = self.data_template.replace(
            qpos=local_qpos,
            qvel=qvel,
            xpos=local_xpos,
            xquat=body_xquat,
            xmat=body_xmat,
            cvel=body_cvel,
        )
        obs = self.env._get_obs(data, info)
        key, action_key = jax.random.split(key)
        action, _ = self.policy(obs, action_key)
        policy_action = jp.clip(
            jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0),
            -1.0,
            1.0,
        )
        previous_action = info["last_action"]
        smoothed_action = (
            self.env._config.action_smoothing * policy_action
            + (1.0 - self.env._config.action_smoothing) * previous_action
        )
        targets = self.env._clip_motor_targets(
            self.env._policy_action_to_motor_targets(smoothed_action, info)
        )
        updated = dict(info)
        updated["last_action"] = smoothed_action
        updated["motion_step"] = info["motion_step"] + jp.array(
            1, dtype=info["motion_step"].dtype
        )
        updated["command_step"] = info["command_step"] + jp.array(
            1, dtype=info["command_step"].dtype
        )
        updated["gait_step"] = jp.mod(
            info["gait_step"] + jp.array(1, dtype=info["gait_step"].dtype),
            self.env.GAIT_PERIOD_STEPS,
        )
        return targets, policy_action, updated, key

    def control(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """Return one motor-target vector for the current native MuJoCo state."""
        if self.info is None or self.key is None:
            raise RuntimeError(f"{self.skill} controller is not active")
        xmat_shape = self.data_template.xmat.shape
        with jax.default_device(self.device):
            targets, action, self.info, self.key = self._control(
                jp.asarray(data.qpos),
                jp.asarray(data.qvel),
                jp.asarray(data.xpos),
                jp.asarray(data.xquat),
                jp.asarray(data.xmat.reshape(xmat_shape)),
                jp.asarray(data.cvel),
                jp.asarray(self.origin_xy),
                self.info,
                self.key,
            )
            targets, action = jax.device_get((targets, action))
        return np.asarray(targets, dtype=np.float64), float(np.linalg.norm(action))


def foot_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_ids: tuple[int, int],
    floor_id: int,
) -> tuple[bool, bool]:
    """Return exact floor-contact flags for the two sole geoms."""
    contacts = [False, False]
    for contact in data.contact[: data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        for index, foot_id in enumerate(foot_ids):
            contacts[index] |= pair == {foot_id, floor_id}
    return tuple(contacts)  # type: ignore[return-value]


def phase_summary(
    rows: list[dict[str, Any]], phase_name: str, dt: float
) -> dict[str, Any]:
    """Summarize one named interval using physical, not reward, quantities."""
    phase_rows = [row for row in rows if row["phase"] == phase_name]
    if not phase_rows:
        raise ValueError(f"Missing transition phase: {phase_name}")
    final = phase_rows[-1]
    tail_steps = max(1, round(1.0 / dt))
    tail = phase_rows[-tail_steps:]
    stand_phase = phase_name.startswith("stand")
    passed = not any(row["physical_failure"] for row in phase_rows)
    if stand_phase:
        passed = passed and all(row["stand_eligible"] for row in tail)
    elif phase_name.startswith("walk"):
        displacement = np.linalg.norm(
            np.array([final["root_x"], final["root_y"]])
            - np.array([phase_rows[0]["root_x"], phase_rows[0]["root_y"]])
        )
        passed = passed and displacement > 0.5
    return {
        "phase": phase_name,
        "seconds": len(phase_rows) * dt,
        "steps": len(phase_rows),
        "root_xy_displacement": float(
            np.linalg.norm(
                np.array([final["root_x"], final["root_y"]])
                - np.array([phase_rows[0]["root_x"], phase_rows[0]["root_y"]])
            )
        ),
        "forward_displacement": final["root_x"] - phase_rows[0]["root_x"],
        "final_speed": final["speed"],
        "max_speed": max(row["speed"] for row in phase_rows),
        "min_torso_up": min(row["torso_up"] for row in phase_rows),
        "min_height_ratio": min(row["height_ratio"] for row in phase_rows),
        "max_contact_foot_slip_speed": max(
            row["contact_foot_slip_speed"] for row in phase_rows
        ),
        "tail_stand_eligible_fraction": (
            sum(row["stand_eligible"] for row in tail) / len(tail)
        ),
        "passed": bool(passed),
    }


def run_probe(
    walk_checkpoint: Path,
    stand_checkpoint: Path,
    *,
    seed: int,
    device: str,
    walk_phase_candidates: int,
    brake_checkpoint: Path | None = None,
    brake_phase_candidates: int = 64,
    protocol: str = "walk_stop",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[tuple[int, np.ndarray, np.ndarray]],
]:
    """Execute the connected two-start/two-stop transition schedule."""
    print("[transition] loading stand actor", flush=True)
    stand = FrozenSkillController(stand_checkpoint, "stand", seed=seed, device=device)
    print("[transition] loading walk actor", flush=True)
    walk = FrozenSkillController(walk_checkpoint, "walk", seed=seed + 37, device=device)
    brake = None
    if protocol == "brake_handoff":
        if brake_checkpoint is None:
            raise ValueError("brake_handoff requires --brake-checkpoint")
        print("[transition] loading brake actor", flush=True)
        brake = FrozenSkillController(
            brake_checkpoint,
            "brake_transition",
            seed=seed + 73,
            device=device,
        )
    if file_sha256(stand.env.xml_path) != file_sha256(walk.env.xml_path):
        raise ValueError("Stand and walk checkpoints use different XML models")
    if stand.env.action_size != walk.env.action_size or stand.env.dt != walk.env.dt:
        raise ValueError("Stand and walk control interfaces differ")

    model = mujoco.MjModel.from_xml_path(str(Path(stand.env.xml_path).resolve()))
    data = mujoco.MjData(model)
    if protocol == "walk_stop":
        phases = WALK_STOP_PROBE_PHASES
    elif protocol == "brake_handoff":
        phases = BRAKE_HANDOFF_PHASES
    else:
        phases = DEFAULT_TRANSITION_PHASES
    directives = build_transition_directives(dt=float(stand.env.dt), phases=phases)
    initial_skill = directives[0].source
    controllers = {"stand": stand, "walk": walk}
    if brake is not None:
        if file_sha256(brake.env.xml_path) != file_sha256(walk.env.xml_path):
            raise ValueError("Brake and final policies use different XML models")
        if brake.env.action_size != walk.env.action_size or brake.env.dt != walk.env.dt:
            raise ValueError("Brake and final policy control interfaces differ")
        controllers["brake_transition"] = brake
    initial_controller = controllers[initial_skill]
    initial_state = initial_controller.activate(np.zeros(2, dtype=np.float32))
    data.qpos[:] = np.asarray(initial_state.data.qpos)
    data.qvel[:] = np.asarray(initial_state.data.qvel)
    mujoco.mj_forward(model, data)
    initial_controller.origin_xy = data.qpos[:2].astype(np.float32).copy()

    dt = float(stand.env.dt)
    substeps = round(dt / model.opt.timestep)
    if substeps < 1 or not np.isclose(substeps * model.opt.timestep, dt):
        raise ValueError("Control dt must be an integer multiple of simulation dt")
    boundaries = dict(transition_boundaries(directives))
    torso_id = model.body("thorax").id
    floor_id = model.geom("floor").id
    foot_ids = (
        model.geom("left_foot_sole").id,
        model.geom("right_foot_sole").id,
    )
    standing_height = float(model.qpos0[2])
    snapshots = [(0, data.qpos.copy(), data.qvel.copy())]
    rows: list[dict[str, Any]] = []
    previous_targets: np.ndarray | None = None
    previous_foot_xy = data.geom_xpos[list(foot_ids), :2].copy()
    stand_anchor = data.qpos[:2].copy()
    terminated = False

    for step, directive in enumerate(directives):
        incoming = boundaries.get(step)
        if step and incoming is not None:
            print(
                f"[transition] t={step * dt:.2f}s phase={directive.phase_name} "
                f"activate={incoming}",
                flush=True,
            )
            match_physical_state = incoming != "stand"
            candidate_count = 1
            if incoming == "walk":
                candidate_count = walk_phase_candidates
            elif incoming == "brake_transition":
                candidate_count = brake_phase_candidates
            controllers[incoming].activate(
                data.qpos[:2],
                physical_qpos=(data.qpos.copy() if match_physical_state else None),
                physical_qvel=(data.qvel.copy() if match_physical_state else None),
                candidate_count=candidate_count,
            )
            if incoming == "stand":
                stand_anchor = data.qpos[:2].copy()

        source_targets, source_action_l2 = controllers[directive.source].control(data)
        targets = source_targets
        target_action_l2 = 0.0
        if directive.target is not None:
            target_targets, target_action_l2 = controllers[directive.target].control(
                data
            )
            targets = (
                1.0 - directive.blend
            ) * source_targets + directive.blend * target_targets

        target_jump = (
            0.0
            if previous_targets is None
            else float(np.max(np.abs(targets - previous_targets)))
        )
        data.ctrl[:] = targets
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        if not np.isclose(data.time, (step + 1) * dt, atol=1e-8):
            raise RuntimeError("Physical time discontinuity detected")

        torso_up = float(data.xmat[torso_id].reshape(3, 3)[2, 1])
        speed = float(np.linalg.norm(data.qvel[:2]))
        height_ratio = float(data.qpos[2] / standing_height)
        contacts = foot_contacts(model, data, foot_ids, floor_id)
        foot_xy = data.geom_xpos[list(foot_ids), :2].copy()
        foot_speed = np.linalg.norm(foot_xy - previous_foot_xy, axis=1) / dt
        contact_slip = float(
            max(
                (foot_speed[index] if contact else 0.0)
                for index, contact in enumerate(contacts)
            )
        )
        physical_failure = bool(
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or height_ratio < 0.30
            or torso_up < 0.25
        )
        distance_to_anchor = float(np.linalg.norm(data.qpos[:2] - stand_anchor))
        stand_eligible = bool(
            speed < STAND_SPEED_LIMIT
            and distance_to_anchor < STAND_DISTANCE_LIMIT
            and torso_up >= STAND_TORSO_UP_LIMIT
            and height_ratio >= STAND_HEIGHT_RATIO_LIMIT
            and all(contacts)
            and not physical_failure
        )
        rows.append(
            {
                "step": step + 1,
                "seconds": (step + 1) * dt,
                "phase": directive.phase_name,
                "source": directive.source,
                "target": directive.target or "",
                "blend": directive.blend,
                "root_x": float(data.qpos[0]),
                "root_y": float(data.qpos[1]),
                "height": float(data.qpos[2]),
                "height_ratio": height_ratio,
                "speed": speed,
                "torso_up": torso_up,
                "left_contact": contacts[0],
                "right_contact": contacts[1],
                "distance_to_stand_anchor": distance_to_anchor,
                "stand_eligible": stand_eligible,
                "contact_foot_slip_speed": contact_slip,
                "target_jump": target_jump,
                "source_action_l2": source_action_l2,
                "target_action_l2": target_action_l2,
                "physical_failure": physical_failure,
            }
        )
        if (step + 1) % 5 == 0 or physical_failure:
            snapshots.append((step + 1, data.qpos.copy(), data.qvel.copy()))
        previous_targets = targets
        previous_foot_xy = foot_xy
        if physical_failure:
            terminated = True
            break

    phase_names = list(dict.fromkeys(row["phase"] for row in rows))
    summaries = [phase_summary(rows, name, dt) for name in phase_names]
    required_names = {phase.name for phase in phases}
    completed = required_names == set(phase_names) and len(rows) == len(directives)
    report = {
        "protocol": "frozen_skill_transition_probe_v1",
        "scope": protocol,
        "walk_policy": FINAL_WALKING.label,
        "walk_checkpoint": str(walk_checkpoint.resolve()),
        "stand_policy": FINAL_STANDING.label,
        "stand_checkpoint": str(stand_checkpoint.resolve()),
        "brake_policy": "BRAKE_TRANSITION" if brake is not None else None,
        "brake_checkpoint": (
            str(brake_checkpoint.resolve()) if brake_checkpoint is not None else None
        ),
        "xml_sha256": file_sha256(stand.env.xml_path),
        "seed": seed,
        "device": device,
        "control_dt": dt,
        "simulation_dt": float(model.opt.timestep),
        "requested_steps": len(directives),
        "completed_steps": len(rows),
        "requested_seconds": len(directives) * dt,
        "completed_seconds": len(rows) * dt,
        "physics_resets": 0,
        "walk_phase_candidates": walk_phase_candidates,
        "brake_phase_candidates": brake_phase_candidates,
        "stand_activations": stand.activation_records,
        "walk_activations": walk.activation_records,
        "brake_activations": [] if brake is None else brake.activation_records,
        "terminated": terminated,
        "phases": summaries,
        "max_motor_target_jump": max(row["target_jump"] for row in rows),
        "max_contact_foot_slip_speed": max(
            row["contact_foot_slip_speed"] for row in rows
        ),
        "final_root_xy": [rows[-1]["root_x"], rows[-1]["root_y"]],
        "passed": bool(completed and all(summary["passed"] for summary in summaries)),
        "limitation": (
            "The archived walk actor is straight-only near 1.1 m/s. This probe "
            "tests frozen-policy transitions, not turning or arbitrary A-to-B."
        ),
    }
    return report, rows, snapshots


def main() -> None:
    """Run the probe and write a report, scalar trace, sparse replay and video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk-checkpoint", type=Path, default=DEFAULT_WALK_CHECKPOINT)
    parser.add_argument(
        "--stand-checkpoint", type=Path, default=DEFAULT_STAND_CHECKPOINT
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--protocol",
        choices=("walk_stop", "brake_handoff", "two_cycle"),
        default="walk_stop",
        help="Isolate walking-to-standing, or run the full two-cycle diagnostic.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--walk-phase-candidates",
        type=int,
        default=64,
        help="Reference reset phases sampled at each stand-to-walk activation.",
    )
    parser.add_argument(
        "--brake-checkpoint",
        type=Path,
        help="Trained BRAKE_TRANSITION checkpoint for the three-policy gate.",
    )
    parser.add_argument(
        "--brake-phase-candidates",
        type=int,
        default=64,
        help="Bridge reference phases matched to the incoming walking state.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        raise ValueError("Transition output must be a fresh directory under runs/")
    output.mkdir(parents=True)

    report, rows, snapshots = run_probe(
        args.walk_checkpoint,
        args.stand_checkpoint,
        seed=args.seed,
        device=args.device,
        walk_phase_candidates=args.walk_phase_candidates,
        brake_checkpoint=args.brake_checkpoint,
        brake_phase_candidates=args.brake_phase_candidates,
        protocol=args.protocol,
    )
    trace_path = output / "trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    replay_path = output / "replay.npz"
    np.savez_compressed(
        replay_path,
        qpos=np.stack([snapshot[1] for snapshot in snapshots]),
        qvel=np.stack([snapshot[2] for snapshot in snapshots]),
        steps=np.asarray([snapshot[0] for snapshot in snapshots], dtype=np.int64),
        dt=report["control_dt"],
    )
    report.update(trace=str(trace_path), replay=str(replay_path))

    if args.video:
        model = mujoco.MjModel.from_xml_path(
            str(Path(load_stand_config(args.stand_checkpoint).xml_path).resolve())
        )
        frames = render_host_replay_frames(
            model,
            [(snapshot[1], snapshot[2]) for snapshot in snapshots],
            960,
            720,
            layout="quad",
        )
        video_path = (
            PROJECT_ROOT
            / "videos"
            / "final_skill_transitions"
            / f"{output.name}_quad.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(video_path, frames, fps=10)
        report["video"] = str(video_path)
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
