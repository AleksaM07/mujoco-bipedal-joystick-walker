"""Native, free-root reference replay diagnostics without a policy or training.

This deliberately idealized position-control replay is not the environment's
action path and is not a certificate of RL feasibility. It changes no gains,
limits, damping, contacts, or root constraints in the supplied native model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from tools.check_reference_curves import full_state

if TYPE_CHECKING:
    from biomechanics_env import BiomechanicsJoystickEnv


def _validate_position_actuators(env: BiomechanicsJoystickEnv) -> None:
    """Reject transmissions for which a joint angle is not a position control."""
    model = env._mj_model
    if model.nu == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Replay requires a free-root model with position actuators.")
    if model.jnt_qposadr[0] != 0 or model.jnt_dofadr[0] != 0:
        raise ValueError("Replay requires the free root at qpos/dof address zero.")
    expected_gear = np.zeros_like(model.actuator_gear)
    expected_gear[:, 0] = 1.0
    if not np.allclose(model.actuator_gear, expected_gear, atol=1e-12, rtol=0):
        raise ValueError("Direct reference-position replay requires gear == 1.")
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("Replay only supports direct joint transmissions.")
    joint_ids = model.actuator_trnid[:, 0]
    if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Replay only supports scalar hinge actuators.")
    if not (
        np.array_equal(model.jnt_qposadr[joint_ids], env._actuator_qpos_indices)
        and np.array_equal(model.jnt_dofadr[joint_ids], env._actuator_dof_indices)
    ):
        raise ValueError("Reference actuator indices do not match native ordering.")
    gains = model.actuator_gainprm[:, 0]
    biases = model.actuator_biasprm
    if not (
        np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
        and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        and np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
        and np.all(gains > 0)
        and np.allclose(biases[:, 0], 0)
        and np.allclose(biases[:, 1], -gains)
        and np.all(biases[:, 2] <= 0)
    ):
        raise ValueError(
            "Replay requires stateless position actuators (kp, optional kv)."
        )


def _physical_failure(
    env: BiomechanicsJoystickEnv, data: mujoco.MjData, minimum_height: float
) -> list[str]:
    """Mirror environment physical gates, with stronger finite-value validation."""
    reasons = []
    if not all(
        np.isfinite(values).all()
        for values in (data.qpos, data.qvel, data.qacc, data.actuator_force)
    ):
        reasons.append("nonfinite_state_or_force")
    if data.qpos[2] < minimum_height:
        reasons.append("root_height_below_environment_threshold")
    if data.xmat[env._torso_body_id].reshape(3, 3)[2, 1] < 0.25:
        reasons.append("torso_anatomical_up_below_0.25")
    for warning in (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
    ):
        if data.warning[warning].number:
            reasons.append(f"native_numerical_warning_{warning.name}")
    return reasons


def _floor_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
    foot_ids: set[int],
) -> tuple[float, float, int]:
    """Sum solved floor normal force, separating foot and non-foot collisions."""
    foot_force, other_force, active_contacts = 0.0, 0.0, 0
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        if floor_id not in geom_ids:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        normal = max(0.0, float(force[0]))
        active_contacts += int(normal > 1e-6)
        other_geom = geom_ids[1] if geom_ids[0] == floor_id else geom_ids[0]
        if other_geom in foot_ids:
            foot_force += normal
        else:
            other_force += normal
    return foot_force, other_force, active_contacts


def replay_reference(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    cycles: float = 3.0,
    start_phases: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    *,
    control_dt: float | None = None,
    target_lookahead_seconds: float = 0.0,
) -> dict[str, Any]:
    """Replay joint targets with an unassisted root from several exact RSI phases.

    Args:
        env: Loaded environment providing a native model and reference query.
        clip_id: Index of a nontrivial, repeating reference clip.
        cycles: Requested number of cycles per starting phase.
        start_phases: Fractions in [0, 1), independently reset before each replay.
        control_dt: Hold interval, a multiple of the physics timestep. None
            updates at every physics step, unlike the slower policy action path.
        target_lookahead_seconds: Advance only control targets, not initialization
            or the reference used to score tracking. Zero is contemporaneous.

    Returns:
        JSON-serializable diagnostics. Survival includes the terminal physics
        step and is quantized to the native timestep; completed horizons can
        overshoot the requested length by less than one timestep.

    Raises:
        TypeError: clip_id is not an integer.
        ValueError: Invalid inputs or unsupported actuator/reference semantics.
    """
    _validate_position_actuators(env)
    model = env._mj_model
    lengths = np.asarray(env._bvh_reference_motion_lengths)
    if isinstance(clip_id, bool) or not isinstance(clip_id, (int, np.integer)):
        raise TypeError("clip_id must be an integer.")
    if not 0 <= clip_id < len(lengths):
        raise ValueError("clip_id is outside the prepared reference batch.")
    duration = float(lengths[clip_id])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Replay requires a positive reference duration.")
    if int(env._bvh_reference_loop_modes[clip_id]) != 1:
        raise ValueError("Multi-cycle replay requires a WRAP reference.")
    if not np.isfinite(cycles) or cycles <= 0:
        raise ValueError("cycles must be finite and positive.")
    phases = np.asarray(start_phases, dtype=float)
    if (
        phases.ndim != 1
        or not phases.size
        or not np.isfinite(phases).all()
        or np.any((phases < 0) | (phases >= 1))
    ):
        raise ValueError("start_phases must be a nonempty sequence in [0, 1).")
    timestep = float(model.opt.timestep)
    hold_dt = timestep if control_dt is None else float(control_dt)
    if not np.isfinite(hold_dt) or hold_dt < timestep:
        raise ValueError("control_dt must be at least the physics timestep.")
    hold_steps = round(hold_dt / timestep)
    if not np.isclose(hold_steps * timestep, hold_dt, atol=1e-10, rtol=1e-8):
        raise ValueError("control_dt must be an integer multiple of physics timestep.")
    if not np.isfinite(target_lookahead_seconds) or target_lookahead_seconds < 0:
        raise ValueError("target_lookahead_seconds must be finite and nonnegative.")
    minimum_height = float(env.MIN_STANDING_HEIGHT_RATIO * env._standing_height())
    joint_indices = np.asarray(env._actuator_qpos_indices, dtype=int)
    limited = np.asarray(model.actuator_forcelimited, dtype=bool)
    force_ranges = np.asarray(model.actuator_forcerange)
    control_limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    control_ranges = np.asarray(model.actuator_ctrlrange)
    foot_ids = set(map(int, env._foot_geom_ids_np))
    force_tolerance = 1e-5 * np.maximum(1.0, np.max(np.abs(force_ranges), axis=1))
    step_count = int(np.ceil(cycles * duration / timestep))
    elapsed_times = np.arange(step_count + 1) * timestep
    fields = (
        "qpos",
        "qvel",
        "sim_root_pos",
        "sim_root_quat",
        "sim_root_vel",
        "sim_root_angvel",
    )

    def state_query(time: jax.Array) -> dict[str, jax.Array]:
        reference = env._query_bvh_reference_clip(jp.array(clip_id), time)
        return {name: reference[name] for name in fields}

    query = jax.jit(jax.vmap(state_query))
    runs = []
    for phase in phases:
        start_time = phase * duration
        batch = jax.tree.map(np.asarray, query(jp.array(start_time + elapsed_times)))
        control_times = (
            start_time
            + (np.arange(step_count) // hold_steps) * hold_steps * timestep
            + target_lookahead_seconds
        )
        controls = np.asarray(query(jp.array(control_times))["qpos"])
        if not all(np.isfinite(value).all() for value in batch.values()) or not (
            np.isfinite(controls).all()
        ):
            raise ValueError("Reference query returned nonfinite data.")
        if np.any(
            control_limited
            & (
                (controls < control_ranges[:, 0] - 1e-6)
                | (controls > control_ranges[:, 1] + 1e-6)
            )
        ):
            raise ValueError("Reference controls exceed native actuator ctrlrange.")
        initial = {name: values[0] for name, values in batch.items()}
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = full_state(env, initial)
        data.ctrl[:] = controls[0]
        mujoco.mj_forward(model, data)
        reasons = _physical_failure(env, data, minimum_height)
        initial_contacts = int(data.ncon)
        joint_errors, root_errors = [], []
        saturation_counts = np.zeros(model.nu, dtype=int)
        force_peaks = np.zeros(model.nu)
        foot_force_peak, other_force_peak = 0.0, 0.0
        floor_contact_steps = 0
        minimum_root_height = float(data.qpos[2])
        minimum_torso_up = float(data.xmat[env._torso_body_id].reshape(3, 3)[2, 1])
        simulated_steps = 0
        final_root_error = data.qpos[:3] - batch["sim_root_pos"][0]
        for index in range(step_count):
            if reasons:
                break
            data.ctrl[:] = controls[index]
            mujoco.mj_step(model, data)
            simulated_steps += 1
            # Forces/contact solve belong to the state evaluated by mj_step,
            # before refreshing kinematics at the integrated, end-step pose.
            forces = data.actuator_force.copy()
            foot_force, other_force, contacts = _floor_forces(
                model, data, int(env._floor_geom_id), foot_ids
            )
            mujoco.mj_forward(model, data)
            reasons = _physical_failure(env, data, minimum_height)
            if not all(
                np.isfinite(values).all() for values in (data.qpos, data.qvel, forces)
            ):
                break
            force_peaks = np.maximum(force_peaks, np.abs(forces))
            saturated = limited & (
                (forces <= force_ranges[:, 0] + force_tolerance)
                | (forces >= force_ranges[:, 1] - force_tolerance)
            )
            saturation_counts += saturated
            floor_contact_steps += int(contacts > 0)
            foot_force_peak = max(foot_force_peak, foot_force)
            other_force_peak = max(other_force_peak, other_force)
            joint_errors.append(data.qpos[joint_indices] - batch["qpos"][index + 1])
            final_root_error = data.qpos[:3] - batch["sim_root_pos"][index + 1]
            root_errors.append(final_root_error.copy())
            minimum_root_height = min(minimum_root_height, float(data.qpos[2]))
            minimum_torso_up = min(
                minimum_torso_up,
                float(data.xmat[env._torso_body_id].reshape(3, 3)[2, 1]),
            )
        samples = len(joint_errors)
        completed = simulated_steps == step_count and not reasons
        runs.append(
            {
                "start_phase": float(phase),
                "requested_seconds": cycles * duration,
                "simulated_seconds": simulated_steps * timestep,
                "survived_cycles": simulated_steps * timestep / duration,
                "completed_requested_horizon": completed,
                "stop_reasons": reasons or ["requested_horizon_completed"],
                "simulated_steps": simulated_steps,
                "finite_metric_samples": samples,
                "joint_position_rmse_rad": float(
                    np.sqrt(np.mean(np.square(joint_errors)))
                )
                if samples
                else None,
                "joint_position_max_error_rad": float(np.max(np.abs(joint_errors)))
                if samples
                else None,
                "root_position_rmse_m": float(np.sqrt(np.mean(np.square(root_errors))))
                if samples
                else None,
                "final_free_root_position_error_xyz_m": final_root_error.tolist(),
                "minimum_root_height_m": minimum_root_height,
                "minimum_torso_anatomical_up": minimum_torso_up,
                "actuator_peak_absolute_force": force_peaks.tolist(),
                "actuator_force_limit_sample_fraction": (
                    saturation_counts / max(samples, 1)
                ).tolist(),
                "initial_geometric_contact_count": initial_contacts,
                "active_floor_contact_sample_fraction": floor_contact_steps
                / max(samples, 1),
                "peak_total_foot_floor_normal_force_n": foot_force_peak,
                "peak_total_nonfoot_floor_normal_force_n": other_force_peak,
            }
        )
    return {
        "scope": (
            "Idealized native reference-position replay; not policy evaluation "
            "or an RL feasibility certificate."
        ),
        "root_assistance": "None after exact reference qpos/qvel initialization.",
        "clip_id": int(clip_id),
        "reference_duration_seconds": duration,
        "physics_timestep_seconds": timestep,
        "control_hold_seconds": hold_dt,
        "target_lookahead_seconds": target_lookahead_seconds,
        "control_semantics": (
            "Direct reference joint positions; existing native kp/kv/force limits; "
            "no action scaling, residual, smoothing, or gain changes."
        ),
        "termination": {
            "minimum_root_qpos_z_m": minimum_height,
            "minimum_torso_anatomical_up": 0.25,
            "other_conditions": (
                "Any nonfinite state/force or MuJoCo bad-qpos/qvel/qacc warning; "
                "no optional pose termination."
            ),
        },
        "force_sampling": (
            "actuator_force and solved contacts from each mj_step, before post-step "
            "kinematic refresh; saturation counts only for actuator_forcelimited=True."
        ),
        "actuator_names": [model.actuator(index).name for index in range(model.nu)],
        "actuator_force_limited": limited.tolist(),
        "actuator_force_ranges": force_ranges.tolist(),
        "all_phases_completed": all(run["completed_requested_horizon"] for run in runs),
        "runs": runs,
        "limitations": [
            "Native CPU physics, not MJX/Warp backend equivalence.",
            "Open-loop position replay lacks learned feedback, velocity feed-forward, "
            "and balance corrections.",
            "Actuator saturation does not include joint aggregate actuatorfrcrange "
            "or unmodeled hardware limits.",
            "Failure is evidence against this controller/reference pairing, "
            "not proof no policy can walk.",
        ],
    }
