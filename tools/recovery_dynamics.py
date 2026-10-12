"""Native inverse-dynamics diagnostics for a prescribed reference trajectory.

``qfrc_inverse`` is the generalized force still required after MuJoCo's passive
and constraint forces. It is not an actuator's scalar force. For the supported
one-actuator-per-hinge/slide transmissions, tau = gear * actuator_force.

This is an exact-trajectory diagnostic under the model's current soft-contact
law, not proof that no nearby trajectory or feedback controller is feasible.
See https://mujoco.readthedocs.io/en/stable/computation.html#actuation-model and
https://mujoco.readthedocs.io/en/stable/programming/simulation.html#inverse-dynamics.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def _stats(values: np.ndarray) -> dict[str, float]:
    """Summarize a nonempty scalar series without mixing physical units."""
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "abs_max": float(np.max(np.abs(values))),
        "abs_p95": float(np.percentile(np.abs(values), 95)),
    }


def _limit_summary(
    values: np.ndarray, limits: np.ndarray, enabled: bool
) -> dict[str, Any]:
    """Check both sides of an asymmetric range, only when clamping is enabled."""
    if not enabled:
        return {"enabled": False, "range": None, "exceedance": None}
    excess = np.maximum(np.maximum(limits[0] - values, values - limits[1]), 0.0)
    return {
        "enabled": True,
        "range": limits.tolist(),
        "exceedance": {
            "sample_count": int(np.count_nonzero(excess > 1e-8)),
            "sample_fraction": float(np.mean(excess > 1e-8)),
            "max": float(excess.max()),
        },
    }


def _validate_model(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Reject models for which scalar actuator inversion would be misleading."""
    if model.nmocap or model.nplugin:
        raise ValueError(
            "Mocap bodies and plugins require additional state; unsupported."
        )
    if mujoco.get_mjcb_passive() is not None:
        raise ValueError("Custom passive-force callbacks are unsupported.")
    unsupported_flags = (
        mujoco.mjtEnableBit.mjENBL_INVDISCRETE | mujoco.mjtEnableBit.mjENBL_SLEEP
    )
    if model.opt.enableflags & unsupported_flags:
        raise ValueError("Discrete inverse dynamics and sleeping are unsupported.")
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION:
        raise ValueError("Disabled actuation is unsupported.")
    if np.any(model.jnt_actgravcomp):
        raise ValueError("Actuator-routed gravity compensation is unsupported.")
    joints: list[int] = []
    dofs: list[int] = []
    for index in range(model.nu):
        if model.actuator_trntype[index] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"Actuator {index}: only joint transmission is supported.")
        joint = int(model.actuator_trnid[index, 0])
        if model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Actuator {index}: requires a hinge or slide joint.")
        if joint in joints:
            raise ValueError(
                "Multiple actuators on one joint require force allocation."
            )
        gear = model.actuator_gear[index, 0]
        if not np.isfinite(gear) or abs(gear) < 1e-12:
            raise ValueError(f"Actuator {index}: zero or nonfinite transmission gear.")
        if model.actuator_dyntype[index] != mujoco.mjtDyn.mjDYN_NONE:
            raise ValueError(f"Actuator {index}: stateful dynamics are unsupported.")
        if model.actuator_delay[index] != 0:
            raise ValueError(f"Actuator {index}: delayed control is unsupported.")
        if model.actuator_gaintype[index] not in (
            mujoco.mjtGain.mjGAIN_FIXED,
            mujoco.mjtGain.mjGAIN_AFFINE,
        ) or model.actuator_biastype[index] not in (
            mujoco.mjtBias.mjBIAS_NONE,
            mujoco.mjtBias.mjBIAS_AFFINE,
        ):
            raise ValueError(f"Actuator {index}: unsupported gain/bias law.")
        group = int(model.actuator_group[index])
        if 0 <= group < 31 and model.opt.disableactuator & (1 << group):
            raise ValueError(f"Actuator {index}: disabled actuator group.")
        joints.append(joint)
        dofs.append(int(model.jnt_dofadr[joint]))
    return np.asarray(joints, dtype=int), np.asarray(dofs, dtype=int)


def inspect_dynamics(
    model: mujoco.MjModel,
    qpos_batch: np.ndarray,
    qvel_batch: np.ndarray,
    qacc_batch: np.ndarray,
) -> dict[str, Any]:
    """Measure exact-reference inverse forces and independent actuator limits.

    Args:
        model: Native model; not modified. Contact settings are used as-is.
        qpos_batch: Finite configurations with shape (samples, model.nq).
        qvel_batch: Generalized velocities, shape (samples, model.nv).
        qacc_batch: Time derivatives of qvel, shape (samples, model.nv).

    Returns:
        JSON-compatible summary. Free-joint force and torque norms are reported
        separately; unactuated hinge/slide residuals retain their own units.
        Controls are raw solutions to force = gain * ctrl + bias. Native forward
        actuation then checks the effect of active control, force and joint limits.

    Raises:
        ValueError: Invalid samples or unsupported actuator/model configuration.
        FloatingPointError: Dynamics or actuator inversion produces nonfinite data.
    """
    joint_ids, dof_ids = _validate_model(model)
    positions = np.asarray(qpos_batch, dtype=np.float64)
    velocities = np.asarray(qvel_batch, dtype=np.float64)
    accelerations = np.asarray(qacc_batch, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != model.nq or not len(positions):
        raise ValueError(f"qpos_batch must have nonempty shape (samples, {model.nq}).")
    count = len(positions)
    for name, values in (("qvel", velocities), ("qacc", accelerations)):
        if values.shape != (count, model.nv):
            raise ValueError(f"{name}_batch must have shape ({count}, {model.nv}).")
    if not all(
        np.isfinite(array).all() for array in (positions, velocities, accelerations)
    ):
        raise ValueError("Trajectory samples must be finite.")
    for joint in range(model.njnt):
        kind = model.jnt_type[joint]
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            address = int(model.jnt_qposadr[joint])
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                address += 3
            norms = np.linalg.norm(positions[:, address : address + 4], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
                raise ValueError(f"Joint {joint}: qpos quaternions must be normalized.")

    required = np.empty((count, model.nv))
    scalar_force = np.empty((count, model.nu))
    control = np.empty((count, model.nu))
    achieved = np.empty((count, model.nv))
    data = mujoco.MjData(model)
    gear = np.asarray(model.actuator_gear[:, 0])
    for sample in range(count):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = positions[sample]
        data.qvel[:] = velocities[sample]
        data.qacc[:] = accelerations[sample]
        mujoco.mj_inverse(model, data)
        required[sample] = data.qfrc_inverse
        scalar_force[sample] = required[sample, dof_ids] / gear
        # Native sparse moments verify that the assumed scalar map really holds.
        for actuator, dof in enumerate(dof_ids):
            address = data.moment_rowadr[actuator]
            if (
                data.moment_rownnz[actuator] != 1
                or data.moment_colind[address] != dof
                or not np.isclose(data.actuator_moment[address], gear[actuator])
            ):
                raise ValueError("Unexpected actuator moment matrix; unsupported map.")
        gain = model.actuator_gainprm[:, 0].copy()
        affine = model.actuator_gaintype == mujoco.mjtGain.mjGAIN_AFFINE
        gain[affine] += (
            model.actuator_gainprm[affine, 1] * data.actuator_length[affine]
            + model.actuator_gainprm[affine, 2] * data.actuator_velocity[affine]
        )
        bias = np.zeros(model.nu)
        affine = model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE
        bias[affine] = (
            model.actuator_biasprm[affine, 0]
            + model.actuator_biasprm[affine, 1] * data.actuator_length[affine]
            + model.actuator_biasprm[affine, 2] * data.actuator_velocity[affine]
        )
        if np.any(np.abs(gain) < 1e-12) or not np.isfinite(gain).all():
            raise ValueError(f"Sample {sample}: singular/nonfinite actuator gain.")
        control[sample] = (scalar_force[sample] - bias) / gain
        data.ctrl[:] = control[sample]
        mujoco.mj_fwdActuation(model, data)
        achieved[sample] = data.qfrc_actuator
    if not all(
        np.isfinite(array).all()
        for array in (required, scalar_force, control, achieved)
    ):
        raise FloatingPointError(
            "Inverse dynamics or actuator inversion was nonfinite."
        )

    ctrl_clamping = not bool(
        model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
    )
    actuators = []
    force_fail = np.zeros(count, dtype=bool)
    ctrl_fail = np.zeros(count, dtype=bool)
    for actuator, (joint, dof) in enumerate(zip(joint_ids, dof_ids, strict=True)):
        force_limit = _limit_summary(
            scalar_force[:, actuator],
            model.actuator_forcerange[actuator],
            bool(model.actuator_forcelimited[actuator]),
        )
        ctrl_limit = _limit_summary(
            control[:, actuator],
            model.actuator_ctrlrange[actuator],
            bool(model.actuator_ctrllimited[actuator]) and ctrl_clamping,
        )
        if force_limit["enabled"]:
            lower, upper = model.actuator_forcerange[actuator]
            force_fail |= (scalar_force[:, actuator] < lower - 1e-8) | (
                scalar_force[:, actuator] > upper + 1e-8
            )
        if ctrl_limit["enabled"]:
            lower, upper = model.actuator_ctrlrange[actuator]
            ctrl_fail |= (control[:, actuator] < lower - 1e-8) | (
                control[:, actuator] > upper + 1e-8
            )
        actuators.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                or f"actuator_{actuator}",
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
                or f"joint_{joint}",
                "dof": int(dof),
                "gear": float(gear[actuator]),
                "generalized_force_unit": "Nm"
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                else "N",
                "required_generalized_force": _stats(required[:, dof]),
                "required_scalar_force": _stats(scalar_force[:, actuator]),
                "required_ctrl": _stats(control[:, actuator]),
                "reference_velocity": _stats(velocities[:, dof]),
                "reference_acceleration": _stats(accelerations[:, dof]),
                "force_limit": force_limit,
                "ctrl_limit": ctrl_limit,
                "joint_actuator_force_limit": _limit_summary(
                    required[:, dof],
                    model.jnt_actfrcrange[joint],
                    bool(model.jnt_actfrclimited[joint]),
                ),
                "generalized_force_shortfall_after_native_clamps": _stats(
                    required[:, dof] - achieved[:, dof]
                ),
            }
        )

    unactuated = []
    free_joints = []
    for dof in sorted(set(range(model.nv)) - set(dof_ids)):
        joint = int(model.dof_jntid[dof])
        kind = model.jnt_type[joint]
        component = dof - int(model.jnt_dofadr[joint])
        linear = kind == mujoco.mjtJoint.mjJNT_SLIDE or (
            kind == mujoco.mjtJoint.mjJNT_FREE and component < 3
        )
        unactuated.append(
            {
                "dof": dof,
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                or f"joint_{joint}",
                "component": component,
                "unit": "N" if linear else "Nm",
                "required_force": _stats(required[:, dof]),
            }
        )
    for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        address = int(model.jnt_dofadr[joint])
        free_joints.append(
            {
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
                or f"joint_{joint}",
                "force_norm_N": _stats(
                    np.linalg.norm(required[:, address : address + 3], axis=1)
                ),
                "torque_norm_Nm": _stats(
                    np.linalg.norm(required[:, address + 3 : address + 6], axis=1)
                ),
                "linear_acceleration_norm_m_s2": _stats(
                    np.linalg.norm(accelerations[:, address : address + 3], axis=1)
                ),
                "angular_acceleration_norm_rad_s2": _stats(
                    np.linalg.norm(accelerations[:, address + 3 : address + 6], axis=1)
                ),
            }
        )
    # Diagnose the force decomposition at separate linear/rotational extrema;
    # selecting these samples does not impose a mixed-unit feasibility threshold.
    selected: dict[str, int] = {}
    if model.nu:
        selected["largest_absolute_actuator_scalar_force"] = int(
            np.unravel_index(np.argmax(np.abs(scalar_force)), scalar_force.shape)[0]
        )
    for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        address = int(model.jnt_dofadr[joint])
        selected[f"free_joint_{joint}_force_peak"] = int(
            np.argmax(np.linalg.norm(required[:, address : address + 3], axis=1))
        )
        selected[f"free_joint_{joint}_torque_peak"] = int(
            np.argmax(np.linalg.norm(required[:, address + 3 : address + 6], axis=1))
        )
    extremes = {}
    for reason, sample in selected.items():
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.qacc[:] = (
            positions[sample],
            velocities[sample],
            accelerations[sample],
        )
        mujoco.mj_inverse(model, data)
        inertial_force = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, inertial_force, data.qacc)
        contact_rows = []
        for index in range(data.ncon):
            contact = data.contact[index]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            names = []
            for geom in (contact.geom1, contact.geom2):
                body = int(model.geom_bodyid[geom])
                names.append(
                    model.geom(int(geom)).name or f"{model.body(body).name}/geom{geom}"
                )
            contact_rows.append(
                {
                    "geoms": names,
                    "distance_m": float(contact.dist),
                    "normal_force_n": float(force[0]),
                    "tangential_force_norm_n": float(np.linalg.norm(force[1:3])),
                }
            )
        contact_rows.sort(key=lambda row: abs(row["normal_force_n"]), reverse=True)
        extremes[reason] = {
            "sample_index": sample,
            "qacc": data.qacc.tolist(),
            "mass_times_qacc": inertial_force.tolist(),
            "bias": data.qfrc_bias.tolist(),
            "passive": data.qfrc_passive.tolist(),
            "constraint": data.qfrc_constraint.tolist(),
            "inverse": data.qfrc_inverse.tolist(),
            "equation_error_abs_max": float(
                np.max(
                    np.abs(
                        data.qfrc_inverse
                        - (
                            inertial_force
                            + data.qfrc_bias
                            - data.qfrc_passive
                            - data.qfrc_constraint
                        )
                    )
                )
            ),
            "contact_count": int(data.ncon),
            "largest_contact_normal_forces": contact_rows[:8],
        }
    return {
        "samples": count,
        "interpretation": (
            "Exact prescribed qpos/qvel/qacc under current MuJoCo soft-contact law; "
            "not a proof that nearby reference tracking or walking is impossible. "
            "Passive and constraint forces are already subtracted from qfrc_inverse. "
            "Unactuated generalized residual requires external force or changed motion."
        ),
        "any_actuator_force_limit_exceeded_fraction": float(force_fail.mean()),
        "any_actuator_ctrl_limit_exceeded_fraction": float(ctrl_fail.mean()),
        "actuators": actuators,
        "unactuated_dofs": unactuated,
        "free_joint_residuals": free_joints,
        "extreme_sample_force_decomposition": extremes,
    }
