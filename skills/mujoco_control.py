"""Explicit actuator-force bridge for the navigation model's hinge controls.

This maps physical MuJoCo control targets, not normalized policy actions. Passive
forces remain in the simulator and must not be added to this output a second time.
"""

import mujoco
import numpy as np


def validate_hinge_actuators(model: mujoco.MjModel) -> None:
    """Reject actuator semantics not implemented by the force bridge."""
    joints = model.actuator_trnid[:, 0]
    if (
        model.nu == 0
        or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
        or np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE)
    ):
        raise ValueError("Expected direct hinge-joint actuators")
    if (
        np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)
        or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
        or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
    ):
        raise ValueError("Expected stateless fixed-gain, affine-bias actuators")
    if np.any(model.jnt_actfrclimited) or np.any(model.jnt_actgravcomp):
        raise ValueError("Joint-level actuator force limits/gravcomp are unsupported")
    if model.opt.disableflags or model.opt.disableactuator:
        raise ValueError("Disabled physics/actuator features are unsupported")


def hinge_actuator_forces(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return actuator forces and generalized torques for one physical state.

    Call ``validate_hinge_actuators`` when loading the model. Position targets
    use actuator length units, including gear, as in MuJoCo ``data.ctrl``.
    Control clipping precedes force clipping; transmission then maps forces to
    joint torques. Passive springs/damping, constraints and gravity stay separate.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    for values, size in ((qpos, model.nq), (qvel, model.nv), (targets, model.nu)):
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite vector with shape ({size},)")
    joints = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joints]
    gear = model.actuator_gear[:, 0]
    length = gear * qpos[model.jnt_qposadr[joints]]
    velocity = gear * qvel[dofs]
    control = np.where(
        model.actuator_ctrllimited,
        np.clip(targets, *model.actuator_ctrlrange.T),
        targets,
    )
    force = (
        model.actuator_gainprm[:, 0] * control
        + model.actuator_biasprm[:, 0]
        + model.actuator_biasprm[:, 1] * length
        + model.actuator_biasprm[:, 2] * velocity
    )
    force = np.where(
        model.actuator_forcelimited,
        np.clip(force, *model.actuator_forcerange.T),
        force,
    )
    torque = np.zeros(model.nv)
    np.add.at(torque, dofs, gear * force)
    return force, torque
