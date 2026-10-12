"""Native reference root locking with an existing-contact support handoff."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from biomechanics_env import BiomechanicsJoystickEnv


def lock_with_contact_history(
    env: BiomechanicsJoystickEnv,
    joints: np.ndarray,
    roots: np.ndarray,
    quaternions: np.ndarray,
    count: int,
) -> np.ndarray:
    """Preserve an incoming support foot's previous world-space center.

    If the selected foot was within the height threshold in both frames, its
    previous position is an available anchor even when another foot was selected
    before. Genuine landings and flight inherit the input root displacement.
    Center locking does not constrain rotation or certify material-point slip.
    """
    locked = np.asarray(roots, dtype=np.float32).copy()
    if count <= 1:
        return locked
    data = mujoco.MjData(env._mj_model)
    feet = np.asarray(env._foot_geom_ids_np, dtype=int)
    threshold = float(env._config.get("reference_foot_lock_height", 0.035))
    active: int | None = None
    previous_contact = np.zeros(len(feet), dtype=bool)
    previous_local = np.zeros((len(feet), 2))
    for frame in range(count):
        data.qpos[:] = env._init_q_np
        data.qpos[:3] = roots[frame]
        data.qpos[3:7] = quaternions[frame]
        data.qpos[env._actuator_qpos_indices_np] = joints[frame]
        mujoco.mj_forward(env._mj_model, data)
        # Keep the historical height classification precision at the threshold.
        heights = np.array(
            [env._geom_min_z(data, int(foot)) for foot in feet], dtype=np.float32
        )
        contact = heights <= threshold
        local = data.geom_xpos[feet, :2] - roots[frame, :2]
        if not np.any(contact):
            active = None
        elif active is None or not contact[active]:
            active = int(np.argmin(np.where(contact, heights, np.inf)))
        if frame:
            displacement = roots[frame, :2] - roots[frame - 1, :2]
            if active is not None and previous_contact[active]:
                displacement = previous_local[active] - local[active]
            locked[frame, :2] = locked[frame - 1, :2] + displacement
        previous_contact = contact
        previous_local = local.copy()
    locked[count:] = locked[count - 1]
    return locked
