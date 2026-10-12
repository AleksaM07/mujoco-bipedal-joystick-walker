"""Counterfactual IK fits on captured frames; never alter the reference pipeline.

The production solver is intentionally not reproduced here. SciPy independently
minimizes its Cartesian/pose objective using the environment's native FK and
Jacobian. All fits freeze the root, passive joints and excluded trunk actuators.
Individual-frame convergence does not certify temporal smoothness or dynamics.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

from biomechanics_env import ARM_ACTUATED_JOINTS, LEG_ACTUATED_JOINTS


class _FrameObjective:
    """Small, independent native-MuJoCo least-squares objective for one frame."""

    def __init__(
        self,
        env: Any,
        markers: list[dict[str, Any]],
        full_pose: np.ndarray,
        targets: np.ndarray,
        active: np.ndarray,
        seed: np.ndarray,
        previous: np.ndarray | None,
        pose_reg: float,
        smooth_reg: float,
    ) -> None:
        self.env = env
        self.data = mujoco.MjData(env._mj_model)
        self.full_pose = full_pose.copy()
        self.markers = markers
        self.targets = targets.copy()
        for index, marker in enumerate(markers):
            if marker["is_foot"]:
                self.targets[index, 2] = max(
                    self.targets[index, 2], float(env.FOOT_CONTACT_HEIGHT)
                )
        self.active = active
        self.seed = seed[active]
        self.previous = None if previous is None else previous[active]
        self.pose_reg = pose_reg
        self.smooth_reg = smooth_reg
        self.weights = np.sqrt([float(marker["weight"]) for marker in markers])

    def forward(self, values: np.ndarray) -> None:
        """Set only selected scalar actuator coordinates on a private data object."""
        self.data.qpos[:] = self.full_pose
        self.data.qpos[self.env._actuator_qpos_indices_np[self.active]] = values
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env._mj_model, self.data)

    def marker_errors(self, values: np.ndarray) -> np.ndarray:
        """Return unweighted XYZ marker errors in meters."""
        self.forward(values)
        return (
            np.asarray(
                [
                    self.env._host_marker_position(self.data, marker).copy()
                    for marker in self.markers
                ]
            )
            - self.targets
        )

    def residual(self, values: np.ndarray) -> np.ndarray:
        """Return weighted Cartesian errors and optional pose penalties."""
        rows = [(self.marker_errors(values) * self.weights[:, None]).ravel()]
        if self.pose_reg:
            rows.append(np.sqrt(self.pose_reg) * (values - self.seed))
        if self.previous is not None and self.smooth_reg:
            rows.append(np.sqrt(self.smooth_reg) * (values - self.previous))
        return np.concatenate(rows)

    def jacobian(self, values: np.ndarray) -> np.ndarray:
        """Differentiate the residual through actual MuJoCo marker Jacobians."""
        self.forward(values)
        rows = [
            weight * self.env._host_marker_jacobian(self.data, marker)[:, self.active]
            for weight, marker in zip(self.weights, self.markers, strict=True)
        ]
        identity = np.eye(len(values))
        if self.pose_reg:
            rows.append(np.sqrt(self.pose_reg) * identity)
        if self.previous is not None and self.smooth_reg:
            rows.append(np.sqrt(self.smooth_reg) * identity)
        return np.vstack(rows)


def _pose_metrics(
    objective: _FrameObjective,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    errors = objective.marker_errors(values)
    env, data = objective.env, objective.data
    collisions = []
    for contact in data.contact:
        if contact.dist < -1e-6:
            collisions.append(
                {
                    "geoms": [
                        env._mj_model.geom(int(contact.geom1)).name,
                        env._mj_model.geom(int(contact.geom2)).name,
                    ],
                    "distance_m": float(contact.dist),
                }
            )
    joint_values = data.qpos[env._actuator_qpos_indices_np]
    joint_ids = env._mj_model.actuator_trnid[:, 0]
    joint_limits = env._mj_model.jnt_range[joint_ids]
    violations = np.maximum(joint_limits[:, 0] - joint_values, 0) + np.maximum(
        joint_values - joint_limits[:, 1], 0
    )
    violations = np.where(env._mj_model.jnt_limited[joint_ids], violations, 0)
    names = np.asarray(env._actuator_joint_names)[objective.active]
    return {
        "weighted_marker_squared_error_m2": float(
            np.sum((errors * objective.weights[:, None]) ** 2)
        ),
        "objective_squared_norm": float(np.sum(objective.residual(values) ** 2)),
        "marker_errors": {
            str(marker["name"]): {
                "xyz_m": error.tolist(),
                "distance_m": float(np.linalg.norm(error)),
            }
            for marker, error in zip(objective.markers, errors, strict=True)
        },
        "active_lower_bounds": names[np.abs(values - lower) <= 1e-6].tolist(),
        "active_upper_bounds": names[np.abs(values - upper) <= 1e-6].tolist(),
        "max_physical_joint_limit_violation_rad": float(violations.max(initial=0)),
        "penetrating_contacts": collisions,
        "actuator_pose_rad": joint_values.tolist(),
    }


def inspect_ik_fit(
    env: Any,
    metadata: dict[str, Any],
    stages: list[dict[str, Any]],
    clip_id: int = 0,
    frame_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Compare saved IK against converged regularized and unregularized fits.

    Inputs come directly from ``capture_retarget_stages``. Each frame uses its
    actual captured raw seed and previous production IK pose; counterfactual
    frames are never fed into the next frame. Production control bounds remain
    intact, with physical joint-limit violations reported separately. Reported
    costs exclude frozen-coordinate regularization constants. No collision or
    force term is optimized; those contacts are measured after each fit.
    """
    captured = {stage["name"]: stage for stage in stages}
    required = {"normalized_ik_input", "ik_solution", "stability_prior"}
    if not required.issubset(captured):
        raise ValueError(
            f"Missing captured stages: {sorted(required - captured.keys())}"
        )
    if not 0 <= clip_id < len(metadata["counts"]):
        raise ValueError("clip_id is outside the captured clips.")
    count = int(metadata["counts"][clip_id])
    if frame_ids is None:
        frame_ids = np.unique(
            np.linspace(0, count - 1, min(count, 9), dtype=int)
        ).tolist()
    if not frame_ids or any(
        not isinstance(frame, (int, np.integer)) or not 0 <= frame < count
        for frame in frame_ids
    ):
        raise ValueError("frame_ids must be nonempty valid integer frame indices.")
    frame_ids = [int(frame) for frame in frame_ids]
    names = tuple(metadata["names"])
    markers = env._resolve_host_ik_markers(names)
    if tuple(marker["name"] for marker in markers) != names:
        raise ValueError("Unresolved or reordered host IK marker names.")
    active = np.flatnonzero(
        [
            name in LEG_ACTUATED_JOINTS or name in ARM_ACTUATED_JOINTS
            for name in env._actuator_joint_names
        ]
    )
    lower = np.asarray(env._actuator_ctrl_lower_limits_np, dtype=float)[active]
    upper = np.asarray(env._actuator_ctrl_upper_limits_np, dtype=float)[active]
    if (
        not len(active)
        or np.any(lower >= upper)
        or np.isnan(lower).any()
        or np.isnan(upper).any()
    ):
        raise ValueError("Active IK coordinates require valid nonzero-width bounds.")
    joint_ids = env._mj_model.actuator_trnid[:, 0]
    if not np.isin(
        env._mj_model.jnt_type[joint_ids],
        [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE],
    ).all():
        raise ValueError("This diagnostic requires scalar actuator joint coordinates.")

    source, fitted, prior = (
        captured[name]
        for name in ("normalized_ik_input", "ik_solution", "stability_prior")
    )
    warmstart = getattr(env, "_reference_ik_warmstart_diagnostics", [])
    rows = []
    for frame in frame_ids:
        seed = np.asarray(source["joints"][clip_id, frame], dtype=float)
        baseline = np.asarray(fitted["joints"][clip_id, frame], dtype=float)
        previous = None if frame == 0 else fitted["joints"][clip_id, frame - 1]
        if frame == 0 and warmstart:
            initial_previous = warmstart[clip_id]["initial_previous_pose_rad"]
            if initial_previous is not None:
                previous = np.asarray(initial_previous, dtype=float)
        full_pose = np.asarray(env._init_q_np, dtype=float).copy()
        full_pose[:3] = source["roots"][clip_id, frame]
        full_pose[3:7] = source["quaternions"][clip_id, frame]
        full_pose[env._actuator_qpos_indices_np] = baseline
        targets = source["targets"][clip_id, frame]
        if not all(np.isfinite(array).all() for array in (full_pose, targets, seed)):
            raise ValueError("Captured IK frame contains non-finite values.")
        arguments = (env, markers, full_pose, targets, active, seed, previous)
        regularized = _FrameObjective(*arguments, pose_reg=0.20, smooth_reg=0.08)
        unregularized = _FrameObjective(*arguments, pose_reg=0.0, smooth_reg=0.0)
        row = {
            "frame_id": int(frame),
            "previous_pose_rad": None if previous is None else previous.tolist(),
            "source_frame_id": int(metadata["source_start_frames"][clip_id])
            + int(frame),
            "root_xyz_m": full_pose[:3].tolist(),
            "production_ik": _pose_metrics(regularized, baseline[active], lower, upper),
            "fits": {},
        }
        prior_pose = full_pose.copy()
        prior_pose[env._actuator_qpos_indices_np] = prior["joints"][clip_id, frame]
        prior_objective = _FrameObjective(
            env, markers, prior_pose, targets, active, seed, previous, 0.0, 0.0
        )
        row["stability_prior"] = _pose_metrics(
            prior_objective, prior["joints"][clip_id, frame, active], lower, upper
        )
        for label, objective in (
            ("converged_regularized", regularized),
            ("no_regularization", unregularized),
        ):
            starts = [baseline[active]]
            if label == "no_regularization":
                starts.append(seed[active])
            candidates = [
                least_squares(
                    objective.residual,
                    np.clip(start, lower, upper),
                    jac=objective.jacobian,
                    bounds=(lower, upper),
                    ftol=1e-11,
                    xtol=1e-11,
                    gtol=1e-11,
                    max_nfev=300,
                )
                for start in starts
            ]
            best = min(candidates, key=lambda result: result.cost)
            row["fits"][label] = {
                **_pose_metrics(objective, best.x, lower, upper),
                "success": bool(best.success),
                "status": int(best.status),
                "message": str(best.message),
                "function_evaluations": int(best.nfev),
                "optimality": float(best.optimality),
                "multistart_costs": [float(candidate.cost) for candidate in candidates],
                "joint_change_from_production_l2_rad": float(
                    np.linalg.norm(best.x - baseline[active])
                ),
            }
        rows.append(row)
    return {
        "clip_id": clip_id,
        "selected_frames": frame_ids,
        "active_joint_names": np.asarray(env._actuator_joint_names)[active].tolist(),
        "bounds_lower_rad": lower.tolist(),
        "bounds_upper_rad": upper.tolist(),
        "production_pose_reg": 0.20,
        "production_smooth_reg": 0.08,
        "production_max_iterations": 10,
        "production_max_coordinate_update_rad": 0.12,
        "limitations": [
            "Root, passive joints and excluded trunk coordinates are frozen.",
            "Each frame freezes the actual previous production IK pose, not a refit.",
            "Multistart local optimization is not a proof of global reachability.",
            "No dynamics, temporal continuity or collision constraints are optimized.",
            "The stability-prior fit is measured before subsequent floor alignment.",
            "Costs omit constant regularization of excluded coordinates.",
        ],
        "frames": rows,
    }
