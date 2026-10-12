"""Opt-in continuous reference curves for the recovery experiment.

The historical interpolators are deliberately not changed by this module.
WRAP clips include a duplicate endpoint, so their period is ``(count - 1) * dt``.
Bounded closure is an explicit geometric edit, not a contact/dynamics guarantee.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jp
import numpy as np

from reference_kinematics import scale_reference_frame_times

POSITION_TOLERANCE = 1e-6
ORIENTATION_TOLERANCE = 1e-6
MAX_JOINT_CLOSURE = 0.2
MAX_ROOT_HEIGHT_CLOSURE = 0.02
MAX_ORIENTATION_CLOSURE = 0.2


def _monotone_tangents(
    positions: np.ndarray, interval: float, periodic: bool
) -> np.ndarray:
    """Return uniform-grid PCHIP tangents with shared periodic or zero clamp ends."""
    result = np.zeros_like(positions)
    if len(positions) <= 1:
        return result
    slopes = np.diff(positions, axis=0) / interval

    def harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        same_direction = (left > 0.0) & (right > 0.0) | ((left < 0.0) & (right < 0.0))
        denominator = np.where(same_direction, left + right, 1.0)
        return np.where(same_direction, 2.0 * left * right / denominator, 0.0)

    if len(positions) > 2:
        result[1:-1] = harmonic(slopes[:-1], slopes[1:])
    if periodic:
        result[0] = harmonic(slopes[-1], slopes[0])
        result[-1] = result[0]
    return result


def _quaternion_tangents(
    quaternions: np.ndarray, interval: float, periodic: bool
) -> np.ndarray:
    """Return shared nodal derivatives projected onto the unit quaternion sphere."""
    result = np.zeros_like(quaternions)
    if len(quaternions) > 2:
        result[1:-1] = (quaternions[2:] - quaternions[:-2]) / (2.0 * interval)
    if periodic and len(quaternions) > 2:
        result[0] = (quaternions[1] - quaternions[-2]) / (2.0 * interval)
        result[-1] = result[0]
    result -= quaternions * np.sum(quaternions * result, axis=-1, keepdims=True)
    return result


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 1e-12):
        raise ValueError("Reference quaternions must have finite nonzero norm.")
    return quaternions / lengths


def _orientation_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sign-invariant quaternion angle, stable near coincident orientations."""
    sign = np.where(np.sum(left * right, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    aligned = right * sign
    return 4.0 * np.arctan2(
        np.linalg.norm(left - aligned, axis=-1),
        np.linalg.norm(left + aligned, axis=-1),
    )


def prepare_reference_curves(
    joint_positions: np.ndarray,
    root_positions: np.ndarray,
    root_quaternions: np.ndarray,
    frame_times: np.ndarray,
    frame_counts: np.ndarray,
    loop_modes: np.ndarray,
    wrap_deltas: np.ndarray,
    loop_closure: Literal["strict", "bounded"] = "strict",
    joint_lower_limits: np.ndarray | None = None,
    joint_upper_limits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build C1 scalar and normalized-quaternion cubic reference curves.

    Scalar coordinates use monotone cubic Hermite interpolation. A wrapping
    root's XY drift is removed before its residual is made periodic. Clamp
    endpoints have zero tangents, allowing a C1 transition into a constant hold.
    Quaternion component cubics are normalized at query time; their exact
    derivative supplies local free-joint angular velocity.

    ``strict`` rejects nonperiodic WRAP endpoints. ``bounded`` explicitly closes
    gaps of at most 0.2 rad per joint, 0.02 m root height, and 0.2 rad orientation
    using distributed smooth corrections. Those engineering bounds do not prove
    valid contact, actuator feasibility, or a dynamically trackable trajectory.
    WRAP XY displacement must already match ``wrap_deltas``; net Z must be zero.

    Inputs are never modified. Position padding is preserved; tangent padding is
    zero and must not be queried. Quaternion sign normalization only touches valid
    samples. The returned arrays are float32 and diagnostics contain plain values.

    Raises:
        ValueError: Invalid shapes/data, unsupported loop modes, inconsistent
            translation drift, an antipodal quaternion seam, or excessive closure.
    """
    if loop_closure not in {"strict", "bounded"}:
        raise ValueError("loop_closure must be 'strict' or 'bounded'.")
    joints = np.array(joint_positions, dtype=np.float64, copy=True)
    roots = np.array(root_positions, dtype=np.float64, copy=True)
    quats = np.array(root_quaternions, dtype=np.float64, copy=True)
    if joints.ndim != 3 or any(size == 0 for size in joints.shape):
        raise ValueError("joint_positions must have shape (clips, frames, joints).")
    clips, padded_frames, _ = joints.shape
    if roots.shape != (clips, padded_frames, 3):
        raise ValueError("root_positions must have shape (clips, frames, 3).")
    if quats.shape != (clips, padded_frames, 4):
        raise ValueError("root_quaternions must have shape (clips, frames, 4).")
    intervals = scale_reference_frame_times(frame_times, 1.0)
    counts = np.asarray(frame_counts)
    modes = np.asarray(loop_modes)
    deltas = np.asarray(wrap_deltas, dtype=np.float64)
    if intervals.shape != (clips,) or counts.shape != (clips,):
        raise ValueError("frame_times and frame_counts must match the clip count.")
    if (
        not np.all(np.isfinite(counts))
        or np.any(counts != np.floor(counts))
        or np.any(counts < 1)
        or np.any(counts > padded_frames)
    ):
        raise ValueError("frame_counts must be integers in [1, padded_frames].")
    if modes.shape != (clips,) or not np.all(np.isin(modes, (0, 1))):
        raise ValueError("loop_modes must contain CLAMP=0 or WRAP=1 per clip.")
    if deltas.shape != (clips, 3) or not np.all(np.isfinite(deltas)):
        raise ValueError("wrap_deltas must contain three finite coordinates per clip.")
    if (joint_lower_limits is None) != (joint_upper_limits is None):
        raise ValueError("Provide both joint limit arrays, or neither.")
    lower = np.full(joints.shape[-1], -np.inf)
    upper = np.full(joints.shape[-1], np.inf)
    if joint_lower_limits is not None:
        lower = np.asarray(joint_lower_limits, dtype=np.float64)
        upper = np.asarray(joint_upper_limits, dtype=np.float64)
        if (
            lower.shape != (joints.shape[-1],)
            or upper.shape != lower.shape
            or np.any(np.isnan(lower))
            or np.any(np.isnan(upper))
            or np.any(lower > upper)
        ):
            raise ValueError(
                "Joint limits must be ordered arrays matching joint count."
            )
    joint_tangents = np.zeros_like(joints)
    root_tangents = np.zeros_like(roots)
    quat_tangents = np.zeros_like(quats)
    diagnostics: list[dict[str, Any]] = []
    for clip, count in enumerate(counts.astype(np.int64)):
        interval = float(intervals[clip])
        joint = joints[clip, :count]
        root = roots[clip, :count]
        quat = quats[clip, :count]
        if not all(np.all(np.isfinite(value)) for value in (joint, root, quat)):
            raise ValueError(f"Clip {clip} contains non-finite valid samples.")
        quat[:] = _normalize_quaternions(quat)
        for frame in range(1, count):
            if np.dot(quat[frame - 1], quat[frame]) < 0.0:
                quat[frame] *= -1.0
        original_joint = joint.copy()
        original_root = root.copy()
        original_quat = quat.copy()
        source_violation = float(np.max(np.maximum(lower - joint, joint - upper)))
        if source_violation > POSITION_TOLERANCE:
            raise ValueError(f"Clip {clip}: source joint nodes violate limits.")
        periodic = bool(modes[clip] == 1)
        phase = np.linspace(0.0, 1.0, count)[:, None]
        delta = deltas[clip] if periodic else np.zeros(3)
        residual_root = root - phase * delta
        joint_gap = joint[-1] - joint[0]
        root_gap = residual_root[-1] - residual_root[0]
        orientation_gap = float(_orientation_distance(quat[0], quat[-1]))
        joint_velocity_gap = 0.0
        if count > 1:
            joint_velocity_gap = float(
                np.max(np.abs((joint[-1] - joint[-2] - joint[1] + joint[0]) / interval))
            )
        if periodic:
            if delta[2] != 0.0:
                raise ValueError(f"Clip {clip}: WRAP delta must have zero net Z.")
            if count == 1 and np.any(np.abs(delta) > POSITION_TOLERANCE):
                raise ValueError(
                    f"Clip {clip}: a one-frame WRAP clip cannot translate."
                )
            if np.max(np.abs(root_gap[:2])) > POSITION_TOLERANCE:
                raise ValueError(
                    f"Clip {clip}: WRAP XY delta does not match endpoint displacement."
                )
            if np.dot(quat[0], quat[-1]) < 0.0:
                raise ValueError(
                    f"Clip {clip}: antipodal quaternion seam/whole-spin lift "
                    "is unsupported."
                )
            joint_limit = (
                MAX_JOINT_CLOSURE if loop_closure == "bounded" else POSITION_TOLERANCE
            )
            root_limit = (
                MAX_ROOT_HEIGHT_CLOSURE
                if loop_closure == "bounded"
                else POSITION_TOLERANCE
            )
            rotation_limit = (
                MAX_ORIENTATION_CLOSURE
                if loop_closure == "bounded"
                else ORIENTATION_TOLERANCE
            )
            if (
                np.max(np.abs(joint_gap)) > joint_limit
                or abs(float(root_gap[2])) > root_limit
                or orientation_gap > rotation_limit
            ):
                raise ValueError(
                    f"Clip {clip}: {loop_closure} loop closure rejected "
                    f"(joint={np.max(np.abs(joint_gap)):.6g} rad, "
                    f"root_z={abs(float(root_gap[2])):.6g} m, "
                    f"orientation={orientation_gap:.6g} rad)."
                )
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            joint += (0.5 - smooth_phase) * joint_gap
            residual_root += (0.5 - smooth_phase) * root_gap
            shared_quat = _normalize_quaternions(quat[0] + quat[-1])
            quat += (1.0 - smooth_phase) * (shared_quat - quat[0]) + smooth_phase * (
                shared_quat - quat[-1]
            )
            quat[:] = _normalize_quaternions(quat)
            # Explicit equality avoids floating-point reintroducing a seam.
            joint[-1] = joint[0]
            residual_root[-1] = residual_root[0]
            quat[-1] = quat[0]
            root[:] = residual_root + phase * delta
            # Vertical wrap drift is forbidden, including tiny numeric noise.
            root[-1, 2] = root[0, 2]
        # Project the edited NODES, not runtime output: recomputing monotone
        # tangents then keeps both exact derivatives and between-node limits.
        before_projection = joint.copy()
        joint[:] = np.clip(joint, lower, upper)
        projection = np.abs(joint - before_projection)
        joint_tangents[clip, :count] = _monotone_tangents(joint, interval, periodic)
        if periodic and count > 1:
            root_tangents[clip, :count] = _monotone_tangents(
                residual_root, interval, True
            ) + delta / ((count - 1) * interval)
        else:
            root_tangents[clip, :count] = _monotone_tangents(root, interval, False)
        quat_tangents[clip, :count] = _quaternion_tangents(quat, interval, periodic)
        diagnostics.append(
            {
                "clip_id": clip,
                "frame_count": int(count),
                "loop_mode": "wrap" if periodic else "clamp",
                "loop_closure": loop_closure,
                "source_joint_limit_violation_rad": max(source_violation, 0.0),
                "joint_limit_projected_coordinates": int(np.count_nonzero(projection)),
                "max_joint_limit_projection_rad": float(np.max(projection)),
                "original_joint_gap_rad": float(np.max(np.abs(joint_gap))),
                "original_root_gap_m": float(np.linalg.norm(root_gap)),
                "original_root_z_gap_m": abs(float(root_gap[2])),
                "original_orientation_gap_rad": orientation_gap,
                "original_joint_velocity_gap_rad_s": joint_velocity_gap,
                "max_joint_correction_rad": float(
                    np.max(np.abs(joint - original_joint))
                ),
                "max_root_correction_m": float(
                    np.max(np.linalg.norm(root - original_root, axis=-1))
                ),
                "max_orientation_correction_rad": float(
                    np.max(_orientation_distance(quat, original_quat))
                ),
                "closure_applied": bool(
                    periodic
                    and (np.any(joint_gap) or np.any(root_gap) or orientation_gap > 0.0)
                ),
            }
        )
    arrays = {
        "joint_positions": joints,
        "joint_tangents": joint_tangents,
        "root_positions": roots,
        "root_tangents": root_tangents,
        "root_quaternions": quats,
        "quaternion_tangents": quat_tangents,
    }
    with np.errstate(over="ignore", invalid="ignore"):
        result: dict[str, Any] = {
            name: value.astype(np.float32) for name, value in arrays.items()
        }
    for clip, count in enumerate(counts.astype(np.int64)):
        if any(
            not np.all(np.isfinite(value[clip, :count])) for value in result.values()
        ):
            raise ValueError(
                f"Clip {clip}: curves cannot be represented as finite float32."
            )
    result["closure_diagnostics"] = diagnostics
    return result


def interpolate_segment(
    p0: jax.Array,
    p1: jax.Array,
    v0: jax.Array,
    v1: jax.Array,
    alpha: jax.Array | float,
    dt: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate one cubic Hermite segment and its exact time derivative.

    ``alpha`` is the scalar fraction in [0, 1]; ``dt`` must be positive.
    The caller handles clamp/wrap selection, not this interval-local function.
    """
    # Difference form avoids subtracting large absolute-position terms in qdot.
    displacement = p1 - p0
    linear = dt * v0
    quadratic = 3.0 * displacement - dt * (2.0 * v0 + v1)
    cubic = -2.0 * displacement + dt * (v0 + v1)
    value = p0 + alpha * (linear + alpha * (quadratic + alpha * cubic))
    derivative = v0 + alpha * (2.0 * quadratic + 3.0 * alpha * cubic) / dt
    return value, derivative


def interpolate_quaternion_segment(
    q0: jax.Array,
    q1: jax.Array,
    dq0: jax.Array,
    dq1: jax.Array,
    alpha: jax.Array | float,
    dt: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate a normalized quaternion cubic and LOCAL angular velocity.

    Use sign-continuous endpoints and tangents from ``prepare_reference_curves``.
    The returned angular velocity is MuJoCo free-joint qvel[3:6], not world-frame
    angular velocity. Both quaternion value and derivative are normalized
    analytically; a derivative quaternion must never itself be normalized.
    """
    raw, raw_derivative = interpolate_segment(q0, q1, dq0, dq1, alpha, dt)
    norm = jp.maximum(jp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
    quat = raw / norm
    derivative = (
        raw_derivative - quat * jp.sum(quat * raw_derivative, axis=-1, keepdims=True)
    ) / norm
    angular_velocity = 2.0 * (
        quat[..., :1] * derivative[..., 1:]
        - derivative[..., :1] * quat[..., 1:]
        - jp.cross(quat[..., 1:], derivative[..., 1:])
    )
    return quat, angular_velocity
