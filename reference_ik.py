"""Opt-in periodic initialization around the unchanged reference IK kernel."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import Any

import numpy as np

IKFit = Callable[
    [np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]


def fit_periodic_ik(
    fit_once: IKFit,
    qpos_targets: np.ndarray,
    ik_target_positions: np.ndarray,
    ik_target_names: tuple[str, ...],
    root_pos_targets: np.ndarray,
    root_quat_targets: np.ndarray,
    frame_counts: np.ndarray,
    loop_modes: np.ndarray,
    *,
    max_warmup_cycles: int = 4,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Warm a causal IK kernel on unique WRAP nodes before the retained solve.

    WRAP (mode 1) includes an endpoint at the next cycle's phase zero. Its
    predecessor is therefore node ``count - 2``, not the duplicate endpoint.
    Repeat the unique nodes before the original complete clip, without changing
    source poses, targets, root transforms or the kernel's objective. CLAMP
    (mode 0) and singleton clips receive no warmup.

    One kernel call is made per clip. The last warmup block and the retained
    solve's unique nodes must agree within ``tolerance`` radians in every joint;
    otherwise fail explicitly. This tests initialization convergence, not exact
    endpoint closure, IK residual acceptance or physical feasibility. World
    translations restart together with targets; the kernel must be invariant
    to a common translation and keep only pose, not world-contact, history.

    Input arrays are never passed to the callback without copying. Valid output
    is float32, with padding held at the last valid pose as in the legacy kernel.
    """
    if not callable(fit_once):
        raise ValueError("fit_once must be an IK fitting callable.")
    if (
        isinstance(max_warmup_cycles, bool)
        or not isinstance(max_warmup_cycles, Integral)
        or max_warmup_cycles < 1
    ):
        raise ValueError("max_warmup_cycles must be a positive integer.")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be finite and positive.")

    joints = np.asarray(qpos_targets)
    targets = np.asarray(ik_target_positions)
    roots = np.asarray(root_pos_targets)
    quaternions = np.asarray(root_quat_targets)
    counts = np.asarray(frame_counts)
    modes = np.asarray(loop_modes)
    if joints.ndim != 3 or any(size == 0 for size in joints.shape):
        raise ValueError(
            "qpos_targets must have nonempty (clips, frames, joints) shape."
        )
    clips, frames, _ = joints.shape
    if (
        not isinstance(ik_target_names, tuple)
        or not ik_target_names
        or any(not isinstance(name, str) or not name for name in ik_target_names)
        or len(set(ik_target_names)) != len(ik_target_names)
    ):
        raise ValueError("ik_target_names must be a nonempty tuple of unique names.")
    if targets.shape != (clips, frames, len(ik_target_names), 3):
        raise ValueError("IK target positions must match clip/frame/name dimensions.")
    if roots.shape != (clips, frames, 3):
        raise ValueError("Root positions must have (clips, frames, 3) shape.")
    if quaternions.shape != (clips, frames, 4):
        raise ValueError("Root quaternions must have (clips, frames, 4) shape.")
    for name, values in (
        ("joint positions", joints),
        ("IK target positions", targets),
        ("root positions", roots),
        ("root quaternions", quaternions),
    ):
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"Reference {name} must be finite real numbers.")
    if np.any(np.linalg.norm(quaternions, axis=-1) < 1e-12):
        raise ValueError("Root quaternions must have nonzero norm.")
    if (
        counts.shape != (clips,)
        or counts.dtype.kind not in "iu"
        or np.any(counts < 1)
        or np.any(counts > frames)
    ):
        raise ValueError("frame_counts must contain valid positive integer lengths.")
    if (
        modes.shape != (clips,)
        or modes.dtype.kind not in "iu"
        or not np.isin(modes, [0, 1]).all()
    ):
        raise ValueError("loop_modes must contain integer CLAMP (0) or WRAP (1).")

    fitted = np.empty(joints.shape, dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []
    for clip in range(clips):
        count = int(counts[clip])
        periodic = int(modes[clip]) == 1 and count > 1
        unique_count = count - 1 if periodic else count
        warmup_cycles = int(max_warmup_cycles) if periodic else 0
        prefix_count = warmup_cycles * unique_count

        def extend(values: np.ndarray) -> np.ndarray:
            valid = values[clip : clip + 1, :count]
            if not periodic:
                return valid.copy()
            return np.concatenate(
                [valid[:, :unique_count]] * warmup_cycles + [valid], axis=1
            )

        result = np.asarray(
            fit_once(
                extend(joints),
                extend(targets),
                ik_target_names,
                extend(roots),
                extend(quaternions),
                np.array([prefix_count + count], dtype=np.int32),
            )
        )
        expected_shape = (1, prefix_count + count, joints.shape[-1])
        if (
            result.shape != expected_shape
            or result.dtype.kind not in "fiu"
            or not np.isfinite(result).all()
        ):
            raise ValueError(
                f"IK kernel returned invalid output for clip {clip}; "
                f"expected finite real array {expected_shape}."
            )
        cycle_difference = None
        if periodic:
            previous = result[0, prefix_count - unique_count : prefix_count]
            retained = result[0, prefix_count : prefix_count + unique_count]
            cycle_difference = float(
                np.max(np.abs(retained.astype(float) - previous.astype(float)))
            )
            if cycle_difference > tolerance:
                raise ValueError(
                    f"Periodic IK warmup did not converge for clip {clip} after "
                    f"{warmup_cycles} cycles: maximum cycle difference "
                    f"{cycle_difference:.9g} rad exceeds {tolerance:.9g} rad."
                )
        with np.errstate(over="ignore"):
            valid_result = result[0, prefix_count:].astype(np.float32)
        if not np.isfinite(valid_result).all():
            raise ValueError(f"IK output exceeds float32 range for clip {clip}.")
        fitted[clip, :count] = valid_result
        fitted[clip, count:] = valid_result[-1]
        diagnostics.append(
            {
                "clip_id": clip,
                "loop_mode": int(modes[clip]),
                "frame_count": count,
                "unique_frame_count": unique_count,
                "warmup_cycles": warmup_cycles,
                "converged": True,
                "max_cycle_difference_rad": cycle_difference,
                "tolerance_rad": float(tolerance),
                "initial_previous_pose_rad": (
                    result[0, prefix_count - 1].astype(float).tolist()
                    if periodic
                    else None
                ),
            }
        )
    return fitted, diagnostics
