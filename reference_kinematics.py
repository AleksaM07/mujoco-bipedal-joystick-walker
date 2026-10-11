"""Small host-side timing helpers for retargeted reference motion.

These functions validate and differentiate sampled trajectories. They do not
repair a loop seam, validate contact dynamics, or change the runtime interpolator.
In particular, interpolating nodal finite-difference velocities is generally not
the exact derivative of linearly interpolated positions between sample frames.
"""

from __future__ import annotations

import numpy as np


def scale_reference_frame_times(
    frame_times: np.ndarray, speed_scale: float
) -> np.ndarray:
    """Return playback frame intervals for a positive finite speed multiplier.

    For example, ``speed_scale=0.25`` makes intervals four times longer. The
    environment clock must continue in real seconds and all derived velocities
    must use these intervals, with no additional multiplier of 0.25.

    Args:
        frame_times: One positive interval in seconds per source clip.
        speed_scale: Playback speed relative to the source motion.

    Returns:
        Float32 intervals in seconds, without modifying the source array.

    Raises:
        ValueError: The scale or intervals are invalid or cannot be represented
            as finite, positive float32 intervals.
    """
    if not np.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError("reference_motion_speed_scale must be positive and finite.")
    intervals = np.asarray(frame_times, dtype=np.float64)
    if intervals.ndim != 1 or intervals.size == 0:
        raise ValueError("frame_times must contain one interval per clip.")
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0.0):
        raise ValueError("frame_times must be positive and finite.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = (intervals / speed_scale).astype(np.float32)
    if not np.all(np.isfinite(scaled)) or np.any(scaled <= 0.0):
        raise ValueError("Scaled frame_times must be positive finite float32 values.")
    return scaled


def differentiate_reference_positions(
    positions: np.ndarray,
    frame_times: np.ndarray,
    frame_counts: np.ndarray,
) -> np.ndarray:
    """Differentiate padded scalar-joint or Cartesian trajectories per clip.

    Uses central differences inside a clip and one-sided differences at its
    ends, matching the existing reference convention. A one-frame clip is
    stationary. Padded frames repeat the last valid velocity and never influence
    differences. Quaternion components must not be differentiated with this
    function; they require rotational differences instead.

    Args:
        positions: Sample positions of shape (clips, padded_frames, coordinates).
        frame_times: Already speed-adjusted time interval in seconds per clip.
        frame_counts: Number of valid frames per clip; at least one per clip.

    Returns:
        Float32 velocities with the same shape as ``positions``.

    Raises:
        ValueError: Shapes, counts, valid positions, or intervals are invalid.
    """
    samples = np.asarray(positions, dtype=np.float64)
    if samples.ndim != 3 or any(size == 0 for size in samples.shape):
        raise ValueError("positions must have shape (clips, frames, coordinates).")
    intervals = scale_reference_frame_times(frame_times, 1.0)
    counts = np.asarray(frame_counts)
    clip_count, padded_frames, _ = samples.shape
    if intervals.shape != (clip_count,) or counts.shape != (clip_count,):
        raise ValueError("frame_times and frame_counts must match the clip count.")
    if (
        not np.all(np.isfinite(counts))
        or np.any(counts != np.floor(counts))
        or np.any(counts < 1)
        or np.any(counts > padded_frames)
    ):
        raise ValueError("frame_counts must be integers in [1, padded_frames].")
    velocities = np.zeros(samples.shape, dtype=np.float32)
    for clip_id, count in enumerate(counts.astype(np.int64)):
        valid_samples = samples[clip_id, :count]
        if not np.all(np.isfinite(valid_samples)):
            raise ValueError(f"Clip {clip_id} has non-finite valid positions.")
        if count == 1:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            valid_velocities = np.gradient(
                valid_samples, float(intervals[clip_id]), axis=0
            ).astype(np.float32)
        if not np.all(np.isfinite(valid_velocities)):
            raise ValueError(f"Clip {clip_id} has non-finite derived velocities.")
        velocities[clip_id, :count] = valid_velocities
        velocities[clip_id, count:] = valid_velocities[-1]
    return velocities
