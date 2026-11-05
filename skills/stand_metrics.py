"""Physical acceptance criteria for standing, independent of imitation reward."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jp


@dataclass(frozen=True)
class StandThresholds:
    """Engineering gates for the first stand experiment, in SI units."""

    max_speed: float = 0.15
    max_distance: float = 0.25
    min_torso_up: float = 0.9
    min_height_ratio: float = 0.85
    dwell_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_speed", "max_distance", "dwell_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("min_torso_up", "min_height_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")


def required_dwell_steps(dt: float, thresholds: StandThresholds) -> int:
    """Count complete control steps without accumulated float32 time drift."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    return max(1, math.ceil(thresholds.dwell_seconds / dt - 1e-9))


def update_stand_metrics(
    previous_steps: jax.Array,
    root_xy: jax.Array,
    velocity_xy: jax.Array,
    anchor_xy: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    both_feet_contact: jax.Array,
    physical_failure: jax.Array,
    *,
    dt: float,
    thresholds: StandThresholds = StandThresholds(),
) -> dict[str, jax.Array]:
    """Update continuous dwell; any failed gate immediately clears success.

    Scalar state and XY vectors are expected; use vmap for batches. This is a
    metric, not a reset or terminal condition. Continue checking after success.
    """
    required = required_dwell_steps(dt, thresholds)
    speed = jp.linalg.norm(velocity_xy)
    distance = jp.linalg.norm(root_xy - anchor_xy)
    finite = jp.all(jp.isfinite(jp.stack([speed, distance, torso_up, height_ratio])))
    eligible = (
        finite
        & (speed < thresholds.max_speed)
        & (distance < thresholds.max_distance)
        & (torso_up >= thresholds.min_torso_up)
        & (height_ratio >= thresholds.min_height_ratio)
        & both_feet_contact
        & ~physical_failure
    )
    steps = jp.where(eligible, previous_steps + 1, 0).astype(jp.int32)
    return {
        "stand_speed": speed,
        "stand_distance": distance,
        "stand_eligible": eligible.astype(jp.float32),
        "stand_dwell_steps": steps,
        "stand_dwell_seconds": steps.astype(jp.float32) * dt,
        "stand_success": (steps >= required).astype(jp.float32),
    }
