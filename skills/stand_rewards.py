"""Dense stand feedback, versioned separately from physical acceptance gates."""

import jax
import jax.numpy as jp

from skills.stand_metrics import StandThresholds

DEFAULT_THRESHOLDS = StandThresholds()


def stand_balance_score(
    speed: jax.Array,
    distance: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    both_feet_contact: jax.Array,
    thresholds: StandThresholds = DEFAULT_THRESHOLDS,
) -> jax.Array:
    """Score instantaneous balance smoothly on the acceptance criteria's scales.

    No elapsed dwell or success flag enters the reward. A value of one requires
    zero speed/drift, upright posture, reference height and both feet touching.
    This is shaping feedback; only the separate 500-step gate certifies stand.
    """
    error = (
        jp.square(speed / thresholds.max_speed)
        + jp.square(distance / thresholds.max_distance)
        + jp.square(
            (1.0 - jp.clip(torso_up, -1.0, 1.0))
            / max(1.0 - thresholds.min_torso_up, 1e-6)
        )
        + jp.square((height_ratio - 1.0) / max(1.0 - thresholds.min_height_ratio, 1e-6))
    )
    score = jp.exp(-error) * both_feet_contact.astype(jp.float32)
    finite = jp.all(jp.isfinite(jp.stack([speed, distance, torso_up, height_ratio])))
    return jp.where(finite, jp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
