"""Dense stand feedback, versioned separately from physical acceptance gates."""

import jax
import jax.numpy as jp

from skills.stand_metrics import StandThresholds

DEFAULT_THRESHOLDS = StandThresholds()


def _stand_state_score(
    speed: jax.Array,
    distance: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    thresholds: StandThresholds = DEFAULT_THRESHOLDS,
) -> jax.Array:
    """Score finite speed, drift, posture and height without contact semantics."""
    error = (
        jp.square(speed / thresholds.max_speed)
        + jp.square(distance / thresholds.max_distance)
        + jp.square(
            (1.0 - jp.clip(torso_up, -1.0, 1.0))
            / max(1.0 - thresholds.min_torso_up, 1e-6)
        )
        + jp.square((height_ratio - 1.0) / max(1.0 - thresholds.min_height_ratio, 1e-6))
    )
    score = jp.exp(-error)
    finite = jp.all(jp.isfinite(jp.stack([speed, distance, torso_up, height_ratio])))
    return jp.where(finite, jp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def stand_balance_score(
    speed: jax.Array,
    distance: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    both_feet_contact: jax.Array,
    thresholds: StandThresholds = DEFAULT_THRESHOLDS,
) -> jax.Array:
    """Score balance while requiring both feet to remain in contact."""
    score = _stand_state_score(speed, distance, torso_up, height_ratio, thresholds)
    return score * both_feet_contact.astype(jp.float32)


def stand_recovery_score(
    speed: jax.Array,
    distance: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    thresholds: StandThresholds = DEFAULT_THRESHOLDS,
) -> jax.Array:
    """Score recovery state without punishing a temporary corrective step.

    The independent success gate still requires both feet in contact and a full
    one-second dwell. This reward only removes the hard per-step contact zero
    that can suppress learning a recovery step after a backward impulse.
    """
    return _stand_state_score(speed, distance, torso_up, height_ratio, thresholds)
