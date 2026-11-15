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


def stand_recovery_v3_score(
    speed: jax.Array,
    distance: jax.Array,
    angular_speed: jax.Array,
    support_distance: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    recovery_active: jax.Array,
) -> jax.Array:
    """Dense physical recovery score with wider tolerances during a disturbance.

    Each term contributes independently, so one initially bad quantity does not
    erase the learning signal from improving another.  The anchor is deliberately
    loose while recovery is active: a corrective step may move the root, but the
    center of mass should still return over the feet and motion should settle.
    """
    recovery_active = recovery_active.astype(bool)
    speed_scale = jp.where(recovery_active, 0.75, 0.20)
    distance_scale = jp.where(recovery_active, 0.60, 0.25)
    angular_scale = jp.where(recovery_active, 1.50, 0.50)
    support_scale = jp.where(recovery_active, 0.20, 0.10)
    torso_scale = jp.where(recovery_active, 0.25, 0.10)
    height_scale = jp.where(recovery_active, 0.25, 0.15)

    components = jp.stack(
        [
            jp.exp(-jp.square(speed / speed_scale)),
            jp.exp(-jp.square(distance / distance_scale)),
            jp.exp(-jp.square(angular_speed / angular_scale)),
            jp.exp(-jp.square(support_distance / support_scale)),
            jp.exp(-jp.square((1.0 - jp.clip(torso_up, -1.0, 1.0)) / torso_scale)),
            jp.exp(-jp.square((height_ratio - 1.0) / height_scale)),
        ]
    )
    weights = jp.array([0.22, 0.08, 0.18, 0.20, 0.18, 0.14])
    score = jp.sum(weights * components)
    finite = jp.all(
        jp.isfinite(
            jp.stack(
                [
                    speed,
                    distance,
                    angular_speed,
                    support_distance,
                    torso_up,
                    height_ratio,
                ]
            )
        )
    )
    return jp.where(finite, jp.nan_to_num(score, nan=0.0), 0.0)


def stand_recovery_v4_score(
    capture_distance: jax.Array,
    support_distance: jax.Array,
    velocity_target_error: jax.Array,
    angular_speed: jax.Array,
    torso_up: jax.Array,
    height_ratio: jax.Array,
    any_foot_contact: jax.Array,
    anchor_distance: jax.Array,
    recovery_active: jax.Array,
) -> jax.Array:
    """Physics-guided capturability score for reactive stepping recovery.

    Capture-point distance supplies the missing dynamic signal from v3: a state
    can have its CoM over the feet and still be unrecoverable because momentum
    carries it outside the support region.  Independent terms preserve useful
    gradients when another component is already poor.
    """
    recovery_active = recovery_active.astype(bool)
    anchor_scale = jp.where(recovery_active, 0.75, 0.25)
    components = jp.stack(
        [
            jp.exp(-jp.square(capture_distance / 0.04)),
            jp.exp(-jp.square(support_distance / 0.08)),
            jp.exp(-jp.square(velocity_target_error / 0.15)),
            jp.exp(-jp.square(angular_speed / 0.80)),
            jp.exp(-jp.square((1.0 - jp.clip(torso_up, -1.0, 1.0)) / 0.15)),
            jp.exp(-jp.square((height_ratio - 1.0) / 0.20)),
            any_foot_contact.astype(jp.float32),
            jp.exp(-jp.square(anchor_distance / anchor_scale)),
        ]
    )
    weights = jp.array([0.32, 0.08, 0.22, 0.08, 0.10, 0.08, 0.07, 0.05])
    values = jp.stack(
        [
            capture_distance,
            support_distance,
            velocity_target_error,
            angular_speed,
            torso_up,
            height_ratio,
            anchor_distance,
        ]
    )
    score = jp.sum(weights * components)
    return jp.where(jp.all(jp.isfinite(values)), jp.nan_to_num(score, nan=0.0), 0.0)
