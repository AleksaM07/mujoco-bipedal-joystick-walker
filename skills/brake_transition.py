"""Specialized bridge from FINAL_WALKING into FINAL_STANDING's capture region."""

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from biomechanics_env import BiomechanicsJoystickEnv
from skills.walk_stop import WalkStopEnv


class BrakeTransitionEnv(WalkStopEnv):
    """Train only the short braking maneuver, not walking or long standing.

    Resets sample shortly before the reference starts its stop blend. Unlike the
    legacy walk-stop reward, physical terms are additive and remain informative
    while the humanoid is still moving quickly.
    """

    CAPTURE_DWELL_SECONDS = 0.20

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        minimum = float(self._config.get("brake_lead_seconds_min", 0.2))
        maximum = float(self._config.get("brake_lead_seconds_max", 0.8))
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("brake_lead_seconds_min must be finite and nonnegative.")
        if not np.isfinite(maximum) or maximum < minimum:
            raise ValueError(
                "brake_lead_seconds_max must be finite and at least the minimum."
            )
        self._brake_lead_seconds_min = minimum
        self._brake_lead_seconds_max = maximum
        self._capture_dwell_steps = max(
            1, round(self.CAPTURE_DWELL_SECONDS / self.dt)
        )

    def _sample_bvh_motion_time(
        self, rng: jax.Array, clip_id: jax.Array
    ) -> jax.Array:
        """Place every reset shortly before the calibrated double-support stop."""
        duration = self._bvh_reference_motion_lengths[clip_id]
        lead_seconds = jax.random.uniform(
            rng,
            shape=(),
            minval=self._brake_lead_seconds_min,
            maxval=self._brake_lead_seconds_max,
        )
        blend_seconds = self._transition_blend_steps * self.dt
        stop_phase_time = duration * self._transition_stop_phase
        return jp.mod(stop_phase_time - blend_seconds - lead_seconds, duration)

    def _capture_terms(
        self, data: mjx.Data, info: dict
    ) -> dict[str, jax.Array]:
        feet = self._foot_xy(data)
        foot_delta = feet[1] - feet[0]
        denominator = jp.maximum(jp.dot(foot_delta, foot_delta), 1e-8)
        center_of_mass = self._center_of_mass(data)
        com_xy = center_of_mass[:2]
        com_velocity = self._center_of_mass_velocity(data)[:2]

        def distance_to_support(point: jax.Array) -> jax.Array:
            fraction = jp.clip(
                jp.dot(point - feet[0], foot_delta) / denominator, 0.0, 1.0
            )
            closest = feet[0] + fraction * foot_delta
            return jp.maximum(jp.linalg.norm(point - closest) - 0.06, 0.0)

        gravity = float(np.linalg.norm(self.mj_model.opt.gravity))
        omega = jp.sqrt(gravity / jp.maximum(center_of_mass[2], 0.25))
        capture_point = com_xy + com_velocity / jp.maximum(omega, 1e-6)
        local_angvel = self._local_root_angvel(data)
        return {
            "capture_distance": distance_to_support(capture_point),
            "support_distance": distance_to_support(com_xy),
            "angular_speed": jp.linalg.norm(local_angvel[jp.array([0, 2])]),
            "any_foot_contact": jp.any(self._foot_contact(data, info) > 0),
        }

    def _capture_ready(self, data: mjx.Data, info: dict) -> jax.Array:
        terms = self._capture_terms(data, info)
        return (
            (self._transition_alpha(info) >= 1.0)
            & (jp.linalg.norm(data.qvel[:2]) < self.thresholds.max_speed)
            & (terms["capture_distance"] < 0.08)
            & (self._torso_up(data) >= self.thresholds.min_torso_up)
            & (data.qpos[2] / self._init_q[2] >= self.thresholds.min_height_ratio)
            & terms["any_foot_contact"]
            & (~self._get_done(data, info))
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> jax.Array:
        """Use broad additive shaping that has a gradient before capture."""
        imitation = BiomechanicsJoystickEnv._get_reward(
            self, data, action, previous_action, info
        )
        imitation_quality = jp.clip(imitation / self.REWARD_MAX, 0.0, 1.0)
        alpha = self._transition_alpha(info)
        terms = self._capture_terms(data, info)
        speed = jp.linalg.norm(data.qvel[:2])
        walking_reference = BiomechanicsJoystickEnv._query_bvh_reference(
            self, info, 0
        )
        walking_speed = jp.linalg.norm(walking_reference["sim_root_vel"][:2])
        desired_speed = (1.0 - alpha) * walking_speed
        speed_score = jp.exp(-jp.square((speed - desired_speed) / 0.55))
        capture_score = jp.exp(-jp.square(terms["capture_distance"] / 0.25))
        support_score = jp.exp(-jp.square(terms["support_distance"] / 0.16))
        angular_score = jp.exp(-jp.square(terms["angular_speed"] / 1.5))
        torso_score = jp.clip((self._torso_up(data) - 0.45) / 0.50, 0.0, 1.0)
        height_score = jp.clip(
            (data.qpos[2] / self._init_q[2] - 0.55) / 0.35, 0.0, 1.0
        )
        contact_score = terms["any_foot_contact"].astype(jp.float32)

        early_score = (
            0.45 * imitation_quality
            + 0.30 * speed_score
            + 0.15 * torso_score
            + 0.10 * height_score
        )
        braking_score = (
            0.25 * speed_score
            + 0.25 * capture_score
            + 0.15 * support_score
            + 0.10 * angular_score
            + 0.10 * torso_score
            + 0.10 * height_score
            + 0.05 * contact_score
        )
        score = (1.0 - alpha) * early_score + alpha * braking_score
        score += 0.10 * self._capture_ready(data, info).astype(jp.float32)
        reward = self.REWARD_MAX * score
        reward -= self.ACTION_RATE_COST_SCALE * jp.sum(
            jp.square(action - previous_action)
        )
        reward -= 0.05 * jp.clip(self._get_foot_slip_cost(data, info), 0.0, 3.0)
        return jp.where(
            self._get_done(data, info),
            imitation,
            jp.clip(jp.nan_to_num(reward, nan=0.0), 0.0, self.REWARD_MAX),
        )

    def _brake_metrics(self, state: mjx_env.State) -> dict[str, jax.Array]:
        terms = self._capture_terms(state.data, state.info)
        capture_steps = state.info.get(
            "brake_capture_steps", jp.array(0, dtype=jp.int32)
        )
        return {
            "brake_speed": jp.linalg.norm(state.data.qvel[:2]),
            "brake_capture_distance": terms["capture_distance"],
            "brake_support_distance": terms["support_distance"],
            "brake_angular_speed": terms["angular_speed"],
            "brake_capture_ready": self._capture_ready(
                state.data, state.info
            ).astype(jp.float32),
            "brake_capture_steps": capture_steps.astype(jp.float32),
            "brake_handoff_ready": (
                capture_steps >= self._capture_dwell_steps
            ).astype(jp.float32),
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        info = {**state.info, "brake_capture_steps": jp.array(0, dtype=jp.int32)}
        state = state.replace(info=info, obs=self._get_obs(state.data, info))
        return state.replace(metrics={**state.metrics, **self._brake_metrics(state)})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        ready = self._capture_ready(state.data, state.info)
        previous = state.info.get(
            "brake_capture_steps", jp.array(0, dtype=jp.int32)
        )
        capture_steps = jp.where(ready, previous + 1, 0)
        info = {**state.info, "brake_capture_steps": capture_steps}
        state = state.replace(info=info, obs=self._get_obs(state.data, info))
        return state.replace(metrics={**state.metrics, **self._brake_metrics(state)})
