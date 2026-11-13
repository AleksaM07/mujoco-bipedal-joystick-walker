"""Opt-in training perturbations for stand recovery."""

import math

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import mjx_env, wrapper

from skills.stand import StandEnv


class StandTrainingPushWrapper(wrapper.Wrapper):
    """Mix nominal episodes with short thorax forces, without changing observations.

    Active episodes sample a world-X force uniformly between ``min_fraction``
    and ``max_fraction`` of body weight. Direction can be fixed or sampled per
    episode. One 0.2 s pulse repeats every 10 s, with an episode-specific phase
    uniformly sampled from 2 to 8 s. Neither schedule nor force is observed.
    """

    def __init__(
        self,
        env: StandEnv,
        max_fraction: float,
        *,
        min_fraction: float = 0.0,
        active_probability: float = 0.5,
        direction: str = "negative",
    ) -> None:
        if not all(
            math.isfinite(value)
            for value in (min_fraction, max_fraction, active_probability)
        ):
            raise ValueError("Stand push distribution values must be finite.")
        if not 0 <= min_fraction <= max_fraction <= 0.05:
            raise ValueError(
                "Stand push fractions must satisfy 0 <= min <= max <= 0.05."
            )
        if not 0 <= active_probability <= 1:
            raise ValueError("Stand push probability must be in [0, 1].")
        if direction not in {"negative", "positive", "bidirectional"}:
            raise ValueError(
                "Stand push direction must be negative, positive or bidirectional."
            )
        if not isinstance(env, StandEnv) or env._config.physics_backend != "mjx_jax":
            raise ValueError("Training pushes require StandEnv with mjx_jax physics.")
        super().__init__(env)
        model_weight = float(np.sum(env.mj_model.body_mass)) * float(
            np.linalg.norm(env.mj_model.opt.gravity)
        )
        self.min_force = min_fraction * model_weight
        self.max_force = max_fraction * model_weight
        self.active_probability = active_probability
        self.direction = direction
        self.body_id = env.mj_model.body("thorax").id
        self.period_steps = round(10.0 / env.dt)
        self.duration_steps = round(0.2 / env.dt)
        self.first_start_step = round(2.0 / env.dt)
        self.last_start_step = round(8.0 / env.dt)
        if (
            self.duration_steps < 1
            or self.last_start_step + self.duration_steps >= self.period_steps
        ):
            raise ValueError("Control dt cannot represent the stand push schedule.")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Sample a nominal/pushed episode, magnitude and pulse phase on each reset."""
        env_key, active_key, force_key, phase_key, direction_key = jax.random.split(
            rng, 5
        )
        state = self.env.reset(env_key)
        magnitude = jax.random.uniform(
            force_key, minval=self.min_force, maxval=self.max_force
        )
        if self.direction == "negative":
            sign = jp.array(-1.0)
        elif self.direction == "positive":
            sign = jp.array(1.0)
        else:
            sign = jp.where(jax.random.bernoulli(direction_key), 1.0, -1.0)
        force = sign * magnitude
        force = jp.where(
            jax.random.bernoulli(active_key, self.active_probability), force, 0.0
        )
        start = jax.random.randint(
            phase_key, (), self.first_start_step, self.last_start_step + 1
        )
        return state.replace(
            info={
                **state.info,
                "stand_push_steps": jp.array(0, dtype=jp.int32),
                "stand_push_start": start,
                "stand_push_force_x": force,
            },
            metrics={**state.metrics, "stand_training_push_x_newtons": jp.array(0.0)},
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Apply a world force through the normal physics step; clear it off-pulse."""
        phase = state.info["stand_push_steps"] % self.period_steps
        start = state.info["stand_push_start"]
        active = (phase >= start) & (phase < start + self.duration_steps)
        force = jp.where(active, state.info["stand_push_force_x"], 0.0)
        forces = jp.zeros_like(state.data.xfrc_applied).at[self.body_id, 0].set(force)
        state = self.env.step(
            state.replace(data=state.data.replace(xfrc_applied=forces)), action
        )
        return state.replace(
            info={
                **state.info,
                "stand_push_steps": state.info["stand_push_steps"] + 1,
            },
            metrics={**state.metrics, "stand_training_push_x_newtons": force},
        )
