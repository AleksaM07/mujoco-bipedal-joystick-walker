"""Opt-in training perturbations for the remaining negative-X stand weakness."""

import math

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import mjx_env, wrapper

from skills.stand import StandEnv


class StandTrainingPushWrapper(wrapper.Wrapper):
    """Mix nominal episodes with short thorax forces, without changing observations.

    Half of episodes are nominal in expectation. Other episodes sample a -world X
    force uniformly up to ``max_fraction`` of body weight. One 0.2 s pulse repeats
    every 10 s, with an episode-specific phase uniformly sampled from 2 to 8 s.
    Neither the pulse phase nor force is added to the actor's observations.
    """

    def __init__(self, env: StandEnv, max_fraction: float) -> None:
        if not math.isfinite(max_fraction) or not 0 <= max_fraction <= 0.05:
            raise ValueError("Stand push fraction must be finite and in [0, 0.05].")
        if not isinstance(env, StandEnv) or env._config.physics_backend != "mjx_jax":
            raise ValueError("Training pushes require StandEnv with mjx_jax physics.")
        super().__init__(env)
        self.max_force = (
            max_fraction
            * float(np.sum(env.mj_model.body_mass))
            * float(np.linalg.norm(env.mj_model.opt.gravity))
        )
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
        env_key, active_key, force_key, phase_key = jax.random.split(rng, 4)
        state = self.env.reset(env_key)
        force = -self.max_force * jax.random.uniform(force_key)
        force = jp.where(jax.random.bernoulli(active_key), force, 0.0)
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
