"""Opt-in training perturbations for stand recovery."""

import math

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env, wrapper

from skills.stand import StandEnv


class StandTrainingPushWrapper(wrapper.Wrapper):
    """Mix nominal episodes with short thorax forces, without new observations.

    Legacy modes preserve their fixed world-X schedule. Recovery v4 samples a
    new horizontal direction and magnitude for every 0.2 s pulse, with three
    seconds of recovery time. Neither the schedule nor force enters the policy
    observation.
    """

    WAITING = 0
    PULSE = 1
    RECOVERY = 2
    INACTIVE = 3
    CURRICULUM_LEVELS = 5

    def __init__(
        self,
        env: StandEnv,
        max_fraction: float,
        *,
        min_fraction: float = 0.0,
        active_probability: float = 0.5,
        direction: str = "negative",
        success_curriculum: bool = False,
        random_pushes: bool = False,
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
        if direction not in {
            "negative",
            "positive",
            "bidirectional",
            "omnidirectional",
        }:
            raise ValueError(
                "Stand push direction must be negative, positive, bidirectional or "
                "omnidirectional."
            )
        if success_curriculum and random_pushes:
            raise ValueError("Select either success curriculum or random pushes.")
        if not isinstance(env, StandEnv) or env._config.physics_backend != "mjx_jax":
            raise ValueError("Training pushes require StandEnv with mjx_jax physics.")
        super().__init__(env)
        model_weight = float(np.sum(env.mj_model.body_mass)) * float(
            np.linalg.norm(env.mj_model.opt.gravity)
        )
        self.model_weight = model_weight
        self.min_force = min_fraction * model_weight
        self.max_force = max_fraction * model_weight
        self.active_probability = active_probability
        self.direction = direction
        self.success_curriculum = success_curriculum
        self.random_pushes = random_pushes
        self.body_id = env.mj_model.body("thorax").id
        self.period_steps = round(10.0 / env.dt)
        self.duration_steps = round(0.2 / env.dt)
        self.first_start_step = round(2.0 / env.dt)
        self.last_start_step = round(8.0 / env.dt)
        self.curriculum_wait_steps = round(2.0 / env.dt)
        self.curriculum_dwell_steps = round(1.0 / env.dt)
        self.random_recovery_steps = round(3.0 / env.dt)
        self.random_first_min_steps = round(2.0 / env.dt)
        self.random_first_max_steps = round(4.0 / env.dt)
        self.random_interval_min_steps = round(4.0 / env.dt)
        self.random_interval_max_steps = round(6.0 / env.dt)
        if (
            self.duration_steps < 1
            or self.last_start_step + self.duration_steps >= self.period_steps
        ):
            raise ValueError("Control dt cannot represent the stand push schedule.")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Sample a nominal/pushed episode, magnitude and pulse phase on each reset."""
        if self.success_curriculum:
            return self._reset_curriculum(rng)
        if self.random_pushes:
            return self._reset_random_pushes(rng)
        env_key, active_key, force_key, phase_key, direction_key = jax.random.split(
            rng, 5
        )
        state = self.env.reset(env_key)
        magnitude = jax.random.uniform(
            force_key, minval=self.min_force, maxval=self.max_force
        )
        sign = self._direction_sign(direction_key)
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

    def _direction_sign(self, key: jax.Array) -> jax.Array:
        """Sample the configured world-X direction as a scalar sign."""
        if self.direction == "negative":
            return jp.array(-1.0)
        if self.direction == "positive":
            return jp.array(1.0)
        if self.direction == "bidirectional":
            return jp.where(jax.random.bernoulli(key), 1.0, -1.0)
        return self._direction_vector(key)[0]

    def _direction_vector(self, key: jax.Array) -> jax.Array:
        """Sample a horizontal unit vector from the configured distribution."""
        if self.direction == "negative":
            return jp.array([-1.0, 0.0])
        if self.direction == "positive":
            return jp.array([1.0, 0.0])
        if self.direction == "bidirectional":
            return jp.array([self._direction_sign(key), 0.0])
        angle = jax.random.uniform(key, minval=-jp.pi, maxval=jp.pi)
        return jp.array([jp.cos(angle), jp.sin(angle)])

    def _reset_random_pushes(self, rng: jax.Array) -> mjx_env.State:
        """Schedule direct randomized pushes without a per-episode unlock gate."""
        env_key, active_key, start_key, schedule_key = jax.random.split(rng, 4)
        state = self.env.reset(env_key)
        active = jax.random.bernoulli(active_key, self.active_probability)
        start = jax.random.randint(
            start_key,
            (),
            self.random_first_min_steps,
            self.random_first_max_steps + 1,
        )
        phase = jp.where(active, self.WAITING, self.INACTIVE).astype(jp.int32)
        zero_vector = jp.zeros(2)
        info = {
            **state.info,
            "stand_push_steps": jp.array(0, dtype=jp.int32),
            "stand_push_start": start,
            "stand_push_force_x": jp.array(0.0),
            "stand_push_force_xy": zero_vector,
            "stand_push_rng": schedule_key,
            "stand_curriculum_phase": phase,
            "stand_curriculum_pulse_steps": jp.array(0, dtype=jp.int32),
            "stand_random_recovery_end": jp.array(0, dtype=jp.int32),
            "stand_random_push_count": jp.array(0, dtype=jp.int32),
            "stand_recovery_active": jp.array(False),
            "stand_current_force_fraction": jp.array(0.0),
        }
        return state.replace(
            info=info,
            metrics={
                **state.metrics,
                "stand_training_push_x_newtons": jp.array(0.0),
                "stand_training_push_y_newtons": jp.array(0.0),
                "stand_training_push_fraction": jp.array(0.0),
                "stand_random_push_count": jp.array(0.0),
                "stand_recovery_active": jp.array(0.0),
            },
        )

    def _reset_curriculum(self, rng: jax.Array) -> mjx_env.State:
        """Start active episodes with a small recoverable lean and velocity."""
        (
            env_key,
            active_key,
            direction_key,
            tilt_key,
            velocity_key,
            angular_key,
        ) = jax.random.split(rng, 6)
        state = self.env.reset(env_key)
        active = jax.random.bernoulli(active_key, self.active_probability)
        sign = self._direction_sign(direction_key)
        tilt = sign * jax.random.uniform(
            tilt_key,
            minval=math.radians(1.0),
            maxval=math.radians(3.0),
        )
        linear_velocity = sign * jax.random.uniform(
            velocity_key, minval=0.05, maxval=0.20
        )
        angular_velocity = sign * jax.random.uniform(
            angular_key, minval=0.05, maxval=0.20
        )
        tilt = jp.where(active, tilt, 0.0)
        linear_velocity = jp.where(active, linear_velocity, 0.0)
        angular_velocity = jp.where(active, angular_velocity, 0.0)
        tilt_quaternion = jp.array([jp.cos(tilt / 2.0), 0.0, jp.sin(tilt / 2.0), 0.0])
        qpos = state.data.qpos.at[3:7].set(
            self.env._quat_mul(tilt_quaternion, state.data.qpos[3:7])
        )
        qvel = state.data.qvel.at[0].add(linear_velocity)
        qvel = qvel.at[4].add(angular_velocity)
        data = mjx.forward(self.env.mjx_model, state.data.replace(qpos=qpos, qvel=qvel))
        info = {
            **state.info,
            "last_foot_xy": self.env._foot_xy(data),
            "stand_dwell_steps": jp.array(0, dtype=jp.int32),
            "stand_push_steps": jp.array(0, dtype=jp.int32),
            "stand_push_start": jp.array(0, dtype=jp.int32),
            "stand_push_force_x": jp.array(0.0),
            "stand_push_direction_sign": sign,
            "stand_curriculum_level": jp.array(0, dtype=jp.int32),
            "stand_curriculum_phase": jp.where(
                active, self.RECOVERY, self.INACTIVE
            ).astype(jp.int32),
            "stand_curriculum_stable_steps": jp.array(0, dtype=jp.int32),
            "stand_curriculum_pulse_steps": jp.array(0, dtype=jp.int32),
            "stand_curriculum_next_start": jp.array(
                self.curriculum_wait_steps, dtype=jp.int32
            ),
            "stand_curriculum_recoveries": jp.array(0, dtype=jp.int32),
            "stand_recovery_active": active,
            "stand_initial_tilt_radians": tilt,
            "stand_initial_velocity_x": linear_velocity,
        }
        metrics = {
            **state.metrics,
            "stand_training_push_x_newtons": jp.array(0.0),
            "stand_curriculum_level": jp.array(0.0),
            "stand_recovery_active": active.astype(jp.float32),
            "stand_curriculum_recoveries": jp.array(0.0),
            "stand_initial_tilt_radians": tilt,
            "stand_initial_velocity_x": linear_velocity,
        }
        return state.replace(
            data=data,
            obs=self.env._get_obs(data, info),
            info=info,
            metrics=metrics,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Apply a world force through the normal physics step; clear it off-pulse."""
        if self.success_curriculum:
            return self._step_curriculum(state, action)
        if self.random_pushes:
            return self._step_random_pushes(state, action)
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

    def _step_random_pushes(
        self, state: mjx_env.State, action: jax.Array
    ) -> mjx_env.State:
        """Apply independently sampled horizontal impulses every four to six seconds."""
        info = state.info
        steps = info["stand_push_steps"]
        phase = info["stand_curriculum_phase"]
        next_rng, magnitude_key, direction_key, interval_key = jax.random.split(
            info["stand_push_rng"], 4
        )
        start_pulse = (phase == self.WAITING) & (steps >= info["stand_push_start"])
        candidate_magnitude = jax.random.uniform(
            magnitude_key, minval=self.min_force, maxval=self.max_force
        )
        candidate_force = candidate_magnitude * self._direction_vector(direction_key)
        force_vector = jp.where(
            start_pulse, candidate_force, info["stand_push_force_xy"]
        )
        force_fraction = jp.where(
            start_pulse,
            candidate_magnitude / self.model_weight,
            info["stand_current_force_fraction"],
        )
        pulse_steps = jp.where(
            start_pulse,
            jp.array(0, dtype=jp.int32),
            info["stand_curriculum_pulse_steps"],
        )
        interval = jax.random.randint(
            interval_key,
            (),
            self.random_interval_min_steps,
            self.random_interval_max_steps + 1,
        )
        next_start = jp.where(
            start_pulse, steps + interval, info["stand_push_start"]
        ).astype(jp.int32)
        phase = jp.where(start_pulse, self.PULSE, phase).astype(jp.int32)
        active_force = jp.where(phase == self.PULSE, force_vector, jp.zeros(2))
        forces = (
            jp.zeros_like(state.data.xfrc_applied)
            .at[self.body_id, :2]
            .set(active_force)
        )
        recovery_active = (phase == self.PULSE) | (phase == self.RECOVERY)
        state = self.env.step(
            state.replace(
                data=state.data.replace(xfrc_applied=forces),
                info={**info, "stand_recovery_active": recovery_active},
            ),
            action,
        )

        next_steps = steps + 1
        pulse_steps = jp.where(phase == self.PULSE, pulse_steps + 1, pulse_steps)
        pulse_finished = (phase == self.PULSE) & (pulse_steps >= self.duration_steps)
        recovery_end = jp.where(
            pulse_finished,
            next_steps + self.random_recovery_steps,
            info["stand_random_recovery_end"],
        ).astype(jp.int32)
        phase = jp.where(pulse_finished, self.RECOVERY, phase).astype(jp.int32)
        recovery_finished = (phase == self.RECOVERY) & (next_steps >= recovery_end)
        phase = jp.where(recovery_finished, self.WAITING, phase).astype(jp.int32)
        recovery_active = (phase == self.PULSE) | (phase == self.RECOVERY)
        push_count = info["stand_random_push_count"] + start_pulse.astype(jp.int32)
        next_info = {
            **state.info,
            "stand_push_steps": next_steps,
            "stand_push_start": next_start,
            "stand_push_force_x": force_vector[0],
            "stand_push_force_xy": force_vector,
            "stand_push_rng": next_rng,
            "stand_curriculum_phase": phase,
            "stand_curriculum_pulse_steps": pulse_steps,
            "stand_random_recovery_end": recovery_end,
            "stand_random_push_count": push_count,
            "stand_recovery_active": recovery_active,
            "stand_current_force_fraction": force_fraction,
        }
        return state.replace(
            info=next_info,
            metrics={
                **state.metrics,
                "stand_training_push_x_newtons": active_force[0],
                "stand_training_push_y_newtons": active_force[1],
                "stand_training_push_fraction": force_fraction,
                "stand_random_push_count": push_count.astype(jp.float32),
                "stand_recovery_active": recovery_active.astype(jp.float32),
            },
        )

    def _step_curriculum(
        self, state: mjx_env.State, action: jax.Array
    ) -> mjx_env.State:
        """Advance only after one second of stable recovery at the current level."""
        info = state.info
        steps = info["stand_push_steps"]
        phase = info["stand_curriculum_phase"]
        start_pulse = (phase == self.WAITING) & (
            steps >= info["stand_curriculum_next_start"]
        )
        phase = jp.where(start_pulse, self.PULSE, phase).astype(jp.int32)
        pulse_steps = jp.where(
            start_pulse,
            jp.array(0, dtype=jp.int32),
            info["stand_curriculum_pulse_steps"],
        )
        level_fraction = info["stand_curriculum_level"].astype(jp.float32) / max(
            self.CURRICULUM_LEVELS - 1, 1
        )
        magnitude = self.min_force + level_fraction * (self.max_force - self.min_force)
        force = jp.where(
            phase == self.PULSE,
            info["stand_push_direction_sign"] * magnitude,
            0.0,
        )
        recovery_active = (phase == self.PULSE) | (phase == self.RECOVERY)
        forces = jp.zeros_like(state.data.xfrc_applied).at[self.body_id, 0].set(force)
        state = self.env.step(
            state.replace(
                data=state.data.replace(xfrc_applied=forces),
                info={**info, "stand_recovery_active": recovery_active},
            ),
            action,
        )

        pulse_steps = jp.where(phase == self.PULSE, pulse_steps + 1, pulse_steps)
        pulse_finished = (phase == self.PULSE) & (pulse_steps >= self.duration_steps)
        phase = jp.where(pulse_finished, self.RECOVERY, phase).astype(jp.int32)
        stable = (
            (phase == self.RECOVERY)
            & (state.metrics["stand_speed"] < self.env.thresholds.max_speed)
            & (self.env._torso_up(state.data) >= self.env.thresholds.min_torso_up)
            & (
                state.data.qpos[2] / self.env._init_q[2]
                >= self.env.thresholds.min_height_ratio
            )
            & jp.all(self.env._foot_contact(state.data, state.info) > 0)
            & (state.done == 0)
        )
        stable_steps = jp.where(
            stable,
            info["stand_curriculum_stable_steps"] + 1,
            jp.array(0, dtype=jp.int32),
        )
        recovered = (phase == self.RECOVERY) & (
            stable_steps >= self.curriculum_dwell_steps
        )
        level = jp.where(
            recovered,
            jp.minimum(
                info["stand_curriculum_level"] + 1,
                self.CURRICULUM_LEVELS - 1,
            ),
            info["stand_curriculum_level"],
        ).astype(jp.int32)
        phase = jp.where(recovered, self.WAITING, phase).astype(jp.int32)
        stable_steps = jp.where(recovered, 0, stable_steps).astype(jp.int32)
        next_start = jp.where(
            recovered,
            steps + 1 + self.curriculum_wait_steps,
            info["stand_curriculum_next_start"],
        ).astype(jp.int32)
        recoveries = info["stand_curriculum_recoveries"] + recovered.astype(jp.int32)
        new_anchor = jp.where(
            recovered, state.data.qpos[:2], state.info["stand_anchor_xy"]
        )
        recovery_active = (phase == self.PULSE) | (phase == self.RECOVERY)
        next_info = {
            **state.info,
            "stand_anchor_xy": new_anchor,
            "stand_push_steps": steps + 1,
            "stand_curriculum_level": level,
            "stand_curriculum_phase": phase,
            "stand_curriculum_stable_steps": stable_steps,
            "stand_curriculum_pulse_steps": pulse_steps,
            "stand_curriculum_next_start": next_start,
            "stand_curriculum_recoveries": recoveries,
            "stand_recovery_active": recovery_active,
        }
        return state.replace(
            info=next_info,
            metrics={
                **state.metrics,
                "stand_training_push_x_newtons": force,
                "stand_curriculum_level": level.astype(jp.float32),
                "stand_recovery_active": recovery_active.astype(jp.float32),
                "stand_curriculum_recoveries": recoveries.astype(jp.float32),
                "stand_initial_tilt_radians": info["stand_initial_tilt_radians"],
                "stand_initial_velocity_x": info["stand_initial_velocity_x"],
            },
        )
