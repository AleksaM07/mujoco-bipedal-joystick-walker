"""Walk-to-stand curriculum built around the proven cyclic walking reference."""

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from biomechanics_env import BiomechanicsJoystickEnv
from skills.stand_metrics import StandThresholds, update_stand_metrics
from skills.stand_rewards import stand_balance_score


class WalkStopEnv(BiomechanicsJoystickEnv):
    """Track a stable walk, brake at double support, and hold neutral stance."""

    thresholds = StandThresholds()
    FOOT_CONTACT_DISTANCE = 0.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._bvh_reference_clip_count != 1:
            raise ValueError("walk_stop requires exactly one cyclic walking clip.")
        if self._config.get("deepmimic_reward_mode") != "pure":
            raise ValueError("walk_stop requires pure imitation plus balance shaping.")
        if self._config.get("reference_action_mode") != "residual":
            raise ValueError("walk_stop requires residual reference actions.")
        if not self._continuous_reference:
            raise ValueError("walk_stop requires consistent_v2 or newer physics.")

        blend_seconds = float(self._config.get("transition_blend_seconds", 0.8))
        min_walk_seconds = float(
            self._config.get("transition_min_walk_seconds", 2.0)
        )
        stop_phase = float(self._config.get("transition_stop_phase", 70.0 / 133.0))
        if not np.isfinite(blend_seconds) or blend_seconds <= 0.0:
            raise ValueError("transition_blend_seconds must be finite and positive.")
        if not np.isfinite(min_walk_seconds) or min_walk_seconds < 0.0:
            raise ValueError(
                "transition_min_walk_seconds must be finite and nonnegative."
            )
        if not np.isfinite(stop_phase) or not 0.0 <= stop_phase < 1.0:
            raise ValueError("transition_stop_phase must be in [0, 1).")
        self._transition_blend_steps = max(1, round(blend_seconds / self.dt))
        self._transition_min_walk_seconds = min_walk_seconds
        self._transition_stop_phase = stop_phase

    def _transition_alpha(self, info: dict, future_step: int = 0) -> jax.Array:
        if "transition_start_step" not in info:
            return jp.array(0.0, dtype=jp.float32)
        step = info["motion_step"].astype(jp.float32) + jp.array(
            future_step, dtype=jp.float32
        )
        start = info["transition_start_step"].astype(jp.float32)
        duration = jp.array(self._transition_blend_steps, dtype=jp.float32)
        linear = jp.clip((step - start) / duration, 0.0, 1.0)
        return linear * linear * (3.0 - 2.0 * linear)

    def _transition_alpha_rate(self, info: dict, future_step: int) -> jax.Array:
        step = info["motion_step"].astype(jp.float32) + jp.array(
            future_step, dtype=jp.float32
        )
        start = info["transition_start_step"].astype(jp.float32)
        duration_steps = jp.array(self._transition_blend_steps, dtype=jp.float32)
        linear = (step - start) / duration_steps
        rate = 6.0 * linear * (1.0 - linear) / (duration_steps * self.dt)
        return jp.where((linear > 0.0) & (linear < 1.0), rate, 0.0)

    def _anchored_stand_reference(
        self, stop_reference: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        standing_root = self._standing_reference_root_pos_targets[0, 0]
        standing_sim_root = self._standing_reference_sim_root_pos_targets[0, 0]
        root_xy = stop_reference["root_pos"][:2] - standing_root[:2]
        sim_xy = stop_reference["sim_root_pos"][:2] - standing_sim_root[:2]
        root_shift = jp.array([root_xy[0], root_xy[1], 0.0])
        sim_shift = jp.array([sim_xy[0], sim_xy[1], 0.0])
        return {
            "qpos": self._standing_reference_qpos_targets[0, 0],
            "qvel": self._standing_reference_qvel_targets[0, 0],
            "root_pos": standing_root + root_shift,
            "root_quat": self._standing_reference_root_quat_targets[0, 0],
            "root_vel": self._standing_reference_root_vel_targets[0, 0],
            "root_angvel": self._standing_reference_root_angvel_targets[0, 0],
            "sim_root_pos": standing_sim_root + sim_shift,
            "sim_root_quat": self._standing_reference_sim_root_quat_targets[0, 0],
            "sim_root_vel": self._standing_reference_sim_root_vel_targets[0, 0],
            "sim_root_angvel": self._standing_reference_sim_root_angvel_targets[0, 0],
            "reset_root_pos": (
                self._standing_reference_reset_root_pos_targets[0, 0] + sim_shift
            ),
            "reset_root_quat": self._standing_reference_reset_root_quat_targets[0, 0],
            "reset_root_vel": self._standing_reference_reset_root_vel_targets[0, 0],
            "reset_root_angvel": (
                self._standing_reference_reset_root_angvel_targets[0, 0]
            ),
            "key_pos": self._standing_reference_key_pos_targets[0, 0] + root_shift,
        }

    def _query_bvh_reference(
        self, info: dict, future_step: int
    ) -> dict[str, jax.Array]:
        walking = super()._query_bvh_reference(info, future_step)
        if "transition_stop_motion_time" not in info:
            return walking

        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        stop_reference = self._query_bvh_reference_clip(
            clip_id, info["transition_stop_motion_time"]
        )
        standing = self._anchored_stand_reference(stop_reference)
        anchor_delta_xy = (
            info["transition_anchor_xy"] - standing["sim_root_pos"][:2]
        )
        anchor_shift = jp.array(
            [anchor_delta_xy[0], anchor_delta_xy[1], 0.0], dtype=jp.float32
        )
        for position_key in (
            "root_pos",
            "sim_root_pos",
            "reset_root_pos",
            "key_pos",
        ):
            standing[position_key] = standing[position_key] + anchor_shift
        alpha = self._transition_alpha(info, future_step)
        alpha_rate = self._transition_alpha_rate(info, future_step)

        blended = dict(walking)
        for position_key, velocity_key in (
            ("qpos", "qvel"),
            ("root_pos", "root_vel"),
            ("sim_root_pos", "sim_root_vel"),
            ("reset_root_pos", "reset_root_vel"),
        ):
            start_position = walking[position_key]
            end_position = standing[position_key]
            blended[position_key] = start_position + alpha * (
                end_position - start_position
            )
            blended[velocity_key] = (
                (1.0 - alpha) * walking[velocity_key]
                + alpha * standing[velocity_key]
                + alpha_rate * (end_position - start_position)
            )
        for quaternion_key, angular_velocity_key in (
            ("root_quat", "root_angvel"),
            ("sim_root_quat", "sim_root_angvel"),
            ("reset_root_quat", "reset_root_angvel"),
        ):
            blended[quaternion_key] = self._quat_slerp(
                walking[quaternion_key], standing[quaternion_key], alpha
            )
            blended[angular_velocity_key] = (1.0 - alpha) * walking[
                angular_velocity_key
            ]
        blended["key_pos"] = walking["key_pos"] + alpha * (
            standing["key_pos"] - walking["key_pos"]
        )
        return blended

    def _get_bvh_reference_phase_angle(self, info: dict) -> jax.Array:
        walking_phase = super()._get_bvh_reference_phase_angle(info)
        stop_phase = jp.array(2.0 * jp.pi * self._transition_stop_phase)
        return jp.where(self._transition_alpha(info) >= 1.0, stop_phase, walking_phase)

    def _balance_score(self, data: mjx.Data, info: dict) -> jax.Array:
        return stand_balance_score(
            jp.linalg.norm(data.qvel[:2]),
            jp.linalg.norm(data.qpos[:2] - info["transition_anchor_xy"]),
            self._torso_up(data),
            data.qpos[2] / self._init_q[2],
            jp.all(self._foot_contact(data, info) > 0),
            self.thresholds,
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> jax.Array:
        imitation = super()._get_reward(data, action, previous_action, info)
        if "transition_anchor_xy" not in info:
            return imitation
        shaped = imitation * (0.2 + 0.8 * self._balance_score(data, info))
        alpha = self._transition_alpha(info)
        return (1.0 - alpha) * imitation + alpha * shaped

    def _transition_metrics(self, state: mjx_env.State) -> dict[str, jax.Array]:
        alpha = self._transition_alpha(state.info)
        low, tipped, invalid = self._get_done_reasons(state.data)
        metrics = update_stand_metrics(
            state.info["stand_dwell_steps"],
            state.data.qpos[:2],
            state.data.qvel[:2],
            state.info["transition_anchor_xy"],
            self._torso_up(state.data),
            state.data.qpos[2] / self._init_q[2],
            jp.all(self._foot_contact(state.data, state.info) > 0),
            low | tipped | invalid,
            dt=self.dt,
            thresholds=self.thresholds,
        )
        active = alpha >= 1.0
        dwell_steps = jp.where(active, metrics["stand_dwell_steps"], 0)
        return {
            **metrics,
            "stand_dwell_steps": dwell_steps,
            "stand_dwell_seconds": dwell_steps.astype(jp.float32) * self.dt,
            "stand_success": jp.where(active, metrics["stand_success"], 0.0),
            "transition_alpha": alpha,
            "stand_balance_score": self._balance_score(state.data, state.info),
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        info = dict(state.info)
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        duration = self._bvh_reference_motion_lengths[clip_id]
        offset = info["bvh_reference_time_offset"]
        stop_phase_time = duration * self._transition_stop_phase
        phase_time = offset - jp.floor(offset / duration) * duration
        first_stop_delta = jp.mod(stop_phase_time - phase_time, duration)
        minimum_end = self._transition_min_walk_seconds + (
            self._transition_blend_steps * self.dt
        )
        extra_cycles = jp.maximum(
            jp.ceil((minimum_end - first_stop_delta) / duration), 0.0
        )
        stop_delta = first_stop_delta + extra_cycles * duration
        stop_step = jp.ceil(stop_delta / self.dt).astype(jp.int32)
        start_step = stop_step - jp.array(
            self._transition_blend_steps, dtype=jp.int32
        )
        stop_motion_time = offset + stop_step.astype(jp.float32) * self.dt
        start_reference = self._query_bvh_reference_clip(clip_id, offset)
        stop_reference = self._query_bvh_reference_clip(clip_id, stop_motion_time)
        standing = self._anchored_stand_reference(stop_reference)
        expected_travel_xy = (
            standing["sim_root_pos"][:2] - start_reference["sim_root_pos"][:2]
        )
        info.update(
            transition_start_step=start_step,
            transition_stop_step=stop_step,
            transition_stop_motion_time=stop_motion_time,
            transition_anchor_xy=state.data.qpos[:2] + expected_travel_xy,
            stand_dwell_steps=jp.array(0, dtype=jp.int32),
        )
        state = state.replace(info=info, obs=self._get_obs(state.data, info))
        metrics = self._transition_metrics(state)
        return state.replace(
            metrics={
                **state.metrics,
                **{name: value.astype(jp.float32) for name, value in metrics.items()},
            }
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        previous_alpha = self._transition_alpha(state.info)
        state = super().step(state, action)
        current_alpha = self._transition_alpha(state.info)
        lock_anchor = (previous_alpha < 1.0) & (current_alpha >= 1.0)
        info = dict(state.info)
        info["transition_anchor_xy"] = jp.where(
            lock_anchor,
            state.data.qpos[:2],
            info["transition_anchor_xy"],
        )
        state = state.replace(info=info, obs=self._get_obs(state.data, info))
        metrics = self._transition_metrics(state)
        info = {**state.info, "stand_dwell_steps": metrics["stand_dwell_steps"]}
        return state.replace(
            info=info,
            metrics={
                **state.metrics,
                **{name: value.astype(jp.float32) for name, value in metrics.items()},
            },
        )
