"""Stationary imitation skill with the natural walker's action/observation layout.

The target is the model's neutral pose under full gravity, not a frozen stride.
This environment does not implement braking, switching policies, or navigation.
"""

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import LoopMode
from skills.stand_metrics import StandThresholds, update_stand_metrics
from skills.stand_rewards import (
    stand_balance_score,
    stand_recovery_score,
    stand_recovery_v3_score,
    stand_recovery_v4_score,
)


class StandEnv(BiomechanicsJoystickEnv):
    """Reuse imitation and physics with one coherent stationary reference."""

    thresholds = StandThresholds()
    # Stand success requires touching, not merely being within 1 cm of the floor.
    FOOT_CONTACT_DISTANCE = 0.0

    def _configure_bvh_reference(self) -> None:
        """Keep the neutral FK reference prepared by the parent constructor."""
        if self._config.get("stand_reward_mode", "imitation") not in {
            "imitation",
            "balance_v1",
            "recovery_v2",
            "recovery_v3",
            "recovery_v4",
        }:
            raise ValueError(
                "stand_reward_mode must be imitation, balance_v1, recovery_v2 or "
                "recovery_v3/recovery_v4."
            )
        if (
            self._config.get("reference_gait") != "bvh"
            or self._config.get("deepmimic_reward_mode") != "pure"
            or self._config.get("reference_action_mode") != "residual"
            or not self._continuous_reference
        ):
            raise ValueError(
                "Stand requires the BVH reference interface, pure imitation, "
                "residual actions and consistent_v2 or newer physics."
            )
        if self._config.get("reference_forced_clip_id") not in (None, 0):
            raise ValueError("Stand has exactly one reference: force clip 0 or None.")
        self._bvh_reference_loop_modes = jp.array([int(LoopMode.WRAP)])
        self._bvh_reference_source_paths = ("generated:neutral_stand",)
        self._bvh_reference_source_start_frames = np.array([0], dtype=np.int32)
        self._bvh_reference_source_end_frames = np.array([1], dtype=np.int32)
        self._bvh_reference_support_feet = ("both",)
        self._bvh_reference_diagnostic_summaries = ()

    def _sample_bvh_motion_time(self, rng: jax.Array, clip_id: jax.Array) -> jax.Array:
        del rng, clip_id
        return jp.array(0.0, dtype=jp.float32)

    def _get_bvh_reference_motion_time(self, info: dict, future_step: int) -> jax.Array:
        del info, future_step
        return jp.array(0.0, dtype=jp.float32)

    def _stand_metrics(self, state: mjx_env.State) -> dict[str, jax.Array]:
        data = state.data
        low, tipped, invalid = self._get_done_reasons(data)
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        return update_stand_metrics(
            state.info["stand_dwell_steps"],
            data.qpos[:2],
            data.qvel[:2],
            state.info["stand_anchor_xy"],
            self._torso_up(data),
            data.qpos[2] / self._init_q[2],
            jp.all(self._foot_contact(data, state.info) > 0),
            low | tipped | invalid | ~finite,
            dt=self.dt,
            thresholds=self.thresholds,
        )

    def _balance_score(self, data: mjx.Data, info: dict[str, jax.Array]) -> jax.Array:
        common = (
            jp.linalg.norm(data.qvel[:2]),
            jp.linalg.norm(data.qpos[:2] - info["stand_anchor_xy"]),
            self._torso_up(data),
            data.qpos[2] / self._init_q[2],
        )
        reward_mode = self._config.get("stand_reward_mode")
        if reward_mode == "recovery_v4":
            terms = self._recovery_v4_terms(data, info)
            return stand_recovery_v4_score(
                terms["capture_distance"],
                terms["support_distance"],
                terms["velocity_target_error"],
                terms["angular_speed"],
                common[2],
                common[3],
                terms["any_foot_contact"],
                common[1],
                info.get("stand_recovery_active", jp.array(False)),
            )
        if reward_mode == "recovery_v3":
            feet = self._foot_xy(data)
            foot_delta = feet[1] - feet[0]
            segment_fraction = jp.clip(
                jp.dot(self._center_of_mass(data)[:2] - feet[0], foot_delta)
                / jp.maximum(jp.dot(foot_delta, foot_delta), 1e-8),
                0.0,
                1.0,
            )
            closest_support_point = feet[0] + segment_fraction * foot_delta
            support_distance = jp.linalg.norm(
                self._center_of_mass(data)[:2] - closest_support_point
            )
            local_angvel = self._local_root_angvel(data)
            angular_speed = jp.linalg.norm(local_angvel[jp.array([0, 2])])
            return stand_recovery_v3_score(
                *common[:2],
                angular_speed,
                support_distance,
                *common[2:],
                info.get("stand_recovery_active", jp.array(False)),
            )
        if reward_mode == "recovery_v2":
            return stand_recovery_score(*common, self.thresholds)
        return stand_balance_score(
            *common,
            jp.all(self._foot_contact(data, info) > 0),
            self.thresholds,
        )

    def _recovery_v4_terms(
        self, data: mjx.Data, info: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """Return support and capture-point quantities used by recovery v4."""
        feet = self._foot_xy(data)
        foot_delta = feet[1] - feet[0]
        denominator = jp.maximum(jp.dot(foot_delta, foot_delta), 1e-8)
        center_of_mass = self._center_of_mass(data)
        com_xy = center_of_mass[:2]
        com_velocity = self._center_of_mass_velocity(data)[:2]

        def closest_support_point(point: jax.Array) -> jax.Array:
            fraction = jp.clip(
                jp.dot(point - feet[0], foot_delta) / denominator, 0.0, 1.0
            )
            return feet[0] + fraction * foot_delta

        gravity = float(np.linalg.norm(self.mj_model.opt.gravity))
        omega = jp.sqrt(gravity / jp.maximum(center_of_mass[2], 0.25))
        capture_point = com_xy + com_velocity / jp.maximum(omega, 1e-6)
        closest_com = closest_support_point(com_xy)
        closest_capture = closest_support_point(capture_point)
        # Approximate the two sole rectangles by their centre segment expanded
        # with a conservative isotropic footprint radius.
        support_radius = jp.array(0.06, dtype=jp.float32)
        support_distance = jp.maximum(
            jp.linalg.norm(com_xy - closest_com) - support_radius, 0.0
        )
        capture_distance = jp.maximum(
            jp.linalg.norm(capture_point - closest_capture) - support_radius, 0.0
        )
        target_velocity = omega * (closest_com - com_xy)
        local_angvel = self._local_root_angvel(data)
        return {
            "capture_distance": capture_distance,
            "support_distance": support_distance,
            "velocity_target_error": jp.linalg.norm(com_velocity - target_velocity),
            "angular_speed": jp.linalg.norm(local_angvel[jp.array([0, 2])]),
            "any_foot_contact": jp.any(self._foot_contact(data, info) > 0),
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict[str, jax.Array],
    ) -> jax.Array:
        """Optionally gate imitation quality with dense physical balance feedback."""
        imitation = super()._get_reward(data, action, previous_action, info)
        reward_mode = self._config.get("stand_reward_mode", "imitation")
        if reward_mode == "imitation":
            return imitation
        if reward_mode == "recovery_v3":
            physical_score = self._balance_score(data, info)
            recovery_active = info.get("stand_recovery_active", jp.array(False))
            imitation_quality = jp.clip(imitation / self.REWARD_MAX, 0.0, 1.0)
            imitation_weight = jp.where(recovery_active, 0.10, 0.30)
            stable = (
                (jp.linalg.norm(data.qvel[:2]) < self.thresholds.max_speed)
                & (
                    jp.linalg.norm(data.qpos[:2] - info["stand_anchor_xy"])
                    < self.thresholds.max_distance
                )
                & (self._torso_up(data) >= self.thresholds.min_torso_up)
                & (data.qpos[2] / self._init_q[2] >= self.thresholds.min_height_ratio)
                & jp.all(self._foot_contact(data, info) > 0)
            )
            reward = self.REWARD_MAX * (
                (1.0 - imitation_weight) * physical_score
                + imitation_weight * imitation_quality
            )
            reward += 0.15 * stable.astype(jp.float32)
            reward -= self.ACTION_RATE_COST_SCALE * jp.sum(
                jp.square(action - previous_action)
            )
            reward -= 0.10 * jp.clip(self._get_foot_slip_cost(data, info), 0.0, 3.0)
            reward = jp.clip(jp.nan_to_num(reward, nan=0.0), 0.0, self.REWARD_MAX)
            return jp.where(self._get_done(data, info), imitation, reward)
        if reward_mode == "recovery_v4":
            physical_score = self._balance_score(data, info)
            recovery_active = info.get("stand_recovery_active", jp.array(False))
            imitation_quality = jp.clip(imitation / self.REWARD_MAX, 0.0, 1.0)
            imitation_weight = jp.where(recovery_active, 0.05, 0.20)
            reward = self.REWARD_MAX * (
                (1.0 - imitation_weight) * physical_score
                + imitation_weight * imitation_quality
            )
            reward -= self.ACTION_RATE_COST_SCALE * jp.sum(
                jp.square(action - previous_action)
            )
            reward -= 0.05 * jp.clip(self._get_foot_slip_cost(data, info), 0.0, 3.0)
            reward = jp.clip(jp.nan_to_num(reward, nan=0.0), 0.0, self.REWARD_MAX)
            return jp.where(self._get_done(data, info), imitation, reward)
        # Keep a small pose signal outside the balance region. In particular,
        # a nearly perfect joint pose must not mask large root speed/drift.
        shaped = imitation * (0.2 + 0.8 * self._balance_score(data, info))
        return jp.where(self._get_done(data, info), imitation, shaped)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Start in neutral stance with no elapsed dwell time."""
        state = super().reset(rng)
        info = dict(state.info)
        info["stand_anchor_xy"] = state.data.qpos[:2]
        info["stand_dwell_steps"] = jp.array(0, dtype=jp.int32)
        state = state.replace(info=info)
        metrics = self._stand_metrics(state)
        metrics.update(
            stand_dwell_steps=jp.array(0, dtype=jp.int32),
            stand_dwell_seconds=jp.array(0.0),
            stand_success=jp.array(0.0),
            stand_balance_score=self._balance_score(state.data, state.info),
        )
        if self._config.get("stand_reward_mode") == "recovery_v4":
            terms = self._recovery_v4_terms(state.data, state.info)
            metrics.update(
                stand_capture_distance=terms["capture_distance"],
                stand_support_distance=terms["support_distance"],
                stand_velocity_target_error=terms["velocity_target_error"],
            )
        return state.replace(
            metrics={
                **state.metrics,
                **{name: value.astype(jp.float32) for name, value in metrics.items()},
            }
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Measure stand success without early termination or success bonuses."""
        state = super().step(state, action)
        metrics = self._stand_metrics(state)
        metrics["stand_balance_score"] = self._balance_score(state.data, state.info)
        if self._config.get("stand_reward_mode") == "recovery_v4":
            terms = self._recovery_v4_terms(state.data, state.info)
            metrics.update(
                stand_capture_distance=terms["capture_distance"],
                stand_support_distance=terms["support_distance"],
                stand_velocity_target_error=terms["velocity_target_error"],
            )
        info = {**state.info, "stand_dwell_steps": metrics["stand_dwell_steps"]}
        return state.replace(
            info=info,
            metrics={
                **state.metrics,
                **{name: value.astype(jp.float32) for name, value in metrics.items()},
            },
        )
