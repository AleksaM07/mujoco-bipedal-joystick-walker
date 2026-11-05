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
from skills.stand_rewards import stand_balance_score


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
        }:
            raise ValueError("stand_reward_mode must be imitation or balance_v1.")
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
        return stand_balance_score(
            jp.linalg.norm(data.qvel[:2]),
            jp.linalg.norm(data.qpos[:2] - info["stand_anchor_xy"]),
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
        info: dict[str, jax.Array],
    ) -> jax.Array:
        """Optionally gate imitation quality with dense physical balance feedback."""
        imitation = super()._get_reward(data, action, previous_action, info)
        if self._config.get("stand_reward_mode", "imitation") == "imitation":
            return imitation
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
        info = {**state.info, "stand_dwell_steps": metrics["stand_dwell_steps"]}
        return state.replace(
            info=info,
            metrics={
                **state.metrics,
                **{name: value.astype(jp.float32) for name, value in metrics.items()},
            },
        )
