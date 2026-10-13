"""Exercise production imitation rewards without constructing a simulator."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jp

from biomechanics_env import BiomechanicsJoystickEnv


class RewardFixture(BiomechanicsJoystickEnv):
    """Supply state/reference extraction only; keep all reward math unchanged."""

    def _query_bvh_reference(self, info: dict, offset: int) -> dict:
        self.query_calls.append((info, offset))
        return info["reference"]

    def _key_marker_positions(self, data: SimpleNamespace) -> jax.Array:
        return data.key_pos

    def _reference_anchor_pos(self, data: SimpleNamespace) -> jax.Array:
        return data.root_pos

    def _reference_anchor_quat(self, data: SimpleNamespace) -> jax.Array:
        return data.root_quat

    def _reference_anchor_linvel(self, data: SimpleNamespace) -> jax.Array:
        return data.root_vel

    def _reference_anchor_angvel(self, data: SimpleNamespace) -> jax.Array:
        return data.root_angvel


class DeepMimicRewardContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RewardFixture.__new__(RewardFixture)
        self.env._config = {}
        self.env._actuator_qpos_indices = jp.arange(18)
        self.env._actuator_dof_indices = jp.arange(18)
        self.env.query_calls = []
        self.reference = {
            "qpos": jp.zeros(18),
            "qvel": jp.zeros(18),
            "key_pos": jp.array([[0.1, -0.1, 0.0], [0.1, 0.1, 0.0]]),
            "root_pos": jp.array([0.0, 0.0, 1.0]),
            "root_quat": jp.array([1.0, 0.0, 0.0, 0.0]),
            "root_vel": jp.zeros(3),
            "root_angvel": jp.zeros(3),
        }
        self.data = SimpleNamespace(**self.reference)
        self.info = {"reference": self.reference, "motion_step": 7}

    def score(self) -> dict[str, jax.Array]:
        return self.env._get_bvh_deepmimic_reward(self.data, self.info)

    def test_exact_match_has_unit_reward_with_quaternion_numerical_floor(self) -> None:
        reward = self.score()
        for name in ("pose", "velocity", "root_velocity", "key_position"):
            self.assertEqual(float(reward[name]), 1.0)
        # Production quaternion clipping leaves a tiny nonzero angular error.
        self.assertAlmostEqual(float(reward["root_pose"]), 1.0, delta=5e-6)
        self.assertAlmostEqual(float(reward["total"]), 1.0, delta=1e-6)
        for name in ("pose_error", "velocity_error", "key_pos_error"):
            self.assertEqual(float(reward[name]), 0.0)

    def test_pose_uses_mean_squared_error_over_eighteen_dofs(self) -> None:
        self.data.qpos = self.data.qpos.at[0].set(1.0)
        reward = self.score()
        self.assertEqual(self.env.DEEPMIMIC_POSE_SCALE, 0.25)
        self.assertAlmostEqual(float(reward["pose_error"]), 1.0 / 18.0)
        expected = math.exp(-self.env.DEEPMIMIC_POSE_SCALE / 18.0)
        self.assertAlmostEqual(float(reward["pose"]), expected)
        self.assertGreater(float(reward["pose"]), math.exp(-0.25))

    def test_velocity_uses_mean_squared_error_over_eighteen_dofs(self) -> None:
        self.data.qvel = self.data.qvel.at[0].set(3.0)
        reward = self.score()
        self.assertEqual(self.env.DEEPMIMIC_VELOCITY_SCALE, 0.01)
        self.assertAlmostEqual(float(reward["velocity_error"]), 9.0 / 18.0)
        expected = math.exp(-self.env.DEEPMIMIC_VELOCITY_SCALE * 9.0 / 18.0)
        self.assertAlmostEqual(float(reward["velocity"]), expected)
        self.assertGreater(float(reward["velocity"]), math.exp(-0.09))

    def test_single_joint_error_monotonically_lowers_pose_and_total(self) -> None:
        previous = self.score()
        for error in (0.1, 0.5, 1.0):
            self.data.qpos = self.data.qpos.at[4].set(error)
            reward = self.score()
            for name in ("pose", "total"):
                self.assertLess(float(reward[name]), float(previous[name]))
            previous = reward

    def test_root_xy_translation_is_reported_but_not_penalized(self) -> None:
        baseline = self.score()
        translation = jp.array([3.0, 4.0, 0.0])
        self.data.root_pos = self.data.root_pos + translation
        self.data.key_pos = self.data.key_pos + translation
        reward = self.score()
        self.assertAlmostEqual(float(reward["root_xy_error"]), 5.0)
        for name in baseline.keys() - {"root_xy_error"}:
            self.assertAlmostEqual(float(reward[name]), float(baseline[name]), places=6)

    def test_root_height_translation_remains_penalized(self) -> None:
        baseline = self.score()
        translation = jp.array([0.0, 0.0, 0.2])
        self.data.root_pos = self.data.root_pos + translation
        self.data.key_pos = self.data.key_pos + translation
        reward = self.score()
        self.assertAlmostEqual(float(reward["root_height_error"]), 0.2)
        self.assertLess(float(reward["root_pose"]), float(baseline["root_pose"]))
        self.assertAlmostEqual(float(reward["key_position"]), 1.0)

    def test_half_meter_velocity_change_only_changes_root_velocity_component(
        self,
    ) -> None:
        baseline = self.score()
        self.data.root_vel = jp.array([0.5, 0.0, 0.0])
        reward = self.score()
        self.assertAlmostEqual(float(reward["root_velocity"]), math.exp(-0.25))
        self.assertEqual(float(reward["root_vel_error"]), 0.5)
        changed = {"total", "com", "root_velocity", "root_vel_error"}
        for name in baseline.keys() - changed:
            self.assertAlmostEqual(float(reward[name]), float(baseline[name]))
        self.assertAlmostEqual(
            float(baseline["total"] - reward["total"]),
            0.1 * (1.0 - math.exp(-0.25)),
        )

    def test_root_velocity_weight_scale_is_normalized(self) -> None:
        base_weight = self.env.DEEPMIMIC_ROOT_VELOCITY_WEIGHT
        other_weight = sum(
            getattr(self.env, f"DEEPMIMIC_{name}_WEIGHT")
            for name in ("POSE", "VELOCITY", "ROOT_POSE", "KEY_POSITION")
        )
        for scale in (0.0, 1.0, 4.0):
            with self.subTest(scale=scale):
                self.env._config["deepmimic_root_velocity_weight_scale"] = scale
                self.data.root_vel = jp.zeros(3)
                baseline = self.score()
                self.data.root_vel = jp.array([0.5, 0.0, 0.0])
                reward = self.score()
                fraction = scale * base_weight / (other_weight + scale * base_weight)
                self.assertAlmostEqual(
                    float(baseline["total"] - reward["total"]),
                    fraction * (1.0 - math.exp(-0.25)),
                    delta=2e-7,
                )
                self.assertAlmostEqual(
                    float(reward["total_no_root_velocity"]),
                    float(baseline["total_no_root_velocity"]),
                )

    def test_scores_only_info_reference_without_temporal_best_match(self) -> None:
        matched = self.score()
        self.assertEqual(len(self.env.query_calls), 1)
        self.assertIs(self.env.query_calls[0][0], self.info)
        self.assertEqual(self.env.query_calls[0][1], 0)
        self.info["reference"] = {**self.reference, "qpos": jp.full(18, 0.5)}
        self.env.query_calls.clear()
        reward = self.env._get_bvh_deepmimic_frame_reward(
            self.data, self.info, jp.array(99), jp.array(999)
        )
        self.assertEqual(len(self.env.query_calls), 1)
        self.assertIs(self.env.query_calls[0][0], self.info)
        self.assertEqual(self.env.query_calls[0][1], 0)
        self.assertAlmostEqual(float(reward["pose_error"]), 0.25)
        self.assertLess(float(reward["total"]), float(matched["total"]))


if __name__ == "__main__":
    unittest.main()
