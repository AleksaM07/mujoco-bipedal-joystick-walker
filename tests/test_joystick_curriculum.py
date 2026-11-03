"""Protect learned gait rewards and reject command-independent promotion."""

import unittest
from dataclasses import asdict
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

import evaluate
import train
from biomechanics_env import BiomechanicsJoystickEnv
from config import EnvConfig
from scripts.train_joystick_curriculum import promotion_gate
from tests import test_deepmimic_reward_contract as reward_contract


class JoystickCurriculumTests(unittest.TestCase):
    def test_curriculum_samples_separated_endpoint_commands(self):
        env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
        env._config = config_dict.ConfigDict(dict(
            command_profile="joystick_curriculum", joystick_speed_center=.5,
            joystick_speed_radius=.2, joystick_yaw_limit=.1,
        ))
        samples = np.asarray(jax.jit(jax.vmap(env.sample_command))(
            jax.random.split(jax.random.PRNGKey(1), 1000)))
        np.testing.assert_allclose(np.unique(samples[:, 0]), [.3, .7], atol=1e-6)
        np.testing.assert_array_equal(samples[:, 1], 0.)
        np.testing.assert_allclose(np.unique(samples[:, 2]), [-.1, .1], atol=1e-6)

    def test_command_reward_is_bounded_and_ignores_bvh_root_speed(self):
        fixture = reward_contract.DeepMimicRewardContractTests()
        fixture.setUp()
        env, data, info = fixture.env, fixture.data, fixture.info
        env._config = config_dict.ConfigDict(dict(
            deepmimic_reward_mode="command", joystick_reward_weight=.2,
            tracking_sigma=.25, tracking_yaw_sigma=.35,
        ))
        info["command"] = jp.array([.5, 0., 0.])
        measured = jp.array([.5, 0., 0.])
        with (
            patch.object(env, "_measured_command", side_effect=lambda _: measured),
            patch.object(env, "_get_done", return_value=jp.array(False)),
            patch.object(env, "_get_physical_fail_done", return_value=jp.array(False)),
        ):
            matched = float(env._get_reward(data, jp.zeros(18), jp.zeros(18), info))
            self.assertAlmostEqual(matched, env.REWARD_MAX, places=5)
            fixture.reference["root_vel"] = jp.array([-7., 3., 0.])
            unchanged = float(env._get_command_imitation_reward(data, info))
            self.assertAlmostEqual(matched, unchanged, places=6)
            for speed in (0., 1., 10.):
                measured = jp.array([speed, 0., 0.])
                reward = float(env._get_command_imitation_reward(data, info))
                self.assertGreaterEqual(reward, 0.)
                self.assertLess(reward, matched)
            with patch.object(env, "_get_done", return_value=jp.array(True)):
                self.assertEqual(float(env._get_command_imitation_reward(data, info)), 0.)
                with patch.object(env, "_get_physical_fail_done", return_value=jp.array(True)):
                    self.assertEqual(float(env._get_command_imitation_reward(data, info)), env.FALL_REWARD)

    def test_config_round_trip_and_train_eval_agree(self):
        config = EnvConfig(command_profile="joystick_curriculum",
                           deepmimic_reward_mode="command", joystick_speed_center=.4,
                           joystick_speed_radius=.03, joystick_yaw_limit=.12,
                           joystick_reward_weight=.3, arm_reference_mode="cmu_wrist")
        restored = evaluate.restore_saved_env_config({"env": asdict(config)})
        self.assertEqual(config, restored)
        for module in (evaluate, train):
            with patch.object(module, "BiomechanicsJoystickEnv") as constructor:
                module.make_environment(restored)
            forwarded = constructor.call_args.kwargs["config_overrides"]
            for key in ("command_profile", "deepmimic_reward_mode", "joystick_speed_center",
                        "joystick_speed_radius", "joystick_yaw_limit", "joystick_reward_weight",
                        "arm_reference_mode"):
                self.assertEqual(forwarded[key], getattr(config, key))
        legacy = evaluate.restore_saved_env_config({"env": {"deepmimic_reward_mode": "pure"}})
        self.assertEqual(legacy.arm_reference_mode, "legacy")
        self.assertEqual(legacy.deepmimic_reward_mode, "pure")

    def test_high_reward_walker_ignoring_commands_does_not_advance(self):
        rows = []
        low = .5 - .25
        high = .5 + .25
        for command in (low, high):
            for seed in (11, 22):
                rows.append(dict(seed=seed, command=[command, 0., 0.], length=500,
                                 survived=True, pose=.99, linear_mae=.05, yaw_mae=0.,
                                 reward=1450., measured=[.5, 0., 0.]))
        gate = promotion_gate(rows, 0, .5)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["speed_response"])
        for row in rows:
            row["measured"] = row["command"]
        self.assertTrue(promotion_gate(rows, 0, .5)["passed"])
        rows[-1]["survived"] = False
        self.assertFalse(promotion_gate(rows, 0, .5)["passed"])


if __name__ == "__main__":
    unittest.main()
