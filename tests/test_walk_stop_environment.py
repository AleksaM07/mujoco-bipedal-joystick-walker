"""Focused compatibility tests for the opt-in walk-stop curriculum."""

import unittest
from unittest.mock import patch

import jax.numpy as jp

import evaluate
import train
from config import EnvConfig
from skills.walk_stop import WalkStopEnv


class WalkStopEnvironmentTests(unittest.TestCase):
    def test_transition_clock_is_smooth_and_bounded(self) -> None:
        env = object.__new__(WalkStopEnv)
        env._transition_blend_steps = 40
        info = {
            "motion_step": jp.array(0, dtype=jp.int32),
            "transition_start_step": jp.array(100, dtype=jp.int32),
        }
        self.assertEqual(float(env._transition_alpha(info)), 0.0)
        halfway = {**info, "motion_step": jp.array(120, dtype=jp.int32)}
        self.assertAlmostEqual(float(env._transition_alpha(halfway)), 0.5, places=6)
        finished = {**info, "motion_step": jp.array(140, dtype=jp.int32)}
        self.assertEqual(float(env._transition_alpha(finished)), 1.0)

    def test_training_factory_selects_walk_stop_without_changing_default(self) -> None:
        self.assertEqual(EnvConfig().locomotion_skill, "walk")
        config = EnvConfig(locomotion_skill="walk_stop")
        with patch("skills.walk_stop.WalkStopEnv") as constructor:
            train.make_environment(config)
        constructor.assert_called_once()
        forwarded = constructor.call_args.kwargs["config_overrides"]
        self.assertEqual(forwarded["transition_blend_seconds"], 0.8)
        self.assertEqual(forwarded["transition_min_walk_seconds"], 2.0)

    def test_evaluation_factory_selects_same_skill(self) -> None:
        config = EnvConfig(locomotion_skill="walk_stop")
        with patch("skills.walk_stop.WalkStopEnv") as constructor:
            evaluate.make_environment(config)
        constructor.assert_called_once()
        self.assertNotIn(
            "locomotion_skill", constructor.call_args.kwargs["config_overrides"]
        )


if __name__ == "__main__":
    unittest.main()
