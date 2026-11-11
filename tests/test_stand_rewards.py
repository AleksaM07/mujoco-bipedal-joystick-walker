"""Stand shaping must distinguish stable poses from drifting or falling ones."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from skills.stand_metrics import StandThresholds
from skills.stand_rewards import stand_balance_score, stand_recovery_score


class StandRewardTests(unittest.TestCase):
    def score(self, **changes: float) -> float:
        values = {
            "speed": 0.0,
            "distance": 0.0,
            "torso_up": 1.0,
            "height_ratio": 1.0,
            "both_feet_contact": 1.0,
        }
        values.update(changes)
        return float(
            stand_balance_score(
                **{name: jp.array(value) for name, value in values.items()}
            )
        )

    def test_stable_reference_has_maximum_score(self) -> None:
        self.assertEqual(self.score(), 1.0)

    def test_each_physical_error_reduces_score_before_termination(self) -> None:
        for change in (
            {"speed": 0.15},
            {"distance": 0.25},
            {"torso_up": 0.9},
            {"height_ratio": 0.85},
        ):
            with self.subTest(change=change):
                self.assertAlmostEqual(self.score(**change), np.exp(-1), places=5)
        self.assertEqual(self.score(both_feet_contact=0), 0.0)
        self.assertLess(self.score(speed=0.45), 0.001)
        self.assertLess(self.score(distance=0.75), 0.001)

    def test_recovery_reward_allows_a_temporary_corrective_step(self) -> None:
        recovery = stand_recovery_score(
            jp.array(0.0),
            jp.array(0.0),
            jp.array(1.0),
            jp.array(1.0),
        )
        self.assertEqual(float(recovery), 1.0)
        self.assertEqual(self.score(both_feet_contact=0), 0.0)

    def test_recovery_reward_still_penalizes_state_errors_and_nonfinite_values(
        self,
    ) -> None:
        speed_error = stand_recovery_score(
            jp.array(0.15),
            jp.array(0.0),
            jp.array(1.0),
            jp.array(1.0),
        )
        invalid = stand_recovery_score(
            jp.array(float("nan")),
            jp.array(0.0),
            jp.array(1.0),
            jp.array(1.0),
        )
        self.assertAlmostEqual(float(speed_error), np.exp(-1), places=5)
        self.assertEqual(float(invalid), 0.0)

    def test_invalid_state_cannot_receive_balance_reward(self) -> None:
        for field in ("speed", "distance", "torso_up", "height_ratio"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.score(**{field: value}), 0.0)

    def test_jit_vmap_preserves_independent_balance_scores(self) -> None:
        score = jax.jit(jax.vmap(stand_balance_score))(
            jp.array([0.0, 0.15, 0.45]),
            jp.zeros(3),
            jp.ones(3),
            jp.ones(3),
            jp.ones(3, dtype=bool),
        )
        np.testing.assert_allclose(score, np.exp(-np.array([0.0, 1.0, 9.0])), rtol=1e-5)

    def test_strict_valid_thresholds_do_not_divide_by_zero(self) -> None:
        score = stand_balance_score(
            jp.array(0.0),
            jp.array(0.0),
            jp.array(1.0),
            jp.array(1.0),
            jp.array(True),
            StandThresholds(min_torso_up=1.0, min_height_ratio=1.0),
        )
        self.assertEqual(float(score), 1.0)


if __name__ == "__main__":
    unittest.main()
