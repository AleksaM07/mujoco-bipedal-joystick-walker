"""Standing requires continuous physical dwell, not a high imitation score."""

import unittest
from functools import partial

import jax
import jax.numpy as jp

from skills.stand_metrics import (
    StandThresholds,
    required_dwell_steps,
    update_stand_metrics,
)


class StandMetricsTests(unittest.TestCase):
    def update(self, steps=0, **overrides):
        values = dict(
            previous_steps=jp.array(steps),
            root_xy=jp.zeros(2),
            velocity_xy=jp.zeros(2),
            anchor_xy=jp.zeros(2),
            torso_up=jp.array(1.0),
            height_ratio=jp.array(1.0),
            both_feet_contact=jp.array(True),
            physical_failure=jp.array(False),
        )
        values.update(overrides)
        return jax.jit(partial(update_stand_metrics, dt=0.02))(**values)

    def test_success_needs_fifty_complete_steps(self):
        self.assertEqual(float(self.update(48)["stand_success"]), 0)
        self.assertEqual(float(self.update(49)["stand_success"]), 1)
        self.assertEqual(float(self.update(49)["stand_dwell_seconds"]), 1)
        self.assertEqual(required_dwell_steps(0.03, StandThresholds()), 34)

    def test_every_gate_resets_dwell_even_after_success(self):
        cases = [
            {"velocity_xy": jp.array([0.15, 0.0])},
            {"root_xy": jp.array([0.25, 0.0])},
            {"torso_up": jp.array(0.89)},
            {"height_ratio": jp.array(0.84)},
            {"both_feet_contact": jp.array(False)},
            {"physical_failure": jp.array(True)},
            {"velocity_xy": jp.array([jp.nan, 0.0])},
            {"root_xy": jp.array([jp.inf, 0.0])},
            {"torso_up": jp.array(jp.nan)},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                metrics = self.update(100, **overrides)
                self.assertEqual(int(metrics["stand_dwell_steps"]), 0)
                self.assertEqual(float(metrics["stand_success"]), 0)

    def test_rotation_and_translation_do_not_change_distance_gate(self):
        metrics = self.update(
            49, root_xy=jp.array([10.0, -4.9]), anchor_xy=jp.array([10.0, -5.0])
        )
        self.assertEqual(float(metrics["stand_success"]), 1)
        self.assertAlmostEqual(float(metrics["stand_distance"]), 0.1, places=5)

    def test_mixed_batch_preserves_independent_dwell(self):
        def update(steps, failure):
            return update_stand_metrics(
                steps,
                jp.zeros(2),
                jp.zeros(2),
                jp.zeros(2),
                jp.array(1.0),
                jp.array(1.0),
                jp.array(True),
                failure,
                dt=0.02,
            )

        result = jax.jit(jax.vmap(update))(
            jp.array([49, 49, 5]), jp.array([False, True, False])
        )
        self.assertEqual(result["stand_dwell_steps"].tolist(), [50, 0, 6])
        self.assertEqual(result["stand_success"].tolist(), [1.0, 0.0, 0.0])

    def test_invalid_thresholds_and_timestep_fail_early(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    StandThresholds(dwell_seconds=value)
                with self.assertRaises(ValueError):
                    required_dwell_steps(value, StandThresholds())
        with self.assertRaises(ValueError):
            StandThresholds(min_torso_up=1.01)


if __name__ == "__main__":
    unittest.main()
