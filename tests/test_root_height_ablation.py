"""Check smooth floor-height offsets and diagnostic-only override scoping."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from tools.check_root_height_ablation import (
    final_height_ablation,
    smooth_height_raise,
    summarize_clearance,
)


class SmoothHeightRaiseTests(unittest.TestCase):
    def test_equal_heights_have_maximum_offset(self) -> None:
        for height in (-2.0, 0.0, 1.5):
            with self.subTest(height=height):
                self.assertEqual(
                    smooth_height_raise(np.array([height, height]), 0.02), 0.01
                )

    def test_offset_produces_smooth_minimum_without_lowering_root(self) -> None:
        for heights in ([0.01, 0.02], [-0.02, 0.01], [-1.5, -1.5]):
            for epsilon in (0.0, 0.005, 0.02, 0.1):
                with self.subTest(heights=heights, epsilon=epsilon):
                    values = np.array(heights)
                    offset = smooth_height_raise(values, epsilon)
                    self.assertGreaterEqual(offset, 0.0)
                    self.assertLessEqual(offset, epsilon / 2)
                    expected_minimum = (
                        values.sum() - np.hypot(values[0] - values[1], epsilon)
                    ) / 2
                    self.assertAlmostEqual(
                        float(np.min(values) - offset), expected_minimum
                    )

    def test_translation_and_foot_order_do_not_change_offset(self) -> None:
        heights = np.array([-0.013, 0.009])
        expected = smooth_height_raise(heights, 0.01)
        self.assertEqual(smooth_height_raise(heights[::-1], 0.01), expected)
        self.assertAlmostEqual(smooth_height_raise(heights + 20, 0.01), expected)

    def test_zero_epsilon_is_exact_identity_and_input_is_not_mutated(self) -> None:
        heights = np.array([-0.03123456789, 0.06432109876])
        snapshot = heights.copy()
        self.assertEqual(smooth_height_raise(heights, 0.0), 0.0)
        smooth_height_raise(heights, 0.012)
        np.testing.assert_array_equal(heights, snapshot)

    def test_invalid_shapes_and_nonfinite_heights_are_rejected(self) -> None:
        for heights in (
            np.array(0.0),
            np.array([]),
            np.array([0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([[0.0, 0.0]]),
            np.array([np.nan, 0.0]),
            np.array([0.0, np.inf]),
            np.array([-np.inf, 0.0]),
        ):
            with self.subTest(heights=heights), self.assertRaises(ValueError):
                smooth_height_raise(heights, 0.01)

    def test_invalid_epsilon_is_rejected(self) -> None:
        for epsilon in (-0.001, np.nan, np.inf, -np.inf):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                smooth_height_raise(np.zeros(2), epsilon)


class FinalHeightAblationScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SimpleNamespace(_foot_geom_ids_np=np.array([2, 5]))
        self.data = np.array([0.1, 0.2, -0.03123456789, 0.3, 0.4, -0.02654321098])
        self.geom_ids = (2, 5, 3)
        self.lock_calls: list[tuple[tuple, dict]] = []

    def read_heights(self, env: SimpleNamespace) -> np.ndarray:
        """Read both foot heights and an unrelated geometry through the class."""
        return np.array(
            [
                BiomechanicsJoystickEnv._geom_min_z(env, self.data, geom_id)
                for geom_id in self.geom_ids
            ]
        )

    @contextmanager
    def dummy_methods(self) -> Iterator[tuple]:
        """Install original methods that expose when synthetic heights leak."""

        def geom_fixture(env, data, geom_id):
            return float(data[geom_id])

        def lock_fixture(env, *args, **kwargs):
            self.lock_calls.append((args, kwargs))
            return self.read_heights(env)

        def align_fixture(env, *, token=None):
            before = self.read_heights(env)
            during_lock = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                env, "joints", "roots", "quaternions", 3, contact_history=True
            )
            after = self.read_heights(env)
            return before, during_lock, after, token

        with (
            patch.object(BiomechanicsJoystickEnv, "_geom_min_z", geom_fixture),
            patch.object(
                BiomechanicsJoystickEnv,
                "_lock_clip_root_xy_to_stance_feet",
                lock_fixture,
            ),
            patch.object(
                BiomechanicsJoystickEnv,
                "_align_reference_roots_to_sim_floor",
                align_fixture,
            ),
        ):
            yield align_fixture, geom_fixture, lock_fixture

    def test_only_final_alignment_foot_reads_are_shifted_per_environment(self) -> None:
        expected_raw = self.data[list(self.geom_ids)]
        foot_heights = self.data[self.env._foot_geom_ids_np].astype(np.float32)
        offset = smooth_height_raise(foot_heights, 0.02)
        expected_final = expected_raw - np.array([offset, offset, 0.0])
        other_env = SimpleNamespace(_foot_geom_ids_np=np.array([2, 5]))

        with self.dummy_methods(), final_height_ablation(0.02):
            align = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor
            for env in (self.env, other_env):
                np.testing.assert_array_equal(self.read_heights(env), expected_raw)
                initial = align(env, token="initial")
                for result in initial[:3]:
                    np.testing.assert_array_equal(result, expected_raw)
                self.assertEqual(initial[3], "initial")
            for env in (self.env, other_env):
                before, during_lock, after, token = align(env, token="final")
                np.testing.assert_allclose(before, expected_final, rtol=0, atol=1e-15)
                np.testing.assert_array_equal(during_lock, expected_raw)
                np.testing.assert_array_equal(after, before)
                self.assertEqual(token, "final")
                np.testing.assert_array_equal(self.read_heights(env), expected_raw)

        self.assertEqual(len(self.lock_calls), 4)
        for args, kwargs in self.lock_calls:
            self.assertEqual(args, ("joints", "roots", "quaternions", 3))
            self.assertEqual(kwargs, {"contact_history": True})

    def test_zero_epsilon_preserves_raw_double_precision_heights_exactly(self) -> None:
        expected = self.data[list(self.geom_ids)]
        with self.dummy_methods(), final_height_ablation(0.0):
            for _ in range(2):
                results = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                    self.env
                )
                for heights in results[:3]:
                    np.testing.assert_array_equal(heights, expected)

    def test_outside_alignment_reads_remain_raw_even_after_completed_passes(
        self,
    ) -> None:
        expected = self.data[list(self.geom_ids)]
        with self.dummy_methods(), final_height_ablation(0.02):
            for _ in range(2):
                np.testing.assert_array_equal(self.read_heights(self.env), expected)
                BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(self.env)
            np.testing.assert_array_equal(self.read_heights(self.env), expected)

    def test_third_alignment_is_rejected(self) -> None:
        with self.dummy_methods(), final_height_ablation(0.02):
            align = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor
            align(self.env)
            align(self.env)
            with self.assertRaises(ValueError):
                align(self.env)

    def test_nested_alignment_is_rejected_and_state_is_released(self) -> None:
        calls = 0

        def nested_align(env):
            nonlocal calls
            calls += 1
            if calls == 1:
                return BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(env)
            return self.read_heights(env)

        with (
            self.dummy_methods(),
            patch.object(
                BiomechanicsJoystickEnv,
                "_align_reference_roots_to_sim_floor",
                nested_align,
            ),
            final_height_ablation(0.02),
        ):
            with self.assertRaises(ValueError):
                BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(self.env)
            np.testing.assert_array_equal(
                self.read_heights(self.env), self.data[list(self.geom_ids)]
            )
            result = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                self.env
            )
            self.assertLess(result[0], self.data[2])

    def test_all_methods_restore_after_error(self) -> None:
        names = (
            "_align_reference_roots_to_sim_floor",
            "_geom_min_z",
            "_lock_clip_root_xy_to_stance_feet",
        )
        originals = [getattr(BiomechanicsJoystickEnv, name) for name in names]
        with self.dummy_methods() as fixtures:
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                with final_height_ablation(0.02):
                    BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                        self.env
                    )
                    raise RuntimeError("fixture interruption")
            for name, fixture in zip(names, fixtures, strict=True):
                self.assertIs(getattr(BiomechanicsJoystickEnv, name), fixture)
        for name, original in zip(names, originals, strict=True):
            self.assertIs(getattr(BiomechanicsJoystickEnv, name), original)

    def test_lock_error_does_not_leave_geometry_override_suspended(self) -> None:
        lock_heights = []

        def failing_lock(env, *args, **kwargs):
            lock_heights.append(self.read_heights(env))
            raise RuntimeError("lock interrupted")

        def catching_align(env):
            before = self.read_heights(env)
            with self.assertRaisesRegex(RuntimeError, "lock interrupted"):
                BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                    env, "joints", "roots", "quaternions", 3
                )
            return before, self.read_heights(env)

        with (
            self.dummy_methods(),
            patch.object(
                BiomechanicsJoystickEnv,
                "_lock_clip_root_xy_to_stance_feet",
                failing_lock,
            ),
            patch.object(
                BiomechanicsJoystickEnv,
                "_align_reference_roots_to_sim_floor",
                catching_align,
            ),
            final_height_ablation(0.02),
        ):
            align = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor
            align(self.env)
            before, after = align(self.env)
            np.testing.assert_array_equal(before, after)
            self.assertLess(before[0], self.data[2])
        for heights in lock_heights:
            np.testing.assert_array_equal(heights, self.data[list(self.geom_ids)])

    def test_invalid_epsilon_context_does_not_replace_methods(self) -> None:
        with self.dummy_methods() as fixtures:
            for epsilon in (-0.01, np.nan, np.inf, -np.inf):
                with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                    with final_height_ablation(epsilon):
                        self.fail("Invalid epsilon must be rejected before yielding.")
                for name, fixture in zip(
                    (
                        "_align_reference_roots_to_sim_floor",
                        "_geom_min_z",
                        "_lock_clip_root_xy_to_stance_feet",
                    ),
                    fixtures,
                    strict=True,
                ):
                    self.assertIs(getattr(BiomechanicsJoystickEnv, name), fixture)


class ClearanceSummaryTests(unittest.TestCase):
    def test_interleaved_samples_use_lowest_sole_and_strict_thresholds(self) -> None:
        paired_heights = [
            (0.0, -0.005),
            (0.0001, 0.002),
            (0.00010001, 0.003),
            (-0.02, -0.01),
        ]
        rows = [
            {"sample": sample, "surface_height_m": heights[foot]}
            for foot in (0, 1)
            for sample, heights in enumerate(paired_heights)
        ]
        summary = summarize_clearance(rows)
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["maximum_lowest_sole_height_m"], 0.00010001)
        self.assertEqual(summary["both_soles_above_floor_fraction"], 0.5)
        self.assertEqual(summary["both_soles_above_0p1mm_fraction"], 0.25)
        self.assertEqual(summarize_clearance(rows[::-1]), summary)

    def test_empty_and_incomplete_samples_are_rejected(self) -> None:
        for samples in ([], [0], [0, 0, 0], [0, 0, 1]):
            rows = [{"sample": sample, "surface_height_m": 0.0} for sample in samples]
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                summarize_clearance(rows)

    def test_either_nonfinite_foot_is_rejected_even_when_other_foot_is_lower(
        self,
    ) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            for foot in (0, 1):
                heights = [-0.01, -0.01]
                heights[foot] = invalid
                rows = [{"sample": 0, "surface_height_m": height} for height in heights]
                with (
                    self.subTest(invalid=invalid, foot=foot),
                    self.assertRaises(ValueError),
                ):
                    summarize_clearance(rows)


if __name__ == "__main__":
    unittest.main()
