"""Deterministic checks for reference timing; no simulator or training needed."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reference_kinematics import (
    differentiate_reference_positions,
    scale_reference_frame_times,
)


def load_environment_method(name: str):
    """Load one actual pure method without importing the GPU environment."""
    source = Path(__file__).resolve().parents[1] / "biomechanics_env.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for cls in module.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    isolated_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = {"jp": np}
    exec(
        compile(ast.fix_missing_locations(isolated_module), str(source), "exec"),
        namespace,
    )
    return namespace[name]


class ReferenceTimingTests(unittest.TestCase):
    def test_quarter_speed_changes_duration_and_derivatives_together(self) -> None:
        positions = np.array([[[0.0, 1.0], [0.1, 1.2], [0.2, 1.4]]])
        counts = np.array([3])
        source_intervals = np.array([0.1])
        normal_intervals = scale_reference_frame_times(source_intervals, 1.0)
        slow_intervals = scale_reference_frame_times(source_intervals, 0.25)
        normal_velocity = differentiate_reference_positions(
            positions, normal_intervals, counts
        )
        slow_velocity = differentiate_reference_positions(
            positions, slow_intervals, counts
        )
        np.testing.assert_allclose(slow_intervals, normal_intervals * 4.0)
        np.testing.assert_allclose(slow_velocity, normal_velocity * 0.25)
        np.testing.assert_allclose(normal_velocity[0], [[1.0, 2.0]] * 3)
        np.testing.assert_allclose(source_intervals, [0.1])

    def test_actual_environment_clock_is_not_scaled_twice(self) -> None:
        get_motion_time = load_environment_method("_get_bvh_reference_motion_time")
        env = SimpleNamespace(dt=0.02)
        info = {
            "bvh_reference_time_offset": np.array(0.1, dtype=np.float32),
            "motion_step": np.array(10, dtype=np.int32),
        }
        self.assertAlmostEqual(float(get_motion_time(env, info, 5)), 0.4, places=6)
        interval = scale_reference_frame_times(np.array([0.1]), 0.25)[0]
        # Three frames now span 0.8 s, so query at 0.4 s is the middle frame.
        phase = float(get_motion_time(env, info, 5)) / float(interval * (3 - 1))
        self.assertAlmostEqual(phase, 0.5, places=6)

    def test_each_clip_uses_its_own_interval(self) -> None:
        positions = np.array([[[0.0], [1.0]], [[0.0], [1.0]]])
        result = differentiate_reference_positions(
            positions, np.array([0.5, 0.25]), np.array([2, 2])
        )
        np.testing.assert_allclose(result[..., 0], [[2.0, 2.0], [4.0, 4.0]])

    def test_padding_never_affects_finite_differences(self) -> None:
        positions = np.array([[[1.0], [3.0], [5.0], [np.nan], [999.0]]])
        result = differentiate_reference_positions(
            positions, np.array([0.5]), np.array([3])
        )
        np.testing.assert_allclose(result, np.full((1, 5, 1), 4.0))

    def test_single_frame_has_zero_velocity(self) -> None:
        result = differentiate_reference_positions(
            np.array([[[2.0], [999.0]]]), np.array([0.1]), np.array([1])
        )
        np.testing.assert_array_equal(result, np.zeros((1, 2, 1)))

    def test_loop_seams_are_not_silently_repaired(self) -> None:
        positions = np.array([[[0.0], [1.0], [0.0]]])
        result = differentiate_reference_positions(
            positions, np.array([0.1]), np.array([3])
        )
        np.testing.assert_allclose(result[0, :, 0], [10.0, 0.0, -10.0])
        # Position endpoints coincide, but one-sided velocities disagree.
        # A separate loop gate must reject this instead of this helper hiding it.
        self.assertNotEqual(float(result[0, 0, 0]), float(result[0, -1, 0]))

    def test_nodal_gradient_is_not_claimed_as_exact_interpolant_derivative(
        self,
    ) -> None:
        positions = np.array([[[0.0], [1.0], [4.0]]])
        result = differentiate_reference_positions(
            positions, np.array([1.0]), np.array([3])
        )
        interpolated_velocity = float(0.5 * (result[0, 0, 0] + result[0, 1, 0]))
        segment_slope = float(positions[0, 1, 0] - positions[0, 0, 0])
        self.assertEqual(interpolated_velocity, 1.5)
        self.assertEqual(segment_slope, 1.0)

    def test_invalid_speed_scales_are_rejected(self) -> None:
        for scale in (0.0, -1.0, np.nan, np.inf, -np.inf, 1e-300, 1e300):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                scale_reference_frame_times(np.array([0.01]), scale)

    def test_invalid_intervals_are_rejected(self) -> None:
        for intervals in ([0.0], [-1.0], [np.nan], [np.inf], [], [[0.1]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                scale_reference_frame_times(np.array(intervals), 1.0)

    def test_invalid_counts_are_rejected(self) -> None:
        for counts in ([0], [-1], [3], [1.5], [np.nan], [np.inf], [1, 1]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                differentiate_reference_positions(
                    np.zeros((1, 2, 1)), np.array([0.1]), np.array(counts)
                )

    def test_invalid_positions_are_rejected(self) -> None:
        for positions in (
            np.zeros((2, 1)),
            np.zeros((1, 0, 1)),
            np.full((1, 2, 1), np.nan),
        ):
            with self.subTest(shape=positions.shape), self.assertRaises(ValueError):
                differentiate_reference_positions(
                    positions, np.array([0.1]), np.array([2])
                )

    def test_mismatched_interval_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            differentiate_reference_positions(
                np.zeros((1, 2, 1)), np.array([0.1, 0.1]), np.array([2])
            )


if __name__ == "__main__":
    unittest.main()
