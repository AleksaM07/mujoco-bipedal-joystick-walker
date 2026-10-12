"""Curve-level recovery tests, independent of the training environment."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jp
import numpy as np

from reference_interpolation import (
    interpolate_quaternion_segment,
    interpolate_segment,
    prepare_reference_curves,
)


def quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply raw quaternion components without normalizing derivatives."""
    return np.concatenate(
        (
            [left[0] * right[0] - np.dot(left[1:], right[1:])],
            left[0] * right[1:] + right[0] * left[1:] + np.cross(left[1:], right[1:]),
        )
    )


def local_interval_velocity(
    left: np.ndarray, right: np.ndarray, dt: float
) -> np.ndarray:
    """Independent finite-interval logarithm for orientation checks."""
    conjugate = left * [1.0, -1.0, -1.0, -1.0]
    delta = quaternion_product(conjugate, right)
    if delta[0] < 0.0:
        delta *= -1.0
    length = np.linalg.norm(delta[1:])
    if length < 1e-12:
        return np.zeros(3)
    return delta[1:] / length * (2.0 * np.arctan2(length, delta[0])) / dt


def curve_inputs(periodic: bool = True, count: int = 7) -> dict:
    """Construct small nontrivial periodic poses with a translating root."""
    phase = np.linspace(0.0, 1.0, count)
    joints = np.stack(
        (0.2 * np.sin(2.0 * np.pi * phase), 0.1 * np.cos(2.0 * np.pi * phase)), axis=-1
    )
    roots = np.stack(
        (
            0.5 * phase,
            0.02 * np.sin(2.0 * np.pi * phase),
            0.9 + 0.01 * np.sin(2.0 * np.pi * phase),
        ),
        axis=-1,
    )
    base = np.array([0.5, 0.5, 0.5, 0.5])
    quats = []
    for angle in 0.3 * np.sin(2.0 * np.pi * phase):
        quats.append(
            quaternion_product(
                base, np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
            )
        )
    return {
        "joint_positions": joints[None, ...],
        "root_positions": roots[None, ...],
        "root_quaternions": np.array(quats)[None, ...],
        "frame_times": np.array([0.04]),
        "frame_counts": np.array([count]),
        "loop_modes": np.array([int(periodic)]),
        "wrap_deltas": np.array([[0.5 if count > 1 else 0.0, 0.0, 0.0]]),
    }


class ReferenceInterpolationTests(unittest.TestCase):
    def test_even_tiny_vertical_wrap_drift_is_rejected(self) -> None:
        inputs = curve_inputs()
        inputs["wrap_deltas"][0, 2] = 1e-8
        with self.assertRaisesRegex(ValueError, "zero net Z"):
            prepare_reference_curves(**inputs)

    def test_bounded_projection_preserves_limits_closure_and_reports_edit(self) -> None:
        inputs = curve_inputs()
        inputs["joint_positions"][0, -1, 0] += 0.08
        lower = np.min(inputs["joint_positions"], axis=(0, 1))
        upper = np.max(inputs["joint_positions"], axis=(0, 1))
        curves = prepare_reference_curves(
            **inputs,
            loop_closure="bounded",
            joint_lower_limits=lower,
            joint_upper_limits=upper,
        )
        diagnostic = curves["closure_diagnostics"][0]
        self.assertGreater(diagnostic["joint_limit_projected_coordinates"], 0)
        self.assertGreater(diagnostic["max_joint_limit_projection_rad"], 0)
        self.assertLessEqual(diagnostic["max_joint_correction_rad"], 0.04 + 1e-7)
        values = curves["joint_positions"][0]
        tangents = curves["joint_tangents"][0]
        np.testing.assert_array_equal(values[0], values[-1])
        np.testing.assert_array_equal(tangents[0], tangents[-1])
        for index in range(len(values) - 1):
            samples, _ = jax.vmap(
                lambda phase: interpolate_segment(
                    values[index],
                    values[index + 1],
                    tangents[index],
                    tangents[index + 1],
                    phase,
                    0.04,
                )
            )(jp.linspace(0, 1, 33))
            self.assertTrue(np.all(samples >= lower - 1e-6))
            self.assertTrue(np.all(samples <= upper + 1e-6))

    def test_invalid_source_limits_are_not_silently_repaired(self) -> None:
        with self.assertRaisesRegex(ValueError, "source joint nodes"):
            prepare_reference_curves(
                **curve_inputs(),
                loop_closure="bounded",
                joint_lower_limits=np.zeros(2),
                joint_upper_limits=np.ones(2),
            )

    def test_cubic_derivative_matches_jvp_and_finite_difference(self) -> None:
        curves = prepare_reference_curves(**curve_inputs())
        interval = 0.04
        for index in range(6):
            values = [
                jp.array(curves[key][0, index + offset])
                for key, offset in (
                    ("joint_positions", 0),
                    ("joint_positions", 1),
                    ("joint_tangents", 0),
                    ("joint_tangents", 1),
                )
            ]
            for phase in (0.17, 0.5, 0.83):

                def evaluate(alpha):
                    return interpolate_segment(*values, alpha, interval)[0]

                _, velocity = interpolate_segment(*values, phase, interval)
                _, jvp = jax.jvp(
                    evaluate, (jp.array(phase),), (jp.array(1.0 / interval),)
                )
                np.testing.assert_allclose(velocity, jvp, atol=2e-6)
                epsilon = 1e-3
                finite_difference = (
                    evaluate(phase + epsilon) - evaluate(phase - epsilon)
                ) / (2 * epsilon * interval)
                np.testing.assert_allclose(velocity, finite_difference, atol=4e-4)

    def test_normalized_quaternion_derivative_is_local_and_exact(self) -> None:
        curves = prepare_reference_curves(**curve_inputs())
        interval = 0.04
        for index in range(6):
            values = [
                jp.array(curves[key][0, index + offset])
                for key, offset in (
                    ("root_quaternions", 0),
                    ("root_quaternions", 1),
                    ("quaternion_tangents", 0),
                    ("quaternion_tangents", 1),
                )
            ]
            for phase in (0.13, 0.5, 0.87):
                quat, velocity = interpolate_quaternion_segment(
                    *values, phase, interval
                )
                self.assertAlmostEqual(float(jp.linalg.norm(quat)), 1.0, places=6)
                epsilon = 1e-3
                before = np.asarray(
                    interpolate_quaternion_segment(*values, phase - epsilon, interval)[
                        0
                    ],
                    dtype=np.float64,
                )
                after = np.asarray(
                    interpolate_quaternion_segment(*values, phase + epsilon, interval)[
                        0
                    ],
                    dtype=np.float64,
                )
                finite_velocity = local_interval_velocity(
                    before, after, 2.0 * epsilon * interval
                )
                np.testing.assert_allclose(velocity, finite_velocity, atol=2e-3)
                # Root base orientation rotates local X into world Y: using a
                # world-frame quaternion difference would fail this assertion.
                np.testing.assert_allclose(np.asarray(velocity)[1:], 0.0, atol=3e-6)

    def test_periodic_endpoint_values_and_tangents_are_shared(self) -> None:
        inputs = curve_inputs()
        curves = prepare_reference_curves(**inputs)
        for key in (
            "joint_positions",
            "joint_tangents",
            "root_quaternions",
            "quaternion_tangents",
            "root_tangents",
        ):
            np.testing.assert_array_equal(curves[key][0, 0], curves[key][0, -1])
        np.testing.assert_allclose(
            curves["root_positions"][0, -1] - curves["root_positions"][0, 0],
            inputs["wrap_deltas"][0],
            atol=2e-7,
        )
        for position_key, tangent_key, interpolate in (
            ("joint_positions", "joint_tangents", interpolate_segment),
            ("root_positions", "root_tangents", interpolate_segment),
            ("root_quaternions", "quaternion_tangents", interpolate_quaternion_segment),
        ):
            positions = jp.array(curves[position_key][0])
            tangents = jp.array(curves[tangent_key][0])
            left = interpolate(
                positions[-2], positions[-1], tangents[-2], tangents[-1], 1.0, 0.04
            )[1]
            right = interpolate(
                positions[0], positions[1], tangents[0], tangents[1], 0.0, 0.04
            )[1]
            np.testing.assert_allclose(left, right, atol=2e-6)

    def test_monotone_joint_segments_stay_inside_node_limits(self) -> None:
        curves = prepare_reference_curves(**curve_inputs())
        positions = jp.array(curves["joint_positions"][0])
        tangents = jp.array(curves["joint_tangents"][0])
        for index in range(len(positions) - 1):
            phases = jp.linspace(0.0, 1.0, 101)
            samples = jax.vmap(
                lambda phase: interpolate_segment(
                    positions[index],
                    positions[index + 1],
                    tangents[index],
                    tangents[index + 1],
                    phase,
                    0.04,
                )[0]
            )(phases)
            lower = np.minimum(positions[index], positions[index + 1])
            upper = np.maximum(positions[index], positions[index + 1])
            self.assertTrue(np.all(samples >= lower - 2e-7))
            self.assertTrue(np.all(samples <= upper + 2e-7))

    def test_strict_rejects_real_seam_but_bounded_records_correction(self) -> None:
        inputs = curve_inputs()
        inputs["joint_positions"][0, -1, 0] += 0.08
        inputs["root_positions"][0, -1, 2] += 0.008
        inputs["root_quaternions"][0, -1] = quaternion_product(
            inputs["root_quaternions"][0, -1],
            np.array([np.cos(0.025), 0.0, np.sin(0.025), 0.0]),
        )
        originals = {key: value.copy() for key, value in inputs.items()}
        with self.assertRaisesRegex(ValueError, "strict loop closure rejected"):
            prepare_reference_curves(**inputs)
        curves = prepare_reference_curves(**inputs, loop_closure="bounded")
        diagnostic = curves["closure_diagnostics"][0]
        self.assertTrue(diagnostic["closure_applied"])
        self.assertAlmostEqual(diagnostic["original_joint_gap_rad"], 0.08)
        self.assertAlmostEqual(diagnostic["original_root_z_gap_m"], 0.008)
        self.assertAlmostEqual(diagnostic["original_orientation_gap_rad"], 0.05)
        self.assertAlmostEqual(diagnostic["max_joint_correction_rad"], 0.04)
        self.assertAlmostEqual(diagnostic["max_root_correction_m"], 0.004)
        self.assertAlmostEqual(
            diagnostic["max_orientation_correction_rad"], 0.025, places=6
        )
        for key in inputs:
            np.testing.assert_array_equal(inputs[key], originals[key])
        np.testing.assert_array_equal(
            curves["joint_positions"][0, 0], curves["joint_positions"][0, -1]
        )
        np.testing.assert_array_equal(
            curves["root_quaternions"][0, 0], curves["root_quaternions"][0, -1]
        )

    def test_bounded_repair_rejects_excessive_edit(self) -> None:
        for key, coordinate, gap in (
            ("joint_positions", 0, 0.21),
            ("root_positions", 2, 0.021),
        ):
            with self.subTest(key=key):
                inputs = curve_inputs()
                inputs[key][0, -1, coordinate] += gap
                with self.assertRaisesRegex(
                    ValueError, "bounded loop closure rejected"
                ):
                    prepare_reference_curves(**inputs, loop_closure="bounded")
        inputs = curve_inputs()
        inputs["root_quaternions"][0, -1] = quaternion_product(
            inputs["root_quaternions"][0, -1],
            np.array([np.cos(0.11), 0.0, np.sin(0.11), 0.0]),
        )
        with self.assertRaisesRegex(ValueError, "bounded loop closure rejected"):
            prepare_reference_curves(**inputs, loop_closure="bounded")

    def test_clamped_endpoints_have_zero_derivatives(self) -> None:
        inputs = curve_inputs(periodic=False)
        inputs["joint_positions"][0, -1] += 0.4
        curves = prepare_reference_curves(**inputs)
        for key in ("joint_tangents", "root_tangents", "quaternion_tangents"):
            np.testing.assert_array_equal(curves[key][0, [0, -1]], 0.0)
        np.testing.assert_array_equal(
            curves["joint_positions"], inputs["joint_positions"].astype(np.float32)
        )

    def test_single_frame_is_stationary(self) -> None:
        curves = prepare_reference_curves(**curve_inputs(count=1))
        for key in ("joint_tangents", "root_tangents", "quaternion_tangents"):
            np.testing.assert_array_equal(curves[key], 0.0)

    def test_padding_is_preserved_and_ignored(self) -> None:
        inputs = curve_inputs()
        for key in ("joint_positions", "root_positions", "root_quaternions"):
            inputs[key] = np.concatenate(
                (inputs[key], np.full_like(inputs[key][:, :2], np.nan)), axis=1
            )
        curves = prepare_reference_curves(**inputs)
        for key in ("joint_positions", "root_positions", "root_quaternions"):
            self.assertTrue(np.all(np.isnan(curves[key][0, 7:])))
        for key in ("joint_tangents", "root_tangents", "quaternion_tangents"):
            np.testing.assert_array_equal(curves[key][0, 7:], 0.0)

    def test_arbitrary_source_quaternion_signs_do_not_change_curve(self) -> None:
        inputs = curve_inputs()
        expected = prepare_reference_curves(**inputs)
        inputs["root_quaternions"][0, [1, 2, 5, 6]] *= -1.0
        actual = prepare_reference_curves(**inputs)
        np.testing.assert_array_equal(
            actual["root_quaternions"], expected["root_quaternions"]
        )
        np.testing.assert_array_equal(
            actual["quaternion_tangents"], expected["quaternion_tangents"]
        )

    def test_whole_spin_antipodal_lift_is_explicitly_rejected(self) -> None:
        inputs = curve_inputs(count=9)
        angles = np.linspace(0.0, 2.0 * np.pi, 9)
        inputs["root_quaternions"][0] = np.stack(
            (np.cos(angles / 2), np.zeros(9), np.zeros(9), np.sin(angles / 2)), axis=-1
        )
        with self.assertRaisesRegex(ValueError, "antipodal quaternion seam"):
            prepare_reference_curves(**inputs, loop_closure="bounded")

    def test_playback_speed_changes_all_tangents_once(self) -> None:
        inputs = curve_inputs()
        normal = prepare_reference_curves(**inputs)
        inputs["frame_times"] *= 4.0
        slow = prepare_reference_curves(**inputs)
        for key in ("joint_positions", "root_positions", "root_quaternions"):
            np.testing.assert_array_equal(slow[key], normal[key])
        for key in ("joint_tangents", "root_tangents", "quaternion_tangents"):
            np.testing.assert_allclose(slow[key], normal[key] * 0.25, atol=2e-7)

    def test_inconsistent_wrap_translation_is_rejected(self) -> None:
        for delta in ([0.5, 0.0, 0.001], [0.6, 0.0, 0.0]):
            inputs = curve_inputs()
            inputs["wrap_deltas"][0] = delta
            with self.assertRaises(ValueError):
                prepare_reference_curves(**inputs, loop_closure="bounded")

    def test_invalid_shapes_values_and_modes_are_rejected(self) -> None:
        replacements = (
            ("joint_positions", np.zeros((1, 0, 2))),
            ("root_positions", np.zeros((1, 7, 2))),
            ("root_quaternions", np.zeros((1, 7, 4))),
            ("frame_counts", np.array([0])),
            ("frame_counts", np.array([7.5])),
            ("frame_counts", np.array([8])),
            ("frame_times", np.array([0.0])),
            ("frame_times", np.array([np.inf])),
            ("loop_modes", np.array([2])),
            ("wrap_deltas", np.array([[np.nan, 0.0, 0.0]])),
        )
        for name, value in replacements:
            with self.subTest(name=name, shape=value.shape):
                inputs = curve_inputs()
                inputs[name] = value
                with self.assertRaises(ValueError):
                    prepare_reference_curves(**inputs)
        with self.assertRaises(ValueError):
            prepare_reference_curves(**curve_inputs(), loop_closure="automatic")

    def test_jit_and_vmap_return_finite_values(self) -> None:
        curves = prepare_reference_curves(**curve_inputs())
        quats = jp.array(curves["root_quaternions"][0])
        tangents = jp.array(curves["quaternion_tangents"][0])
        evaluate = jax.jit(
            jax.vmap(
                lambda phase: interpolate_quaternion_segment(
                    quats[0], quats[1], tangents[0], tangents[1], phase, 0.04
                )
            )
        )
        positions, velocities = evaluate(jp.linspace(0.0, 1.0, 13))
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(velocities)))


if __name__ == "__main__":
    unittest.main()
