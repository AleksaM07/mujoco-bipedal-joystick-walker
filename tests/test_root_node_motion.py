"""Distinguish smooth source translation from a nodal root-lock jump."""

import unittest

import numpy as np

from tools.check_retarget_stages import inspect_root_node_motion


class RootNodeMotionTests(unittest.TestCase):
    def test_constant_xy_velocity_has_zero_second_difference(self) -> None:
        positions = np.column_stack((np.arange(6) * 0.5, np.zeros(6), np.ones(6)))
        report = inspect_root_node_motion(positions, 0.25)
        self.assertEqual(report["xy_nodal_speed_m_s"]["max"], 2.0)
        self.assertEqual(report["xy_second_difference_m_s2"]["max"], 0.0)

    def test_jump_is_localized_without_mutating_source(self) -> None:
        positions = np.zeros((6, 3))
        positions[3:, 0] = 0.1
        original = positions.copy()
        report = inspect_root_node_motion(positions, 0.01)
        self.assertAlmostEqual(report["xy_second_difference_m_s2"]["max"], 1000.0)
        self.assertEqual(report["worst_center_frame"], 2)
        self.assertEqual(report["worst_window_start_frame"], 0)
        np.testing.assert_array_equal(positions, original)

    def test_invalid_inputs_raise(self) -> None:
        for positions, interval in (
            (np.zeros((2, 3)), 0.1),
            (np.zeros((3, 2)), 0.1),
            (np.zeros((3, 3)), 0.0),
            (np.zeros((3, 3)), float("nan")),
            (np.full((3, 3), float("inf")), 0.1),
        ):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                inspect_root_node_motion(positions, interval)


if __name__ == "__main__":
    unittest.main()
