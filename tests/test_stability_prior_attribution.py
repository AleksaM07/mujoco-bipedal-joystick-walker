"""Check pose-group attribution with independent arrays and a native FK fixture."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from tools.check_stability_prior_attribution import (
    inspect_knee_mapping,
    inspect_local_feet,
    nodal_second_difference,
    restore_joint_group,
)


class RestoreJointGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ("hip", "knee", "ankle")
        self.prior = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.ik = self.prior + 100

    def test_selected_columns_come_from_ik_with_name_order_respected(self) -> None:
        prior_before = self.prior.copy()
        ik_before = self.ik.copy()
        result = restore_joint_group(self.prior, self.ik, self.names, ("ankle", "hip"))
        np.testing.assert_array_equal(result[:, [0, 2]], self.ik[:, [0, 2]])
        np.testing.assert_array_equal(result[:, 1], self.prior[:, 1])
        np.testing.assert_array_equal(self.prior, prior_before)
        np.testing.assert_array_equal(self.ik, ik_before)
        self.assertFalse(np.shares_memory(result, self.prior))
        self.assertFalse(np.shares_memory(result, self.ik))

    def test_empty_group_returns_detached_identity(self) -> None:
        result = restore_joint_group(self.prior, self.ik, self.names, ())
        np.testing.assert_array_equal(result, self.prior)
        self.assertFalse(np.shares_memory(result, self.prior))

    def test_full_group_restores_ik_without_aliasing_it(self) -> None:
        result = restore_joint_group(self.prior, self.ik, self.names, self.names)
        np.testing.assert_array_equal(result, self.ik)
        self.assertFalse(np.shares_memory(result, self.ik))

    def test_mixed_dtypes_do_not_truncate_restored_values(self) -> None:
        prior = self.prior.astype(int)
        ik = self.ik.astype(np.float64) + 0.125
        result = restore_joint_group(prior, ik, self.names, ("knee",))
        np.testing.assert_array_equal(result[:, 1], ik[:, 1])
        np.testing.assert_array_equal(result[:, [0, 2]], prior[:, [0, 2]])

    def test_invalid_shapes_and_nonfinite_arrays_are_rejected(self) -> None:
        cases = [
            (self.prior.ravel(), self.ik.ravel()),
            (self.prior[:, :, None], self.ik[:, :, None]),
            (self.prior, self.ik[:3]),
            (self.prior, self.ik[:, :2]),
            (self.prior[:0], self.ik[:0]),
            (np.full_like(self.prior, np.nan), self.ik),
            (self.prior, np.full_like(self.ik, np.inf)),
            (self.prior.astype(complex) + 1j, self.ik),
        ]
        for prior, ik in cases:
            with (
                self.subTest(prior_shape=prior.shape, ik_shape=ik.shape),
                self.assertRaises(ValueError),
            ):
                restore_joint_group(prior, ik, self.names, ("hip",))

    def test_unknown_or_duplicate_names_are_rejected(self) -> None:
        cases = [
            (self.names[:2], ("hip",)),
            (("hip", "hip", "ankle"), ("hip",)),
            (self.names, ("missing",)),
            (self.names, ("hip", "hip")),
        ]
        for names, group in cases:
            with (
                self.subTest(names=names, group=group),
                self.assertRaises(ValueError),
            ):
                restore_joint_group(self.prior, self.ik, names, group)


class NodalSecondDifferenceTests(unittest.TestCase):
    def test_quadratic_samples_have_known_acceleration(self) -> None:
        times = np.arange(5, dtype=np.float32) * 0.5
        values = np.column_stack((3 * times**2 + 2 * times, -(times**2)))
        before = values.copy()
        result = nodal_second_difference(values, 0.5)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(result, np.tile([6.0, -2.0], (3, 1)))
        np.testing.assert_array_equal(values, before)

    def test_integer_samples_are_converted_before_differencing(self) -> None:
        values = np.array([[0], [255], [0]], dtype=np.uint8)
        np.testing.assert_array_equal(nodal_second_difference(values, 0.5), [[-2040.0]])

    def test_invalid_intervals_are_rejected(self) -> None:
        for interval in (0, -0.1, np.nan, np.inf, -np.inf):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                nodal_second_difference(np.zeros((3, 2)), interval)

    def test_invalid_values_are_rejected(self) -> None:
        values_cases = [
            np.zeros((0, 2)),
            np.zeros((2, 2)),
            np.zeros((3, 0)),
            np.zeros(3),
            np.array(1.0),
            np.full((3, 2), np.nan),
            np.full((3, 2), np.inf),
            np.ones((3, 2), dtype=complex) * (1 + 1j),
        ]
        for values in values_cases:
            with (
                self.subTest(shape=values.shape, dtype=values.dtype),
                self.assertRaises(ValueError),
            ):
                nodal_second_difference(values, 0.1)


class InspectLocalFeetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body>
              <freejoint/>
              <geom type="sphere" size=".01" mass="1"/>
              <body pos="0 .2 .1">
                <joint name="foot_x" type="slide" axis="1 0 0"/>
                <geom name="foot" type="capsule" size=".04 .1" mass="1"/>
              </body>
            </body></worldbody></mujoco>
        """)
        self.env = SimpleNamespace(
            _mj_model=self.model,
            _init_q_np=self.model.qpos0.copy(),
            _actuator_qpos_indices_np=np.array([7]),
            _foot_geom_ids_np=np.array([self.model.geom("foot").id]),
        )
        self.joints = np.array([[0], [0.01], [0.03], [0.13], [0.23]])
        self.roots = np.zeros((5, 3))
        self.quaternions = np.tile([1.0, 0, 0, 0], (5, 1))

    def test_geometry_and_peak_use_center_frame_indices(self) -> None:
        result = inspect_local_feet(
            self.env, self.joints, self.roots, self.quaternions, 0.1
        )["foot"]
        expected_centers = np.column_stack(
            (self.joints[:, 0], np.full(5, 0.2), np.full(5, 0.1))
        )
        np.testing.assert_allclose(result["root_relative_xyz_m"], expected_centers)
        np.testing.assert_allclose(result["capsule_axis_z"], np.ones(5))
        np.testing.assert_allclose(
            result["xy_second_difference_m_s2"], [[1, 0], [8, 0], [0, 0]], atol=1e-12
        )
        self.assertEqual(result["worst_center_frame"], 2)
        summary = result["xy_second_difference_norm_summary_m_s2"]
        self.assertAlmostEqual(summary["min"], 0)
        self.assertAlmostEqual(summary["median"], 1)
        self.assertAlmostEqual(summary["max"], 8)

    def test_framewise_root_translation_does_not_change_local_motion(self) -> None:
        shifted_roots = np.array(
            [[0, 0, 1], [2, -1, 3], [-3, 5, 2], [0.5, 0.7, 6], [7, 1, 0]]
        )
        original = inspect_local_feet(
            self.env, self.joints, self.roots, self.quaternions, 0.1
        )["foot"]
        translated = inspect_local_feet(
            self.env, self.joints, shifted_roots, self.quaternions, 0.1
        )["foot"]
        for key in (
            "root_relative_xyz_m",
            "capsule_axis_z",
            "xy_second_difference_m_s2",
        ):
            np.testing.assert_allclose(translated[key], original[key], atol=1e-12)
        self.assertEqual(translated["worst_center_frame"], 2)

    def test_root_pitch_rotates_foot_center_and_capsule_axis(self) -> None:
        angles = np.linspace(0, np.pi / 2, 5)
        quaternions = np.zeros((5, 4))
        quaternions[:, 0] = np.cos(angles / 2)
        quaternions[:, 2] = np.sin(angles / 2)
        result = inspect_local_feet(
            self.env, self.joints, self.roots, quaternions, 0.1
        )["foot"]
        expected = np.column_stack(
            (
                self.joints[:, 0] * np.cos(angles) + 0.1 * np.sin(angles),
                np.full(5, 0.2),
                -self.joints[:, 0] * np.sin(angles) + 0.1 * np.cos(angles),
            )
        )
        np.testing.assert_allclose(result["root_relative_xyz_m"], expected, atol=1e-12)
        np.testing.assert_allclose(result["capsule_axis_z"], np.cos(angles), atol=1e-12)

    def test_inspection_preserves_input_arrays_and_model_initial_state(self) -> None:
        arrays = [
            self.joints,
            self.roots,
            self.quaternions,
            self.env._init_q_np,
            self.model.qpos0,
        ]
        snapshots = [array.copy() for array in arrays]
        inspect_local_feet(self.env, self.joints, self.roots, self.quaternions, 0.1)
        for array, snapshot in zip(arrays, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)

    def test_invalid_root_shapes_and_quaternions_are_rejected(self) -> None:
        cases = [
            (self.roots[:3], self.quaternions),
            (self.roots[:, :2], self.quaternions),
            (self.roots, self.quaternions[:, :3]),
            (self.roots, np.zeros_like(self.quaternions)),
            (self.roots, self.quaternions * 2),
            (np.full_like(self.roots, np.nan), self.quaternions),
            (self.roots, np.full_like(self.quaternions, np.inf)),
            (self.roots.astype(complex), self.quaternions),
            (self.roots, self.quaternions.astype(complex)),
        ]
        for roots, quaternions in cases:
            with (
                self.subTest(
                    roots_shape=roots.shape, quaternion_shape=quaternions.shape
                ),
                self.assertRaises(ValueError),
            ):
                inspect_local_feet(self.env, self.joints, roots, quaternions, 0.1)

    def test_non_capsule_foot_geometry_is_rejected(self) -> None:
        foot = self.env._foot_geom_ids_np[0]
        self.model.geom_type[foot] = mujoco.mjtGeom.mjGEOM_SPHERE
        with self.assertRaisesRegex(ValueError, "capsule"):
            inspect_local_feet(self.env, self.joints, self.roots, self.quaternions, 0.1)


class InspectKneeMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path("synthetic.bvh")
        self.metadata = {"source_start_frames": [79], "source_end_frames": [84]}
        self.env = SimpleNamespace(
            _actuator_joint_names=("ankle", "right_knee_z", "left_knee_z"),
            _default_ctrl=np.array([0.0, -0.125, -0.25]),
            _actuator_ctrl_lower_limits_np=np.array([-2.0, -0.75, -1.0]),
            _actuator_ctrl_upper_limits_np=np.array([2.0, 0.0, 0.0]),
        )
        self.left_flexion = np.array([0.25, 0.5, 0, -0.25, 1.5])
        self.right_flexion = np.array([0.125, 0.25, 0.5, -0.25, 1.5])
        self.seed = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.125, -0.25],
                [0.0, -0.375, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -0.75, -1.0],
            ],
            dtype=np.float32,
        )
        parsed = Mock()
        self.sliced = parsed.slice.return_value
        self.sliced.frames = 5
        self.parse = self.enterContext(
            patch(
                "tools.check_stability_prior_attribution._parse_bvh",
                return_value=parsed,
            )
        )
        self.slice = parsed.slice
        self.hinges = self.enterContext(
            patch(
                "tools.check_stability_prior_attribution._joint_hinge_angles",
                side_effect=[
                    np.column_stack((np.ones(5), self.left_flexion, np.ones(5))),
                    np.column_stack((np.ones(5), self.right_flexion, np.ones(5))),
                ],
            )
        )
        self.bind = self.enterContext(
            patch(
                "tools.check_stability_prior_attribution._standing_like_frame_index",
                return_value=1,
            )
        )

    def test_offset_clipping_and_plateau_match_captured_seed(self) -> None:
        result = inspect_knee_mapping(self.env, self.source, self.metadata, self.seed)
        self.assertEqual(result["bind_frame"], 1)
        left = result["knees"]["left_knee_z"]
        right = result["knees"]["right_knee_z"]
        np.testing.assert_array_equal(left["raw_hinge_rad"], -self.left_flexion)
        self.assertEqual(left["bind_offset_rad"], 0.25)
        self.assertEqual(right["bind_offset_rad"], 0.125)
        np.testing.assert_array_equal(
            left["before_clip_rad"], [0.0, -0.25, 0.25, 0.5, -1.25]
        )
        np.testing.assert_array_equal(
            right["before_clip_rad"], [0.0, -0.125, -0.375, 0.375, -1.375]
        )
        self.assertEqual(left["upper_clipped_frames"], [2, 3])
        self.assertEqual(right["upper_clipped_frames"], [3])
        for side, index, lower in ((left, 2, -1.0), (right, 1, -0.75)):
            np.testing.assert_array_equal(side["after_clip_rad"], self.seed[:, index])
            self.assertEqual(side["lower_clipped_frames"], [4])
            self.assertEqual(side["lower_limit_rad"], lower)
            self.assertEqual(side["upper_limit_rad"], 0.0)
            self.assertEqual(side["captured_seed_difference_max_rad"], 0.0)

    def test_source_slice_and_bind_use_selected_clip_not_full_source(self) -> None:
        inspect_knee_mapping(self.env, self.source, self.metadata, self.seed)
        self.parse.assert_called_once_with(self.source)
        self.slice.assert_called_once()
        segment = self.slice.call_args.args[0]
        self.assertEqual((segment.start_frame, segment.end_frame), (79, 84))
        self.assertEqual(self.hinges.call_count, 2)
        for call, side in zip(
            self.hinges.call_args_list, ("Left", "Right"), strict=True
        ):
            self.assertIs(call.args[0], self.sliced)
            self.assertEqual(call.args[1], (f"{side}Leg", f"{side}Knee"))
            self.assertEqual(call.kwargs, {"allow_missing": True})
        self.bind.assert_called_once()
        self.assertIs(self.bind.call_args.args[0], self.sliced)
        np.testing.assert_array_equal(
            self.bind.call_args.kwargs["left_knee_flex"], self.left_flexion
        )
        np.testing.assert_array_equal(
            self.bind.call_args.kwargs["right_knee_flex"], self.right_flexion
        )

    def test_mismatching_captured_seed_is_rejected(self) -> None:
        seed = self.seed.copy()
        seed[2, 2] = -0.01
        with self.assertRaisesRegex(ValueError, "left_knee_z.*captured loader seed"):
            inspect_knee_mapping(self.env, self.source, self.metadata, seed)

    def test_invalid_seed_values_and_shapes_are_rejected(self) -> None:
        cases = [
            self.seed[:3],
            self.seed[:, :2],
            self.seed.ravel(),
            self.seed.astype(complex),
            np.full_like(self.seed, np.nan),
            np.full_like(self.seed, np.inf),
        ]
        for seed in cases:
            with (
                self.subTest(shape=seed.shape, dtype=seed.dtype),
                self.assertRaises(ValueError),
            ):
                inspect_knee_mapping(self.env, self.source, self.metadata, seed)
        self.hinges.assert_not_called()

    def test_reconstruction_leaves_seed_and_environment_values_unchanged(self) -> None:
        arrays = [
            self.seed,
            self.env._default_ctrl,
            self.env._actuator_ctrl_lower_limits_np,
            self.env._actuator_ctrl_upper_limits_np,
        ]
        snapshots = [array.copy() for array in arrays]
        inspect_knee_mapping(self.env, self.source, self.metadata, self.seed)
        for array, snapshot in zip(arrays, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)


if __name__ == "__main__":
    unittest.main()
