"""Check constant knee-offset fitting and diagnostic override isolation."""

import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import BvhReferenceBatch
from tools.check_knee_offset_ablation import (
    fit_constant_offset,
    knee_offset_ablation,
    modify_selected_knees,
)


class ConstantOffsetTests(unittest.TestCase):
    def test_feasible_preferred_offset_is_unchanged(self) -> None:
        raw = np.array([-0.6, -0.2, 0.4])
        self.assertEqual(fit_constant_offset(raw, 0.25, -1.0, 1.0), 0.25)

    def test_offset_below_feasible_interval_is_raised(self) -> None:
        raw = np.array([-0.5, 0.0, 1.0])
        offset = fit_constant_offset(raw, -0.5, 0.0, 2.0)
        self.assertEqual(offset, 0.5)
        np.testing.assert_array_equal(raw + offset, [0.0, 0.5, 1.5])

    def test_offset_above_feasible_interval_is_lowered(self) -> None:
        raw = np.array([-0.5, 0.0, 1.0])
        offset = fit_constant_offset(raw, 2.0, -1.0, 1.5)
        self.assertEqual(offset, 0.5)
        np.testing.assert_array_equal(raw + offset, [0.0, 0.5, 1.5])

    def test_exact_amplitude_fit_has_one_possible_offset(self) -> None:
        raw = np.array([-1.0, -0.3, 1.0])
        for preferred in (-5.0, 0.0, 3.0):
            with self.subTest(preferred=preferred):
                self.assertEqual(fit_constant_offset(raw, preferred, 0.0, 2.0), 1.0)

    def test_oversized_motion_is_rejected_instead_of_clipped(self) -> None:
        with self.assertRaises(ValueError):
            fit_constant_offset(np.array([-1.0, 1.1]), 0.0, -1.0, 1.0)

    def test_constant_and_singleton_motion_accept_equal_bounds(self) -> None:
        for raw in (np.zeros(4), np.array([0.0]), np.full(3, 2.0)):
            with self.subTest(raw=raw):
                offset = fit_constant_offset(raw, -10.0, 0.5, 0.5)
                np.testing.assert_array_equal(raw + offset, np.full(raw.shape, 0.5))

    def test_preserves_all_temporal_differences_without_mutating_input(self) -> None:
        raw = np.array([0.4, -0.5, 0.0, 1.0, 0.4])
        snapshot = raw.copy()
        offset = fit_constant_offset(raw, -1.0, 0.0, 2.0)
        fitted = raw + offset
        np.testing.assert_allclose(np.diff(fitted), np.diff(raw), atol=1e-15)
        np.testing.assert_array_equal(raw, snapshot)
        self.assertEqual(fitted[0], fitted[-1])

    def test_translating_raw_and_preferred_preserves_fitted_curve(self) -> None:
        raw = np.array([-0.5, 0.0, 1.0])
        for preferred in (-3.0, 0.7, 5.0):
            offset = fit_constant_offset(raw, preferred, 0.0, 2.0)
            for translation in (-8.0, 2.0):
                with self.subTest(preferred=preferred, translation=translation):
                    translated_offset = fit_constant_offset(
                        raw + translation, preferred - translation, 0.0, 2.0
                    )
                    np.testing.assert_allclose(
                        raw + translation + translated_offset, raw + offset
                    )

    def test_translating_bounds_and_preferred_translates_curve(self) -> None:
        raw = np.array([-0.5, 0.0, 1.0])
        for preferred in (-3.0, 0.7, 5.0):
            offset = fit_constant_offset(raw, preferred, 0.0, 2.0)
            translated = fit_constant_offset(raw, preferred + 4.0, 4.0, 6.0)
            with self.subTest(preferred=preferred):
                self.assertAlmostEqual(translated, offset + 4.0)

    def test_invalid_raw_shapes_are_rejected(self) -> None:
        for raw in (np.array([]), np.array(0.0), np.zeros((1, 2)), np.zeros((2, 1))):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                fit_constant_offset(raw, 0.0, -1.0, 1.0)

    def test_nonfinite_raw_values_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_constant_offset(np.array([0.0, value]), 0.0, -1.0, 1.0)

    def test_nonreal_raw_types_are_rejected(self) -> None:
        for raw in (
            np.array([0.0 + 0.0j, 1.0 + 0.0j]),
            np.array([False, True]),
            np.array(["0", "1"]),
            np.array([0.0, 1.0], dtype=object),
        ):
            with self.subTest(dtype=raw.dtype), self.assertRaises(ValueError):
                fit_constant_offset(raw, 0.0, -1.0, 1.0)

    def test_invalid_scalar_parameters_are_rejected(self) -> None:
        for index, name in enumerate(("preferred", "lower", "upper")):
            for value in (
                np.nan,
                np.inf,
                -np.inf,
                0.0 + 0.0j,
                False,
                "0",
                np.array([0.0]),
                None,
            ):
                arguments = [0.0, -1.0, 1.0]
                arguments[index] = value
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    fit_constant_offset(np.zeros(2), *arguments)

    def test_inverted_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_constant_offset(np.zeros(2), 0.0, 1.0, -1.0)

    def test_numpy_real_scalar_inputs_are_supported(self) -> None:
        offset = fit_constant_offset(
            np.array([-1, 0, 1]), np.float32(0.5), np.int64(-1), np.float64(2.0)
        )
        self.assertEqual(offset, 0.5)


class SelectedKneeBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SimpleNamespace(
            _actuator_joint_names=("torso", "left_knee_z", "right_knee_z"),
            _actuator_ctrl_lower_limits_np=np.array([-2, -2, -2]),
            _actuator_ctrl_upper_limits_np=np.array([2, 0, 0]),
            _actuator_qpos_lower_limits_np=np.array([-3, -0.7, -1]),
            _actuator_qpos_upper_limits_np=np.array([3, 0.1, 0.1]),
        )
        joints = np.array(
            [
                [
                    [0.1, -0.3, -0.3],
                    [0.2, 0, -0.2],
                    [0.3, 0, -0.1],
                    [0.3, 0, -0.1],
                    [0.3, 0, -0.1],
                ]
            ],
            dtype=np.float32,
        )
        self.batch = BvhReferenceBatch(
            qpos_targets=joints,
            qvel_targets=np.arange(15, dtype=np.float32).reshape(1, 5, 3),
            root_pos_targets=np.zeros((1, 5, 3), dtype=np.float32),
            root_quat_targets=np.tile([1.0, 0, 0, 0], (1, 5, 1)),
            root_vel_targets=np.zeros((1, 5, 3), dtype=np.float32),
            root_angvel_targets=np.zeros((1, 5, 3), dtype=np.float32),
            wrap_deltas=np.zeros((1, 3)),
            frame_times=np.array([0.2], dtype=np.float32),
            frame_counts=np.array([3]),
            loop_modes=np.array([1]),
            weights=np.array([1.0]),
            source_paths=(Path("synthetic.bvh"),),
            source_start_frames=np.array([4]),
            source_end_frames=np.array([7]),
            support_feet=("left",),
            diagnostic_summaries=("synthetic",),
            ik_target_positions=np.zeros((1, 5, 2, 3)),
            ik_target_names=("right_foot", "left_foot"),
        )
        self.mapping = {
            "knees": {
                "left_knee_z": {
                    "raw_hinge_rad": [-0.5, -0.2, 0.1],
                    "bind_offset_rad": 0.2,
                },
                "right_knee_z": {
                    "raw_hinge_rad": [-0.3, -0.2, -0.1],
                    "bind_offset_rad": 0.0,
                },
            }
        }
        self.enterContext(
            patch(
                "tools.check_knee_offset_ablation.inspect_knee_mapping",
                return_value=self.mapping,
            )
        )
        self.enterContext(
            patch(
                "tools.check_knee_offset_ablation._parse_bvh",
                return_value=SimpleNamespace(frame_time=0.1),
            )
        )
        self.classify = self.enterContext(
            patch(
                "tools.check_knee_offset_ablation._resolve_loop_mode",
                return_value=1,
            )
        )

    def test_only_changed_knee_columns_are_replaced_with_correct_padding(self) -> None:
        original_pose = self.batch.qpos_targets.copy()
        original_velocity = self.batch.qvel_targets.copy()
        corrected, record = modify_selected_knees(self.env, self.batch)
        np.testing.assert_allclose(
            corrected.qpos_targets[0, :, 1], [-0.6, -0.3, 0, 0, 0]
        )
        # Source interval .1 is intentional: using batch .2 would halve this.
        np.testing.assert_allclose(corrected.qvel_targets[0, :, 1], [3, 3, 3, 0, 0])
        for index in (0, 2):
            np.testing.assert_array_equal(
                corrected.qpos_targets[:, :, index], original_pose[:, :, index]
            )
            np.testing.assert_array_equal(
                corrected.qvel_targets[:, :, index], original_velocity[:, :, index]
            )
        np.testing.assert_array_equal(self.batch.qpos_targets, original_pose)
        np.testing.assert_array_equal(self.batch.qvel_targets, original_velocity)
        self.assertFalse(
            np.shares_memory(corrected.qpos_targets, self.batch.qpos_targets)
        )
        self.assertFalse(
            np.shares_memory(corrected.qvel_targets, self.batch.qvel_targets)
        )
        self.assertAlmostEqual(
            record["knees"]["left_knee_z"]["fitted_offset_rad"], -0.1
        )
        self.assertEqual(
            record["knees"]["left_knee_z"]["unchanged_bounds_rad"], [-0.7, 0]
        )
        self.assertFalse(record["knees"]["right_knee_z"]["changed"])

    def test_all_other_batch_fields_are_preserved_and_classifier_sees_new_nodes(
        self,
    ) -> None:
        corrected, _ = modify_selected_knees(self.env, self.batch)
        for field in fields(self.batch):
            if field.name not in ("qpos_targets", "qvel_targets"):
                self.assertIs(
                    getattr(corrected, field.name), getattr(self.batch, field.name)
                )
        arguments = self.classify.call_args.kwargs
        np.testing.assert_array_equal(
            arguments["qpos_targets"], corrected.qpos_targets[0, :3]
        )
        np.testing.assert_array_equal(
            arguments["qvel_targets"], corrected.qvel_targets[0, :3]
        )

    def test_changed_loop_classification_rejects_candidate(self) -> None:
        self.classify.return_value = 0
        with self.assertRaisesRegex(ValueError, "loop heuristic"):
            modify_selected_knees(self.env, self.batch)

    def test_infeasible_physical_range_does_not_mutate_input(self) -> None:
        self.env._actuator_qpos_lower_limits_np[1] = -0.3
        snapshot = self.batch.qpos_targets.copy()
        with self.assertRaisesRegex(ValueError, "amplitude cannot fit"):
            modify_selected_knees(self.env, self.batch)
        np.testing.assert_array_equal(self.batch.qpos_targets, snapshot)
        self.classify.assert_not_called()

    def test_invalid_clip_count_and_short_sequence_are_rejected(self) -> None:
        for counts in (np.array([2]), np.array([3, 3])):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                modify_selected_knees(
                    self.env, replace(self.batch, frame_counts=counts)
                )


class KneeOffsetAblationScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SimpleNamespace()
        self.input_batch = object()
        self.filtered_batch = object()
        self.modified_batch = object()
        self.record = {"diagnostic": "knee offset"}
        self.filter_calls: list[tuple[object, object]] = []

    def filter_fixture(self, env: object, references: object) -> object:
        self.filter_calls.append((env, references))
        return self.filtered_batch

    def test_enabled_modifier_receives_selected_batch_and_records_result(self) -> None:
        with (
            patch.object(
                BiomechanicsJoystickEnv, "_filter_reference_clips", self.filter_fixture
            ),
            patch(
                "tools.check_knee_offset_ablation.modify_selected_knees",
                return_value=(self.modified_batch, self.record),
            ) as modifier,
        ):
            original = BiomechanicsJoystickEnv._filter_reference_clips
            with knee_offset_ablation(enabled=True) as records:
                result = BiomechanicsJoystickEnv._filter_reference_clips(
                    self.env, references=self.input_batch
                )
            self.assertIs(result, self.modified_batch)
            self.assertEqual(records, [self.record])
            self.assertIs(BiomechanicsJoystickEnv._filter_reference_clips, original)
        self.assertEqual(self.filter_calls, [(self.env, self.input_batch)])
        modifier.assert_called_once_with(self.env, self.filtered_batch)

    def test_disabled_keeps_original_method_and_batch_identity(self) -> None:
        with (
            patch.object(
                BiomechanicsJoystickEnv, "_filter_reference_clips", self.filter_fixture
            ),
            patch("tools.check_knee_offset_ablation.modify_selected_knees") as modifier,
        ):
            original = BiomechanicsJoystickEnv._filter_reference_clips
            with knee_offset_ablation(enabled=False) as records:
                self.assertIs(BiomechanicsJoystickEnv._filter_reference_clips, original)
                result = BiomechanicsJoystickEnv._filter_reference_clips(
                    self.env, self.input_batch
                )
            self.assertIs(result, self.filtered_batch)
            self.assertEqual(records, [])
            self.assertIs(BiomechanicsJoystickEnv._filter_reference_clips, original)
        modifier.assert_not_called()

    def test_multiple_environments_are_modified_independently(self) -> None:
        second_env = SimpleNamespace()
        second_record = {"diagnostic": "second environment"}
        with (
            patch.object(
                BiomechanicsJoystickEnv, "_filter_reference_clips", self.filter_fixture
            ),
            patch(
                "tools.check_knee_offset_ablation.modify_selected_knees",
                side_effect=[
                    (self.modified_batch, self.record),
                    (self.modified_batch, second_record),
                ],
            ) as modifier,
            knee_offset_ablation(enabled=True) as records,
        ):
            for env in (self.env, second_env):
                BiomechanicsJoystickEnv._filter_reference_clips(env, self.input_batch)
        self.assertEqual(records, [self.record, second_record])
        self.assertIsNot(records[0], records[1])
        self.assertEqual(
            modifier.call_args_list[0].args, (self.env, self.filtered_batch)
        )
        self.assertEqual(
            modifier.call_args_list[1].args, (second_env, self.filtered_batch)
        )

    def test_restores_method_when_modifier_raises(self) -> None:
        with (
            patch.object(
                BiomechanicsJoystickEnv, "_filter_reference_clips", self.filter_fixture
            ),
            patch(
                "tools.check_knee_offset_ablation.modify_selected_knees",
                side_effect=ValueError("infeasible knee trajectory"),
            ),
        ):
            original = BiomechanicsJoystickEnv._filter_reference_clips
            with (
                self.assertRaisesRegex(ValueError, "infeasible knee"),
                knee_offset_ablation(enabled=True) as records,
            ):
                BiomechanicsJoystickEnv._filter_reference_clips(
                    self.env, self.input_batch
                )
            self.assertEqual(records, [])
            self.assertIs(BiomechanicsJoystickEnv._filter_reference_clips, original)

    def test_restores_method_when_original_filter_raises(self) -> None:
        with (
            patch.object(
                BiomechanicsJoystickEnv,
                "_filter_reference_clips",
                side_effect=ValueError("invalid clip selection"),
            ) as original,
            patch("tools.check_knee_offset_ablation.modify_selected_knees") as modifier,
        ):
            with (
                self.assertRaisesRegex(ValueError, "invalid clip"),
                knee_offset_ablation(enabled=True) as records,
            ):
                BiomechanicsJoystickEnv._filter_reference_clips(
                    self.env, self.input_batch
                )
            self.assertEqual(records, [])
            self.assertIs(BiomechanicsJoystickEnv._filter_reference_clips, original)
        modifier.assert_not_called()

    def test_restores_method_when_context_body_raises(self) -> None:
        original = BiomechanicsJoystickEnv._filter_reference_clips
        with (
            self.assertRaisesRegex(RuntimeError, "construction failed"),
            knee_offset_ablation(enabled=True),
        ):
            raise RuntimeError("construction failed")
        self.assertIs(BiomechanicsJoystickEnv._filter_reference_clips, original)


if __name__ == "__main__":
    unittest.main()
