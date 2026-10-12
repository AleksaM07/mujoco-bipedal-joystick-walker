"""Regression coverage for explicit loader-clip selection before preprocessing."""

from __future__ import annotations

import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import BvhReferenceBatch


def make_reference_batch() -> BvhReferenceBatch:
    """Return distinct padded clips, with the middle clip intentionally short."""
    joints = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)
    roots = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
    return BvhReferenceBatch(
        qpos_targets=joints,
        qvel_targets=joints + 10,
        root_pos_targets=roots,
        root_quat_targets=np.tile([1.0, 0.0, 0.0, 0.0], (3, 5, 1)),
        root_vel_targets=roots + 10,
        root_angvel_targets=roots + 20,
        wrap_deltas=np.arange(9, dtype=np.float32).reshape(3, 3),
        frame_times=np.array([0.25, 0.1, 0.3], dtype=np.float32),
        frame_counts=np.array([5, 3, 5], dtype=np.int32),
        loop_modes=np.array([1, 0, 1], dtype=np.int32),
        weights=np.array([0.2, 0.3, 0.5], dtype=np.float32),
        source_paths=tuple(Path(f"source-{index}.bvh") for index in range(3)),
        source_start_frames=np.array([11, 25, 40], dtype=np.int32),
        source_end_frames=np.array([16, 28, 45], dtype=np.int32),
        support_feet=("left_foot", "right_foot", "left_foot"),
        diagnostic_summaries=("clip zero", "clip one", "clip two"),
        ik_target_names=("metatarsal_midpoint_right", "metatarsal_midpoint_left"),
        ik_target_positions=np.arange(3 * 5 * 2 * 3, dtype=np.float32).reshape(
            3, 5, 2, 3
        ),
    )


def make_selection_environment(
    selection: list[int] | None = None, minimum: float = 0.8
) -> BiomechanicsJoystickEnv:
    """Build only the configuration needed by the host-side batch selector."""
    env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
    env._config = {
        "reference_clip_ids": selection,
        "reference_min_motion_length": minimum,
        "reference_forced_clip_id": 0,
    }
    return env


class ReferenceClipSelectionTests(unittest.TestCase):
    """Keep runtime order, source provenance, and metadata aligned."""

    def setUp(self) -> None:
        self.batch = make_reference_batch()

    def test_explicit_ids_reorder_every_clip_aligned_field(self) -> None:
        env = make_selection_environment([2, 0])
        result = env._filter_reference_clips(self.batch)
        indices = [2, 0]
        for field in fields(self.batch):
            if field.name in {"weights", "ik_target_names"}:
                continue
            source = getattr(self.batch, field.name)
            selected = getattr(result, field.name)
            with self.subTest(field=field.name):
                if isinstance(source, np.ndarray):
                    np.testing.assert_array_equal(selected, source[indices])
                else:
                    self.assertEqual(
                        selected, tuple(source[index] for index in indices)
                    )
        np.testing.assert_allclose(result.weights, [5 / 7, 2 / 7])
        np.testing.assert_array_equal(env._bvh_reference_source_clip_ids_np, indices)
        self.assertEqual(result.ik_target_names, self.batch.ik_target_names)
        self.assertEqual(env._config["reference_forced_clip_id"], 0)

    def test_selection_ids_are_before_minimum_filter_not_filtered_indices(self) -> None:
        env = make_selection_environment([2])
        result = env._filter_reference_clips(self.batch)
        np.testing.assert_array_equal(result.qpos_targets, self.batch.qpos_targets[[2]])
        np.testing.assert_array_equal(env._bvh_reference_source_clip_ids_np, [2])

    def test_explicit_short_clip_fails_instead_of_silently_dropping_it(self) -> None:
        for selection in ([1], [0, 1], [1, 2]):
            with self.subTest(selection=selection):
                env = make_selection_environment(selection)
                with self.assertRaisesRegex(
                    ValueError, "shorter.*reference_min_motion_length"
                ):
                    env._filter_reference_clips(self.batch)

    def test_explicit_short_clip_can_be_selected_if_duration_filter_is_disabled(
        self,
    ) -> None:
        env = make_selection_environment([1], minimum=0.0)
        result = env._filter_reference_clips(self.batch)
        np.testing.assert_array_equal(result.frame_counts, [3])
        np.testing.assert_array_equal(result.weights, [1.0])

    def test_invalid_selections_raise_clear_errors(self) -> None:
        for selection in (
            [],
            [0, 0],
            [-1],
            [3],
            [1.0],
            [True],
            [np.bool_(False)],
            "1",
            1,
        ):
            with self.subTest(selection=selection):
                env = make_selection_environment()
                env._config["reference_clip_ids"] = selection
                with self.assertRaisesRegex(ValueError, "reference_clip_ids"):
                    env._filter_reference_clips(self.batch)

    def test_numpy_integer_ids_are_supported_without_coercing_floats(self) -> None:
        env = make_selection_environment([np.int64(2)])
        result = env._filter_reference_clips(self.batch)
        np.testing.assert_array_equal(result.qpos_targets, self.batch.qpos_targets[[2]])

    def test_empty_optional_metadata_and_missing_ik_positions_stay_empty(self) -> None:
        batch = replace(self.batch, diagnostic_summaries=(), ik_target_positions=None)
        result = make_selection_environment([2, 0])._filter_reference_clips(batch)
        self.assertEqual(result.diagnostic_summaries, ())
        self.assertIsNone(result.ik_target_positions)

    def test_zero_weights_become_uniform_for_selected_clips(self) -> None:
        batch = replace(self.batch, weights=np.zeros(3, dtype=np.float32))
        result = make_selection_environment([2, 0])._filter_reference_clips(batch)
        np.testing.assert_array_equal(result.weights, [0.5, 0.5])

    def test_selection_does_not_modify_source_arrays_or_config(self) -> None:
        selected_ids = [2, 0]
        env = make_selection_environment(selected_ids)
        original = {
            field.name: getattr(self.batch, field.name).copy()
            for field in fields(self.batch)
            if isinstance(getattr(self.batch, field.name), np.ndarray)
        }
        result = env._filter_reference_clips(self.batch)
        result.qpos_targets.fill(-1)
        result.weights.fill(-1)
        for name, expected in original.items():
            np.testing.assert_array_equal(getattr(self.batch, name), expected)
        self.assertEqual(selected_ids, [2, 0])

    def test_default_filter_keeps_historical_behavior_and_original_ids(self) -> None:
        env = make_selection_environment()
        result = env._filter_reference_clips(self.batch)
        np.testing.assert_array_equal(
            result.qpos_targets, self.batch.qpos_targets[[0, 2]]
        )
        np.testing.assert_allclose(result.weights, [2 / 7, 5 / 7])
        np.testing.assert_array_equal(env._bvh_reference_source_clip_ids_np, [0, 2])

    def test_default_all_short_fallback_returns_original_object(self) -> None:
        env = make_selection_environment(minimum=100.0)
        result = env._filter_reference_clips(self.batch)
        self.assertIs(result, self.batch)
        np.testing.assert_array_equal(env._bvh_reference_source_clip_ids_np, [0, 1, 2])

    def test_default_disabled_filter_returns_original_object(self) -> None:
        for minimum in (0.0, -1.0):
            with self.subTest(minimum=minimum):
                env = make_selection_environment(minimum=minimum)
                result = env._filter_reference_clips(self.batch)
                self.assertIs(result, self.batch)
                np.testing.assert_array_equal(
                    env._bvh_reference_source_clip_ids_np, [0, 1, 2]
                )


if __name__ == "__main__":
    unittest.main()
