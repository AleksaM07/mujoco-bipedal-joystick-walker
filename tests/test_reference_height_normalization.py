"""Versioned regression tests for IK foot-height normalization."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from config import default_biomechanics_env_config


def make_height_environment(version: str) -> BiomechanicsJoystickEnv:
    """Build the normalization fixture without loading models or motion files."""
    env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
    env._framewise_foot_heights = version in {"consistent_v3", "consistent_v4"}
    env._standing_marker_height = Mock(return_value=0.05)
    return env


class ReferenceHeightNormalizationTests(unittest.TestCase):
    """Keep old targets reproducible while correcting v3 frame/foot axes."""

    def test_v4_keeps_v3_framewise_height_normalization(self) -> None:
        targets = np.zeros((1, 4, 2, 3), dtype=np.float32)
        targets[0, :, :, 2] = [[0.2, 0.3], [0.4, 0.1], [0.6, 0.5], [0.8, 0.2]]
        names = ("metatarsal_midpoint_right", "metatarsal_midpoint_left")
        outputs = [
            make_height_environment(version)._normalize_ik_foot_target_heights(
                targets, names, np.array([4])
            )
            for version in ("consistent_v3", "consistent_v4")
        ]
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_allclose(outputs[1][0, :, :, 2].min(axis=1), 0.05, atol=1e-7)

    def test_four_frames_two_feet_use_one_offset_per_frame(self) -> None:
        env = make_height_environment("consistent_v3")
        targets = np.arange(36, dtype=np.float32).reshape(1, 4, 3, 3)
        targets[0, :, [0, 2], 2] = np.array(
            [[0.2, 0.4, 0.6, 0.8], [0.3, 0.1, 0.5, 0.2]], dtype=np.float32
        )
        original = targets.copy()
        result = env._normalize_ik_foot_target_heights(
            targets, ("right_foot", "head", "left_foot"), np.array([4])
        )
        heights = result[0][:, [0, 2], 2]
        original_heights = original[0][:, [0, 2], 2]
        np.testing.assert_allclose(heights.min(axis=1), 0.05, atol=1e-7)
        np.testing.assert_allclose(
            heights[:, 1] - heights[:, 0],
            original_heights[:, 1] - original_heights[:, 0],
            atol=1e-7,
        )
        np.testing.assert_array_equal(result[..., :2], original[..., :2])
        np.testing.assert_array_equal(result[:, :, 1], original[:, :, 1])
        np.testing.assert_array_equal(targets, original)
        self.assertFalse(np.shares_memory(result, targets))

    def test_historical_versions_keep_whole_clip_minimum_behavior(self) -> None:
        targets = np.zeros((1, 4, 2, 3), dtype=np.float32)
        targets[0, :, :, 2] = [[0.2, 0.3], [0.4, 0.1], [0.6, 0.5], [0.8, 0.2]]
        expected = [[0.05, 0.25], [0.25, 0.05], [0.45, 0.45], [0.65, 0.15]]
        for version in ("legacy", "consistent_v1", "consistent_v2"):
            with self.subTest(version=version):
                result = make_height_environment(
                    version
                )._normalize_ik_foot_target_heights(
                    targets, ("right_foot", "left_foot"), np.array([4])
                )
                np.testing.assert_allclose(result[0, :, :, 2], expected, atol=1e-7)

    def test_single_foot_is_normalized_in_every_frame(self) -> None:
        env = make_height_environment("consistent_v3")
        targets = np.zeros((1, 5, 1, 3), dtype=np.float32)
        targets[0, :, 0, 2] = [0.3, -0.1, 0.5, 0.2, 0.4]
        result = env._normalize_ik_foot_target_heights(
            targets, ("metatarsal_midpoint_right",), np.array([5])
        )
        np.testing.assert_allclose(result[0, :, 0, 2], 0.05, atol=1e-7)

    def test_multiple_clips_repeat_last_corrected_frame_in_padding(self) -> None:
        env = make_height_environment("consistent_v3")
        targets = np.arange(2 * 5 * 3 * 3, dtype=np.float32).reshape(2, 5, 3, 3)
        counts = np.array([3, 4])
        for clip, count in enumerate(counts):
            targets[clip, count:] = targets[clip, count - 1]
        original = targets.copy()
        result = env._normalize_ik_foot_target_heights(
            targets, ("left_foot", "head", "right_foot"), counts
        )
        for clip, count in enumerate(counts):
            with self.subTest(clip=clip):
                heights = result[clip, :count][:, [0, 2], 2]
                np.testing.assert_allclose(heights.min(axis=1), 0.05, atol=1e-5)
                expected_padding = np.repeat(
                    result[clip, count - 1 : count], 5 - count, axis=0
                )
                np.testing.assert_array_equal(result[clip, count:], expected_padding)
        np.testing.assert_array_equal(result[..., :2], original[..., :2])
        np.testing.assert_array_equal(result[:, :, 1], original[:, :, 1])
        np.testing.assert_array_equal(targets, original)

    def test_no_foot_markers_leave_targets_unchanged(self) -> None:
        env = make_height_environment("consistent_v3")
        targets = np.arange(18, dtype=np.float32).reshape(1, 3, 2, 3)
        original = targets.copy()
        result = env._normalize_ik_foot_target_heights(
            targets, ("head", "pelvis"), np.array([3])
        )
        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(targets, original)
        env._standing_marker_height.assert_not_called()

    def test_empty_clip_is_not_modified(self) -> None:
        env = make_height_environment("consistent_v3")
        targets = np.zeros((1, 3, 2, 3), dtype=np.float32)
        result = env._normalize_ik_foot_target_heights(
            targets, ("left_foot", "right_foot"), np.array([0])
        )
        np.testing.assert_array_equal(result, targets)

    def test_versions_select_flags_before_model_loading(self) -> None:
        for version in (
            "legacy",
            "consistent_v1",
            "consistent_v2",
            "consistent_v3",
            "consistent_v4",
        ):
            with self.subTest(version=version):
                config = default_biomechanics_env_config()
                config.recovery_physics_version = version
                config.xml_path = "unused-test-model.xml"
                if version in {"consistent_v2", "consistent_v3", "consistent_v4"}:
                    config.reference_loop_closure = "bounded"
                env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
                with patch("biomechanics_env.mujoco.MjModel") as model:
                    model.from_xml_path.side_effect = RuntimeError("model load reached")
                    with self.assertRaisesRegex(RuntimeError, "model load reached"):
                        env.__init__(config=config)
                    model.from_xml_path.assert_called_once()
                self.assertEqual(env._consistent_physics, version != "legacy")
                self.assertEqual(
                    env._continuous_reference,
                    version in {"consistent_v2", "consistent_v3", "consistent_v4"},
                )
                self.assertEqual(
                    env._framewise_foot_heights,
                    version in {"consistent_v3", "consistent_v4"},
                )
                self.assertEqual(env._strict_bvh_ik_targets, version == "consistent_v4")

    def test_unknown_version_is_rejected_before_model_loading(self) -> None:
        config = default_biomechanics_env_config()
        config.recovery_physics_version = "consistent_v999"
        with patch("biomechanics_env.mujoco.MjModel") as model:
            with self.assertRaisesRegex(ValueError, "recovery_physics_version"):
                BiomechanicsJoystickEnv(config=config)
            model.from_xml_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
