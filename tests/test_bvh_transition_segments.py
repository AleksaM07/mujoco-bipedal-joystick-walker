"""Regression tests for opt-in finite transition BVH preservation."""

import unittest
from pathlib import Path

import numpy as np

from bvh_reference import ParsedBvh, ParsedJoint, _motion_segments_for_mode
from config import EnvConfig, default_biomechanics_env_config


def make_bvh(frame_count: int = 480) -> ParsedBvh:
    """Build the minimal parsed motion needed by the segment selector."""
    root = ParsedJoint(
        name="Hips",
        parent=None,
        children=(),
        offset=np.zeros(3, dtype=np.float32),
        channels=("Xposition", "Yposition", "Zposition"),
        channel_indices=(0, 1, 2),
    )
    return ParsedBvh(
        joints={"Hips": root},
        channels=(("Hips", "Xposition"), ("Hips", "Yposition"), ("Hips", "Zposition")),
        motion=np.zeros((frame_count, 3), dtype=np.float32),
        frames=frame_count,
        frame_time=1 / 120,
        root_name="Hips",
    )


class BvhTransitionSegmentTests(unittest.TestCase):
    def test_existing_gait_configs_keep_locomotion_segmentation(self) -> None:
        self.assertEqual(EnvConfig().reference_segment_mode, "locomotion")
        self.assertEqual(
            default_biomechanics_env_config().reference_segment_mode,
            "locomotion",
        )

    def test_full_mode_keeps_entire_take_even_above_cycle_length_limit(self) -> None:
        segments = _motion_segments_for_mode(Path("transition.bvh"), make_bvh(), "full")

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start_frame, segments[0].end_frame), (0, 480))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_segment_mode"):
            _motion_segments_for_mode(Path("transition.bvh"), make_bvh(), "unknown")


if __name__ == "__main__":
    unittest.main()
