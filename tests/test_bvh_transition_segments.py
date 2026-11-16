"""Regression tests for opt-in finite transition BVH preservation."""

import unittest
from pathlib import Path

import numpy as np

from bvh_reference import (
    MotionSegment,
    ParsedBvh,
    ParsedJoint,
    _motion_segments_for_mode,
    _transition_endpoint_segments,
)
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

    def test_endpoint_mode_overlaps_start_and_stop_with_walking_middle(self) -> None:
        detected = (
            MotionSegment(100, 140, "right_foot"),
            MotionSegment(140, 180, "left_foot"),
            MotionSegment(180, 220, "right_foot"),
            MotionSegment(220, 260, "left_foot"),
            MotionSegment(260, 300, "right_foot"),
            MotionSegment(300, 340, "left_foot"),
        )

        start, stop = _transition_endpoint_segments(detected, frame_count=480)

        self.assertEqual((start.start_frame, start.end_frame), (0, 220))
        self.assertEqual((stop.start_frame, stop.end_frame), (180, 480))

    def test_endpoint_mode_needs_a_walking_middle(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least four"):
            _transition_endpoint_segments(
                (MotionSegment(100, 140, "right_foot"),),
                frame_count=480,
            )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_segment_mode"):
            _motion_segments_for_mode(Path("transition.bvh"), make_bvh(), "unknown")


if __name__ == "__main__":
    unittest.main()
