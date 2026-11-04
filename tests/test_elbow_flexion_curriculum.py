"""Tests for the staged elbow-flexion transfer campaign."""

from __future__ import annotations

import unittest

from scripts.train_elbow_flexion_curriculum import ELBOW_LEVELS


class ElbowFlexionCurriculumTests(unittest.TestCase):
    def test_levels_are_strictly_increasing_and_finish_at_full_signal(self) -> None:
        self.assertEqual(ELBOW_LEVELS[-1], 1.0)
        self.assertTrue(
            all(left < right for left, right in zip(ELBOW_LEVELS, ELBOW_LEVELS[1:]))
        )
        self.assertTrue(all(0.0 < level <= 1.0 for level in ELBOW_LEVELS))


if __name__ == "__main__":
    unittest.main()
