import unittest
from itertools import pairwise

import numpy as np

from skills.transitions import (
    BRAKE_HANDOFF_PHASES,
    DEFAULT_TRANSITION_PHASES,
    WALK_STOP_PROBE_PHASES,
    TransitionPhase,
    build_transition_directives,
    transition_boundaries,
)
from tools.evaluate_skill_transitions import physical_match_score


class SkillTransitionScheduleTests(unittest.TestCase):
    def test_phase_match_ignores_world_xy_and_penalizes_pose_jump(self) -> None:
        current_qpos = np.zeros(10)
        current_qpos[2] = 1.5
        current_qpos[3] = 1.0
        current_qvel = np.zeros(9)
        translated = current_qpos.copy()
        translated[:2] = [12.0, -7.0]
        actuator_qpos_indices = np.array([7, 8, 9])
        actuator_dof_indices = np.array([6, 7, 8])

        exact = physical_match_score(
            current_qpos,
            current_qvel,
            translated,
            current_qvel,
            actuator_qpos_indices,
            actuator_dof_indices,
        )
        mismatched = translated.copy()
        mismatched[7] = 0.8
        pose_jump = physical_match_score(
            current_qpos,
            current_qvel,
            mismatched,
            current_qvel,
            actuator_qpos_indices,
            actuator_dof_indices,
        )

        self.assertEqual(exact["total"], 0.0)
        self.assertGreater(pose_jump["total"], exact["total"])

    def test_default_schedule_has_two_starts_and_stops_without_gaps(self) -> None:
        directives = build_transition_directives(0.02)

        self.assertEqual(len(directives), 1200)
        self.assertEqual(
            transition_boundaries(directives),
            [
                (0, "stand"),
                (100, "walk"),
                (325, "stand"),
                (450, "walk"),
                (675, "stand"),
            ],
        )
        self.assertEqual(
            [phase.name for phase in DEFAULT_TRANSITION_PHASES],
            [
                "stand_initial",
                "start_1",
                "walk_1",
                "stop_1",
                "stand_middle",
                "start_2",
                "walk_2",
                "stop_2",
                "stand_final",
            ],
        )

    def test_blends_are_monotonic_and_finish_on_incoming_skill(self) -> None:
        directives = build_transition_directives(0.02)
        for phase_name in ("start_1", "stop_1", "start_2", "stop_2"):
            blend = [
                directive.blend
                for directive in directives
                if directive.phase_name == phase_name
            ]
            self.assertEqual(len(blend), 25)
            self.assertGreater(blend[0], 0.0)
            self.assertEqual(blend[-1], 1.0)
            self.assertTrue(all(left < right for left, right in pairwise(blend)))

    def test_walk_stop_probe_does_not_depend_on_untrained_reverse_transition(self) -> None:
        directives = build_transition_directives(0.02, WALK_STOP_PROBE_PHASES)

        self.assertEqual(len(directives), 725)
        self.assertEqual(
            transition_boundaries(directives), [(0, "walk"), (200, "stand")]
        )

    def test_bridge_schedule_connects_all_three_policies(self) -> None:
        directives = build_transition_directives(0.02, BRAKE_HANDOFF_PHASES)

        self.assertEqual(len(directives), 820)
        self.assertEqual(
            transition_boundaries(directives),
            [(0, "walk"), (200, "brake_transition"), (310, "stand")],
        )

    def test_invalid_or_disconnected_schedules_are_rejected(self) -> None:
        for dt in (0.0, -0.02, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "dt"):
                build_transition_directives(dt)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            build_transition_directives(
                0.02,
                (TransitionPhase("bad_duration", 0.03, "stand"),),
            )
        with self.assertRaisesRegex(ValueError, "expected"):
            build_transition_directives(
                0.02,
                (
                    TransitionPhase("stand", 0.1, "stand"),
                    TransitionPhase("walk", 0.1, "walk"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
