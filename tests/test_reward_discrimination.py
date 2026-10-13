"""Check isolated reward-audit probes and native snapshot ownership."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from bvh_reference import LoopMode
from tools.check_reference_curves import full_state
from tools.check_reward_discrimination import (
    audit_reward,
    build_cases,
    snapshot,
    summarize,
)


class RewardCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SimpleNamespace(
            _actuator_joint_names=[
                "right_knee_z",
                "abdomen_z",
                "left_ankle_z",
                "right_hip_z",
                "left_knee_z",
                "left_hip_z",
                "left_hip_y",
                "right_ankle_z",
            ],
            _actuator_qpos_indices=np.array([12, 7, 14, 9, 11, 13, 8, 10]),
            _actuator_dof_indices=np.array([11, 6, 13, 8, 10, 12, 7, 9]),
            _init_q=np.array([0, 0, 1, 1, 0, 0, 0, *np.arange(9) / 100]),
            _mj_model=SimpleNamespace(nv=15),
        )
        self.references = [
            {
                "qpos": np.arange(8, dtype=float) / 10 + offset,
                "qvel": np.arange(8, dtype=float) + 1 + offset,
                "sim_root_pos": np.array([0.1 + offset, 0.2, 1.0]),
                "sim_root_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "sim_root_vel": np.array([0.4, 0.5 + offset, 0.6]),
                "sim_root_angvel": np.array([0.01, 0.02, 0.03 + offset]),
            }
            for offset in (0.0, 0.25, 0.5)
        ]
        self.frozen_qpos = self.env._init_q.copy()
        self.frozen_qpos[0] = -2.0
        self.cases = build_cases(self.env, *self.references, self.frozen_qpos)

    def test_thirteen_named_cases_are_present(self) -> None:
        expected = {
            "exact_reference",
            "standing_default",
            "frozen_phase_zero",
            "stopped_reference_pose",
            "wrong_phase_quarter",
            "wrong_phase_half",
            "swapped_sagittal_legs",
            "translated_xy_control",
            "root_velocity_plus_0p5_world_x",
            "left_knee_velocity_plus_2",
            "left_knee_velocity_minus_2",
            "right_knee_velocity_plus_2",
            "right_knee_velocity_minus_2",
        }
        self.assertEqual(set(self.cases), expected)

    def test_building_cases_preserves_all_sources(self) -> None:
        sources = [
            self.env._init_q,
            self.frozen_qpos,
            *(value for reference in self.references for value in reference.values()),
        ]
        before = [value.copy() for value in sources]
        build_cases(self.env, *self.references, self.frozen_qpos)
        for actual, expected in zip(sources, before, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_cases_do_not_alias_each_other_or_sources(self) -> None:
        sources = [
            self.env._init_q,
            self.frozen_qpos,
            *(value for reference in self.references for value in reference.values()),
        ]
        arrays = [array for pair in self.cases.values() for array in pair]
        for index, array in enumerate(arrays):
            for other in sources + arrays[index + 1 :]:
                self.assertFalse(np.shares_memory(array, other))

    def test_standing_frozen_and_stopped_have_distinct_positions_zero_velocities(
        self,
    ) -> None:
        exact_qpos, _ = self.cases["exact_reference"]
        expected_positions = {
            "standing_default": self.env._init_q,
            "frozen_phase_zero": self.frozen_qpos,
            "stopped_reference_pose": exact_qpos,
        }
        for name, expected in expected_positions.items():
            with self.subTest(case=name):
                qpos, qvel = self.cases[name]
                np.testing.assert_array_equal(qpos, expected)
                np.testing.assert_array_equal(qvel, np.zeros(15))
        self.assertFalse(np.array_equal(exact_qpos, self.env._init_q))
        self.assertFalse(np.array_equal(self.frozen_qpos, self.env._init_q))

    def test_exact_and_phase_probes_are_complete_named_index_states(self) -> None:
        for name, reference in zip(
            ("exact_reference", "wrong_phase_quarter", "wrong_phase_half"),
            self.references,
            strict=True,
        ):
            with self.subTest(case=name):
                qpos, qvel = self.cases[name]
                expected_qpos, expected_qvel = full_state(self.env, reference)
                np.testing.assert_array_equal(qpos, expected_qpos)
                np.testing.assert_array_equal(qvel, expected_qvel)
                np.testing.assert_array_equal(qpos[:3], reference["sim_root_pos"])
                np.testing.assert_array_equal(qvel[3:6], reference["sim_root_angvel"])
                self.assertEqual(qpos[15], self.env._init_q[15])
                self.assertEqual(qvel[14], 0.0)

    def test_only_named_sagittal_channels_are_swapped_in_position_and_velocity(
        self,
    ) -> None:
        qpos, qvel = self.cases["exact_reference"]
        expected_qpos, expected_qvel = qpos.copy(), qvel.copy()
        # Explicit physical-index pairs catch accidental actuator-order assumptions.
        for left, right in ((13, 9), (11, 12), (14, 10)):
            expected_qpos[[left, right]] = qpos[[right, left]]
            expected_qvel[[left - 1, right - 1]] = qvel[[right - 1, left - 1]]
        actual_qpos, actual_qvel = self.cases["swapped_sagittal_legs"]
        np.testing.assert_array_equal(actual_qpos, expected_qpos)
        np.testing.assert_array_equal(actual_qvel, expected_qvel)

    def test_xy_translation_changes_only_root_position(self) -> None:
        qpos, qvel = self.cases["exact_reference"]
        shifted_qpos, shifted_qvel = self.cases["translated_xy_control"]
        expected_delta = np.zeros_like(qpos)
        expected_delta[:2] = [1.0, -0.5]
        np.testing.assert_allclose(shifted_qpos - qpos, expected_delta)
        np.testing.assert_array_equal(shifted_qvel, qvel)

    def test_root_velocity_probe_changes_only_world_x_velocity(self) -> None:
        qpos, qvel = self.cases["exact_reference"]
        shifted_qpos, shifted_qvel = self.cases["root_velocity_plus_0p5_world_x"]
        expected_delta = np.zeros_like(qvel)
        expected_delta[0] = 0.5
        np.testing.assert_array_equal(shifted_qpos, qpos)
        np.testing.assert_allclose(shifted_qvel - qvel, expected_delta)

    def test_knee_velocity_probes_use_joint_names_and_both_signs(self) -> None:
        qpos, qvel = self.cases["exact_reference"]
        for side, index in (("left", 10), ("right", 11)):
            for label, delta in (("plus", 2.0), ("minus", -2.0)):
                with self.subTest(side=side, sign=label):
                    altered_qpos, altered_qvel = self.cases[
                        f"{side}_knee_velocity_{label}_2"
                    ]
                    expected_delta = np.zeros_like(qvel)
                    expected_delta[index] = delta
                    np.testing.assert_array_equal(altered_qpos, qpos)
                    np.testing.assert_allclose(altered_qvel - qvel, expected_delta)


class RewardSnapshotTests(unittest.TestCase):
    def test_native_snapshot_copies_fields_before_data_reuse(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body pos="0 0 1"><freejoint/>
              <geom type="sphere" size=".1" mass="1"/>
              <site pos=".1 0 0"/>
            </body></worldbody></mujoco>
        """)
        data = mujoco.MjData(model)
        data.qvel[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        mujoco.mj_forward(model, data)
        captured = snapshot(data)
        expected_fields = {
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "xmat",
            "site_xpos",
            "cvel",
            "subtree_com",
        }
        self.assertEqual(set(captured), expected_fields)
        for name, value in captured.items():
            self.assertEqual(value.dtype, np.dtype(np.float32))
            np.testing.assert_array_equal(value, getattr(data, name).astype(np.float32))
            self.assertFalse(np.shares_memory(value, getattr(data, name)))
        before = {name: value.copy() for name, value in captured.items()}
        data.qpos[0] = 3.0
        data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for name in captured:
            np.testing.assert_array_equal(captured[name], before[name])
        self.assertNotEqual(float(data.qpos[0]), float(captured["qpos"][0]))


class RewardAuditValidationTests(unittest.TestCase):
    def test_summary_has_expected_statistics(self) -> None:
        expected = {
            "min": 0.0,
            "mean": 2.0,
            "median": 2.0,
            "max": 4.0,
            "p05": 0.2,
            "p95": 3.8,
        }
        actual = summarize(np.array([0, 1, 2, 3, 4]))
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value)

    def test_summary_rejects_empty_nonfinite_or_nonscalar_samples(self) -> None:
        for values in ([], [np.nan], [np.inf], [-np.inf], [[1, 2]], np.array(3)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "nonempty finite scalar"):
                    summarize(values)

    def test_bad_sample_count_is_rejected_before_environment_access(self) -> None:
        for count in (-4, 0, 1, 2, 3, 5, 6):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "positive multiple of four"):
                    audit_reward(object(), count)

    def test_mixed_reward_mode_is_rejected_before_reference_or_model_access(
        self,
    ) -> None:
        env = SimpleNamespace(_config=SimpleNamespace(deepmimic_reward_mode="mixed"))
        with self.assertRaisesRegex(ValueError, "Only pure imitation"):
            audit_reward(env, 4)

    def test_other_physics_versions_are_rejected_before_reference_access(self) -> None:
        env = SimpleNamespace(
            _config=SimpleNamespace(
                deepmimic_reward_mode="pure", recovery_physics_version="consistent_v5"
            )
        )
        with self.assertRaisesRegex(ValueError, "pins the consistent_v6 control"):
            audit_reward(env, 4)

    def test_multiple_or_missing_clips_are_rejected_before_loop_or_model_access(
        self,
    ) -> None:
        for counts in ([], [4, 4]):
            with self.subTest(counts=counts):
                env = SimpleNamespace(
                    _config=SimpleNamespace(
                        deepmimic_reward_mode="pure",
                        recovery_physics_version="consistent_v6",
                    ),
                    _bvh_reference_frame_counts=np.array(counts),
                )
                with self.assertRaisesRegex(ValueError, "one selected WRAP clip"):
                    audit_reward(env, 4)

    def test_nonwrapping_clip_is_rejected_before_model_access(self) -> None:
        for mode in LoopMode:
            if mode == LoopMode.WRAP:
                continue
            with self.subTest(mode=mode):
                env = SimpleNamespace(
                    _config=SimpleNamespace(
                        deepmimic_reward_mode="pure",
                        recovery_physics_version="consistent_v6",
                    ),
                    _bvh_reference_frame_counts=np.array([4]),
                    _bvh_reference_loop_modes=np.array([mode]),
                )
                with self.assertRaisesRegex(ValueError, "one selected WRAP clip"):
                    audit_reward(env, 4)


if __name__ == "__main__":
    unittest.main()
