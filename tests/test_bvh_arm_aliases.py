"""Preserve archived arm targets and exercise CMU wrists against the real XML."""

from dataclasses import replace
from pathlib import Path
import unittest

import mujoco
import numpy as np

from bvh_reference import (
    BvhMotionLibrary,
    MotionSegment,
    _bvh_arm_swing_targets,
    _parse_bvh,
    load_bvh_references,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
XML_PATH = (
    PROJECT_ROOT / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
)


class BvhArmAliasTests(unittest.TestCase):
    """Use the trained clip, not a fixture with already-supported hand names."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        joint_ids = cls.model.actuator_trnid[:, 0]
        cls.names = tuple(cls.model.joint(int(index)).name for index in joint_ids)
        cls.default_ctrl = cls.model.qpos0[cls.model.jnt_qposadr[joint_ids]]
        cls.loader_args = dict(
            paths=(SOURCE_PATH,),
            actuator_joint_names=cls.names,
            default_ctrl=cls.default_ctrl,
            lower_limits=cls.model.jnt_range[joint_ids, 0],
            upper_limits=cls.model.jnt_range[joint_ids, 1],
            initial_root_pos=cls.model.qpos0[:3],
            initial_root_quat=cls.model.qpos0[3:7],
            strict_ik_targets=True,
        )
        cls.legacy = BvhMotionLibrary.from_bvh_paths(**cls.loader_args)
        cls.corrected = BvhMotionLibrary.from_bvh_paths(
            **cls.loader_args, arm_reference_mode="cmu_wrist"
        )
        cls.source = _parse_bvh(SOURCE_PATH).slice(MotionSegment(79, 213))

    def test_historical_default_keeps_zero_arm_targets(self) -> None:
        self.assertIn("LeftWrist", self.source.joints)
        self.assertNotIn("LeftHand", self.source.joints)
        self.assertEqual(
            _bvh_arm_swing_targets(self.source, self.names, self.default_ctrl), {}
        )
        for clip in self.legacy.clips:
            np.testing.assert_array_equal(clip.qpos_targets[:, 18:], 0.0)
            np.testing.assert_array_equal(clip.qvel_targets[:, 18:], 0.0)

    def test_opt_in_adds_swing_without_changing_locomotion_or_selected_clips(self) -> None:
        self.assertEqual(len(self.legacy.clips), len(self.corrected.clips))
        for original, updated in zip(self.legacy.clips, self.corrected.clips, strict=True):
            self.assertEqual(original.source_start_frame, updated.source_start_frame)
            self.assertEqual(original.source_end_frame, updated.source_end_frame)
            self.assertEqual(original.loop_mode, updated.loop_mode)
            np.testing.assert_array_equal(
                original.qpos_targets[:, :18], updated.qpos_targets[:, :18]
            )
            np.testing.assert_array_equal(
                original.qvel_targets[:, :18], updated.qvel_targets[:, :18]
            )
            for field in (
                "root_pos_targets", "root_quat_targets", "root_vel_targets",
                "root_angvel_targets", "ik_target_positions", "wrap_delta",
            ):
                np.testing.assert_array_equal(getattr(original, field), getattr(updated, field))
        clip = self.corrected.clips[1]
        for side in ("left", "right"):
            shoulder = self.names.index(f"{side}_shoulder_z")
            elbow = self.names.index(f"{side}_elbow_z")
            self.assertGreater(np.ptp(clip.qpos_targets[:, shoulder]), 0.4)
            self.assertGreater(np.max(np.abs(clip.qvel_targets[:, shoulder])), 0.1)
            np.testing.assert_allclose(clip.qpos_targets[:, elbow], 0.32)
            for axis in ("x", "y"):
                index = self.names.index(f"{side}_shoulder_{axis}")
                np.testing.assert_array_equal(clip.qpos_targets[:, index], 0.0)
        # Targets are float32 while MuJoCo stores joint limits as float64.
        self.assertTrue(np.all(clip.qpos_targets >= self.loader_args["lower_limits"] - 1e-7))
        self.assertTrue(np.all(clip.qpos_targets <= self.loader_args["upper_limits"] + 1e-7))

    def test_public_loader_propagates_mode(self) -> None:
        batch = load_bvh_references(**self.loader_args, arm_reference_mode="cmu_wrist")
        np.testing.assert_array_equal(
            batch.qpos_targets, self.corrected.to_reference_batch().qpos_targets
        )

    def test_missing_alias_fails_for_active_arms_instead_of_silent_static_target(self) -> None:
        joints = dict(self.source.joints)
        for name in ("LeftWrist", "lhand"):
            joints.pop(name)
        source = replace(self.source, joints=joints)
        with self.assertRaisesRegex(ValueError, "refusing silent neutral arm targets"):
            _bvh_arm_swing_targets(
                source, self.names, self.default_ctrl, arm_reference_mode="cmu_wrist"
            )
        self.assertEqual(
            _bvh_arm_swing_targets(
                source, self.names[:18], self.default_ctrl[:18],
                arm_reference_mode="cmu_wrist",
            ),
            {},
        )

    def test_unknown_mode_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported arm_reference_mode"):
            _bvh_arm_swing_targets(
                self.source, self.names, self.default_ctrl, arm_reference_mode="typo"
            )

    def test_positive_shoulder_and_elbow_angles_move_both_hands_forward(self) -> None:
        # Check the actual model's axes/signs, not assumptions from joint names.
        for side in ("left", "right"):
            for joint in (f"{side}_shoulder_z", f"{side}_elbow_z"):
                with self.subTest(joint=joint):
                    data = mujoco.MjData(self.model)
                    mujoco.mj_forward(self.model, data)
                    hand_id = self.model.body(f"{side}_hand").id
                    baseline = data.xpos[hand_id].copy()
                    joint_id = self.model.joint(joint).id
                    data.qpos[self.model.jnt_qposadr[joint_id]] += 0.001
                    mujoco.mj_forward(self.model, data)
                    displacement = data.xpos[hand_id] - baseline
                    self.assertGreater(displacement[0], 0.0002)
                    self.assertLess(abs(displacement[1]), 1e-6)
                    self.assertEqual(self.model.jnt_stiffness[joint_id], 0.0)


if __name__ == "__main__":
    unittest.main()
