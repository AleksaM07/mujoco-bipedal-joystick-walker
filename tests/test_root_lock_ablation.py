"""Check contact-history locking with independently specified foot trajectories."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from tools.check_root_lock_ablation import lock_with_contact_history, root_lock_ablation


class ContactHistoryLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body>
              <freejoint/>
              <geom type="sphere" size=".01" mass="1"/>
              <body pos="0 .2 .1">
                <joint name="left_x" type="slide" axis="1 0 0"/>
                <joint name="left_z" type="slide" axis="0 0 1"/>
                <geom name="left_foot" type="sphere" size=".1" mass="1"/>
              </body>
              <body pos="0 -.2 .1">
                <joint name="right_x" type="slide" axis="1 0 0"/>
                <joint name="right_z" type="slide" axis="0 0 1"/>
                <geom name="right_foot" type="sphere" size=".1" mass="1"/>
              </body>
            </body></worldbody></mujoco>
        """)

        def lowest_z(data: mujoco.MjData, geom: int) -> float:
            return float(data.geom_xpos[geom, 2] - self.model.geom_size[geom, 0])

        self.env = SimpleNamespace(
            _mj_model=self.model,
            _init_q_np=self.model.qpos0.copy(),
            _foot_geom_ids_np=np.array(
                [self.model.geom(name).id for name in ("left_foot", "right_foot")]
            ),
            _actuator_qpos_indices_np=np.arange(7, 11),
            _config={"reference_foot_lock_height": 0.035},
            _geom_min_z=lowest_z,
        )
        self.joints = np.array(
            [
                [0, 0, 0, 0.02],
                [0.02, 0.01, 0.01, 0.01],
                [0.04, 0.06, 0.03, 0],
                [0.06, 0.06, 0.05, 0],
            ]
        )
        self.roots = np.zeros((4, 3))
        self.roots[:, 0] = np.arange(4) * 0.2
        self.quaternions = np.tile([1.0, 0, 0, 0], (4, 1))

    def test_supported_incoming_foot_keeps_its_previous_world_center(self) -> None:
        locked = lock_with_contact_history(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        np.testing.assert_allclose(locked[:, 0], [0, -0.02, -0.04, -0.06], atol=1e-7)
        previous_right_x = locked[1, 0] + self.joints[1, 2]
        next_right_x = locked[2, 0] + self.joints[2, 2]
        self.assertAlmostEqual(previous_right_x, next_right_x, places=7)
        legacy = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        self.assertAlmostEqual(float(legacy[2, 0] - legacy[1, 0]), 0.2, places=6)

    def test_landing_without_overlap_and_flight_preserve_input_displacement(
        self,
    ) -> None:
        joints = np.zeros((5, 4))
        joints[:, 0] = np.arange(5) * 0.02
        joints[:, 1] = [0, 0, 0.1, 0.1, 0.1]
        joints[:, 2] = np.arange(5) * 0.01
        joints[:, 3] = [0.1, 0.1, 0, 0.1, 0]
        roots = np.zeros((5, 3))
        roots[:, 0] = np.arange(5) * 0.2
        quaternions = np.tile([1.0, 0, 0, 0], (5, 1))
        locked = lock_with_contact_history(self.env, joints, roots, quaternions, 5)
        legacy = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
            self.env, joints, roots, quaternions, 5
        )
        np.testing.assert_allclose(
            locked[:, 0], [0, -0.02, 0.18, 0.38, 0.58], atol=1e-7
        )
        np.testing.assert_allclose(locked, legacy, atol=1e-7)

    def test_continuous_support_matches_existing_lock_and_preserves_root_z(
        self,
    ) -> None:
        joints = self.joints.copy()
        joints[:, 1] = 0
        joints[:, 3] = 0.1
        roots = self.roots.copy()
        roots[:, 2] = [0, 0.002, 0.004, 0.003]
        locked = lock_with_contact_history(self.env, joints, roots, self.quaternions, 4)
        legacy = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
            self.env, joints, roots, self.quaternions, 4
        )
        np.testing.assert_allclose(locked, legacy, atol=1e-7)
        np.testing.assert_array_equal(locked[:, 2], roots[:, 2].astype(np.float32))

    def test_common_xy_translation_translates_locked_path(self) -> None:
        shift = np.array([2.0, -3.0, 0])
        locked = lock_with_contact_history(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        shifted = lock_with_contact_history(
            self.env, self.joints, self.roots + shift, self.quaternions, 4
        )
        np.testing.assert_allclose(shifted, locked + shift, atol=5e-7)

    def test_changing_root_yaw_keeps_the_selected_world_foot_stationary(self) -> None:
        angles = np.array([0, 0.15, 0.3, 0.45])
        quaternions = np.zeros((4, 4))
        quaternions[:, 0] = np.cos(angles / 2)
        quaternions[:, 3] = np.sin(angles / 2)
        joints = np.zeros_like(self.joints)
        joints[:, 3] = 0.1
        locked = lock_with_contact_history(self.env, joints, self.roots, quaternions, 4)
        expected_xy = np.column_stack(
            (0.2 * np.sin(angles), 0.2 * (1 - np.cos(angles)))
        )
        np.testing.assert_allclose(locked[:, :2], expected_xy, atol=1e-7)

    def test_padding_holds_last_valid_frame_without_mutating_inputs_or_model(
        self,
    ) -> None:
        arrays = [self.joints, self.roots, self.quaternions, self.env._init_q_np]
        snapshots = [array.copy() for array in arrays]
        model_snapshot = self.model.qpos0.copy()
        locked = lock_with_contact_history(
            self.env, self.joints, self.roots, self.quaternions, 3
        )
        np.testing.assert_array_equal(locked[3], locked[2])
        for array, snapshot in zip(arrays, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)
        np.testing.assert_array_equal(self.model.qpos0, model_snapshot)
        self.assertEqual(locked.dtype, np.float32)
        self.assertFalse(np.shares_memory(locked, self.roots))

    def test_zero_or_single_frame_returns_unchanged_detached_copy(self) -> None:
        for count in (0, 1):
            with self.subTest(count=count):
                roots = self.roots[:count]
                locked = lock_with_contact_history(
                    self.env,
                    self.joints[:count],
                    roots,
                    self.quaternions[:count],
                    count,
                )
                np.testing.assert_array_equal(locked, roots.astype(np.float32))
                self.assertFalse(np.shares_memory(locked, roots))

    def test_initial_skip_and_final_handoff_are_scoped_per_environment(self) -> None:
        other_env = SimpleNamespace(**vars(self.env))

        def align_fixture(env):
            return BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                env, self.joints, self.roots, self.quaternions, 4
            )

        expected_final = lock_with_contact_history(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        with patch.object(
            BiomechanicsJoystickEnv,
            "_align_reference_roots_to_sim_floor",
            align_fixture,
        ):
            with root_lock_ablation(skip_initial=True, handoff=True):
                for env in (self.env, other_env):
                    initial = (
                        BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(env)
                    )
                    np.testing.assert_array_equal(
                        initial, self.roots.astype(np.float32)
                    )
                    self.assertFalse(np.shares_memory(initial, self.roots))
                for env in (self.env, other_env):
                    final = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                        env
                    )
                    np.testing.assert_array_equal(final, expected_final)

    def test_each_ablation_flag_changes_only_its_selected_alignment_pass(self) -> None:
        def align_fixture(env):
            return BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                env, self.joints, self.roots, self.quaternions, 4
            )

        legacy = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        contact_history = lock_with_contact_history(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        with patch.object(
            BiomechanicsJoystickEnv,
            "_align_reference_roots_to_sim_floor",
            align_fixture,
        ):
            for skip_initial, handoff, expected in (
                (False, True, (legacy, contact_history)),
                (True, False, (self.roots.astype(np.float32), legacy)),
            ):
                with self.subTest(skip_initial=skip_initial, handoff=handoff):
                    with root_lock_ablation(skip_initial=skip_initial, handoff=handoff):
                        align = (
                            BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor
                        )
                        for expected_pass in expected:
                            result = align(self.env)
                            np.testing.assert_array_equal(result, expected_pass)

    def test_unselected_passes_use_original_lock_and_methods_restore_on_error(
        self,
    ) -> None:
        original_lock = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet
        original_align = BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor

        def align_fixture(env):
            return BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                env, self.joints, self.roots, self.quaternions, 4
            )

        expected_legacy = original_lock(
            self.env, self.joints, self.roots, self.quaternions, 4
        )
        with patch.object(
            BiomechanicsJoystickEnv,
            "_align_reference_roots_to_sim_floor",
            align_fixture,
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                with root_lock_ablation():
                    for _ in range(2):
                        result = (
                            BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                                self.env
                            )
                        )
                        np.testing.assert_array_equal(result, expected_legacy)
                    raise RuntimeError("fixture interruption")
            self.assertIs(
                BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor,
                align_fixture,
            )
            self.assertIs(
                BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet,
                original_lock,
            )
        self.assertIs(
            BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor, original_align
        )

    def test_lock_outside_alignment_and_unexpected_third_pass_are_rejected(
        self,
    ) -> None:
        def align_fixture(env):
            return BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                env, self.joints, self.roots, self.quaternions, 4
            )

        with patch.object(
            BiomechanicsJoystickEnv,
            "_align_reference_roots_to_sim_floor",
            align_fixture,
        ):
            with root_lock_ablation(skip_initial=True, handoff=True):
                with self.assertRaisesRegex(ValueError, "outside.*alignment"):
                    BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
                        self.env, self.joints, self.roots, self.quaternions, 4
                    )
                for _ in range(2):
                    BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                        self.env
                    )
                with self.assertRaisesRegex(ValueError, "two non-nested"):
                    BiomechanicsJoystickEnv._align_reference_roots_to_sim_floor(
                        self.env
                    )


if __name__ == "__main__":
    unittest.main()
