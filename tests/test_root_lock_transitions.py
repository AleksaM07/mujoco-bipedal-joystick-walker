"""Check the root-lock observer against a small native MuJoCo fixture."""

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from tools.check_root_lock_transitions import trace_lock_call


class RootLockTransitionTests(unittest.TestCase):
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
        self.joints = np.array([[0, 0, 0, 0.1], [0.02, 0, 0, 0.1], [0.05, 0, 0, 0.1]])
        self.roots = np.array([[0, 0, 0], [0.2, 0.01, 0], [0.4, 0.02, 0]])
        self.quaternions = np.tile([1.0, 0, 0, 0], (3, 1))
        self.locked = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
            self.env, self.joints, self.roots, self.quaternions, 3
        )

    def test_continued_support_matches_stationary_center_recurrence(self) -> None:
        result = trace_lock_call(
            self.env, self.joints, self.roots, self.quaternions, 3, self.locked, 0.1
        )
        self.assertEqual(
            [row["event"] for row in result["rows"]],
            ["initialize", "continue", "continue"],
        )
        np.testing.assert_allclose(self.locked[:, 0], [0, -0.02, -0.05], atol=1e-7)
        self.assertLess(result["max_stationary_center_recurrence_error_m_s"], 1e-6)
        np.testing.assert_allclose(
            result["rows"][1]["stationary_center_velocity_m_s"], [-0.2, 0], atol=1e-7
        )
        self.assertAlmostEqual(result["worst_xy_second_difference_m_s2"], 1, places=5)
        self.assertEqual(result["worst_xy_second_difference_center_frame"], 1)

    def test_switch_flight_and_new_contact_have_distinct_labels(self) -> None:
        joints = np.zeros((6, 4))
        joints[:, 1] = [0, 0, 0.1, 0.1, 0.1, 0.1]
        joints[:, 3] = [0.1, 0.1, 0, 0.1, 0, 0]
        roots = np.zeros((6, 3))
        roots[:, 0] = np.arange(6) * 0.2
        quaternions = np.tile([1.0, 0, 0, 0], (6, 1))
        locked = BiomechanicsJoystickEnv._lock_clip_root_xy_to_stance_feet(
            self.env, joints, roots, quaternions, 6
        )
        result = trace_lock_call(self.env, joints, roots, quaternions, 6, locked, 0.1)
        self.assertEqual(
            [row["event"] for row in result["rows"]],
            [
                "initialize",
                "continue",
                "switch",
                "no_contact",
                "initialize",
                "continue",
            ],
        )
        self.assertEqual(result["support_transitions"], [0, 2, 3, 4])
        self.assertEqual(result["rows"][2]["active_foot"], "right_foot")
        self.assertIsNone(result["rows"][3]["active_foot"])
        for frame in (2, 3, 4):
            np.testing.assert_allclose(
                result["rows"][frame]["output_xy_nodal_velocity_m_s"],
                result["rows"][frame]["input_xy_nodal_velocity_m_s"],
                atol=1e-6,
            )
            self.assertNotIn(
                "stationary_center_recurrence_error_m_s", result["rows"][frame]
            )

    def test_observer_reports_inconsistent_output_without_repairing_it(self) -> None:
        modified = self.locked.copy()
        modified[1, 0] += 0.1
        result = trace_lock_call(
            self.env, self.joints, self.roots, self.quaternions, 3, modified, 0.1
        )
        self.assertAlmostEqual(
            result["max_stationary_center_recurrence_error_m_s"], 1, places=5
        )
        np.testing.assert_array_equal(
            result["rows"][1]["output_root_xyz_m"], modified[1]
        )

    def test_observer_leaves_inputs_and_model_unchanged(self) -> None:
        arrays = [self.joints, self.roots, self.quaternions, self.locked]
        snapshots = [array.copy() for array in arrays]
        qpos0 = self.model.qpos0.copy()
        trace_lock_call(
            self.env, self.joints, self.roots, self.quaternions, 3, self.locked, 0.1
        )
        for array, snapshot in zip(arrays, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)
        np.testing.assert_array_equal(self.model.qpos0, qpos0)

    def test_invalid_count_or_interval_is_rejected(self) -> None:
        for count, interval in (
            (0, 0.1),
            (2, 0.1),
            (3, 0),
            (3, -0.1),
            (3, np.nan),
            (3, np.inf),
        ):
            with self.subTest(count=count, interval=interval):
                with self.assertRaisesRegex(
                    ValueError, "three frames.*positive interval"
                ):
                    trace_lock_call(
                        self.env,
                        self.joints,
                        self.roots,
                        self.quaternions,
                        count,
                        self.locked,
                        interval,
                    )


if __name__ == "__main__":
    unittest.main()
