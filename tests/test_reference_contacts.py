"""Regression checks for contact labels and reference acceleration diagnostics."""

import unittest
from types import SimpleNamespace

import jax.numpy as jp
import mujoco
import numpy as np

from tools.check_reference_contacts import (
    inspect_contacts,
    sample_reference_states,
    summarize,
)


class ReferenceAccelerationTests(unittest.TestCase):
    def test_acceleration_is_qvel_derivative_with_actuator_mapping(self) -> None:
        def query(clip, time):
            return {
                "sim_root_pos": jp.array([time**2, 0, 1]),
                "sim_root_quat": jp.array([1, 0, 0, 0]),
                "sim_root_vel": jp.array([2 * time, 0, 0]),
                "sim_root_angvel": jp.zeros(3),
                "qpos": jp.array([time**3]),
                "qvel": jp.array([3 * time**2]),
            }

        env = SimpleNamespace(
            _continuous_reference=True,
            _init_q=jp.array([0, 0, 1, 1, 0, 0, 0, 0, 0.7]),
            _mj_model=SimpleNamespace(nv=8),
            _actuator_qpos_indices=jp.array([7]),
            _actuator_dof_indices=jp.array([6]),
            _query_bvh_reference_clip=query,
        )
        times = np.array([-0.2, 0.13, 0.65])
        qpos, qvel, qacc = sample_reference_states(env, 0, times)
        np.testing.assert_allclose(qpos[:, 0], times**2, atol=1e-7)
        np.testing.assert_allclose(qpos[:, 8], 0.7, atol=1e-7)
        np.testing.assert_allclose(qvel[:, 6], 3 * times**2, atol=1e-7)
        np.testing.assert_allclose(qacc[:, 0], 2.0, atol=1e-7)
        np.testing.assert_allclose(qacc[:, 6], 6 * times, atol=1e-6)
        np.testing.assert_array_equal(qacc[:, 7], 0.0)

    def test_invalid_times_and_legacy_are_rejected(self) -> None:
        env = SimpleNamespace(_continuous_reference=False)
        for times in (np.array([np.nan]), np.ones((2, 2))):
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                sample_reference_states(env, 0, times)
        with self.assertRaisesRegex(ValueError, "consistent_v2"):
            sample_reference_states(env, 0, np.array([0.1]))


class ContactMeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody>
              <geom name="floor" type="plane" size="0 0 .1"/>
              <body pos="0 0 .095"><freejoint/>
                <geom name="foot" type="sphere" size=".1" mass="1"/>
              </body>
            </worldbody></mujoco>
        """)
        self.env = SimpleNamespace(
            _mj_model=self.model,
            _foot_geom_ids_np=np.array([self.model.geom("foot").id]),
            _floor_geom_id=self.model.geom("floor").id,
            FOOT_CONTACT_PRELOAD=0.005,
            _config=SimpleNamespace(reference_foot_lock_height=0.035),
        )
        self.times = np.array([0.0, 0.01, 0.02])
        self.qpos = np.repeat(self.model.qpos0[None], 3, axis=0)
        self.qpos[:, 0] = self.times * 0.2
        self.qvel = np.zeros((3, self.model.nv))
        self.qvel[:, 0] = 0.2

    def test_preload_is_distinct_from_extra_penetration(self) -> None:
        result, rows = inspect_contacts(
            self.env, self.times, self.qpos, self.qvel, subdivisions=2
        )
        self.assertAlmostEqual(result["maximum_penetration_m"], 0.005)
        self.assertAlmostEqual(result["maximum_penetration_beyond_preload_m"], 0.0)
        self.assertEqual(len(rows), 3)
        foot = result["feet"]["foot"]["height_threshold_0.01m"]
        self.assertAlmostEqual(foot["center_speed_m_s"]["median"], 0.2)
        self.assertAlmostEqual(foot["material_point_speed_m_s"]["median"], 0.2)
        self.assertAlmostEqual(
            foot["consecutive_near_floor_node_center_displacement_m"]["median"],
            0.004,
        )

    def test_rotating_support_point_differs_from_center(self) -> None:
        self.qvel[:, 0] = 0.0
        self.qvel[:, 4] = 2.0
        result, _ = inspect_contacts(
            self.env, self.times, self.qpos, self.qvel, subdivisions=2
        )
        foot = result["feet"]["foot"]["height_threshold_0.01m"]
        self.assertAlmostEqual(foot["center_speed_m_s"]["median"], 0.0)
        self.assertAlmostEqual(foot["material_point_speed_m_s"]["median"], 0.2)

    def test_invalid_contact_samples_fail_explicitly(self) -> None:
        for times in (np.array([0.0, 0.0, 0.1]), np.array([0.0, np.nan, 0.1])):
            with self.assertRaises(ValueError):
                inspect_contacts(self.env, times, self.qpos, self.qvel, 2)

    def test_empty_statistic_group_has_nulls_not_nan(self) -> None:
        self.assertEqual(
            summarize(np.array([])),
            {"count": 0, "median": None, "p95": None, "max": None},
        )
        with self.assertRaises(ValueError):
            summarize(np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
