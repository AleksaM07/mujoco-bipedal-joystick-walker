"""Exercise free-root replay on tiny native models without loading BVH files."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import jax.numpy as jp
import mujoco
import numpy as np

from tools.recovery_replay import _floor_forces, replay_reference


def make_env(*, gravity: float = 0, moving_target: bool = False) -> SimpleNamespace:
    """Build a freely floating root with a scalar, force-limited position motor."""
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <option gravity="0 0 {gravity}" timestep="0.01"/>
          <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1"/>
            <body name="torso" pos="0 0 2" quat="1 1 0 0">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="10"/>
              <body pos="0 0 0.3">
                <joint name="hinge" axis="0 0 1"/>
                <geom name="foot" type="sphere" pos="0.2 0 0" size="0.1" mass="1"/>
              </body>
            </body>
          </worldbody>
          <actuator><position name="motor" joint="hinge" kp="20" kv="0.1"
            forcelimited="true" forcerange="-0.01 0.01"/>
          </actuator>
        </mujoco>"""
    )

    def reference(clip_id, time):
        del clip_id
        return {
            "qpos": jp.array([0.2 * time if moving_target else 0.0]),
            "qvel": jp.array([0.2 if moving_target else 0.0]),
            "sim_root_pos": jp.array(model.qpos0[:3]),
            "sim_root_quat": jp.array(model.qpos0[3:7]),
            "sim_root_vel": jp.zeros(3),
            "sim_root_angvel": jp.zeros(3),
        }

    return SimpleNamespace(
        _mj_model=model,
        _init_q=jp.array(model.qpos0),
        _actuator_qpos_indices=np.array([7]),
        _actuator_dof_indices=np.array([6]),
        _bvh_reference_motion_lengths=np.array([1.0]),
        _bvh_reference_loop_modes=np.array([1]),
        _query_bvh_reference_clip=reference,
        _standing_height=lambda: model.qpos0[2],
        MIN_STANDING_HEIGHT_RATIO=0.3,
        _torso_body_id=model.body("torso").id,
        _floor_geom_id=model.geom("floor").id,
        _foot_geom_ids_np=np.array([model.geom("foot").id]),
    )


class RecoveryReplayTests(unittest.TestCase):
    def test_stationary_zero_gravity_survives_and_does_not_mutate_model(self):
        env = make_env()
        gains = env._mj_model.actuator_gainprm.copy()
        qpos0 = env._mj_model.qpos0.copy()
        result = replay_reference(env, 0, cycles=0.2)
        self.assertTrue(result["all_phases_completed"])
        self.assertEqual(len(result["runs"]), 4)
        for run in result["runs"]:
            self.assertAlmostEqual(run["simulated_seconds"], 0.2)
            self.assertAlmostEqual(run["joint_position_rmse_rad"], 0.0)
            self.assertAlmostEqual(run["root_position_rmse_m"], 0.0)
            self.assertEqual(run["actuator_force_limit_sample_fraction"], [0.0])
        np.testing.assert_array_equal(env._mj_model.actuator_gainprm, gains)
        np.testing.assert_array_equal(env._mj_model.qpos0, qpos0)
        json.dumps(result, allow_nan=False)

    def test_free_root_falls_in_gravity_and_stops_before_horizon(self):
        result = replay_reference(
            make_env(gravity=-9.81), 0, cycles=2, start_phases=(0,)
        )
        self.assertFalse(result["all_phases_completed"])
        run = result["runs"][0]
        self.assertLess(run["simulated_seconds"], 1)
        self.assertIn("root_height_below_environment_threshold", run["stop_reasons"])
        self.assertLess(run["final_free_root_position_error_xyz_m"][2], -1)

    def test_force_saturation_is_reported_only_for_limited_actuators(self):
        env = make_env(moving_target=True)
        limited = replay_reference(
            env, 0, cycles=0.2, start_phases=(0,), target_lookahead_seconds=0.1
        )
        self.assertGreater(
            limited["runs"][0]["actuator_force_limit_sample_fraction"][0], 0
        )
        env._mj_model.actuator_forcelimited[0] = 0
        unlimited = replay_reference(
            env, 0, cycles=0.2, start_phases=(0,), target_lookahead_seconds=0.1
        )
        self.assertEqual(unlimited["actuator_force_limited"], [False])
        self.assertEqual(
            unlimited["runs"][0]["actuator_force_limit_sample_fraction"], [0]
        )

    def test_control_hold_and_lookahead_are_explicit(self):
        result = replay_reference(
            make_env(),
            0,
            cycles=0.1,
            start_phases=(0,),
            control_dt=0.02,
            target_lookahead_seconds=0.02,
        )
        self.assertEqual(result["control_hold_seconds"], 0.02)
        self.assertEqual(result["target_lookahead_seconds"], 0.02)
        self.assertEqual(result["runs"][0]["simulated_steps"], 10)

    def test_solved_floor_forces_separate_feet_from_other_bodies(self):
        env = make_env(gravity=-9.81)
        model = env._mj_model
        data = mujoco.MjData(model)
        data.qpos[2] = 0.05
        mujoco.mj_forward(model, data)
        foot_force, nonfoot_force, contacts = _floor_forces(
            model, data, env._floor_geom_id, set(env._foot_geom_ids_np)
        )
        self.assertGreater(foot_force, 0)
        self.assertGreater(nonfoot_force, 0)
        self.assertGreaterEqual(contacts, 2)

    def test_bad_initial_pose_is_rejected_without_taking_a_step(self):
        env = make_env()
        env._mj_model.qpos0[3:7] = [1, 0, 0, 0]
        result = replay_reference(env, 0, cycles=0.1, start_phases=(0,))
        self.assertFalse(result["all_phases_completed"])
        self.assertEqual(result["runs"][0]["simulated_steps"], 0)
        self.assertIsNone(result["runs"][0]["joint_position_rmse_rad"])

    def test_rejects_nonunit_gear_and_nonposition_motor(self):
        env = make_env()
        env._mj_model.actuator_gear[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "gear"):
            replay_reference(env, 0)
        env._mj_model.actuator_gear[0, 0] = 1
        env._mj_model.actuator_biasprm[0, 1] = 0
        with self.assertRaisesRegex(ValueError, "position actuators"):
            replay_reference(env, 0)

    def test_rejects_invalid_inputs(self):
        for kwargs in (
            {"cycles": 0},
            {"cycles": float("nan")},
            {"start_phases": ()},
            {"start_phases": (-0.1,)},
            {"start_phases": (1,)},
            {"control_dt": 0.015},
            {"control_dt": 0.001},
            {"target_lookahead_seconds": -0.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replay_reference(make_env(), 0, **kwargs)
        with self.assertRaisesRegex(ValueError, "clip_id"):
            replay_reference(make_env(), -1)
        with self.assertRaisesRegex(TypeError, "clip_id"):
            replay_reference(make_env(), 0.5)
        env = make_env()
        env._bvh_reference_loop_modes[0] = 0
        with self.assertRaisesRegex(ValueError, "WRAP"):
            replay_reference(env, 0)


if __name__ == "__main__":
    unittest.main()
