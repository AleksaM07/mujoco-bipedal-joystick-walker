"""Check the runtime reference, reset, and native MuJoCo kinematics together."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import LoopMode

MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="thorax" pos="0 0 2">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="3" pos="0.1 0.2 0.3"/>
      <site name="marker" pos="0.03 0.05 -0.2"/>
      <body name="pelvis" pos="0.2 -0.1 -0.5">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="2" pos="0.15 0.05 -0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="hinge" kp="20"/></actuator>
</mujoco>
"""


def make_reference_env(
    loop_mode: LoopMode = LoopMode.WRAP, *, single_frame: bool = False
) -> BiomechanicsJoystickEnv:
    """Attach an analytic synthetic clip without loading BVH files or training."""
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    env = object.__new__(BiomechanicsJoystickEnv)
    env._mj_model = model
    env._mjx_model = mjx.put_model(model, impl="jax")
    env._reference_kinematics_model = env._mjx_model
    env._reference_kinematics_data = mjx.make_data(env._mjx_model)
    env._config = {
        "reference_gait": "bvh",
        "recovery_physics_version": "consistent_v2",
    }
    env._consistent_physics = True
    env._continuous_reference = True
    env._init_q = jp.array(model.qpos0)
    env._actuator_qpos_indices = jp.array([7], dtype=jp.int32)
    env._actuator_dof_indices = jp.array([6], dtype=jp.int32)
    env._body_root_ids = jp.array(model.body_rootid)
    env._reference_anchor_body_id = model.body("pelvis").id
    env._torso_body_id = model.body("thorax").id
    env._deepmimic_key_body_ids = jp.array([env._reference_anchor_body_id, 0])
    env._deepmimic_key_site_ids = jp.array([0, model.site("marker").id])
    env._deepmimic_key_is_site = jp.array([False, True])

    times = np.linspace(0.0, 1.0, 5)
    phase = 2.0 * np.pi * times
    joint_positions = 0.2 * np.sin(phase)[:, None]
    joint_tangents = (0.4 * np.pi * np.cos(phase))[:, None]
    roots = np.column_stack((times, 0.03 * np.sin(phase), 2 + 0.02 * np.cos(phase)))
    root_tangents = np.column_stack(
        (
            np.ones_like(times),
            0.06 * np.pi * np.cos(phase),
            -0.04 * np.pi * np.sin(phase),
        )
    )
    axis = np.array([0.4, 0.8, -0.3])
    axis /= np.linalg.norm(axis)
    angles = 0.15 * np.sin(phase) + 0.1 * np.cos(2 * phase)
    angular_speeds = 0.3 * np.pi * np.cos(phase) - 0.4 * np.pi * np.sin(2 * phase)
    quaternions = np.column_stack(
        (np.cos(angles / 2), np.sin(angles / 2)[:, None] * axis)
    )
    quaternion_tangents = (
        0.5
        * angular_speeds[:, None]
        * np.column_stack((-np.sin(angles / 2), np.cos(angles / 2)[:, None] * axis))
    )
    sample_slice = slice(0, 1) if single_frame else slice(None)
    for attribute, values in (
        ("_bvh_reference_qpos_targets", joint_positions),
        ("_bvh_reference_sim_root_pos_targets", roots),
        ("_bvh_reference_sim_root_quat_targets", quaternions),
        ("_reference_joint_tangents", joint_tangents),
        ("_reference_root_tangents", root_tangents),
        ("_reference_quaternion_tangents", quaternion_tangents),
    ):
        setattr(env, attribute, jp.array(values[sample_slice][None], dtype=jp.float32))
    env._bvh_reference_frame_counts = jp.array([1 if single_frame else 5])
    env._bvh_reference_frame_times = jp.array([0.25])
    env._bvh_reference_motion_lengths = jp.array([0.0 if single_frame else 1.0])
    env._bvh_reference_loop_modes = jp.array([int(loop_mode)])
    env._bvh_reference_wrap_deltas = jp.array(
        [[0.0 if single_frame else 1.0, 0.0, 0.0]]
    )
    return env


def full_state(reference: dict[str, jax.Array]) -> tuple[np.ndarray, np.ndarray]:
    """Build free-root native MuJoCo coordinates from the queried reference."""
    qpos = np.concatenate(
        (reference["sim_root_pos"], reference["sim_root_quat"], reference["qpos"])
    ).astype(np.float64)
    qvel = np.concatenate(
        (reference["sim_root_vel"], reference["sim_root_angvel"], reference["qvel"])
    ).astype(np.float64)
    return qpos, qvel


class RecoveryInterpolationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = make_reference_env()
        cls.query = staticmethod(
            jax.jit(lambda time: cls.env._query_bvh_reference_clip(jp.array(0), time))
        )

    def test_reported_velocity_matches_runtime_pose_finite_difference(self) -> None:
        step = 2e-4
        for time in (-1.37, -0.012, 0.071, 0.375, 0.998, 1.003, 2.43):
            with self.subTest(time=time):
                _, velocity = full_state(self.query(jp.array(time)))
                before, _ = full_state(self.query(jp.array(time - step)))
                after, _ = full_state(self.query(jp.array(time + step)))
                difference = np.zeros(self.env._mj_model.nv)
                mujoco.mj_differentiatePos(
                    self.env._mj_model, difference, 2 * step, before, after
                )
                np.testing.assert_allclose(velocity, difference, atol=1e-3, rtol=2e-3)

    def test_anchor_and_key_markers_match_native_forward_kinematics(self) -> None:
        model = self.env._mj_model
        for time in (0.113, 0.413, 1.157):
            with self.subTest(time=time):
                reference = self.query(jp.array(time))
                qpos, qvel = full_state(reference)
                data = mujoco.MjData(model)
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(model, data)
                body_id = self.env._reference_anchor_body_id
                jacobian = np.zeros((3, model.nv))
                angular_jacobian = np.zeros_like(jacobian)
                mujoco.mj_jacBody(model, data, jacobian, angular_jacobian, body_id)
                np.testing.assert_allclose(
                    reference["root_pos"], data.xpos[body_id], atol=2e-6
                )
                np.testing.assert_allclose(
                    reference["root_quat"], data.xquat[body_id], atol=2e-6
                )
                np.testing.assert_allclose(
                    reference["root_vel"], jacobian @ qvel, atol=2e-6
                )
                np.testing.assert_allclose(
                    reference["root_angvel"], angular_jacobian @ qvel, atol=2e-6
                )
                np.testing.assert_allclose(
                    reference["key_pos"],
                    np.stack((data.xpos[body_id], data.site_xpos[0])),
                    atol=2e-6,
                )

    def test_wraps_accumulate_translation_for_positive_and_negative_cycles(
        self,
    ) -> None:
        base = self.query(jp.array(0.137))
        for cycle in (-3, -1, 1, 4):
            with self.subTest(cycle=cycle):
                reference = self.query(jp.array(cycle + 0.137))
                self.assertEqual(float(reference["loop_count"]), cycle)
                for name in (
                    "qpos",
                    "qvel",
                    "sim_root_quat",
                    "sim_root_vel",
                    "sim_root_angvel",
                ):
                    np.testing.assert_allclose(reference[name], base[name], atol=4e-6)
                for name in ("sim_root_pos", "root_pos", "key_pos"):
                    np.testing.assert_allclose(
                        reference[name], base[name] + np.array([cycle, 0, 0]), atol=4e-6
                    )

    def test_pose_and_velocity_are_continuous_at_loop_seam(self) -> None:
        step = 1e-5
        before = self.query(jp.array(1.0 - step))
        after = self.query(jp.array(1.0 + step))
        for name in (
            "qpos",
            "qvel",
            "sim_root_pos",
            "sim_root_quat",
            "sim_root_vel",
            "sim_root_angvel",
        ):
            with self.subTest(field=name):
                np.testing.assert_allclose(before[name], after[name], atol=4e-4)

    def test_clamped_reference_is_stationary_outside_motion_interval(self) -> None:
        env = make_reference_env(LoopMode.CLAMP)
        query = jax.jit(lambda time: env._query_bvh_reference_clip(jp.array(0), time))
        for outside, endpoint in ((-0.25, 0.0), (1.25, 1.0)):
            reference = query(jp.array(outside))
            end_reference = query(jp.array(endpoint))
            for name in (
                "qpos",
                "sim_root_pos",
                "sim_root_quat",
                "root_pos",
                "key_pos",
            ):
                np.testing.assert_allclose(
                    reference[name], end_reference[name], atol=2e-6
                )
            for name in (
                "qvel",
                "sim_root_vel",
                "sim_root_angvel",
                "root_vel",
                "root_angvel",
            ):
                np.testing.assert_array_equal(
                    reference[name], np.zeros_like(reference[name])
                )

    def test_single_frame_clip_has_zero_velocity_and_no_nan(self) -> None:
        for mode in (LoopMode.WRAP, LoopMode.CLAMP):
            env = make_reference_env(mode, single_frame=True)
            query = jax.jit(
                lambda time: env._query_bvh_reference_clip(jp.array(0), time)
            )
            for time in (-1.0, 0.0, 3.0):
                reference = query(jp.array(time))
                for value in reference.values():
                    self.assertTrue(np.isfinite(np.asarray(value)).all())
                for name in (
                    "qvel",
                    "sim_root_vel",
                    "sim_root_angvel",
                    "root_vel",
                    "root_angvel",
                ):
                    np.testing.assert_array_equal(
                        reference[name], np.zeros_like(reference[name])
                    )

    def test_exact_reset_uses_the_queried_pose_without_height_realign(self) -> None:
        time = jp.array(0.137)
        reference = self.query(time)
        expected_qpos, expected_qvel = full_state(reference)
        with patch.object(
            self.env,
            "_align_sampled_reference_reset_height",
            side_effect=AssertionError(
                "Exact continuous-reference reset must not realign"
            ),
        ):
            qpos, qvel, control, action = self.env._sample_initial_state_from_reference(
                jp.array(0), time, exact=True
            )
        np.testing.assert_allclose(qpos, expected_qpos, atol=2e-6)
        np.testing.assert_allclose(qvel, expected_qvel, atol=2e-6)
        np.testing.assert_allclose(control, reference["qpos"], atol=2e-6)
        np.testing.assert_array_equal(action, np.zeros(self.env.action_size))
        for suffix in ("pos", "quat", "vel", "angvel"):
            np.testing.assert_array_equal(
                reference[f"reset_root_{suffix}"], reference[f"sim_root_{suffix}"]
            )

    def test_runtime_query_supports_jit_and_vmap(self) -> None:
        times = jp.array([-0.1, 0.07, 0.41, 1.2])
        batched = jax.jit(jax.vmap(self.query))(times)
        for index, time in enumerate(times):
            expected = self.query(time)
            for name in expected:
                np.testing.assert_allclose(
                    batched[name][index], expected[name], atol=3e-6
                )

    def test_legacy_and_v1_keep_stored_velocity_interpolation(self) -> None:
        env = make_reference_env()
        env._continuous_reference = False
        env._bvh_reference_qvel_targets = jp.full((1, 5, 1), 7.25)
        roots = env._bvh_reference_sim_root_pos_targets
        quaternions = env._bvh_reference_sim_root_quat_targets
        velocity = jp.broadcast_to(jp.array([5.0, 6.0, 7.0]), (1, 5, 3))
        angular_velocity = jp.broadcast_to(jp.array([1.0, 2.0, 3.0]), (1, 5, 3))
        for prefix in ("root", "sim_root", "reset_root"):
            for suffix, values in (
                ("pos", roots),
                ("quat", quaternions),
                ("vel", velocity),
                ("angvel", angular_velocity),
            ):
                setattr(env, f"_bvh_reference_{prefix}_{suffix}_targets", values)
        env._bvh_reference_key_rel_local_targets = jp.zeros((1, 5, 2, 3))
        for version in ("legacy", "consistent_v1"):
            env._config["recovery_physics_version"] = version
            env._consistent_physics = version != "legacy"
            with (
                self.subTest(version=version),
                patch.object(
                    env,
                    "_query_bvh_reference_clip_consistent",
                    side_effect=AssertionError("Compatibility query must not use v2"),
                ),
            ):
                reference = env._query_bvh_reference_clip(jp.array(0), jp.array(0.125))
                np.testing.assert_allclose(reference["qpos"], [0.1], atol=1e-7)
                np.testing.assert_array_equal(reference["qvel"], [7.25])
                np.testing.assert_array_equal(
                    reference["sim_root_vel"], [5.0, 6.0, 7.0]
                )


if __name__ == "__main__":
    unittest.main()
