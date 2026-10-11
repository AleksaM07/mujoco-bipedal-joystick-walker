"""Check live and reference-FK measurements against native MuJoCo Jacobians."""

import unittest
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv

MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="thorax" pos="0 0 2">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="3" pos="0.1 0.2 0.3"/>
      <site name="marker" pos="0 0 0"/>
      <body name="pelvis" pos="0.2 -0.1 -0.5">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="2" pos="0.15 0.05 -0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="hinge" kp="20"/></actuator>
</mujoco>
"""
V21_XML = Path(__file__).resolve().parents[1] / (
    "generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml"
)


def make_measurement_env(model: mujoco.MjModel) -> BiomechanicsJoystickEnv:
    """Attach a native model to actual methods without compiling a simulator."""
    env = object.__new__(BiomechanicsJoystickEnv)
    env._mj_model = model
    env._consistent_physics = True
    env._body_root_ids = jp.array(model.body_rootid)
    env._torso_body_id = model.body("thorax").id
    env._reference_anchor_body_id = model.body("pelvis").id
    env._deepmimic_body_ids_np = np.arange(1, model.nbody)
    env._deepmimic_body_ids = jp.array(env._deepmimic_body_ids_np)
    masses = model.body_mass[env._deepmimic_body_ids_np]
    env._deepmimic_body_weights_np = masses / masses.sum()
    env._deepmimic_body_weights = jp.array(env._deepmimic_body_weights_np)
    return env


class RecoveryMeasurementTests(unittest.TestCase):
    def check_measurements(self, model: mujoco.MjModel) -> None:
        env = make_measurement_env(model)
        random = np.random.default_rng(7)
        for case in range(8):
            with self.subTest(case=case):
                data = mujoco.MjData(model)
                data.qpos[:] = model.qpos0
                data.qpos[2] += 1.0
                quat = random.normal(size=4)
                data.qpos[3:7] = quat / np.linalg.norm(quat)
                data.qpos[7:] += random.normal(0.0, 0.15, model.nq - 7)
                data.qvel[:] = random.normal(0.0, 0.4, model.nv)
                mujoco.mj_forward(model, data)
                mujoco.mj_subtreeVel(model, data)
                jacobian = np.zeros((3, model.nv))
                rotational_jacobian = np.zeros_like(jacobian)
                mujoco.mj_jacBody(
                    model,
                    data,
                    jacobian,
                    rotational_jacobian,
                    env._reference_anchor_body_id,
                )
                anchor_velocity = jacobian @ data.qvel
                np.testing.assert_allclose(
                    env._reference_anchor_linvel(data), anchor_velocity, atol=3e-6
                )
                mujoco.mj_jacBody(
                    model, data, jacobian, rotational_jacobian, env._torso_body_id
                )
                rotation = data.xmat[env._torso_body_id].reshape(3, 3)
                gyro = rotation.T @ (rotational_jacobian @ data.qvel)
                np.testing.assert_allclose(
                    env._local_root_angvel(data), gyro, atol=3e-6
                )
                np.testing.assert_allclose(
                    env._measured_command(data)[2],
                    (rotational_jacobian @ data.qvel)[2],
                    atol=3e-6,
                )
                center = np.average(data.xipos[1:], axis=0, weights=model.body_mass[1:])
                np.testing.assert_allclose(env._center_of_mass(data), center, atol=3e-6)
                np.testing.assert_allclose(
                    env._center_of_mass_velocity(data),
                    data.subtree_linvel[env._torso_body_id],
                    atol=3e-6,
                )
                samples = []
                timestep = 1e-6
                for direction in (-1.0, 1.0):
                    offset = mujoco.MjData(model)
                    offset.qpos[:] = data.qpos
                    mujoco.mj_integratePos(
                        model, offset.qpos, data.qvel, direction * timestep
                    )
                    mujoco.mj_forward(model, offset)
                    samples.append(
                        (
                            offset.xpos[env._reference_anchor_body_id].copy(),
                            np.average(
                                offset.xipos[1:], axis=0, weights=model.body_mass[1:]
                            ),
                        )
                    )
                derivative = (np.asarray(samples[1]) - samples[0]) / (2 * timestep)
                np.testing.assert_allclose(derivative[0], anchor_velocity, atol=1e-7)
                np.testing.assert_allclose(
                    derivative[1], env._center_of_mass_velocity(data), atol=3e-6
                )
                # The compatibility path intentionally retains old formulas.
                env._consistent_physics = False
                np.testing.assert_allclose(
                    env._local_root_angvel(data), rotation.T @ data.qvel[3:6], atol=3e-6
                )
                np.testing.assert_array_equal(
                    env._reference_anchor_linvel(data),
                    data.cvel[env._reference_anchor_body_id, 3:6],
                )
                np.testing.assert_allclose(
                    env._center_of_mass(data),
                    np.average(data.xpos[1:], axis=0, weights=model.body_mass[1:]),
                    atol=3e-6,
                )
                np.testing.assert_allclose(
                    env._center_of_mass_velocity(data),
                    np.average(data.cvel[1:, 3:6], axis=0, weights=model.body_mass[1:]),
                    atol=3e-6,
                )
                self.assertAlmostEqual(
                    float(env._measured_command(data)[2]), data.qvel[5], places=6
                )
                env._consistent_physics = True

    def test_native_jacobians_and_finite_differences(self) -> None:
        self.check_measurements(mujoco.MjModel.from_xml_string(MODEL_XML))

    @unittest.skipUnless(V21_XML.exists(), "Optional local v21 XML is not generated")
    def test_actual_v21_jacobians_and_finite_differences(self) -> None:
        self.check_measurements(mujoco.MjModel.from_xml_path(str(V21_XML)))

    def test_reference_fk_and_live_measurements_agree(self) -> None:
        model = mujoco.MjModel.from_xml_string(MODEL_XML)
        env = make_measurement_env(model)
        env._init_q_np = model.qpos0.copy()
        env._actuator_qpos_indices_np = np.array([7])
        env._actuator_dof_indices_np = np.array([6])
        env._deepmimic_key_count = 1
        env._deepmimic_key_body_ids_np = np.array([env._reference_anchor_body_id])
        env._deepmimic_key_site_ids_np = np.array([0])
        env._deepmimic_key_is_site_np = np.array([False])
        env._reference_anchor_body_local_index = env._reference_anchor_body_id - 1
        env._bvh_reference_frame_times = jp.array([0.1])
        env._bvh_reference_wrap_deltas_np = np.zeros((1, 3))
        qpos = np.array([[[0.1], [0.2], [0.3]]], dtype=np.float32)
        qvel = np.ones_like(qpos)
        roots = np.array([[[0.0, 0.0, 2.0], [0.1, 0.2, 2.0], [0.2, 0.4, 2.0]]])
        root_quats = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (1, 3, 1))
        root_vels = np.tile([1.0, 2.0, 0.0], (1, 3, 1))
        root_angvels = np.tile([0.4, 0.7, -0.2], (1, 3, 1))
        env._configure_deepmimic_reference_from_qpos(
            qpos, qvel, roots, root_quats, root_vels, root_angvels
        )
        for frame in range(3):
            data = mujoco.MjData(model)
            data.qpos[:] = np.concatenate(
                (roots[0, frame], root_quats[0, frame], qpos[0, frame])
            )
            data.qvel[:] = np.concatenate(
                (root_vels[0, frame], root_angvels[0, frame], qvel[0, frame])
            )
            mujoco.mj_forward(model, data)
            for stored_name, actual in (
                ("root_vel", env._reference_anchor_linvel(data)),
                ("root_angvel", env._reference_anchor_angvel(data)),
                ("com", env._center_of_mass(data)),
                ("com_vel", env._center_of_mass_velocity(data)),
            ):
                np.testing.assert_allclose(
                    getattr(env, f"_bvh_reference_{stored_name}_targets")[0, frame],
                    actual,
                    atol=3e-6,
                )

    def test_corrected_measurements_can_be_jitted(self) -> None:
        from types import SimpleNamespace

        model = mujoco.MjModel.from_xml_string(MODEL_XML)
        env = make_measurement_env(model)
        data = mujoco.MjData(model)
        data.qvel[:] = 0.3
        mujoco.mj_forward(model, data)

        def measure(cvel, xipos, subtree_com):
            state = SimpleNamespace(cvel=cvel, xipos=xipos, subtree_com=subtree_com)
            return env._center_of_mass_velocity(state)

        result = jax.jit(measure)(
            jp.array(data.cvel), jp.array(data.xipos), jp.array(data.subtree_com)
        )
        np.testing.assert_allclose(
            result, env._center_of_mass_velocity(data), atol=3e-6
        )


if __name__ == "__main__":
    unittest.main()
