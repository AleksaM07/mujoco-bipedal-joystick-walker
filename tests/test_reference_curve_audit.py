"""Check native geometry and coordinate assembly used by the reference audit."""

import unittest
from itertools import product

import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from tools.check_reference_curves import full_state, lowest_foot_point


def make_geom(
    shape: str, size: str, euler: tuple[float, float, float]
) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Build a translated, rotated native geom without loading project assets."""
    angles = " ".join(str(angle) for angle in euler)
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body pos="0.4 -0.3 0.8" euler="{angles}">
              <geom name="foot" type="{shape}" size="{size}" pos="0.1 0.2 -0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, model.geom("foot").id


class LowestFootPointTests(unittest.TestCase):
    def test_box_matches_lowest_brute_force_corner(self) -> None:
        for euler in ((0.0, 0.0, 0.0), (0.3, -0.7, 0.2), (2.1, 0.4, -0.9)):
            with self.subTest(euler=euler):
                model, data, geom_id = make_geom("box", "0.12 0.05 0.03", euler)
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                center = data.geom_xpos[geom_id]
                local_corners = np.array(list(product((-1.0, 1.0), repeat=3)))
                local_corners *= model.geom_size[geom_id]
                world_corners = center + local_corners @ rotation.T
                point = lowest_foot_point(model, data, geom_id)
                self.assertAlmostEqual(point[2], world_corners[:, 2].min(), places=12)
                local_point = rotation.T @ (point - center)
                self.assertTrue(
                    np.all(np.abs(local_point) <= model.geom_size[geom_id] + 1e-12)
                )

    def test_capsule_matches_lowest_end_sphere(self) -> None:
        for euler in (
            (0.0, 0.0, 0.0),
            (0.3, -0.7, 0.2),
            (2.1, 0.4, -0.9),
            (0.0, np.pi / 2, 0.0),
        ):
            with self.subTest(euler=euler):
                model, data, geom_id = make_geom("capsule", "0.04 0.13", euler)
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                center = data.geom_xpos[geom_id]
                radius, half_length = model.geom_size[geom_id, :2]
                endpoints = np.array(
                    [
                        center - half_length * rotation[:, 2],
                        center + half_length * rotation[:, 2],
                    ]
                )
                expected = endpoints[np.argmin(endpoints[:, 2])].copy()
                expected[2] -= radius
                point = lowest_foot_point(model, data, geom_id)
                self.assertAlmostEqual(point[2], expected[2], places=12)
                # A horizontal capsule has many equally low support points.
                if abs(rotation[2, 2]) > 1e-12:
                    np.testing.assert_allclose(point, expected, atol=1e-12)
                local_point = rotation.T @ (point - center)
                segment_point = np.array(
                    [0.0, 0.0, np.clip(local_point[2], -half_length, half_length)]
                )
                self.assertAlmostEqual(
                    np.linalg.norm(local_point - segment_point), radius, places=12
                )

    def test_sphere_bottom_is_rotation_invariant(self) -> None:
        for euler in ((0.0, 0.0, 0.0), (0.3, -0.7, 0.2), (2.1, 0.4, -0.9)):
            with self.subTest(euler=euler):
                model, data, geom_id = make_geom("sphere", "0.04", euler)
                expected = data.geom_xpos[geom_id] - np.array([0.0, 0.0, 0.04])
                np.testing.assert_allclose(
                    lowest_foot_point(model, data, geom_id), expected, atol=1e-12
                )

    def test_unsupported_shapes_fail_instead_of_guessing(self) -> None:
        for shape, size in (("cylinder", "0.04 0.13"), ("ellipsoid", "0.1 0.2 0.3")):
            with self.subTest(shape=shape):
                model, data, geom_id = make_geom(shape, size, (0.3, -0.7, 0.2))
                with self.assertRaisesRegex(ValueError, "Unsupported foot geom type"):
                    lowest_foot_point(model, data, geom_id)


class FullStateTests(unittest.TestCase):
    def test_actuator_order_and_unactuated_pose_are_preserved(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="root">
                  <freejoint/>
                  <geom type="sphere" size="0.1"/>
                  <body>
                    <joint name="first"/>
                    <geom type="sphere" size="0.05"/>
                    <body>
                      <joint name="unactuated"/>
                      <geom type="sphere" size="0.05"/>
                      <body>
                        <joint name="last"/>
                        <geom type="sphere" size="0.05"/>
                      </body>
                    </body>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
        env._mj_model = model
        env._init_q = model.qpos0.copy()
        env._init_q[8] = 0.7
        original = env._init_q.copy()
        # The actuator order intentionally differs from the model's joint order.
        env._actuator_qpos_indices = np.array([9, 7])
        env._actuator_dof_indices = np.array([8, 6])
        reference = {
            "sim_root_pos": np.array([1.0, 2.0, 3.0]),
            "sim_root_quat": np.array([0.5, 0.5, 0.5, 0.5]),
            "sim_root_vel": np.array([0.1, 0.2, 0.3]),
            "sim_root_angvel": np.array([0.4, 0.5, 0.6]),
            "qpos": np.array([0.8, -0.2]),
            "qvel": np.array([-0.3, 0.9]),
        }
        qpos, qvel = full_state(env, reference)
        np.testing.assert_array_equal(
            qpos, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5, -0.2, 0.7, 0.8]
        )
        np.testing.assert_array_equal(
            qvel, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0, -0.3]
        )
        np.testing.assert_array_equal(env._init_q, original)


if __name__ == "__main__":
    unittest.main()
