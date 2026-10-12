"""Collision ablations must be narrow, measurable and leave the model intact."""

import unittest

import mujoco
import numpy as np

from tools.check_reference_collisions import (
    copy_without_foot_pair_collision,
    inspect_collision_geometry,
)


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom name="floor" type="plane" size="0 0 .1"/>
        <body pos="0 0 .095"><freejoint/>
          <geom name="left" type="sphere" size=".1" mass="1"/>
        </body>
        <body pos=".12 0 .095"><freejoint/>
          <geom name="right" type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody></mujoco>
    """)


class ReferenceCollisionTests(unittest.TestCase):
    def test_copied_masks_remove_only_foot_pair(self) -> None:
        model = make_model()
        original = model.geom_contype.copy(), model.geom_conaffinity.copy()
        positions = np.repeat(model.qpos0[None], 2, axis=0)
        velocities = np.zeros((2, model.nv))
        feet = np.array([model.geom("left").id, model.geom("right").id])
        baseline = inspect_collision_geometry(model, positions, velocities)
        copied = copy_without_foot_pair_collision(model, feet, model.geom("floor").id)
        ablated = inspect_collision_geometry(copied, positions, velocities)
        baseline_pairs = {tuple(row["geoms"]): row for row in baseline["pairs"]}
        ablated_pairs = {tuple(row["geoms"]): row for row in ablated["pairs"]}
        pair = baseline_pairs[("left", "right")]
        self.assertAlmostEqual(pair["min_distance_m"], -0.08)
        self.assertEqual(pair["penetrating_samples"], 2)
        self.assertEqual(pair["penetrating_sample_fraction"], 1.0)
        self.assertNotIn(("left", "right"), ablated_pairs)
        for foot in ("left", "right"):
            self.assertAlmostEqual(
                ablated_pairs[("floor", foot)]["min_distance_m"], -0.005
            )
        np.testing.assert_array_equal(model.geom_contype, original[0])
        np.testing.assert_array_equal(model.geom_conaffinity, original[1])

    def test_unsupported_collision_scope_is_rejected(self) -> None:
        model = make_model()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            copy_without_foot_pair_collision(model, np.array([1]), 0)
        model.geom_contype[2] = model.geom_conaffinity[2] = 0
        with self.assertRaisesRegex(ValueError, "exactly two"):
            copy_without_foot_pair_collision(model, np.array([1, 2]), 0)

    def test_invalid_batches_are_rejected(self) -> None:
        model = make_model()
        with self.assertRaises(ValueError):
            inspect_collision_geometry(model, model.qpos0, np.zeros((1, model.nv)))
        positions = model.qpos0[None].copy()
        positions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            inspect_collision_geometry(model, positions, np.zeros((1, model.nv)))


if __name__ == "__main__":
    unittest.main()
