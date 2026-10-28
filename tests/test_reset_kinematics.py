"""Compare optimized reset with full-forward selection on a small contact model."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from biomechanics_env import BiomechanicsJoystickEnv


MODEL_XML = """
<mujoco>
  <option timestep="0.005" iterations="2" ls_iterations="2"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="torso" pos="0 0 1" quat="1 1 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.15 0.08" mass="3"
            contype="0" conaffinity="0"/>
      <body name="left_leg" pos="-0.15 -0.6 0">
        <joint name="hip" axis="1 0 0" damping="0.2"/>
        <geom name="left_sole" type="sphere" pos="0 -0.25 0"
              size="0.08" mass="1"/>
      </body>
      <body name="right_leg" pos="0.15 -0.85 0">
        <geom name="right_sole" type="sphere" size="0.08" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="hip" kp="20"/></actuator>
</mujoco>
"""

EXACT = 0
PROJECTED = 1
LATER_ATTEMPT = 2
INVALID = 3
TIPPED = 4
NONUNIT_QUATERNION = 5


class ResetFixture(BiomechanicsJoystickEnv):
    """Keep production reset/termination/geometry; supply tiny reference samples."""

    def __init__(self, full_forward: bool = False) -> None:
        self._mj_model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
        # This test-only label selects the retained full-forward reset branch
        # without enabling Warp buffer/contact handling for our JAX Data.
        self._physics_backend = "full_forward_oracle" if full_forward else "mjx_jax"
        self._data_kwargs = {"impl": "jax"}
        self._config = {
            "reference_gait": "none",
            "reset_sample_attempts": 2,
            "reset_projection_levels": (1.0, 0.7, 0.25),
        }
        self._init_q = jp.array(self._mj_model.qpos0)
        self._default_ctrl = jp.zeros(1)
        self._init_actuator_noise = jp.array([0.025])
        self._actuator_qpos_indices = jp.array([7], dtype=jp.int32)
        self._torso_body_id = self._mj_model.body("torso").id
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._left_foot_sole_geom_id = self._mj_model.geom("left_sole").id
        self._right_foot_sole_geom_id = self._mj_model.geom("right_sole").id
        self._foot_geom_ids = jp.array(
            [self._left_foot_sole_geom_id, self._right_foot_sole_geom_id]
        )
        self._foot_geom_sizes = jp.array(self._mj_model.geom_size[self._foot_geom_ids])
        self._foot_geom_types = jp.array(self._mj_model.geom_type[self._foot_geom_ids])
        self._refresh_reset_data_template()
        self.scenario = jp.array(EXACT)

    def reset_case(self, rng: jax.Array, scenario: jax.Array):
        """Make scenarios dynamic JIT/vmap inputs without retaining any tracer."""
        instance = copy.copy(self)
        instance.scenario = scenario
        return instance.reset(rng)

    def _sample_weighted_bvh_clip_id(self, rng: jax.Array) -> jax.Array:
        return jax.random.randint(rng, (), 0, 3)

    def _sample_bvh_motion_time(
        self, rng: jax.Array, clip_id: jax.Array
    ) -> jax.Array:
        del clip_id
        return jax.random.uniform(rng)

    def _sample_initial_state_from_reference(
        self,
        clip_id: jax.Array,
        motion_time: jax.Array,
        projection_level: float = 1.0,
        exact: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        qpos = self._init_q.at[2].set(0.92)
        qpos = qpos.at[0].set(0.1 * clip_id)
        qpos = qpos.at[7].set(0.1 + 0.1 * motion_time)
        tipped = (self.scenario == TIPPED) | (
            (self.scenario == LATER_ATTEMPT) & (motion_time < 0.5)
        )
        quaternion = jp.where(tipped, jp.array([1.0, 0.0, 0.0, 0.0]), qpos[3:7])
        quaternion *= jp.where(self.scenario == NONUNIT_QUATERNION, 3.7, 1.0)
        qpos = qpos.at[3:7].set(quaternion)
        if exact:
            qpos = qpos.at[2].set(jp.where(self.scenario == PROJECTED, 0.05, qpos[2]))
        qvel = jp.arange(self._mjx_model.nv, dtype=jp.float32) * 0.04
        qvel = qvel.at[6].set(jp.where(self.scenario == INVALID, jp.nan, qvel[6]))
        ctrl = qpos[7:] * projection_level
        last_action = ctrl + 0.07
        if not exact:
            qpos = self._align_sampled_reference_reset_height(qpos, qvel, ctrl)
        return qpos, qvel, ctrl, last_action

    def sample_command(self, rng: jax.Array) -> jax.Array:
        return jax.random.uniform(rng, (3,))

    def sample_episode_torque_offset(self, rng: jax.Array) -> jax.Array:
        return jax.random.uniform(rng, (1,))

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        return jp.concatenate(
            (data.qpos, data.qvel, data.qacc, data.cvel.ravel(), info["command"])
        )

    def _get_reference_gait_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        del info
        return jp.exp(-jp.square(data.qpos[7]))

    def _get_reference_velocity_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        del info
        return jp.exp(-jp.sum(jp.square(data.qvel)))


def sample_times(rng: jax.Array) -> list[float]:
    """Predict the production key schedule independently of candidate selection."""
    bvh_key = jax.random.split(rng, 6)[5]
    result = []
    for _ in range(2):
        bvh_key, _, time_key, _ = jax.random.split(bvh_key, 4)
        result.append(float(jax.random.uniform(time_key)))
    return result


class ResetKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = ResetFixture()
        cls.oracle_env = ResetFixture(full_forward=True)
        cls.reset = staticmethod(jax.jit(cls.env.reset_case))
        cls.oracle = staticmethod(jax.jit(cls.oracle_env.reset_case))
        cls.key = next(
            key
            for seed in range(100)
            if (times := sample_times(key := jax.random.PRNGKey(seed)))[0] < 0.5
            and times[1] >= 0.5
        )

    def assert_tree_close(self, actual, expected) -> None:
        self.assertEqual(
            jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected)
        )
        actual_leaves, _ = jax.tree_util.tree_flatten_with_path(actual)
        expected_leaves = jax.tree_util.tree_leaves(expected)
        for (path, actual_leaf), expected_leaf in zip(actual_leaves, expected_leaves):
            np.testing.assert_allclose(
                actual_leaf,
                expected_leaf,
                rtol=3e-5,
                atol=2e-5,
                equal_nan=True,
                err_msg=jax.tree_util.keystr(path),
            )

    def test_all_reset_outcomes_match_full_forward_state_and_rng(self) -> None:
        expectations = {
            EXACT: (1, 0, 0),
            PROJECTED: (0, 1, 0),
            LATER_ATTEMPT: (1, 4, 0),
            INVALID: (0, 8, 1),
            TIPPED: (0, 8, 1),
            NONUNIT_QUATERNION: (1, 0, 0),
        }
        for scenario, (exact, rejected, fallback) in expectations.items():
            with self.subTest(scenario=scenario):
                actual = self.reset(self.key, jp.array(scenario))
                expected = self.oracle(self.key, jp.array(scenario))
                self.assert_tree_close(actual, expected)
                self.assertEqual(float(actual.metrics["init_exact_count"]), exact)
                self.assertEqual(float(actual.metrics["init_rejected_count"]), rejected)
                self.assertEqual(float(actual.metrics["init_fallback_count"]), fallback)
                np.testing.assert_array_equal(actual.data.qpos, expected.data.qpos)
                np.testing.assert_array_equal(
                    actual.info["rng"], jax.random.split(self.key, 6)[0]
                )
                self.assertTrue(np.isfinite(np.asarray(actual.obs)).all())
        selected = self.reset(self.key, jp.array(LATER_ATTEMPT))
        self.assertAlmostEqual(
            float(selected.info["bvh_reference_time_offset"]), sample_times(self.key)[1]
        )

    def test_final_forward_populates_contacts_and_accelerations(self) -> None:
        state = self.reset(self.key, jp.array(EXACT))
        self.assertTrue(np.any(np.asarray(state.data._impl.contact.dist) < 0.0))
        self.assertGreater(float(jp.linalg.norm(state.data.qacc)), 0.1)
        self.assertGreater(float(jp.linalg.norm(state.data.actuator_force)), 0.0)
        self.assertTrue(bool(jp.any(self.env._foot_contact(state.data, state.info))))

    def test_height_alignment_matches_full_forward_for_rotated_nonunit_pose(self) -> None:
        qpos = self.env._init_q.at[2].set(1.4).at[7].set(0.6)
        qpos = qpos.at[3:7].multiply(2.3)
        qvel = jp.linspace(-0.4, 0.4, self.env._mjx_model.nv)
        control = jp.array([0.3])
        actual = jax.jit(self.env._align_sampled_reference_reset_height)(
            qpos, qvel, control
        )
        expected = jax.jit(self.oracle_env._align_sampled_reference_reset_height)(
            qpos, qvel, control
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
        np.testing.assert_array_equal(actual[3:], qpos[3:])

    def test_jitted_vmap_preserves_mixed_acceptance_and_fallback(self) -> None:
        scenarios = jp.arange(6)
        keys = jp.broadcast_to(self.key, (6,) + self.key.shape)
        states = jax.jit(jax.vmap(self.env.reset_case))(keys, scenarios)
        for index, scenario in enumerate(scenarios):
            actual = jax.tree_util.tree_map(lambda value: value[index], states)
            self.assert_tree_close(actual, self.oracle(self.key, scenario))

    def test_traced_reset_has_one_full_forward_instead_of_fifteen(self) -> None:
        for env, expected_count in ((self.env, 1), (self.oracle_env, 15)):
            with self.subTest(backend=env._physics_backend):
                with patch("biomechanics_env.mjx.forward", wraps=mjx.forward) as forward:
                    jax.make_jaxpr(lambda key, case: env.reset_case(key, case))(
                        self.key, jp.array(EXACT)
                    )
                self.assertEqual(forward.call_count, expected_count)


if __name__ == "__main__":
    unittest.main()
