"""Cross-check stand reset, preview, control and reward against the same pose."""

import unittest
from dataclasses import replace

import jax
import jax.numpy as jp
import numpy as np

from evaluate import make_environment
from skills.stand import StandEnv
from tools.check_stand import stand_config
from train import make_environment as make_training_environment
from train import wrap_biomechanics_evaluation


class StandEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_environment(stand_config())
        cls.reset = staticmethod(jax.jit(cls.env.reset))
        cls.step = staticmethod(jax.jit(cls.env.step))

    def test_reset_targets_and_previews_are_stationary_and_consistent(self):
        env = self.env
        state = self.reset(jax.random.PRNGKey(11))
        self.assertEqual(env.action_size, 26)
        self.assertEqual(state.obs["state"].shape, (366,))
        self.assertEqual(float(state.metrics["init_fallback_count"]), 0)
        self.assertEqual(int(state.info["stand_dwell_steps"]), 0)
        np.testing.assert_allclose(state.data.qvel, 0, atol=1e-7)
        np.testing.assert_array_equal(state.info["command"], np.zeros(3))
        query = jax.jit(env._query_bvh_reference)
        ref = query(state.info, 0)
        future_info = {**state.info, "motion_step": jp.array(10000)}
        later = query(future_info, 200)
        for key in ("qpos", "qvel", "root_pos", "root_vel", "key_pos"):
            np.testing.assert_allclose(ref[key], later[key], atol=1e-7)
        np.testing.assert_allclose(
            state.data.qpos[env._actuator_qpos_indices], ref["qpos"], atol=1e-6
        )
        targets = env._policy_action_to_motor_targets(jp.zeros(26), state.info)
        np.testing.assert_allclose(targets, ref["qpos"], atol=1e-6)
        self.assertFalse(bool(env._get_bvh_motion_over(future_info)))
        self.assertEqual(float(env._get_gait_phase_angle(future_info)), 0)
        self.assertGreater(float(state.metrics["deepmimic_pose"]), 0.999)

    def test_step_preserves_jax_tree_and_does_not_end_at_static_clip_length(self):
        state = self.reset(jax.random.PRNGKey(22))
        original_tree = jax.tree_util.tree_structure(state)
        for _ in range(3):
            state = self.step(state, jp.zeros(26))
        self.assertEqual(original_tree, jax.tree_util.tree_structure(state))
        self.assertEqual(float(state.done), 0)
        self.assertEqual(float(state.metrics["done_motion_over"]), 0)
        self.assertTrue(np.isfinite(float(state.reward)))
        self.assertTrue(all(np.isfinite(value).all() for value in state.obs.values()))

    def test_train_and_viewer_select_same_skill_and_layout(self):
        config = stand_config()
        training_env = make_training_environment(config, enable_erfi=False)
        self.assertIsInstance(training_env, StandEnv)
        np.testing.assert_allclose(training_env._init_q, self.env._init_q)
        np.testing.assert_allclose(training_env._action_scale, self.env._action_scale)
        with self.assertRaisesRegex(ValueError, "locomotion_skill"):
            make_environment(replace(config, locomotion_skill="typo"))

    def test_batched_evaluator_preserves_metric_dtypes(self):
        wrapped = wrap_biomechanics_evaluation(self.env, episode_length=500)
        keys = jax.random.split(jax.random.PRNGKey(7), 2)
        state = jax.eval_shape(wrapped.reset, keys)
        stepped = jax.eval_shape(wrapped.step, state, jp.zeros((2, 26)))
        self.assertEqual(state.info["stand_dwell_steps"].dtype, jp.int32)
        self.assertEqual(stepped.metrics["stand_dwell_steps"].dtype, jp.float32)


if __name__ == "__main__":
    unittest.main()
