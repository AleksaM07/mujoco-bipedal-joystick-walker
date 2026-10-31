"""Preserve Brax evaluation scores while skipping work after all episodes end."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_training
from mujoco_playground._src import mjx_env, wrapper

from train import (
    BiomechanicsVmapWrapper,
    CompletedEvaluationWrapper,
    make_wrap_env_fn,
    wrap_biomechanics_training,
    wrap_biomechanics_evaluation,
)


class ToyEnvironment:
    """Provide stochastic resets and controllable physical terminations."""

    _config = SimpleNamespace(physics_backend="mjx_jax")

    def reset(self, rng):
        position = jax.random.uniform(rng)
        zero = jnp.array(0.0)
        return mjx_env.State(
            data=position,
            obs=position[None],
            reward=zero,
            done=zero,
            metrics={
                "position": position,
                "reward": zero,
                "terminated": zero,
                "truncated": zero,
            },
            info={},
        )

    def step(self, state, action):
        position = state.data + 1.0
        return state.replace(
            data=position,
            obs=position[None],
            reward=position,
            done=(action[0] < 0).astype(jnp.float32),
            metrics={
                **state.metrics,
                "position": position,
                "reward": position,
                "terminated": jnp.array(0.0),
                "truncated": jnp.array(0.0),
            },
        )


class CountingWrapper(wrapper.Wrapper):
    def __init__(self, env, calls):
        super().__init__(env)
        self.calls = calls

    def step(self, state, action):
        jax.debug.callback(lambda _: self.calls.append(1), jnp.sum(state.done))
        return self.env.step(state, action)


class EvaluationCompletionTests(unittest.TestCase):
    def run_rollout(
        self, actions, episode_length, optimized, action_repeat=1,
        env_factory=ToyEnvironment,
    ):
        calls = []
        if optimized:
            inner = brax_training.EpisodeWrapper(
                BiomechanicsVmapWrapper(env_factory()), episode_length, action_repeat
            )
            env = CompletedEvaluationWrapper(CountingWrapper(inner, calls))
        else:
            env = CountingWrapper(
                wrap_biomechanics_training(
                    env_factory(), episode_length, action_repeat
                ),
                calls,
            )
        env = brax_training.EvalWrapper(env)

        @jax.jit
        def rollout(keys, controls):
            state = env.reset(keys)

            def step(state, control):
                state = env.step(state, control)
                return state, state.info["eval_metrics"]

            return jax.lax.scan(step, state, controls)

        keys = jax.random.split(jax.random.PRNGKey(7), actions.shape[1])
        result = jax.block_until_ready(rollout(keys, actions))
        jax.effects_barrier()
        return result, len(calls)

    def assert_metrics_equal(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(left, right)

    def test_mixed_failures_and_timeout_preserve_every_evaluation_metric(self):
        actions = jnp.zeros((8, 3, 1)).at[1, 0, 0].set(-1).at[2, 1, 0].set(-1)
        reference, baseline_calls = self.run_rollout(actions, 5, False)
        optimized, optimized_calls = self.run_rollout(actions, 5, True)
        self.assert_metrics_equal(optimized[1], reference[1])
        np.testing.assert_array_equal(
            optimized[0].info["eval_metrics"].episode_steps, [2, 3, 5]
        )
        self.assertEqual(baseline_calls, 8)
        self.assertEqual(optimized_calls, 5)
        initial = wrap_biomechanics_training(ToyEnvironment(), 5).reset(
            jax.random.split(jax.random.PRNGKey(7), 3)
        )
        np.testing.assert_allclose(
            optimized[0].data, initial.data + jnp.array([2, 3, 5]), rtol=1e-6
        )

    def test_inactive_worlds_cannot_poison_other_episode_scores(self):
        class PoisonAfterTermination(ToyEnvironment):
            def step(self, state, action):
                stepped = super().step(state, action)
                return stepped.replace(
                    reward=jnp.where(state.done, jnp.nan, stepped.reward),
                    data=jnp.where(state.done, jnp.nan, stepped.data),
                    metrics={
                        name: jnp.where(state.done, jnp.nan, value)
                        for name, value in stepped.metrics.items()
                    },
                )

        actions = jnp.zeros((8, 2, 1)).at[0, 0, 0].set(-1)
        reference, _ = self.run_rollout(
            actions, 5, False, env_factory=PoisonAfterTermination
        )
        optimized, _ = self.run_rollout(
            actions, 5, True, env_factory=PoisonAfterTermination
        )
        self.assert_metrics_equal(optimized[1], reference[1])

    def test_all_first_step_failures_skip_every_later_physics_call(self):
        actions = -jnp.ones((7, 4, 1))
        reference, _ = self.run_rollout(actions, 10, False)
        optimized, calls = self.run_rollout(actions, 10, True)
        self.assert_metrics_equal(optimized[1], reference[1])
        self.assertEqual(calls, 1)

    def test_surviving_episode_keeps_advancing_to_horizon(self):
        actions = jnp.zeros((5, 2, 1)).at[0, 0, 0].set(-1)
        reference, _ = self.run_rollout(actions, 8, False)
        optimized, calls = self.run_rollout(actions, 8, True)
        self.assert_metrics_equal(optimized[1], reference[1])
        self.assertEqual(calls, 5)
        np.testing.assert_array_equal(
            optimized[0].info["eval_metrics"].active_episodes, [0, 1]
        )

    def test_action_repeat_keeps_timeout_counts_and_scores(self):
        actions = jnp.zeros((6, 2, 1))
        reference, _ = self.run_rollout(actions, 6, False, action_repeat=2)
        optimized, calls = self.run_rollout(actions, 6, True, action_repeat=2)
        self.assert_metrics_equal(optimized[1], reference[1])
        self.assertEqual(calls, 3)

    def test_wrapper_is_used_only_for_evaluation(self):
        training_env = ToyEnvironment()
        evaluation_env = ToyEnvironment()
        wrap = make_wrap_env_fn(evaluation_env)
        self.assertNotIsInstance(wrap(training_env), CompletedEvaluationWrapper)
        self.assertIsInstance(wrap(evaluation_env), CompletedEvaluationWrapper)

    def test_warp_backend_keeps_its_existing_evaluation_path(self):
        evaluation_env = ToyEnvironment()
        evaluation_env._config = SimpleNamespace(physics_backend="mjx_warp")
        wrap = make_wrap_env_fn(evaluation_env)
        self.assertNotIsInstance(wrap(evaluation_env), CompletedEvaluationWrapper)

    def test_reset_reactivates_all_episodes(self):
        env = wrap_biomechanics_evaluation(ToyEnvironment())
        keys = jax.random.split(jax.random.PRNGKey(4), 2)
        reset = jax.jit(env.reset)
        state = jax.jit(env.step)(reset(keys), -jnp.ones((2, 1)))
        np.testing.assert_array_equal(state.info[env.ACTIVE_KEY], [False, False])
        fresh = reset(keys)
        np.testing.assert_array_equal(fresh.info[env.ACTIVE_KEY], [True, True])


if __name__ == "__main__":
    unittest.main()
