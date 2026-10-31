"""Check terminal transition bookkeeping through the production PPO wrappers."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_training
from mujoco_playground._src import mjx_env

from train import (
    BiomechanicsVmapWrapper,
    ConditionalAutoResetWrapper,
    wrap_biomechanics_training,
)


class ToyEnvironment:
    """Use random resets and action-triggered failures without a simulator."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        position = jax.random.uniform(rng)
        zero = jnp.array(0.0)
        return mjx_env.State(
            data=position,
            obs=position[None],
            reward=zero,
            done=zero,
            metrics={"progress": zero, "terminated": zero, "truncated": zero},
            info={"reset_marker": position, "age": zero},
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        position = state.data + 1.0
        return state.replace(
            data=position,
            obs=position[None],
            reward=jnp.array(2.0),
            done=(action[0] < 0).astype(jnp.float32),
            metrics={
                "progress": jnp.array(1.0),
                "terminated": jnp.array(0.0),
                "truncated": jnp.array(0.0),
            },
            info={**state.info, "age": state.info["age"] + 1.0},
        )


class TrainingAutoResetTests(unittest.TestCase):
    def make_environment(self, episode_length: int, batch_size: int = 1):
        env = wrap_biomechanics_training(
            ToyEnvironment(), episode_length=episode_length
        )
        state = env.reset(jax.random.split(jax.random.PRNGKey(7), batch_size))
        return jax.jit(env.step), state

    def test_timeout_preserves_terminal_transition_and_next_episode_restarts(self):
        step, state = self.make_environment(episode_length=2)
        actions = jnp.zeros((1, 1))
        state = step(state, actions)
        terminal = step(state, actions)

        np.testing.assert_array_equal(terminal.done, [1])
        np.testing.assert_array_equal(terminal.info["truncation"], [1])
        np.testing.assert_array_equal(terminal.info["episode_done"], [1])
        np.testing.assert_array_equal(terminal.info["episode_metrics"]["length"], [2])
        np.testing.assert_array_equal(
            terminal.info["episode_metrics"]["sum_reward"], [4]
        )
        np.testing.assert_array_equal(terminal.metrics["terminated"], [0])
        np.testing.assert_array_equal(terminal.metrics["truncated"], [1])
        np.testing.assert_array_equal(terminal.info["age"], [0])
        np.testing.assert_array_equal(terminal.info["steps"], [2])

        next_state = step(terminal, actions)
        np.testing.assert_array_equal(next_state.done, [0])
        np.testing.assert_array_equal(next_state.info["truncation"], [0])
        np.testing.assert_array_equal(next_state.info["episode_done"], [0])
        np.testing.assert_array_equal(
            next_state.info["episode_metrics"]["length"], [1]
        )
        np.testing.assert_array_equal(
            next_state.info["episode_metrics"]["sum_reward"], [2]
        )
        np.testing.assert_array_equal(next_state.info["steps"], [1])

    def test_physical_failure_remains_a_terminal_for_ppo(self):
        step, state = self.make_environment(episode_length=5)
        terminal = step(state, -jnp.ones((1, 1)))

        np.testing.assert_array_equal(terminal.done, [1])
        np.testing.assert_array_equal(terminal.info["truncation"], [0])
        np.testing.assert_array_equal(terminal.info["episode_done"], [1])
        np.testing.assert_array_equal(terminal.info["episode_metrics"]["length"], [1])
        np.testing.assert_array_equal(
            terminal.info["episode_metrics"]["sum_reward"], [2]
        )
        np.testing.assert_array_equal(terminal.metrics["terminated"], [1])
        np.testing.assert_array_equal(terminal.metrics["truncated"], [0])
        np.testing.assert_array_equal(terminal.info["age"], [0])

    def test_mixed_batch_preserves_timeout_failure_and_continuing_worlds(self):
        step, state = self.make_environment(episode_length=2, batch_size=3)
        state.info["steps"] = jnp.array([1.0, 0.0, 0.0])
        state.info["episode_metrics"]["length"] = jnp.array([1.0, 0.0, 0.0])
        state.info["episode_metrics"]["sum_reward"] = jnp.array([2.0, 0.0, 0.0])
        old_marker = np.asarray(state.info["reset_marker"])
        next_state = step(state, jnp.array([[0.0], [-1.0], [0.0]]))

        np.testing.assert_array_equal(next_state.done, [1, 1, 0])
        np.testing.assert_array_equal(next_state.info["truncation"], [1, 0, 0])
        np.testing.assert_array_equal(next_state.info["episode_done"], [1, 1, 0])
        np.testing.assert_array_equal(
            next_state.info["episode_metrics"]["length"], [2, 1, 1]
        )
        np.testing.assert_array_equal(
            next_state.info["episode_metrics"]["sum_reward"], [4, 2, 2]
        )
        np.testing.assert_array_equal(next_state.metrics["terminated"], [0, 1, 0])
        np.testing.assert_array_equal(next_state.metrics["truncated"], [1, 0, 0])
        np.testing.assert_array_equal(next_state.info["age"], [0, 0, 1])
        np.testing.assert_array_equal(next_state.info["steps"], [2, 1, 1])
        np.testing.assert_array_equal(
            np.asarray(next_state.info["reset_marker"]) != old_marker,
            [True, True, False],
        )

    def test_full_reset_randomness_is_fresh_and_reproducible(self):
        step, state = self.make_environment(episode_length=1)
        other_step, other_state = self.make_environment(episode_length=1)
        markers = [float(state.info["reset_marker"][0])]
        for _ in range(3):
            state = step(state, jnp.zeros((1, 1)))
            other_state = other_step(other_state, jnp.zeros((1, 1)))
            markers.append(float(state.info["reset_marker"][0]))
            np.testing.assert_array_equal(
                state.info["reset_marker"], other_state.info["reset_marker"]
            )
        self.assertEqual(len(set(markers)), len(markers))

    def test_reset_does_not_carry_unused_full_state_copies(self):
        _, state = self.make_environment(episode_length=5, batch_size=3)

        self.assertNotIn("AutoResetWrapper_first_data", state.info)
        self.assertNotIn("AutoResetWrapper_first_obs", state.info)

    def test_slim_episode_metrics_preserve_training_transitions(self):
        keys = jax.random.split(jax.random.PRNGKey(13), 3)
        optimized_env = wrap_biomechanics_training(ToyEnvironment(), episode_length=3)
        legacy_env = ConditionalAutoResetWrapper(
            brax_training.EpisodeWrapper(
                BiomechanicsVmapWrapper(ToyEnvironment()),
                episode_length=3,
                action_repeat=1,
            )
        )
        optimized_state = optimized_env.reset(keys)
        legacy_state = legacy_env.reset(keys)

        self.assertEqual(
            set(optimized_state.info["episode_metrics"]),
            {"sum_reward", "length"},
        )
        self.assertGreater(len(legacy_state.info["episode_metrics"]), 2)

        optimized_step = jax.jit(optimized_env.step)
        legacy_step = jax.jit(legacy_env.step)
        actions = (
            jnp.zeros((3, 1)),
            jnp.array([[0.0], [-1.0], [0.0]]),
            jnp.zeros((3, 1)),
            jnp.zeros((3, 1)),
        )
        for action in actions:
            optimized_state = optimized_step(optimized_state, action)
            legacy_state = legacy_step(legacy_state, action)
            for optimized, legacy in (
                (optimized_state.data, legacy_state.data),
                (optimized_state.obs, legacy_state.obs),
                (optimized_state.reward, legacy_state.reward),
                (optimized_state.done, legacy_state.done),
                (optimized_state.metrics, legacy_state.metrics),
            ):
                for optimized_leaf, legacy_leaf in zip(
                    jax.tree.leaves(optimized),
                    jax.tree.leaves(legacy),
                    strict=True,
                ):
                    np.testing.assert_array_equal(optimized_leaf, legacy_leaf)
            for key in (
                "steps",
                "truncation",
                "episode_done",
                "reset_marker",
                "age",
                "AutoResetWrapper_rng",
                "AutoResetWrapper_done_count",
            ):
                np.testing.assert_array_equal(
                    optimized_state.info[key],
                    legacy_state.info[key],
                )
            for key in ("sum_reward", "length"):
                np.testing.assert_array_equal(
                    optimized_state.info["episode_metrics"][key],
                    legacy_state.info["episode_metrics"][key],
                )


if __name__ == "__main__":
    unittest.main()
