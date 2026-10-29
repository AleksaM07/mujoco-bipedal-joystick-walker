"""Check evaluation episode accounting without expensive physics or rendering."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

import evaluate
from config import COMMAND_OBS_END, COMMAND_OBS_START, KEY_SPACE, KEY_UP


class ToyData(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    xmat: jax.Array


class ToyEnvironment:
    """Terminate after N steps and make accidental post-terminal steps obvious."""

    def __init__(self, terminal_step: int = 1, pure: bool = False) -> None:
        self.terminal_step = terminal_step
        self.pure = pure
        self.mj_model = SimpleNamespace(nq=3, nv=3)

    def _is_pure_reference_imitation(self) -> jax.Array:
        return jnp.array(self.pure)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        del rng
        upright = jnp.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        data = ToyData(
            jnp.array([0.0, 0, 1.0]), jnp.zeros(3), jnp.stack([upright] * 2)
        )
        obs = jnp.zeros(COMMAND_OBS_END + 1)
        return mjx_env.State(
            data=data,
            obs={"state": obs, "privileged_state": obs},
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics={},
            info={"age": jnp.array(0), "command": jnp.zeros(3)},
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        del action
        age = state.info["age"] + 1
        return state.replace(
            data=state.data._replace(qpos=state.data.qpos.at[0].set(age)),
            reward=jnp.where(state.done, -100.0, 1.0),
            done=(age >= self.terminal_step).astype(jnp.float32),
            info={**state.info, "age": age},
        )


def command_policy(obs: dict, rng: jax.Array) -> tuple[jax.Array, dict]:
    """Expose the command actually received by the policy as its action."""
    del rng
    return obs["state"][COMMAND_OBS_START:COMMAND_OBS_END], {}


class EvaluationRolloutTests(unittest.TestCase):
    def test_compile_preserves_initial_state_and_random_key(self) -> None:
        env = ToyEnvironment()
        key = jax.random.PRNGKey(7)
        command = np.zeros(3, dtype=np.float32)
        state = evaluate.reset_state(env, key, command)
        original_leaves = [np.array(leaf) for leaf in jax.tree.leaves(state)]
        original_key = np.array(key)
        evaluate.compile_simulation_step(env, command_policy, state, key, command)
        for before, after in zip(original_leaves, jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(key, original_key)
        first, _ = evaluate.simulation_step(env, command_policy, state, key, command)
        self.assertEqual(float(first.reward), 1.0)
        self.assertEqual(float(first.done), 1.0)

    def test_inspect_counts_first_terminal_step(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            evaluate.inspect_policy(
                ToyEnvironment(), command_policy, jax.random.PRNGKey(7),
                np.zeros(3, dtype=np.float32), steps=3,
            )
        self.assertIn("resets=3", output.getvalue())
        self.assertIn("total_reward=3.000", output.getvalue())
        self.assertNotIn("reward=-100", output.getvalue())

    def test_record_counts_first_terminal_frame_and_reward(self) -> None:
        for continue_after_done, expected_frames in ((False, 1), (True, 3)):
            with self.subTest(continue_after_done=continue_after_done):
                output = io.StringIO()
                with (
                    redirect_stdout(output),
                    patch.object(evaluate, "render_host_replay_frames") as render,
                    patch.object(evaluate, "write_video") as write,
                ):
                    render.return_value = [np.zeros((2, 2, 3))] * expected_frames
                    evaluate.record_policy_video(
                        ToyEnvironment(), command_policy, jax.random.PRNGKey(7),
                        np.zeros(3, dtype=np.float32), Path("unused.mp4"),
                        3, 50, 2, 2, continue_after_done, "single",
                    )
                snapshots = render.call_args.args[1]
                self.assertEqual(len(snapshots), expected_frames)
                self.assertEqual(float(snapshots[0][0][0]), 1.0)
                self.assertIn(
                    f"total_reward={expected_frames:.3f}", output.getvalue()
                )
                write.assert_called_once()

    def test_pure_policy_never_sees_injected_joystick_command(self) -> None:
        key = jax.random.PRNGKey(7)
        command = np.array([0.3, -0.1, 0.2], dtype=np.float32)
        for pure in (True, False):
            with self.subTest(pure=pure):
                env = ToyEnvironment(pure=pure)
                state = evaluate.reset_state(env, key, command)
                expected = np.zeros(3) if pure else command
                for obs in state.obs.values():
                    np.testing.assert_array_equal(
                        obs[COMMAND_OBS_START:COMMAND_OBS_END], expected
                    )
                state, action = evaluate.simulation_step(
                    env, command_policy, state, key, command
                )
                np.testing.assert_array_equal(action, expected)
                np.testing.assert_array_equal(state.info["command"], expected)

    def test_disabled_joystick_still_allows_pause(self) -> None:
        controller = evaluate.JoystickController(
            np.zeros(3), 0.05, command_enabled=False
        )
        with redirect_stdout(io.StringIO()):
            controller.key_callback(KEY_UP)
            controller.key_callback(KEY_SPACE)
        np.testing.assert_array_equal(controller.command, np.zeros(3))
        self.assertTrue(controller.paused)

    def test_inspect_enforces_episode_limit_without_counting_it_as_fall(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            evaluate.inspect_policy(
                ToyEnvironment(terminal_step=100), command_policy,
                jax.random.PRNGKey(7), np.zeros(3, dtype=np.float32),
                steps=5, episode_length=2,
            )
        self.assertIn("resets=2", output.getvalue())
        self.assertIn("terminations=0", output.getvalue())
        self.assertIn("timeouts=2", output.getvalue())
        self.assertIn("mean_episode_length=2.0", output.getvalue())
        self.assertIn("partial_episode_length=1", output.getvalue())
        self.assertIn("mean_survive_pct=100.0", output.getvalue())

    def test_inspect_rejects_nonpositive_budgets(self) -> None:
        for steps, episode_length in ((0, 2), (-1, 2), (3, 0), (3, -1)):
            with self.subTest(steps=steps, episode_length=episode_length):
                with self.assertRaises(ValueError):
                    evaluate.inspect_policy(
                        ToyEnvironment(), command_policy, jax.random.PRNGKey(7),
                        np.zeros(3), steps, episode_length=episode_length,
                    )

    def test_partial_episode_is_not_reported_as_completed_survival(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            evaluate.inspect_policy(
                ToyEnvironment(terminal_step=100), command_policy,
                jax.random.PRNGKey(7), np.zeros(3, dtype=np.float32),
                steps=1, episode_length=2,
            )
        self.assertIn("resets=0", output.getvalue())
        self.assertIn("mean_episode_length=n/a", output.getvalue())
        self.assertIn("mean_survive_pct=n/a", output.getvalue())
        self.assertIn("partial_episode_length=1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
