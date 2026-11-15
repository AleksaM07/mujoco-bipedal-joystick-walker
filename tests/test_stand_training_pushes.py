"""Verify training forces through MJX and the production reset/evaluation wrappers."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from config import EnvConfig, TrainConfig
from skills.stand import StandEnv
from skills.stand_training_pushes import StandTrainingPushWrapper
from train import make_wrap_env_fn, run_training, wrap_biomechanics_training


class FreeBodyStandEnv(StandEnv):
    """Use a 75 kg free body to isolate real MJX force integration from balance."""

    def __init__(self) -> None:
        self._ctrl_dt = 0.02
        self._config = SimpleNamespace(physics_backend="mjx_jax", ctrl_dt=0.02)
        self._mj_model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 -9.81" timestep="0.005"/>'
            '<worldbody><body name="thorax"><freejoint/>'
            '<geom type="sphere" size="0.1" mass="75"/></body></worldbody></mujoco>'
        )
        self._mjx_model = mjx.put_model(self._mj_model)
        self._init_q = jp.asarray(self._mj_model.qpos0).at[2].set(1.0)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        del rng
        data = mjx.make_data(self.mj_model)
        data = mjx.forward(self.mjx_model, data.replace(qpos=data.qpos.at[2].set(1.0)))
        zero = jp.array(0.0)
        return mjx_env.State(
            data=data,
            obs=jp.concatenate([data.qpos, data.qvel]),
            reward=zero,
            done=zero,
            metrics={"terminated": zero, "truncated": zero, "stand_speed": zero},
            info={"stand_anchor_xy": data.qpos[:2]},
        )

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        del info
        return jp.concatenate([data.qpos, data.qvel])

    def _foot_xy(self, data: mjx.Data) -> jax.Array:
        del data
        return jp.zeros((2, 2))

    def _torso_up(self, data: mjx.Data) -> jax.Array:
        del data
        return jp.array(1.0)

    def _foot_contact(self, data: mjx.Data, info: dict) -> jax.Array:
        del data, info
        return jp.ones(2)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = jax.lax.fori_loop(
            0, 4, lambda _, data: mjx.step(self.mjx_model, data), state.data
        )
        return state.replace(
            data=data,
            obs=jp.concatenate([data.qpos, data.qvel]),
            done=(action[0] < 0).astype(jp.float32),
            reward=jp.array(1.0),
            metrics={
                **state.metrics,
                "stand_speed": jp.linalg.norm(data.qvel[:2]),
            },
        )


class StandTrainingPushTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = FreeBodyStandEnv()
        cls.env = StandTrainingPushWrapper(cls.base, 0.05)

    def test_seeded_reset_mixes_nominal_episodes_and_varies_magnitude_and_phase(
        self,
    ) -> None:
        keys = jax.random.split(jax.random.PRNGKey(7), 512)
        reset = jax.jit(jax.vmap(self.env.reset))
        state = reset(keys)
        force = np.asarray(state.info["stand_push_force_x"])
        np.testing.assert_array_equal(force, reset(keys).info["stand_push_force_x"])
        self.assertGreater(np.mean(force == 0), 0.4)
        self.assertLess(np.mean(force == 0), 0.6)
        self.assertTrue(np.all((force <= 0) & (force >= -36.7875)))
        starts = np.asarray(state.info["stand_push_start"])
        self.assertTrue(np.all((starts >= 100) & (starts <= 400)))
        self.assertGreater(len(np.unique(starts)), 100)
        np.testing.assert_array_equal(state.info["stand_push_steps"], 0)
        self.assertEqual(state.obs.shape, (512, 13))

    def test_real_mjx_pulse_impulse_force_removal_and_periodic_timing(self) -> None:
        initial = self.env.reset(jax.random.PRNGKey(7))
        initial = initial.replace(
            info={
                **initial.info,
                "stand_push_steps": jp.array(249),
                "stand_push_start": jp.array(250),
                "stand_push_force_x": jp.array(-36.7875),
            }
        )

        @jax.jit
        def pulse(state):
            def advance(state, _):
                state = self.env.step(state, jp.zeros(1))
                return state, state.data.xfrc_applied

            return jax.lax.scan(advance, state, None, length=12)

        final, forces = pulse(initial)
        expected = np.zeros((12, 2, 6))
        expected[1:11, 1, 0] = -36.7875
        np.testing.assert_allclose(forces, expected, rtol=1e-6)
        self.assertAlmostEqual(float(final.data.qvel[0]), -0.0981, places=6)
        np.testing.assert_array_equal(final.data.xfrc_applied, 0)
        np.testing.assert_allclose(final.data.qvel[3:], 0, atol=1e-6)
        repeated = initial.replace(
            info={**initial.info, "stand_push_steps": jp.array(749)}
        )
        _, later_forces = pulse(repeated)
        np.testing.assert_array_equal(forces, later_forces)

    def test_hard_push_distribution_preserves_a_nominal_subset(self) -> None:
        env = StandTrainingPushWrapper(
            self.base,
            0.05,
            min_fraction=0.04,
            active_probability=0.75,
        )
        keys = jax.random.split(jax.random.PRNGKey(19), 2048)
        state = jax.jit(jax.vmap(env.reset))(keys)
        force = np.asarray(state.info["stand_push_force_x"])
        active = force < 0
        self.assertGreater(np.mean(active), 0.70)
        self.assertLess(np.mean(active), 0.80)
        self.assertTrue(np.all(force[active] <= -(0.04 * 75 * 9.81)))
        self.assertTrue(np.all(force[active] >= -(0.05 * 75 * 9.81)))

    def test_bidirectional_distribution_samples_both_world_x_signs(self) -> None:
        env = StandTrainingPushWrapper(
            self.base,
            0.05,
            min_fraction=0.04,
            active_probability=1.0,
            direction="bidirectional",
        )
        keys = jax.random.split(jax.random.PRNGKey(23), 2048)
        force = np.asarray(
            jax.jit(jax.vmap(env.reset))(keys).info["stand_push_force_x"]
        )
        self.assertGreater(np.mean(force > 0), 0.45)
        self.assertLess(np.mean(force > 0), 0.55)
        self.assertTrue(np.all(np.abs(force) >= 0.04 * 75 * 9.81))
        self.assertTrue(np.all(np.abs(force) <= 0.05 * 75 * 9.81))

    def test_success_curriculum_starts_with_recoverable_state_and_advances(
        self,
    ) -> None:
        env = StandTrainingPushWrapper(
            self.base,
            0.05,
            min_fraction=0.01,
            active_probability=1.0,
            direction="negative",
            success_curriculum=True,
        )
        state = jax.jit(env.reset)(jax.random.PRNGKey(31))
        self.assertEqual(int(state.info["stand_curriculum_phase"]), env.RECOVERY)
        self.assertTrue(bool(state.info["stand_recovery_active"]))
        self.assertGreaterEqual(float(state.data.qvel[0]), -0.20)
        self.assertLessEqual(float(state.data.qvel[0]), -0.05)
        self.assertGreaterEqual(float(state.data.qvel[4]), -0.20)
        self.assertLessEqual(float(state.data.qvel[4]), -0.05)
        self.assertLess(float(state.info["stand_initial_tilt_radians"]), 0.0)

        state = state.replace(
            data=state.data.replace(qvel=jp.zeros_like(state.data.qvel)),
            info={
                **state.info,
                "stand_curriculum_stable_steps": jp.array(
                    env.curriculum_dwell_steps - 1, dtype=jp.int32
                ),
            },
        )
        state = jax.jit(env.step)(state, jp.zeros(1))
        self.assertEqual(int(state.info["stand_curriculum_level"]), 1)
        self.assertEqual(int(state.info["stand_curriculum_phase"]), env.WAITING)
        self.assertFalse(bool(state.info["stand_recovery_active"]))
        np.testing.assert_allclose(
            state.info["stand_anchor_xy"], state.data.qpos[:2], atol=1e-7
        )

        state = state.replace(
            info={
                **state.info,
                "stand_curriculum_next_start": state.info["stand_push_steps"],
            }
        )
        state = jax.jit(env.step)(state, jp.zeros(1))
        expected_force = -(0.02 * 75 * 9.81)
        self.assertAlmostEqual(
            float(state.metrics["stand_training_push_x_newtons"]),
            expected_force,
            places=4,
        )
        self.assertEqual(int(state.info["stand_curriculum_phase"]), env.PULSE)

    def test_random_pushes_resample_full_horizontal_force_without_unlock(self) -> None:
        env = StandTrainingPushWrapper(
            self.base,
            0.05,
            min_fraction=0.05,
            active_probability=1.0,
            direction="omnidirectional",
            random_pushes=True,
        )
        state = jax.jit(env.reset)(jax.random.PRNGKey(41))
        state = state.replace(
            info={**state.info, "stand_push_start": jp.array(0, dtype=jp.int32)}
        )
        step = jax.jit(env.step)
        first = step(state, jp.zeros(1))
        force = np.asarray(first.info["stand_push_force_xy"])
        self.assertAlmostEqual(float(np.linalg.norm(force)), 0.05 * 75 * 9.81, places=4)
        self.assertEqual(int(first.info["stand_random_push_count"]), 1)
        self.assertEqual(int(first.info["stand_curriculum_phase"]), env.PULSE)
        self.assertTrue(bool(first.info["stand_recovery_active"]))
        self.assertEqual(jax.tree.structure(state), jax.tree.structure(first))

        waiting = first.replace(
            info={
                **first.info,
                "stand_curriculum_phase": jp.array(env.WAITING, dtype=jp.int32),
                "stand_push_start": first.info["stand_push_steps"],
            }
        )
        second = step(waiting, jp.zeros(1))
        self.assertEqual(int(second.info["stand_random_push_count"]), 2)
        self.assertFalse(
            np.allclose(force, np.asarray(second.info["stand_push_force_xy"]))
        )

    def test_auto_reset_clears_counter_and_force_and_resamples_schedule(self) -> None:
        env = wrap_biomechanics_training(self.env, episode_length=2)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(11), 8))
        initial_starts = np.asarray(state.info["stand_push_start"])
        state = state.replace(
            info={
                **state.info,
                "stand_push_start": jp.zeros(8, dtype=jp.int32),
                "stand_push_force_x": jp.full(8, -36.7875),
            }
        )
        step = jax.jit(env.step)
        state = step(state, jp.zeros((8, 1)))
        self.assertTrue(np.all(np.asarray(state.data.qvel[:, 0]) < 0))
        state = step(state, jp.zeros((8, 1)))
        np.testing.assert_array_equal(state.done, 1)
        np.testing.assert_array_equal(state.info["stand_push_steps"], 0)
        np.testing.assert_array_equal(state.data.xfrc_applied, 0)
        np.testing.assert_array_equal(state.data.qvel, 0)
        self.assertTrue(np.any(initial_starts != state.info["stand_push_start"]))
        state = step(state, jp.zeros((8, 1)))
        np.testing.assert_array_equal(state.info["stand_push_steps"], 1)
        np.testing.assert_array_equal(state.data.xfrc_applied, 0)

    def test_eval_and_default_training_never_receive_push_wrapper(self) -> None:
        evaluation = FreeBodyStandEnv()
        wrap = make_wrap_env_fn(evaluation, stand_push_max_fraction=0.05)
        keys = jax.random.split(jax.random.PRNGKey(7), 2)
        eval_state = wrap(evaluation, episode_length=500).reset(keys)
        self.assertNotIn("stand_push_steps", eval_state.info)
        training = wrap(self.base, episode_length=500).reset(keys)
        self.assertIn("stand_push_steps", training.info)
        default_wrap = make_wrap_env_fn(evaluation)
        nominal = default_wrap(self.base, episode_length=500).reset(keys)
        self.assertNotIn("stand_push_steps", nominal.info)
        self.assertEqual(TrainConfig().stand_push_min_fraction, 0)
        self.assertEqual(TrainConfig().stand_push_max_fraction, 0)
        self.assertEqual(TrainConfig().stand_push_probability, 0.5)
        self.assertEqual(TrainConfig().stand_push_direction, "negative")

    def test_invalid_fraction_and_non_stand_training_are_rejected(self) -> None:
        for value in (-0.01, 0.06, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StandTrainingPushWrapper(self.base, value)
        with self.assertRaises(ValueError):
            StandTrainingPushWrapper(self.base, 0.04, min_fraction=0.05)
        for probability in (-0.01, 1.01, float("nan")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                StandTrainingPushWrapper(
                    self.base, 0.05, active_probability=probability
                )
        with self.assertRaisesRegex(ValueError, "direction"):
            StandTrainingPushWrapper(self.base, 0.05, direction="sideways")
        with self.assertRaisesRegex(ValueError, "either"):
            StandTrainingPushWrapper(
                self.base,
                0.05,
                success_curriculum=True,
                random_pushes=True,
            )
        with self.assertRaisesRegex(ValueError, "locomotion_skill=stand"):
            run_training(
                EnvConfig(), TrainConfig(stand_push_max_fraction=0.05), Path("unused")
            )
        with self.assertRaisesRegex(ValueError, "direction"):
            run_training(
                EnvConfig(locomotion_skill="stand"),
                TrainConfig(stand_push_direction="sideways"),
                Path("unused"),
            )


if __name__ == "__main__":
    unittest.main()
