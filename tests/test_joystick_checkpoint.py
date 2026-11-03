"""Behavioral checks for introducing commands into the pure walking checkpoint."""

from __future__ import annotations

import copy
import unittest

import jax
import jax.numpy as jp
import numpy as np
from brax.training import types
from brax.training.acme import running_statistics

from tools.prepare_joystick_checkpoint import (
    COMMAND_CHANNELS,
    OBSERVATION_SIZES,
    prepare_command_inputs,
)


class JoystickCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        count = 1234
        epsilon = 0.001
        mean = {}
        std = {}
        variance = {}
        params = []
        for index, (key, size) in enumerate(OBSERVATION_SIZES.items()):
            mean[key] = rng.normal(size=size).astype(np.float32)
            mean[key][COMMAND_CHANNELS] = 0.0
            variance[key] = rng.uniform(1, 3, size).astype(np.float32) * count
            variance[key][COMMAND_CHANNELS] = 0.0
            std[key] = np.sqrt(variance[key] / count + epsilon)
            output_size = 52 if index == 0 else 1
            params.append(
                {
                    "params": {
                        "hidden_0": {
                            "kernel": rng.normal(size=(size, 7)).astype(np.float32),
                            "bias": rng.normal(size=7).astype(np.float32),
                        },
                        "hidden_1": {
                            "kernel": rng.normal(size=(7, 7)).astype(np.float32),
                            "bias": rng.normal(size=7).astype(np.float32),
                        },
                        "hidden_2": {
                            "kernel": rng.normal(size=(7, output_size)).astype(np.float32),
                            "bias": rng.normal(size=output_size).astype(np.float32),
                        },
                    }
                }
            )
        stats = running_statistics.RunningStatisticsState(
            mean=mean,
            std=std,
            summed_variance=variance,
            count=types.UInt64(hi=0, lo=count),
            std_eps=epsilon,
        )
        self.source = [stats, *params]

    @staticmethod
    def network_output(observations, stats, params, key):
        values = (observations - stats.mean[key]) / stats.std[key]
        for index in range(3):
            layer = params["params"][f"hidden_{index}"]
            values = values @ layer["kernel"] + layer["bias"]
            if index < 2:
                values = np.asarray(jax.nn.silu(jp.asarray(values)))
        return values

    def test_actor_and_critic_keep_zero_command_outputs_and_ignore_new_commands(self) -> None:
        prepared = prepare_command_inputs(self.source)
        rng = np.random.default_rng(22)
        for index, (key, size) in enumerate(OBSERVATION_SIZES.items(), start=1):
            observations = rng.normal(size=(8, size)).astype(np.float32)
            observations[:, COMMAND_CHANNELS] = 0
            expected = self.network_output(observations, self.source[0], self.source[index], key)
            actual = self.network_output(observations, prepared[0], prepared[index], key)
            np.testing.assert_array_equal(actual, expected)
            observations[:, COMMAND_CHANNELS] = rng.uniform(-0.65, 0.65, size=(8, 3))
            commanded = self.network_output(observations, prepared[0], prepared[index], key)
            np.testing.assert_array_equal(commanded, expected)
            self.assertTrue(np.isfinite(commanded).all())

    def test_changes_are_confined_to_command_kernel_and_statistics(self) -> None:
        untouched = copy.deepcopy(self.source)
        prepared = prepare_command_inputs(self.source)
        for index, key in enumerate(OBSERVATION_SIZES, start=1):
            indices = np.r_[0:9, 12:OBSERVATION_SIZES[key]]
            for field in ("mean", "std", "summed_variance"):
                np.testing.assert_array_equal(getattr(prepared[0], field)[key][indices], getattr(untouched[0], field)[key][indices])
                np.testing.assert_array_equal(getattr(self.source[0], field)[key], getattr(untouched[0], field)[key])
            for name, layer in untouched[index]["params"].items():
                for field, value in layer.items():
                    result = prepared[index]["params"][name][field]
                    if name == "hidden_0" and field == "kernel":
                        np.testing.assert_array_equal(result[indices], value[indices])
                        np.testing.assert_array_equal(result[COMMAND_CHANNELS], 0.0)
                    else:
                        np.testing.assert_array_equal(result, value)
                    np.testing.assert_array_equal(self.source[index]["params"][name][field], value)

    def test_unit_deviation_matches_welford_variance_including_epsilon(self) -> None:
        prepared = prepare_command_inputs(self.source)
        stats = prepared[0]
        count = int(stats.count.lo)
        for key in OBSERVATION_SIZES:
            computed = np.sqrt(stats.summed_variance[key][COMMAND_CHANNELS] / count + stats.std_eps)
            np.testing.assert_allclose(computed, 1.0, rtol=1e-7)
            np.testing.assert_array_equal(stats.std[key][COMMAND_CHANNELS], 1.0)
        self.assertEqual(int(stats.count.lo), int(self.source[0].count.lo))

    def test_rejects_nonzero_command_history(self) -> None:
        self.source[0].mean["state"][9] = 0.2
        with self.assertRaisesRegex(ValueError, "zero recorded means"):
            prepare_command_inputs(self.source)

    def test_rejects_wrong_observation_layout(self) -> None:
        self.source[1]["params"]["hidden_0"]["kernel"] = np.zeros((294, 7))
        with self.assertRaisesRegex(ValueError, "input shape"):
            prepare_command_inputs(self.source)


if __name__ == "__main__":
    unittest.main()
