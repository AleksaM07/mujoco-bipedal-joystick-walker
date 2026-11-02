"""Regression tests for the 18-to-26 DOF checkpoint migration."""

from __future__ import annotations

import unittest

import numpy as np

from tools.migrate_checkpoint_to_arms import (
    ObservationLayout,
    migrate_policy_params,
    privileged_index_mapping,
    state_index_mapping,
)


class CheckpointArmMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ObservationLayout(action_size=18)
        self.target = ObservationLayout(action_size=26)

    def test_expected_project_observation_sizes(self) -> None:
        self.assertEqual(self.source.state_size, 294)
        self.assertEqual(self.target.state_size, 366)
        self.assertEqual(self.source.privileged_size, 721)
        self.assertEqual(self.target.privileged_size, 809)

    def test_state_mapping_preserves_every_source_field(self) -> None:
        old_indices, new_indices = state_index_mapping(self.source, self.target)
        self.assertEqual(len(old_indices), self.source.state_size)
        np.testing.assert_array_equal(
            np.sort(old_indices),
            np.arange(self.source.state_size),
        )
        self.assertEqual(len(np.unique(new_indices)), self.source.state_size)
        self.assertTrue(np.all(new_indices < self.target.state_size))

    def test_privileged_mapping_preserves_every_source_field(self) -> None:
        old_indices, new_indices = privileged_index_mapping(
            self.source,
            self.target,
        )
        self.assertEqual(len(old_indices), self.source.privileged_size)
        np.testing.assert_array_equal(
            np.sort(old_indices),
            np.arange(self.source.privileged_size),
        )
        self.assertEqual(len(np.unique(new_indices)), self.source.privileged_size)
        self.assertTrue(np.all(new_indices < self.target.privileged_size))

    def test_policy_migration_keeps_old_logits_exactly(self) -> None:
        rng = np.random.default_rng(7)
        hidden_size = 11
        source_params = {
            "params": {
                "hidden_0": {
                    "kernel": rng.normal(
                        size=(self.source.state_size, hidden_size)
                    ).astype(np.float32),
                    "bias": rng.normal(size=hidden_size).astype(np.float32),
                },
                "hidden_1": {
                    "kernel": rng.normal(size=(hidden_size, hidden_size)).astype(
                        np.float32
                    ),
                    "bias": rng.normal(size=hidden_size).astype(np.float32),
                },
                "hidden_2": {
                    "kernel": rng.normal(
                        size=(hidden_size, 2 * self.source.action_size)
                    ).astype(np.float32),
                    "bias": rng.normal(
                        size=2 * self.source.action_size
                    ).astype(np.float32),
                },
            }
        }
        migrated = migrate_policy_params(
            source_params,
            self.source,
            self.target,
            arm_scale_raw=-3.0,
        )
        old_indices, new_indices = state_index_mapping(self.source, self.target)
        old_observation = rng.normal(size=self.source.state_size).astype(np.float32)
        new_observation = np.zeros(self.target.state_size, dtype=np.float32)
        new_observation[new_indices] = old_observation[old_indices]

        old_hidden = old_observation @ source_params["params"]["hidden_0"]["kernel"]
        new_hidden = new_observation @ migrated["params"]["hidden_0"]["kernel"]
        np.testing.assert_allclose(new_hidden, old_hidden, rtol=1e-6, atol=1e-6)

        old_logits = old_hidden @ source_params["params"]["hidden_2"]["kernel"]
        old_logits += source_params["params"]["hidden_2"]["bias"]
        new_logits = new_hidden @ migrated["params"]["hidden_2"]["kernel"]
        new_logits += migrated["params"]["hidden_2"]["bias"]
        np.testing.assert_allclose(
            new_logits[: self.source.action_size],
            old_logits[: self.source.action_size],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            new_logits[
                self.target.action_size : self.target.action_size
                + self.source.action_size
            ],
            old_logits[self.source.action_size :],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            new_logits[self.source.action_size : self.target.action_size],
            np.zeros(self.target.action_size - self.source.action_size),
        )
        np.testing.assert_allclose(
            new_logits[self.target.action_size + self.source.action_size :],
            -3.0,
            rtol=0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
