"""Keep the stand comparison controlled and rank physics before reward."""

import unittest
from dataclasses import asdict

from scripts.run_stand_ablation import build_configs, checkpoint_rank


class StandAblationTests(unittest.TestCase):
    def test_variants_share_archived_physics_and_training_budget(self) -> None:
        transfer_env, transfer = build_configs("transfer", 1_000_000)
        scratch_env, scratch = build_configs("scratch", 1_000_000)
        self.assertEqual(transfer_env, scratch_env)
        self.assertEqual(transfer_env.physics_backend, "mjx_jax")
        self.assertEqual(transfer_env.locomotion_skill, "stand")
        self.assertEqual(transfer_env.deepmimic_arm_reward_weight, 1.0)
        self.assertTrue(transfer_env.arm_actuators)
        self.assertEqual(transfer_env.reference_gait_file, [])
        transfer_values = asdict(transfer)
        scratch_values = asdict(scratch)
        for key in ("resume_from", "restore_value_fn", "run_tag"):
            transfer_values.pop(key)
            scratch_values.pop(key)
        self.assertEqual(transfer_values, scratch_values)
        self.assertIsNotNone(transfer.resume_from)
        self.assertFalse(transfer.restore_value_fn)
        self.assertIsNone(scratch.resume_from)
        self.assertEqual(transfer.num_timesteps, 1_000_000)
        self.assertEqual(transfer.learning_rate, 2e-5)
        self.assertTrue(transfer.bare)

    def test_physical_gate_and_drift_outrank_reward(self) -> None:
        def report(passed: bool, drift: float, reward: float) -> dict:
            return {
                "passed": passed,
                "rollouts": [
                    {
                        "steps": 500 if passed else 499,
                        "finite": True,
                        "terminated": not passed,
                        "final_success": passed,
                        "max_drift": drift,
                        "mean_speed": 0.02,
                        "reward": reward,
                    }
                ]
                * 4,
            }

        stable = report(True, 0.02, 800)
        high_reward_fall = report(False, 0.01, 1500)
        high_reward_drift = report(True, 0.2, 1500)
        self.assertGreater(checkpoint_rank(stable), checkpoint_rank(high_reward_fall))
        self.assertGreater(checkpoint_rank(stable), checkpoint_rank(high_reward_drift))


if __name__ == "__main__":
    unittest.main()
