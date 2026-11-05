"""Keep training callbacks visible without confusing them with evaluation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import train
from config import TrainConfig
from train import (
    TrainingProgressLogger,
    log_eval_environment_summary,
    make_ppo_config,
)


class TrainingProgressTests(unittest.TestCase):
    def test_cli_forwards_callback_interval_to_saved_training_config(self):
        with (
            patch("sys.argv", ["train.py", "--training-metrics-steps", "100000"]),
            patch.object(train, "choose_device"),
            patch.object(train.jax.config, "update"),
            patch.object(train, "run_training") as run_training,
        ):
            train.main()
        requested = run_training.call_args.args[1]
        self.assertEqual(requested.training_metrics_steps, 100000)

    def test_callbacks_are_opt_in_and_do_not_change_training_schedule(self):
        plain = make_ppo_config(TrainConfig())
        logged = make_ppo_config(TrainConfig(training_metrics_steps=100000))
        self.assertFalse(plain.get("log_training_metrics", False))
        self.assertTrue(logged.log_training_metrics)
        self.assertEqual(logged.training_metrics_steps, 100000)
        for key in plain:
            self.assertEqual(plain[key], logged[key])

    def test_nonpositive_callback_interval_is_rejected(self):
        for interval in (0, -1):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                make_ppo_config(TrainConfig(training_metrics_steps=interval))

    def test_training_callbacks_report_speed_without_becoming_best_eval(self):
        progress = TrainingProgressLogger(500)
        with patch("train.logger.info") as log:
            progress(
                102400,
                {
                    "episode/sps": 1234.5,
                    "episode/sum_reward": 42.0,
                    "episode/length": 30.0,
                    "episode/total_loss": 0.2,
                },
            )
        line = log.call_args.args[0].format(*log.call_args.args[1:])
        self.assertIn("train progress | step=102400", line)
        self.assertIn("sps=1234.500", line)
        self.assertIn("episode_reward=42.000", line)
        self.assertIn("loss=0.200", line)
        self.assertIsNone(progress.best_reward)
        self.assertIsNone(progress.final_reward)

    def test_eval_retains_epoch_speed_and_best_checkpoint_tracking(self):
        progress = TrainingProgressLogger(500)
        with patch("train.logger.info") as log:
            progress(
                2001920,
                {
                    "eval/episode_reward": 100.0,
                    "eval/avg_episode_length": 60.0,
                    "training/sps": 500.0,
                    "training/walltime": 4000.0,
                    "eval/epoch_eval_time": 107.4,
                },
            )
        line = log.call_args.args[0].format(*log.call_args.args[1:])
        self.assertIn("eval | step=2001920", line)
        self.assertIn("training_sps=500.000", line)
        self.assertIn("training_seconds=4000.000", line)
        self.assertIn("eval_seconds=107.400", line)
        self.assertEqual(progress.best_step, 2001920)
        self.assertEqual(progress.best_reward, 100.0)

    def test_warp_capacity_is_not_reported_as_jax_world_count(self):
        for backend, expected in (("mjx_jax", "None"), ("mjx_warp", "16")):
            with self.subTest(backend=backend), patch("train.logger.info") as log:
                env = SimpleNamespace(
                    _config=SimpleNamespace(physics_backend=backend, warp_num_worlds=16)
                )
                log_eval_environment_summary(env)
                line = log.call_args.args[0].format(*log.call_args.args[1:])
                self.assertIn(f"warp_worlds={expected}", line)

    def test_stand_reports_step_means_and_episode_termination_fraction(self):
        # Episodes of 100 and 500 steps: ratio of mean sums to mean length
        # weights each physical step equally despite early termination.
        progress = TrainingProgressLogger(500)
        with patch("train.logger.info") as log:
            progress(
                204800,
                {
                    "eval/episode_reward": 850.0,
                    "eval/avg_episode_length": 300.0,
                    "eval/episode_stand_speed": 60.0,
                    "eval/episode_stand_distance": 30.0,
                    "eval/episode_stand_balance_score": 150.0,
                    "eval/episode_stand_eligible": 225.0,
                    "eval/episode_terminated": 0.5,
                    "eval/episode_stand_success": 200.0,
                    "eval/episode_stand_dwell_seconds": 1000.0,
                },
            )
        line = log.call_args.args[0].format(*log.call_args.args[1:])
        self.assertIn("episode_length=300.0", line)
        self.assertIn("stand_speed_mps=0.200", line)
        self.assertIn("stand_drift_m=0.100", line)
        self.assertIn("stand_balance_step=0.500", line)
        self.assertIn("stand_eligible_pct=75.0", line)
        self.assertIn("physical_termination_pct=50.0", line)
        self.assertIn("stand_gate=requires_checkpoint_rollout", line)
        self.assertNotIn("stand_success=", line)
        self.assertNotIn("stand_dwell", line)

    def test_stand_means_require_a_positive_episode_length(self):
        for length in (None, 0.0):
            with self.subTest(length=length), patch("train.logger.info") as log:
                TrainingProgressLogger(500)(
                    0,
                    {
                        "eval/episode_reward": 0.0,
                        "eval/avg_episode_length": length,
                        "eval/episode_stand_speed": 10.0,
                        "eval/episode_stand_eligible": 5.0,
                    },
                )
                line = log.call_args.args[0].format(*log.call_args.args[1:])
                self.assertNotIn("stand_speed_mps", line)
                self.assertNotIn("stand_eligible_pct", line)


if __name__ == "__main__":
    unittest.main()
