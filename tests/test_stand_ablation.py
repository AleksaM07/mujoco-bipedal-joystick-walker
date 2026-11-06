"""Keep the stand comparison controlled and rank physics before reward."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from scripts.run_stand_ablation import build_configs, checkpoint_rank, main, run_variant


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

    def test_balance_followup_changes_only_reward_and_run_label(self) -> None:
        old_env, old_training = build_configs("scratch", 1_000_000)
        new_env, new_training = build_configs("scratch", 1_000_000, "balance_v1")
        old_values, new_values = asdict(old_env), asdict(new_env)
        self.assertEqual(new_values.pop("stand_reward_mode"), "balance_v1")
        self.assertEqual(old_values.pop("stand_reward_mode"), "imitation")
        self.assertEqual(old_values, new_values)
        old_values, new_values = asdict(old_training), asdict(new_training)
        old_values.pop("run_tag")
        new_values.pop("run_tag")
        self.assertEqual(old_values, new_values)

    def test_learning_rate_sweep_changes_only_rate_and_label(self) -> None:
        baseline_env, baseline = build_configs("scratch", 1_000_000, "balance_v1")
        changed_env, changed = build_configs("scratch", 1_000_000, "balance_v1", 1e-4)
        self.assertEqual(baseline_env, changed_env)
        baseline_values, changed_values = asdict(baseline), asdict(changed)
        self.assertEqual(changed_values.pop("learning_rate"), 1e-4)
        baseline_values.pop("learning_rate")
        baseline_values.pop("run_tag")
        changed_values.pop("run_tag")
        self.assertEqual(baseline_values, changed_values)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                build_configs("scratch", 1_000_000, "balance_v1", rate)

    def test_resume_restores_stand_weights_with_matching_env_and_ppo_config(
        self,
    ) -> None:
        environment, training = build_configs("scratch", 1_000_000, "balance_v1", 3e-4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source/checkpoints/000001024000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "ppo_network_config.json").write_text("{}", encoding="utf-8")
            saved = {"env": asdict(environment), "train": asdict(training)}
            config_path = checkpoint.parent.parent / "config.json"
            config_path.write_text(json.dumps(saved), encoding="utf-8")
            manifest_path = checkpoint.parent.parent / "xml_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "xml_sha256": sha256(
                            Path(environment.xml_path).read_bytes()
                        ).hexdigest()
                    }
                ),
                encoding="utf-8",
            )
            with patch("scripts.run_stand_ablation.RUNS_DIR", root):
                resumed_env, resumed = build_configs(
                    "resume", 1_000_000, "balance_v1", 3e-4, checkpoint
                )
                self.assertEqual(environment, resumed_env)
                self.assertTrue(resumed.restore_value_fn)
                self.assertEqual(resumed.resume_from, str(checkpoint.resolve()))
                before, after = asdict(training), asdict(resumed)
                for key in ("run_tag", "resume_from"):
                    before.pop(key)
                    after.pop(key)
                self.assertEqual(before, after)
                output = io.StringIO()
                arguments = [
                    "run_stand_ablation.py",
                    "--worker",
                    "resume",
                    "--resume-from",
                    str(checkpoint),
                    "--stand-reward-mode",
                    "balance_v1",
                    "--learning-rate",
                    "0.0003",
                    "--dry-run",
                    "--output",
                    str(root / "continued"),
                ]
                with patch("sys.argv", arguments), contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
                self.assertEqual(set(json.loads(output.getvalue())["plan"]), {"resume"})
                saved["env"]["locomotion_skill"] = "walk"
                config_path.write_text(json.dumps(saved), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "same stand environment"):
                    build_configs("resume", 1_000_000, "balance_v1", 3e-4, checkpoint)
                saved["env"]["locomotion_skill"] = "stand"
                saved["train"]["seed"] = 99
                config_path.write_text(json.dumps(saved), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "matching controlled PPO"):
                    build_configs("resume", 1_000_000, "balance_v1", 3e-4, checkpoint)
                manifest_path.write_text('{"xml_sha256": "wrong"}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "matching source XML"):
                    build_configs("resume", 1_000_000, "balance_v1", 3e-4, checkpoint)

    def test_resume_is_explicit_and_forwarded_to_worker(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --resume-from"):
            build_configs("resume", 1_000_000)
        with self.assertRaisesRegex(ValueError, "requires --resume-from"):
            build_configs("scratch", 1_000_000, resume_from=Path("unexpected"))
        checkpoint = Path("runs/source/checkpoints/000001024000")
        with patch("scripts.run_stand_ablation.run_command", return_value=2) as run:
            result = run_variant(
                "resume",
                Path("runs/continuation"),
                1_000_000,
                "balance_v1",
                3e-4,
                checkpoint,
            )
        self.assertEqual(result["status"], "training_failed")
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--resume-from") + 1], str(checkpoint.resolve())
        )
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0003")


if __name__ == "__main__":
    unittest.main()
