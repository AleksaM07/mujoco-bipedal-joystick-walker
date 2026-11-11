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
                recovery_env, recovery_training = build_configs(
                    "resume",
                    1_000_000,
                    "recovery_v2",
                    3e-4,
                    checkpoint,
                )
                self.assertEqual(recovery_env.stand_reward_mode, "recovery_v2")
                source_values = asdict(environment)
                recovery_values = asdict(recovery_env)
                source_values.pop("stand_reward_mode")
                recovery_values.pop("stand_reward_mode")
                self.assertEqual(source_values, recovery_values)
                self.assertTrue(recovery_training.restore_value_fn)
                self.assertEqual(
                    recovery_training.resume_from, str(checkpoint.resolve())
                )
                longer_env, longer_training = build_configs(
                    "resume",
                    1_000_000,
                    "balance_v1",
                    3e-4,
                    checkpoint,
                    episode_length=3000,
                )
                self.assertEqual(environment, longer_env)
                longer_values = asdict(longer_training)
                self.assertEqual(longer_values.pop("episode_length"), 3000)
                self.assertTrue(longer_values.pop("restore_value_fn"))
                for key in ("run_tag", "resume_from"):
                    longer_values.pop(key)
                baseline_values = asdict(training)
                for key in (
                    "episode_length",
                    "restore_value_fn",
                    "run_tag",
                    "resume_from",
                ):
                    baseline_values.pop(key)
                self.assertEqual(longer_values, baseline_values)
                pushed_env, pushed_training = build_configs(
                    "resume",
                    1_000_000,
                    "balance_v1",
                    3e-4,
                    checkpoint,
                    episode_length=3000,
                    stand_push_min_fraction=0.04,
                    stand_push_max_fraction=0.05,
                    stand_push_probability=0.75,
                )
                self.assertEqual(longer_env, pushed_env)
                pushed_values, longer_values = (
                    asdict(pushed_training),
                    asdict(longer_training),
                )
                self.assertEqual(pushed_values.pop("stand_push_min_fraction"), 0.04)
                self.assertEqual(longer_values.pop("stand_push_min_fraction"), 0.0)
                self.assertEqual(pushed_values.pop("stand_push_max_fraction"), 0.05)
                self.assertEqual(longer_values.pop("stand_push_max_fraction"), 0.0)
                self.assertEqual(pushed_values.pop("stand_push_probability"), 0.75)
                self.assertEqual(longer_values.pop("stand_push_probability"), 0.5)
                self.assertEqual(pushed_values, longer_values)
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

    def test_resume_rejects_uncontrolled_reward_migration(self) -> None:
        environment, training = build_configs("scratch", 1_000_000, "imitation")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source/checkpoints/000001024000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "ppo_network_config.json").write_text("{}", encoding="utf-8")
            (checkpoint.parent.parent / "config.json").write_text(
                json.dumps({"env": asdict(environment), "train": asdict(training)}),
                encoding="utf-8",
            )
            (checkpoint.parent.parent / "xml_manifest.json").write_text(
                json.dumps(
                    {
                        "xml_sha256": sha256(
                            Path(environment.xml_path).read_bytes()
                        ).hexdigest()
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("scripts.run_stand_ablation.RUNS_DIR", root),
                self.assertRaisesRegex(ValueError, "balance_v1 to recovery_v2"),
            ):
                build_configs(
                    "resume", 1_000_000, "recovery_v2", 3e-4, checkpoint
                )

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
                episode_length=3000,
                stand_push_min_fraction=0.04,
                stand_push_max_fraction=0.05,
                stand_push_probability=0.75,
            )
        self.assertEqual(result["status"], "training_failed")
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--resume-from") + 1], str(checkpoint.resolve())
        )
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0003")
        self.assertEqual(command[command.index("--episode-length") + 1], "3000")
        self.assertEqual(
            command[command.index("--stand-push-min-fraction") + 1], "0.04"
        )
        self.assertEqual(
            command[command.index("--stand-push-max-fraction") + 1], "0.05"
        )
        self.assertEqual(
            command[command.index("--stand-push-probability") + 1], "0.75"
        )

    def test_push_fraction_is_explicit_and_bounded(self) -> None:
        for fraction in (-0.01, 0.06, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                build_configs("scratch", 1_000_000, stand_push_max_fraction=fraction)
        with self.assertRaises(ValueError):
            build_configs(
                "scratch",
                1_000_000,
                stand_push_min_fraction=0.05,
                stand_push_max_fraction=0.04,
            )
        for probability in (-0.01, 1.01, float("nan")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                build_configs(
                    "scratch",
                    1_000_000,
                    stand_push_probability=probability,
                )

    def test_long_episode_requires_at_least_original_gate_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 500"):
            build_configs("scratch", 1_000_000, episode_length=499)

    def test_robustness_stage_distinguishes_physical_failure_from_missing_cases(
        self,
    ) -> None:
        for missing_case, passed in ((False, False), (False, True), (True, True)):
            with (
                self.subTest(missing_case=missing_case, passed=passed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                checkpoint = root / "training/checkpoints/000000204800"
                checkpoint.mkdir(parents=True)
                result = {
                    "variant": "scratch",
                    "status": "evaluated",
                    "run_dir": str(checkpoint.parent.parent),
                    "best_checkpoint": str(checkpoint),
                    "best_gate": {
                        "passed": True,
                        "rollouts": [
                            {
                                "steps": 500,
                                "finite": True,
                                "terminated": False,
                                "final_success": True,
                                "max_drift": 0.02,
                                "mean_speed": 0.01,
                                "reward": 1400,
                            }
                        ],
                    },
                }
                output = root / "experiment"

                def fake_robustness(
                    command: list[str],
                    log_path: Path,
                    missing_case: bool = missing_case,
                    checkpoint: Path = checkpoint,
                    passed: bool = passed,
                ) -> int:
                    destination = Path(command[command.index("--output-dir") + 1])
                    destination.mkdir()
                    names = [
                        "hold_60s",
                        "push_x_pos",
                        "push_x_neg",
                        "push_y_pos",
                        "push_y_neg",
                    ]
                    if missing_case:
                        names.pop()
                    report = {
                        "status": "complete",
                        "results": [
                            {
                                "checkpoint": str(checkpoint.resolve()),
                                "probe": {"name": name},
                                "passed": passed,
                            }
                            for name in names
                        ],
                    }
                    (destination / "report.json").write_text(
                        json.dumps(report), encoding="utf-8"
                    )
                    return 0 if passed else 1

                arguments = [
                    "runner",
                    "--variants",
                    "scratch",
                    "--output",
                    str(output),
                    "--skip-video",
                    "--robustness-check",
                ]
                with (
                    patch("scripts.run_stand_ablation.RUNS_DIR", root),
                    patch("sys.argv", arguments),
                    patch(
                        "scripts.run_stand_ablation.run_variant", return_value=result
                    ),
                    patch(
                        "scripts.run_stand_ablation.run_command",
                        side_effect=fake_robustness,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    exit_code = main()
                saved = json.loads(
                    (output / "comparison.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved["robustness_complete"], not missing_case)
                self.assertEqual(
                    saved["robustness_candidates_passed"],
                    [str(checkpoint.resolve())] if passed and not missing_case else [],
                )
                self.assertEqual(exit_code, 2 if missing_case else 0)


if __name__ == "__main__":
    unittest.main()
