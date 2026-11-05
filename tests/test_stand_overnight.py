"""Exercise the overnight queue without training or using the GPU."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts.run_stand_ablation import build_configs, write_report
from scripts.run_stand_overnight import (
    completed_result,
    run_queue,
    validate_baseline,
    wait_for_dependency,
)


def experiment_report(passed: bool = False) -> dict:
    """Create one complete stand experiment with the real baseline config."""
    env, training = build_configs("scratch", 1_000_000, "balance_v1")
    gate = {
        "checkpoint": "candidate/000000204800",
        "passed": passed,
        "rollouts": [
            {
                "seed": seed,
                "steps": 500 if passed else 140,
                "finite": True,
                "terminated": not passed,
                "final_success": passed,
                "final_dwell_seconds": 1.0 if passed else 0.0,
                "passed": passed,
                "max_drift": 0.02 if passed else 1.5,
                "mean_speed": 0.01 if passed else 0.6,
                "reward": 400.0,
            }
            for seed in (11, 22, 33, 44)
        ],
    }
    return {
        "status": "complete",
        "video_complete": True,
        "video_exit": 0 if passed else 1,
        "stand_gate_passed": passed,
        "plan": {"scratch": {"env": asdict(env), "train": asdict(training)}},
        "variants": [
            {
                "variant": "scratch",
                "status": "evaluated",
                "training_exit": 0,
                "gate_exit": 0 if passed else 1,
                "best_checkpoint": gate["checkpoint"],
                "best_gate": gate,
            }
        ],
    }


class StandOvernightTests(unittest.TestCase):
    def test_report_requires_complete_video_and_consistent_four_seed_gate(self) -> None:
        valid = experiment_report(True)
        self.assertTrue(completed_result(valid)["best_gate"]["passed"])
        for field, value in (
            ("status", "running"),
            ("video_complete", False),
            ("stand_gate_passed", False),
            ("video_exit", 1),
        ):
            invalid = {**valid, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                completed_result(invalid)
        invalid = copy.deepcopy(valid)
        invalid["variants"][0]["best_gate"]["rollouts"][0]["steps"] = 499
        with self.assertRaises(ValueError):
            completed_result(invalid)
        invalid = copy.deepcopy(valid)
        invalid["variants"][0]["best_gate"]["rollouts"][0]["seed"] = 22
        with self.assertRaises(ValueError):
            completed_result(invalid)

    def test_baseline_config_rejects_changed_physics_or_training(self) -> None:
        valid = experiment_report()
        validate_baseline(valid)
        valid["plan"]["scratch"]["train"]["run_tag"] = "older_label"
        validate_baseline(valid)
        for section, field, value in (
            ("env", "stand_reward_mode", "imitation"),
            ("train", "learning_rate", 1e-3),
        ):
            invalid = copy.deepcopy(valid)
            invalid["plan"]["scratch"][section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_baseline(invalid)

    def test_waits_for_parent_exit_even_after_report_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root / "comparison.json", experiment_report())
            with (
                patch(
                    "scripts.run_stand_overnight.process_matches",
                    side_effect=[True, False],
                ),
                patch("scripts.run_stand_overnight.time.sleep") as sleep,
            ):
                result = wait_for_dependency(root, 123, root / "STOP")
            self.assertEqual(result["status"], "complete")
            sleep.assert_called_once_with(15)

    def test_dead_dependency_or_stop_file_never_starts_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root / "comparison.json", {"status": "running"})
            with (
                patch(
                    "scripts.run_stand_overnight.process_matches", return_value=False
                ),
                self.assertRaises(ValueError),
            ):
                wait_for_dependency(root, 123, root / "STOP")
            (root / "STOP").touch()
            with self.assertRaises(InterruptedError):
                wait_for_dependency(root, 123, root / "STOP")

    def test_queue_stops_on_success_failure_and_authorized_budget(self) -> None:
        scenarios = [
            (True, False, 0, "success", 0),
            (False, True, 0, "success", 1),
            (False, False, 0, "budget_exhausted", 4),
            (False, False, 2, "error", 1),
        ]
        for (
            initial_success,
            next_success,
            exit_code,
            expected_status,
            count,
        ) in scenarios:
            with (
                self.subTest(status=expected_status, count=count),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                calls = []

                def fake_run(
                    command: list[str],
                    log_path: Path,
                    recorded_calls: list[list[str]] = calls,
                    success: bool = next_success,
                    return_code: int = exit_code,
                ) -> int:
                    recorded_calls.append(command)
                    output = Path(command[command.index("--output") + 1])
                    output.mkdir()
                    write_report(output / "comparison.json", experiment_report(success))
                    return return_code

                with (
                    patch("scripts.run_stand_overnight.check_branch"),
                    patch(
                        "scripts.run_stand_overnight.wait_for_dependency",
                        return_value=experiment_report(initial_success),
                    ),
                    patch(
                        "scripts.run_stand_overnight.run_command", side_effect=fake_run
                    ),
                ):
                    result = run_queue(root / "previous", 123, root / "night", 4)
                state = json.loads((root / "night/queue.json").read_text("utf-8"))
                self.assertEqual(state["status"], expected_status)
                self.assertEqual(len(calls), count)
                self.assertEqual(result, 2 if exit_code else 0)
                self.assertTrue((root / "night/summary.md").is_file())
                for command in calls:
                    self.assertEqual(
                        command[command.index("--timesteps") + 1], "1000000"
                    )
                    self.assertNotIn("--skip-video", command)

    def test_existing_output_and_excess_budget_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                run_queue(root, 123, root, 4)
            with self.assertRaises(ValueError):
                run_queue(root, 123, root / "new", 5)

    def test_stop_during_run_finishes_it_but_never_starts_next(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "night"

            def fake_run(command: list[str], log_path: Path) -> int:
                run_dir = Path(command[command.index("--output") + 1])
                run_dir.mkdir()
                write_report(run_dir / "comparison.json", experiment_report())
                (output / "STOP").touch()
                return 0

            with (
                patch("scripts.run_stand_overnight.check_branch"),
                patch(
                    "scripts.run_stand_overnight.wait_for_dependency",
                    return_value=experiment_report(),
                ),
                patch(
                    "scripts.run_stand_overnight.run_command", side_effect=fake_run
                ) as run,
            ):
                self.assertEqual(run_queue(root / "previous", 123, output, 4), 0)
            run.assert_called_once()
            state = json.loads((output / "queue.json").read_text("utf-8"))
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(len(state["results"]), 2)


if __name__ == "__main__":
    unittest.main()
