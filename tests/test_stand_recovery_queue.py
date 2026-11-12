"""Exercise the fixed stand recovery queue without training on the GPU."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_stand_ablation import write_report
from scripts.run_stand_recovery_queue import (
    RUNS,
    build_command,
    check_repository,
    run_queue,
    validate_completed_report,
)
from tests.test_stand_overnight import experiment_report


def robust_report(root: Path, passed: bool = False) -> dict:
    """Create one complete five-checkpoint, five-probe runner report."""
    report = experiment_report(passed)
    run_dir = root / "training"
    checkpoints = [
        run_dir / "checkpoints" / f"{step:012d}"
        for step in (404_480, 808_960, 1_213_440, 1_617_920, 2_022_400)
    ]
    for checkpoint in checkpoints:
        checkpoint.mkdir(parents=True)
    report["variants"][0]["run_dir"] = str(run_dir)
    report["variants"][0]["best_checkpoint"] = str(checkpoints[-1])
    cases = ("hold_60s", "push_x_pos", "push_x_neg", "push_y_pos", "push_y_neg")
    robustness = {
        "status": "complete",
        "results": [
            {
                "checkpoint": str(checkpoint.resolve()),
                "probe": {"name": case},
                "passed": passed and checkpoint == checkpoints[-1],
            }
            for checkpoint in checkpoints
            for case in cases
        ],
    }
    robustness_path = root / "robustness/report.json"
    robustness_path.parent.mkdir()
    robustness_path.write_text(json.dumps(robustness), encoding="utf-8")
    report.update(
        robustness_requested=True,
        robustness_complete=True,
        robustness_exit=0 if passed else 1,
        robustness_report=str(robustness_path),
        robustness_candidates_passed=[str(checkpoints[-1].resolve())] if passed else [],
    )
    return report


class StandRecoveryQueueTests(unittest.TestCase):
    def test_repository_check_normalizes_windows_line_endings_under_wsl(self) -> None:
        with patch(
            "scripts.run_stand_recovery_queue.subprocess.check_output",
            side_effect=[
                "skills_walking_with_anatomical_human_robot\n",
                "",
            ],
        ) as check:
            check_repository()
        status_command = check.call_args_list[1].args[0]
        self.assertEqual(
            status_command[:4], ["git", "-c", "core.autocrlf=true", "status"]
        )

    def test_commands_are_a_matched_two_run_push_distribution_test(self) -> None:
        source = Path("runs/source/checkpoints/000000409600")
        output = Path("runs/night")
        commands = [build_command(spec, source, output / spec.name) for spec in RUNS]
        for command in commands:
            self.assertEqual(command[command.index("--timesteps") + 1], "2000000")
            self.assertEqual(
                command[command.index("--stand-reward-mode") + 1], "recovery_v2"
            )
            self.assertEqual(command[command.index("--learning-rate") + 1], "5e-05")
            self.assertIn("--robustness-check", command)
        self.assertEqual(
            commands[0][commands[0].index("--stand-push-min-fraction") + 1], "0.0"
        )
        self.assertEqual(
            commands[1][commands[1].index("--stand-push-min-fraction") + 1], "0.04"
        )

    def test_completed_report_requires_all_25_robustness_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = robust_report(Path(directory), passed=True)
            self.assertEqual(len(validate_completed_report(report)), 1)
            invalid = copy.deepcopy(report)
            robustness_path = Path(invalid["robustness_report"])
            robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
            robustness["results"].pop()
            robustness_path.write_text(json.dumps(robustness), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "25 unique"):
                validate_completed_report(invalid)

    def test_dependency_success_stops_before_either_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / "active"
            report = robust_report(dependency, passed=True)
            output = root / "queue"
            with (
                patch("scripts.run_stand_recovery_queue.check_repository"),
                patch(
                    "scripts.run_stand_recovery_queue.wait_for_dependency",
                    return_value=report,
                ),
                patch("scripts.run_stand_recovery_queue.run_command") as run,
            ):
                self.assertEqual(
                    run_queue(dependency, 436, root / "source", output), 0
                )
            run.assert_not_called()
            state = json.loads((output / "queue.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "success")
            self.assertEqual(len(state["results"]), 1)

    def test_queue_runs_broad_then_hard_and_stops_on_second_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / "active"
            source = root / "source"
            initial = robust_report(dependency, passed=False)
            calls: list[list[str]] = []

            def fake_run(command: list[str], log_path: Path) -> int:
                calls.append(command)
                run_dir = Path(command[command.index("--output") + 1])
                report = robust_report(run_dir, passed=len(calls) == 2)
                run_dir.mkdir(exist_ok=True)
                write_report(run_dir / "comparison.json", report)
                return 0

            with (
                patch("scripts.run_stand_recovery_queue.check_repository"),
                patch(
                    "scripts.run_stand_recovery_queue.wait_for_dependency",
                    return_value=initial,
                ),
                patch(
                    "scripts.run_stand_recovery_queue.run_command", side_effect=fake_run
                ),
            ):
                self.assertEqual(
                    run_queue(dependency, 436, source, root / "queue"), 0
                )
            self.assertEqual(len(calls), 2)
            self.assertIn("run_1_recovery_broad", calls[0][-1])
            self.assertIn("run_2_recovery_hard", calls[1][-1])
            state = json.loads(
                (root / "queue/queue.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "success")
            self.assertEqual(len(state["results"]), 3)


if __name__ == "__main__":
    unittest.main()
