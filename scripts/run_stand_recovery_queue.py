"""Wait for the active stand run, then execute two controlled recovery tests.

The queue is deliberately finite. It does not edit code or choose new
experiments: it runs the two declared A/B conditions from the same checkpoint
and stops early when any checkpoint passes all five robustness probes.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stand_ablation import (
    RUNS_DIR,
    build_configs,
    run_command,
    write_report,
)
from scripts.run_stand_overnight import completed_result

logger = logging.getLogger(__name__)

BRANCH = "skills_walking_with_anatomical_human_robot"
TIMESTEPS = 2_000_000
LEARNING_RATE = 5e-5
EPISODE_LENGTH = 3_000
ROBUSTNESS_CASES = {
    "hold_60s",
    "push_x_pos",
    "push_x_neg",
    "push_y_pos",
    "push_y_neg",
}


@dataclass(frozen=True)
class RecoveryRun:
    """One fixed recovery-reward training condition."""

    name: str
    push_min_fraction: float
    push_max_fraction: float
    push_probability: float


RUNS = (
    RecoveryRun("run_1_recovery_broad", 0.0, 0.05, 0.5),
    RecoveryRun("run_2_recovery_hard", 0.04, 0.05, 0.75),
)


def process_matches(pid: int, experiment: Path) -> bool:
    """Check that a Linux PID is still the requested ablation runner."""
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    decoded = [os.fsdecode(value) for value in arguments if value]
    if not any(value.endswith("run_stand_ablation.py") for value in decoded):
        return False
    return any(
        value == "--output"
        and index + 1 < len(decoded)
        and (PROJECT_ROOT / decoded[index + 1]).resolve() == experiment
        for index, value in enumerate(decoded)
    )


def validate_completed_report(report: dict[str, Any]) -> list[str]:
    """Require the gate, videos and every checkpoint/probe robustness pair."""
    completed_result(report)
    if (
        report.get("robustness_requested") is not True
        or report.get("robustness_complete") is not True
        or report.get("robustness_exit") not in (0, 1)
    ):
        raise ValueError("Experiment robustness evaluation is incomplete.")
    passed = report.get("robustness_candidates_passed")
    if not isinstance(passed, list) or not all(
        isinstance(checkpoint, str) for checkpoint in passed
    ):
        raise ValueError("Invalid robustness pass list.")

    variants = report["variants"]
    if len(variants) != 1:
        raise ValueError("Recovery queue expects exactly one evaluated variant.")
    checkpoints = sorted(
        path.resolve()
        for path in (Path(variants[0]["run_dir"]) / "checkpoints").iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if len(checkpoints) != 5:
        raise ValueError("Expected five saved checkpoints for 25 robustness probes.")
    robust = json.loads(Path(report["robustness_report"]).read_text(encoding="utf-8"))
    observed = {
        (Path(row["checkpoint"]).resolve(), row["probe"]["name"])
        for row in robust["results"]
    }
    expected = {
        (checkpoint, case) for checkpoint in checkpoints for case in ROBUSTNESS_CASES
    }
    if robust.get("status") != "complete" or observed != expected:
        raise ValueError("Expected all 25 unique checkpoint/probe robustness results.")
    if set(map(Path, passed)) - set(checkpoints):
        raise ValueError("Robustness pass list contains an unknown checkpoint.")
    return passed


def wait_for_dependency(
    experiment: Path,
    pid: int,
    stop_file: Path,
    timeout_seconds: float = 64_800,
) -> dict[str, Any]:
    """Wait for the existing runner and its post-training evaluations."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if stop_file.exists():
            raise InterruptedError("STOP requested while waiting for the active run.")
        try:
            report = json.loads((experiment / "comparison.json").read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            report = None
        if not process_matches(pid, experiment):
            if report is None:
                raise RuntimeError("Active runner exited without a readable report.")
            validate_completed_report(report)
            return report
        time.sleep(15)
    raise TimeoutError("Active stand runner did not finish within 18 hours.")


def check_repository() -> None:
    """Refuse to train from another branch or tracked uncommitted source."""
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch}.")
    changed = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if changed:
        raise RuntimeError(
            "Tracked repository files changed while the queue was waiting."
        )


def build_command(spec: RecoveryRun, source: Path, output: Path) -> list[str]:
    """Build one fully specified resume experiment command."""
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_stand_ablation.py"),
        "--variants",
        "resume",
        "--resume-from",
        str(source),
        "--timesteps",
        str(TIMESTEPS),
        "--stand-reward-mode",
        "recovery_v2",
        "--learning-rate",
        str(LEARNING_RATE),
        "--episode-length",
        str(EPISODE_LENGTH),
        "--stand-push-min-fraction",
        str(spec.push_min_fraction),
        "--stand-push-max-fraction",
        str(spec.push_max_fraction),
        "--stand-push-probability",
        str(spec.push_probability),
        "--robustness-check",
        "--output",
        str(output),
    ]


def summarize_result(
    label: str, experiment: Path, report: dict[str, Any], passed: list[str]
) -> dict[str, Any]:
    """Keep queue state compact while retaining the decision evidence."""
    return {
        "label": label,
        "experiment": str(experiment),
        "best_checkpoint": report["variants"][0]["best_checkpoint"],
        "robustness_candidates_passed": passed,
    }


def save_status(output: Path, state: dict[str, Any]) -> None:
    """Persist machine-readable state and a compact morning report."""
    state["updated_utc"] = datetime.now(UTC).isoformat()
    write_report(output / "queue.json", state)
    lines = [
        "# Stand recovery queue",
        "",
        f"Status: {state['status']}",
        f"Reason: {state.get('reason', '')}",
        f"Updated UTC: {state['updated_utc']}",
        "",
        "Both recovery tests use the same source checkpoint, LR and 2M budget.",
        "The only A/B difference is the training push distribution.",
        "",
        "| Result | Robust checkpoint passes | Experiment |",
        "| --- | ---: | --- |",
    ]
    for result in state["results"]:
        lines.append(
            f"| {result['label']} | "
            f"{len(result['robustness_candidates_passed'])} | "
            f"`{result['experiment']}` |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_queue(experiment: Path, pid: int, source: Path, output: Path) -> int:
    """Run at most two declared tests, stopping on success or technical failure."""
    output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "waiting",
        "dependency": str(experiment),
        "dependency_pid": pid,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "source_checkpoint": str(source),
        "tests": [asdict(spec) for spec in RUNS],
        "results": [],
    }
    save_status(output, state)
    try:
        check_repository()
        report = wait_for_dependency(experiment, pid, output / "STOP")
        passed = validate_completed_report(report)
        state["results"].append(
            summarize_result("active_balance_v1_hard", experiment, report, passed)
        )
        if passed:
            state.update(
                status="success",
                reason="The active run already passed all five robustness probes.",
            )
        else:
            for spec in RUNS:
                if (output / "STOP").exists():
                    state.update(
                        status="stopped", reason="STOP requested between experiments."
                    )
                    break
                check_repository()
                run_dir = output / spec.name
                state.update(status="running", current_run=str(run_dir))
                save_status(output, state)
                exit_code = run_command(
                    build_command(spec, source, run_dir),
                    output / f"{spec.name}.log",
                )
                if exit_code != 0:
                    raise RuntimeError(
                        f"{spec.name} exited with technical code {exit_code}."
                    )
                report = json.loads(
                    (run_dir / "comparison.json").read_text(encoding="utf-8")
                )
                passed = validate_completed_report(report)
                state["results"].append(
                    summarize_result(spec.name, run_dir, report, passed)
                )
                save_status(output, state)
                if passed:
                    state.update(
                        status="success",
                        reason=f"{spec.name} passed all five robustness probes.",
                    )
                    break
            else:
                state.update(
                    status="budget_exhausted",
                    reason=(
                        "Both authorized recovery tests completed without "
                        "a robust pass."
                    ),
                )
    except InterruptedError as exc:
        state.update(status="stopped", reason=str(exc))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        logger.exception("Stand recovery queue stopped")
        state.update(status="error", reason=str(exc))
    save_status(output, state)
    return 2 if state["status"] == "error" else 0


def main() -> int:
    """Validate paths and reserve the single recovery queue lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment = args.after.resolve()
    source = args.source_checkpoint.resolve()
    output = args.output.resolve()
    root = RUNS_DIR.resolve()
    if any(
        path == root or not path.is_relative_to(root)
        for path in (experiment, source, output)
    ):
        parser.error("All paths must stay under runs/.")
    if not (experiment / "comparison.json").is_file():
        parser.error("The dependency needs an existing comparison.json.")
    if (
        source.parent.name != "checkpoints"
        or not source.name.isdigit()
        or not (source / "ppo_network_config.json").is_file()
    ):
        parser.error("Expected one saved numeric source checkpoint.")
    if output.exists() or output.is_relative_to(experiment):
        parser.error("Output must be fresh and separate from the dependency.")
    if args.after_pid <= 0:
        parser.error("after-pid must be positive")

    for spec in RUNS:
        build_configs(
            "resume",
            TIMESTEPS,
            "recovery_v2",
            LEARNING_RATE,
            source,
            EPISODE_LENGTH,
            spec.push_min_fraction,
            spec.push_max_fraction,
            spec.push_probability,
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "after": str(experiment),
                    "source_checkpoint": str(source),
                    "stop_on_robustness_success": True,
                    "commands": [
                        build_command(spec, source, output / spec.name) for spec in RUNS
                    ],
                },
                indent=2,
            )
        )
        return 0
    if sys.platform != "linux":
        parser.error("Run this queue with the WSL Python environment.")

    import fcntl

    with (root / ".stand_recovery_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another stand recovery queue is already running.")
        return run_queue(experiment, args.after_pid, source, output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
