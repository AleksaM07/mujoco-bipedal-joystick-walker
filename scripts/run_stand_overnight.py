"""Wait for one stand result, then run a bounded, sequential learning-rate sweep.

This is a fixed experiment queue, not an autonomous coding agent. It never
edits model code, switches branches, pushes commits, or starts navigation work.
Run with the WSL Python environment used for stand training.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stand_ablation import (
    RUNS_DIR,
    build_configs,
    checkpoint_rank,
    run_command,
    write_report,
)

logger = logging.getLogger(__name__)
BRANCH = "navigation_walking_with_anatomical_human_robot"
LEARNING_RATES = (6e-5, 1e-4, 2e-4, 3e-4)


def process_matches(pid: int, experiment: Path) -> bool:
    """Check Linux process identity, not just a potentially recycled PID."""
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


def completed_result(report: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete jobs; a failed physical gate is still a valid result."""
    if report.get("status") != "complete" or report.get("video_complete") is not True:
        raise ValueError("Experiment or required videos are incomplete.")
    variants = report.get("variants", [])
    if not variants or any(
        row.get("status") != "evaluated"
        or row.get("training_exit") != 0
        or row.get("gate_exit") not in (0, 1)
        for row in variants
    ):
        raise ValueError("Training or checkpoint evaluation failed.")
    best = max(variants, key=lambda row: checkpoint_rank(row["best_gate"]))
    gate = best["best_gate"]
    rollouts = gate["rollouts"]
    if {row["seed"] for row in rollouts} != {11, 22, 33, 44} or len(rollouts) != 4:
        raise ValueError("Expected four mandatory physical rollouts.")
    passed = all(
        row["passed"] is True
        and row["steps"] == 500
        and row["finite"] is True
        and row["terminated"] is False
        and row["final_success"] is True
        and row["final_dwell_seconds"] >= 1.0
        for row in rollouts
    )
    if (
        gate.get("passed") is not passed
        or report.get("stand_gate_passed") is not passed
    ):
        raise ValueError("Inconsistent stand success flags.")
    if report.get("video_exit") != (0 if passed else 1):
        raise ValueError("Selected video rollout disagrees with the physical gate.")
    return best


def wait_for_dependency(
    experiment: Path, pid: int, stop_file: Path, timeout_seconds: float = 10800
) -> dict[str, Any]:
    """Wait for the whole parent pipeline, including videos, to release the GPU."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if stop_file.exists():
            raise InterruptedError("STOP requested while waiting for previous run.")
        try:
            report = json.loads((experiment / "comparison.json").read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            # The already-running older runner writes its report non-atomically.
            report = None
        if not process_matches(pid, experiment):
            if report is None:
                raise RuntimeError("Previous runner exited without a readable report.")
            completed_result(report)
            return report
        time.sleep(15)
    raise TimeoutError("Previous runner did not finish within three hours.")


def check_branch() -> None:
    """Stop before another run if someone changed the working branch."""
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch}.")


def validate_baseline(report: dict[str, Any]) -> None:
    """Only compare against the authorized scratch balance_v1 baseline."""
    environment, training = build_configs("scratch", 1_000_000, "balance_v1")
    plan = report["plan"]
    if set(plan) != {"scratch"} or plan["scratch"]["env"] != asdict(environment):
        raise ValueError(
            "Dependency does not match the scratch balance_v1 environment."
        )
    actual, expected = dict(plan["scratch"]["train"]), asdict(training)
    actual.pop("run_tag")
    expected.pop("run_tag")
    if actual != expected:
        raise ValueError(
            "Dependency does not match the controlled baseline PPO config."
        )


def save_status(output: Path, state: dict[str, Any]) -> None:
    """Persist machine-readable state and a compact morning report."""
    state["updated_utc"] = datetime.now(UTC).isoformat()
    write_report(output / "queue.json", state)
    lines = [
        "# Overnight stand experiments",
        "",
        f"Status: {state['status']}",
        f"Reason: {state.get('reason', '')}",
        f"Updated UTC: {state['updated_utc']}",
        "",
        "Each run uses balance_v1, seed 7 and scratch initialization.",
        "Physical starts are identical across eval seeds; robustness is not proven.",
        "",
        "| Run | LR | Min steps / 500 | Final success / 4 | Max drift (m) | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in state["results"]:
        rows = result["best_gate"]["rollouts"]
        drift = [row["max_drift"] for row in rows]
        max_drift = max(drift) if all(value is not None for value in drift) else None
        lines.append(
            f"| {Path(result['experiment']).name} | {result['learning_rate']:g} | "
            f"{min(row['steps'] for row in rows)} | "
            f"{sum(row['final_success'] for row in rows)} | "
            f"{max_drift} | {result['best_gate']['passed']} |"
        )
    if state["results"]:
        best = max(state["results"], key=lambda row: checkpoint_rank(row["best_gate"]))
        lines.extend(["", f"Best physical candidate: `{best['best_checkpoint']}`"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_queue(experiment: Path, pid: int, output: Path, max_runs: int) -> int:
    """Stop on success, technical failure, STOP, or the authorized run limit."""
    if not 1 <= max_runs <= len(LEARNING_RATES):
        raise ValueError("max_runs must be between 1 and 4.")
    output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "waiting",
        "dependency": str(experiment),
        "dependency_pid": pid,
        "max_additional_runs": max_runs,
        "learning_rates": LEARNING_RATES[:max_runs],
        "results": [],
    }
    save_status(output, state)
    try:
        check_branch()
        logger.info(
            "Waiting for training, all checkpoint gates and videos: %s", experiment
        )
        report = wait_for_dependency(experiment, pid, output / "STOP")
        validate_baseline(report)
        candidate = completed_result(report)
        state["results"].append(
            {**candidate, "experiment": str(experiment), "learning_rate": 2e-5}
        )
        for index, learning_rate in enumerate(LEARNING_RATES[:max_runs], start=1):
            if candidate["best_gate"]["passed"]:
                state.update(status="success", reason="Four-seed stand gate passed.")
                break
            if (output / "STOP").exists():
                state.update(
                    status="stopped", reason="STOP requested; current run finished."
                )
                break
            check_branch()
            run_dir = output / f"run_{index}_lr{learning_rate:g}"
            state.update(status="running", current_run=str(run_dir))
            save_status(output, state)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run_stand_ablation.py"),
                "--variants",
                "scratch",
                "--stand-reward-mode",
                "balance_v1",
                "--learning-rate",
                str(learning_rate),
                "--timesteps",
                "1000000",
                "--output",
                str(run_dir),
            ]
            logger.info("Starting run %s/%s with LR=%g", index, max_runs, learning_rate)
            exit_code = run_command(command, output / f"run_{index}.log")
            if exit_code != 0:
                raise RuntimeError(f"Run {index} exited with code {exit_code}.")
            report = json.loads((run_dir / "comparison.json").read_text("utf-8"))
            candidate = completed_result(report)
            state["results"].append(
                {
                    **candidate,
                    "experiment": str(run_dir),
                    "learning_rate": learning_rate,
                }
            )
            save_status(output, state)
        else:
            passed = candidate["best_gate"]["passed"]
            state.update(
                status="success" if passed else "budget_exhausted",
                reason="Four-seed stand gate passed."
                if passed
                else "Additional run limit reached.",
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
        logger.exception("Overnight queue stopped")
        state.update(status="error", reason=str(exc))
    save_status(output, state)
    logger.info(
        "Queue finished: %s; summary: %s", state["status"], output / "summary.md"
    )
    return 2 if state["status"] == "error" else 0


def main() -> int:
    """Reserve one WSL queue and wait for the identified existing experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment, output = args.after.resolve(), args.output.resolve()
    root = RUNS_DIR.resolve()
    if any(
        path == root or not path.is_relative_to(root) for path in (experiment, output)
    ):
        parser.error("Input and output must be experiment directories under runs/.")
    if not (experiment / "comparison.json").is_file():
        parser.error("The dependency needs an existing comparison.json.")
    if output.exists() or output.is_relative_to(experiment):
        parser.error("Output must be fresh and separate from the dependency.")
    if args.after_pid <= 0:
        parser.error("after-pid must be positive")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "after": str(experiment),
                    "output": str(output),
                    "learning_rates": LEARNING_RATES[: args.max_runs],
                    "timesteps_each": 1_000_000,
                    "stop_on_success": True,
                },
                indent=2,
            )
        )
        return 0
    if sys.platform != "linux":
        parser.error("Run this queue with the WSL Python environment.")
    import fcntl

    # The OS releases this lock even if the process exits unexpectedly.
    with (root / ".stand_overnight.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another stand overnight queue is already running.")
        return run_queue(experiment, args.after_pid, output, args.max_runs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
