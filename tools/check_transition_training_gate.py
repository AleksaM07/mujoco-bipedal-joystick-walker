"""Select a finite-motion checkpoint by completion rather than raw reward."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC_PATTERN = re.compile(r"([a-zA-Z_]+)=([-+0-9.eE]+)")


def parse_eval_rows(log_path: Path) -> list[dict[str, float]]:
    """Parse structured evaluation metrics from a training log."""
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " - eval | step=" not in line:
            continue
        values = {
            key: float(value) for key, value in METRIC_PATTERN.findall(line)
        }
        if "step" in values:
            rows.append(values)
    if not rows:
        raise ValueError(f"No evaluation rows found in {log_path}")
    return rows


def checkpoint_for_step(run_dir: Path, step: int) -> Path | None:
    """Return the checkpoint written for an evaluation step, if present."""
    checkpoint_path = run_dir / "checkpoints" / f"{step:012d}"
    return checkpoint_path if checkpoint_path.is_dir() else None


def select_finite_checkpoint(
    run_dir: Path,
    *,
    min_motion_completion: float,
    max_fall_fraction: float,
) -> dict[str, float | str | bool]:
    """Choose the strongest saved finite-motion evaluation and apply its gate."""
    candidates: list[tuple[dict[str, float], Path]] = []
    for row in parse_eval_rows(run_dir / "train.log"):
        checkpoint_path = checkpoint_for_step(run_dir, int(row["step"]))
        if checkpoint_path is not None:
            candidates.append((row, checkpoint_path))
    if not candidates:
        raise ValueError(f"No evaluated checkpoints found under {run_dir}")

    def rank(candidate: tuple[dict[str, float], Path]) -> tuple[float, ...]:
        row, _ = candidate
        fall_fraction = sum(
            row.get(key, 0.0) for key in ("done_low", "done_tip", "done_nan")
        )
        return (
            row.get("done_motion", 0.0),
            -fall_fraction,
            row.get("episode_length", 0.0),
            row.get("reward", float("-inf")),
        )

    row, checkpoint_path = max(candidates, key=rank)
    fall_fraction = sum(
        row.get(key, 0.0) for key in ("done_low", "done_tip", "done_nan")
    )
    passed = (
        row.get("done_motion", 0.0) >= min_motion_completion
        and fall_fraction <= max_fall_fraction
    )
    return {
        "passed": passed,
        "checkpoint": str(checkpoint_path.resolve()),
        "step": int(row["step"]),
        "reward": row.get("reward", 0.0),
        "episode_length": row.get("episode_length", 0.0),
        "done_motion": row.get("done_motion", 0.0),
        "done_low": row.get("done_low", 0.0),
        "done_tip": row.get("done_tip", 0.0),
        "done_nan": row.get("done_nan", 0.0),
        "fall_fraction": fall_fraction,
        "min_motion_completion": min_motion_completion,
        "max_fall_fraction": max_fall_fraction,
    }


def main() -> None:
    """Run the finite transition checkpoint gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-motion-completion", type=float, default=0.75)
    parser.add_argument("--max-fall-fraction", type=float, default=0.25)
    parser.add_argument("--print-checkpoint", action="store_true")
    args = parser.parse_args()

    result = select_finite_checkpoint(
        args.run_dir.resolve(),
        min_motion_completion=args.min_motion_completion,
        max_fall_fraction=args.max_fall_fraction,
    )
    if args.print_checkpoint:
        print(result["checkpoint"])
    else:
        print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
