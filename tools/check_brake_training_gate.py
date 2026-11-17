"""Select a BRAKE_TRANSITION checkpoint by capture readiness, not reward."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_transition_training_gate import checkpoint_for_step, parse_eval_rows


def select_brake_checkpoint(
    run_dir: Path,
    *,
    min_handoff_fraction: float = 0.02,
    max_fall_fraction: float = 0.25,
) -> dict[str, float | str | bool]:
    """Rank saved evaluations by handoff occupancy and physical safety."""
    candidates: list[tuple[dict[str, float], Path]] = []
    for row in parse_eval_rows(run_dir / "train.log"):
        checkpoint = checkpoint_for_step(run_dir, int(row["step"]))
        if checkpoint is not None:
            candidates.append((row, checkpoint))
    if not candidates:
        raise ValueError(f"No evaluated checkpoints found under {run_dir}")

    def fractions(row: dict[str, float]) -> tuple[float, float, float]:
        length = max(row.get("episode_length", 0.0), 1.0)
        handoff = row.get("brake_handoff_ready", 0.0)
        capture = row.get("brake_capture_ready", 0.0)
        fall = sum(
            row.get(key, 0.0) for key in ("done_low", "done_tip", "done_nan")
        )
        return handoff, capture, fall / length

    row, checkpoint = max(
        candidates,
        key=lambda candidate: (
            fractions(candidate[0])[0],
            fractions(candidate[0])[1],
            -fractions(candidate[0])[2],
            candidate[0].get("episode_length", 0.0),
        ),
    )
    handoff_fraction, capture_fraction, fall_fraction = fractions(row)
    passed = (
        handoff_fraction >= min_handoff_fraction
        and fall_fraction <= max_fall_fraction
    )
    return {
        "passed": passed,
        "checkpoint": str(checkpoint.resolve()),
        "step": int(row["step"]),
        "reward": row.get("reward", 0.0),
        "episode_length": row.get("episode_length", 0.0),
        "brake_handoff_fraction": handoff_fraction,
        "brake_capture_fraction": capture_fraction,
        "fall_fraction": fall_fraction,
        "min_handoff_fraction": min_handoff_fraction,
        "max_fall_fraction": max_fall_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-handoff-fraction", type=float, default=0.02)
    parser.add_argument("--max-fall-fraction", type=float, default=0.25)
    parser.add_argument("--print-checkpoint", action="store_true")
    args = parser.parse_args()
    result = select_brake_checkpoint(
        args.run_dir.resolve(),
        min_handoff_fraction=args.min_handoff_fraction,
        max_fall_fraction=args.max_fall_fraction,
    )
    output = result["checkpoint"] if args.print_checkpoint else json.dumps(result, indent=2)
    print(output)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
