"""Tests for finite transition checkpoint selection."""

import tempfile
import unittest
from pathlib import Path

from tools.check_transition_training_gate import select_finite_checkpoint


def eval_line(
    step: int,
    *,
    reward: float,
    done_motion: float,
    done_low: float,
) -> str:
    """Build the relevant part of one loguru evaluation line."""
    return (
        "2026-11-16 | INFO | train - eval | "
        f"step={step} | reward={reward} | episode_length=100 | "
        f"done_low={done_low} | done_tip=0 | done_nan=0 | "
        f"done_motion={done_motion}"
    )


class TransitionTrainingGateTests(unittest.TestCase):
    def test_completion_beats_higher_reward_with_more_falls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "checkpoints" / "000000200000").mkdir(parents=True)
            (run_dir / "checkpoints" / "000000400000").mkdir(parents=True)
            (run_dir / "train.log").write_text(
                "\n".join(
                    (
                        eval_line(
                            200_000,
                            reward=500.0,
                            done_motion=0.5,
                            done_low=0.5,
                        ),
                        eval_line(
                            400_000,
                            reward=350.0,
                            done_motion=1.0,
                            done_low=0.0,
                        ),
                    )
                ),
                encoding="utf-8",
            )

            result = select_finite_checkpoint(
                run_dir,
                min_motion_completion=0.75,
                max_fall_fraction=0.25,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["step"], 400_000)
        self.assertEqual(result["done_motion"], 1.0)

    def test_gate_rejects_saved_checkpoints_that_still_fall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "checkpoints" / "000000200000").mkdir(parents=True)
            (run_dir / "train.log").write_text(
                eval_line(
                    200_000,
                    reward=500.0,
                    done_motion=0.5,
                    done_low=0.5,
                ),
                encoding="utf-8",
            )

            result = select_finite_checkpoint(
                run_dir,
                min_motion_completion=0.75,
                max_fall_fraction=0.25,
            )

        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
