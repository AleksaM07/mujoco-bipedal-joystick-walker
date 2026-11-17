import tempfile
import unittest
from pathlib import Path

from tools.check_brake_training_gate import select_brake_checkpoint


class BrakeTrainingGateTests(unittest.TestCase):
    def test_selects_handoff_checkpoint_instead_of_highest_reward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoints = run_dir / "checkpoints"
            (checkpoints / "000000000100").mkdir(parents=True)
            (checkpoints / "000000000200").mkdir()
            (run_dir / "train.log").write_text(
                "2026-01-01 - eval | step=100 | reward=999 | "
                "episode_length=200 | brake_capture_ready=0.000 | "
                "brake_handoff_ready=0.000 | done_low=0 | done_tip=0 | done_nan=0\n"
                "2026-01-01 - eval | step=200 | reward=500 | "
                "episode_length=200 | brake_capture_ready=0.080 | "
                "brake_handoff_ready=0.050 | done_low=0 | done_tip=0 | done_nan=0\n",
                encoding="utf-8",
            )

            result = select_brake_checkpoint(run_dir)

        self.assertTrue(result["passed"])
        self.assertEqual(result["step"], 200)
        self.assertEqual(result["brake_handoff_fraction"], 0.05)

    def test_does_not_divide_physical_termination_by_episode_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "checkpoints" / "000000000100"
            checkpoint.mkdir(parents=True)
            (run_dir / "train.log").write_text(
                "2026-01-01 - eval | step=100 | reward=500 | "
                "episode_length=200 | brake_capture_ready=0.080 | "
                "brake_handoff_ready=0.050 | terminated=0.900\n",
                encoding="utf-8",
            )

            result = select_brake_checkpoint(run_dir)

        self.assertFalse(result["passed"])
        self.assertEqual(result["physical_termination_fraction"], 0.9)

    def test_prefers_a_safe_passing_checkpoint_over_larger_unsafe_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoints = run_dir / "checkpoints"
            (checkpoints / "000000000100").mkdir(parents=True)
            (checkpoints / "000000000200").mkdir()
            (run_dir / "train.log").write_text(
                "2026-01-01 - eval | step=100 | reward=500 | "
                "episode_length=200 | brake_capture_ready=0.100 | "
                "brake_handoff_ready=0.080 | terminated=0.900\n"
                "2026-01-01 - eval | step=200 | reward=450 | "
                "episode_length=200 | brake_capture_ready=0.060 | "
                "brake_handoff_ready=0.030 | terminated=0.100\n",
                encoding="utf-8",
            )

            result = select_brake_checkpoint(run_dir)

        self.assertTrue(result["passed"])
        self.assertEqual(result["step"], 200)


if __name__ == "__main__":
    unittest.main()
