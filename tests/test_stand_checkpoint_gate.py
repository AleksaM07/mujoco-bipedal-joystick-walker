"""Physical acceptance must survive transient successes and terminal failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.evaluate_stand import load_stand_config, summarize_rollout


class StandCheckpointGateTests(unittest.TestCase):
    def sample(self, **changes):
        values = {
            "finite": 1.0,
            "physical_failure": 0.0,
            "done": 0.0,
            "success": 1.0,
            "dwell_steps": 50.0,
            "distance": 0.02,
            "speed": 0.01,
            "torso_up": 0.99,
            "height": 0.9,
            "height_ratio": 0.99,
            "eligible": 1.0,
            "reward": 3.0,
        }
        return values | changes

    def summarize(self, samples):
        return summarize_rollout(samples, seed=11, dt=0.02, dwell_steps=50)

    def test_complete_physical_success(self):
        report = self.summarize([self.sample()] * 500)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["final_dwell_seconds"], 1)
        self.assertAlmostEqual(report["mean_speed"], 0.01)
        self.assertAlmostEqual(report["reward"], 1500)

    def test_transient_success_does_not_replace_final_success(self):
        samples = [self.sample()] * 499 + [self.sample(success=0, dwell_steps=0)]
        self.assertFalse(self.summarize(samples)["passed"])

    def test_last_step_fall_and_early_stop_fail_even_with_success_metric(self):
        for changes in ({"physical_failure": 1}, {"done": 1}):
            with self.subTest(changes=changes):
                samples = [self.sample()] * 499 + [self.sample(**changes)]
                self.assertFalse(self.summarize(samples)["passed"])
        self.assertFalse(self.summarize([self.sample()] * 499)["passed"])

    def test_nonfinite_is_failure_and_report_is_valid_json(self):
        samples = [self.sample()] * 499 + [self.sample(speed=float("nan"), finite=0)]
        report = self.summarize(samples)
        self.assertFalse(report["passed"])
        self.assertFalse(report["finite"])
        self.assertIsNone(report["max_speed"])
        json.dumps(report, allow_nan=False)

    def test_saved_settings_are_preserved_and_archive_xml_resolves_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (root / "robot.xml").write_text("<mujoco/>", encoding="utf-8")
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "env": {
                            "locomotion_skill": "stand",
                            "xml_path": "robot.xml",
                            "action_smoothing": 0.37,
                            "reference_residual_scale": 0.23,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_stand_config(checkpoint)
            self.assertEqual(config.action_smoothing, 0.37)
            self.assertEqual(config.reference_residual_scale, 0.23)
            self.assertEqual(Path(config.xml_path), root / "robot.xml")
            self.assertEqual(config.reference_gait_file, ())

    @patch("tools.evaluate_stand.find_run_config", return_value={"env": {}})
    def test_walk_conversion_requires_explicit_baseline_flag(self, _):
        with self.assertRaisesRegex(ValueError, "walking-baseline"):
            load_stand_config(Path("checkpoint"))

    @patch(
        "tools.evaluate_stand.find_run_config",
        return_value={
            "env": {"locomotion_skill": "stand", "physics_backend": "mjx_warp"}
        },
    )
    def test_saved_backend_cannot_silently_migrate(self, _):
        with self.assertRaisesRegex(ValueError, "backend migration"):
            load_stand_config(Path("checkpoint"))


if __name__ == "__main__":
    unittest.main()
