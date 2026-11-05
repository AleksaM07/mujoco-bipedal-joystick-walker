"""Tests for the deterministic natural-walk ablation report helpers."""

from pathlib import Path
import unittest

from tools.audit_natural_walk_ablations import ABLATIONS, load_base_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NaturalWalkAblationAuditTests(unittest.TestCase):
    def test_ablations_change_one_reference_dimension_each(self) -> None:
        self.assertEqual(ABLATIONS["current"], {})
        for name, overrides in ABLATIONS.items():
            if name == "current":
                continue
            expected = 2 if name.endswith("_no_lock") and name != "no_stance_lock" else 1
            self.assertEqual(len(overrides), expected, name)

    def test_archived_config_loads_as_single_clip_full_arm_baseline(self) -> None:
        config = load_base_config(
            PROJECT_ROOT
            / "prezentacija/5th_success_final_natural_walk/config.json"
        )
        self.assertEqual(config.arm_reference_mode, "cmu_elbow_wrist_ik")
        self.assertEqual(config.reference_clip_ids, [1])
        self.assertEqual(config.reference_forced_clip_id, 0)
        self.assertTrue(config.arm_actuators)
        self.assertEqual(config.warp_num_worlds, 1)
        self.assertEqual(config.reference_elbow_flexion_alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
