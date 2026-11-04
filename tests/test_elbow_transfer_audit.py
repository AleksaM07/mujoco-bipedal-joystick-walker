"""Tests for elbow-transfer promotion criteria."""

from __future__ import annotations

import unittest

from tools.audit_elbow_transfer import evaluate_promotion_gate


def passing_record(seed: int) -> dict[str, object]:
    """Return one record exactly on the promotion-safe side of every gate."""
    return {
        "seed": seed,
        "length": 500,
        "physical_termination": False,
        "reward_per_step": 2.61,
        "deepmimic_pose": 0.91,
        "deepmimic_arm_pose": 0.71,
        "deepmimic_arm_velocity": 0.70,
        "left_elbow_amplitude_ratio": 0.46,
        "right_elbow_amplitude_ratio": 0.46,
    }


class ElbowTransferGateTests(unittest.TestCase):
    def test_gate_passes_retained_walk_with_moving_elbows(self) -> None:
        report = evaluate_promotion_gate([passing_record(11), passing_record(22)])
        self.assertTrue(report["passed"])

    def test_gate_rejects_frozen_elbow_despite_good_reward(self) -> None:
        records = [passing_record(11), passing_record(22)]
        records[1]["right_elbow_amplitude_ratio"] = 0.10
        report = evaluate_promotion_gate(records)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["right_elbow_motion"])

    def test_gate_rejects_physical_fall(self) -> None:
        records = [passing_record(11), passing_record(22)]
        records[0]["physical_termination"] = True
        report = evaluate_promotion_gate(records)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["no_physical_termination"])


if __name__ == "__main__":
    unittest.main()
