"""Test that retarget-stage observation preserves data and production failures."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import biomechanics_env
from biomechanics_env import BiomechanicsJoystickEnv
from config import EnvConfig
from tools import check_retarget_stages as audit


class RetargetStageCaptureTests(unittest.TestCase):
    """Use a tiny fake pipeline to isolate observation from MuJoCo and JAX."""

    def setUp(self) -> None:
        self.env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
        self.references = SimpleNamespace(
            qpos_targets=np.array([[[0.1], [0.2]]], dtype=np.float32),
            root_pos_targets=np.array([[[0, 0, 1], [1, 0, 1]]], dtype=np.float32),
            root_quat_targets=np.tile([1.0, 0.0, 0.0, 0.0], (1, 2, 1)),
            ik_target_positions=np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
            ik_target_names=("metatarsal_midpoint_right", "metatarsal_midpoint_left"),
            frame_counts=np.array([2]),
            source_start_frames=np.array([3]),
            source_end_frames=np.array([5]),
        )

        def align(env, joints, roots, quaternions, *args):
            return roots + [0, 0, 0.5], np.zeros_like(roots), np.zeros((1, 3))

        self.method_fakes = {
            "_filter_reference_clips": lambda env, references: references,
            "_align_reference_roots_to_sim_floor": align,
            "_retarget_reference_qpos_to_ik_targets": (
                lambda env, joints, targets, names, roots, quaternions, counts: (
                    joints + 0.1
                )
            ),
            "_apply_reference_stability_prior": lambda env, joints: 0.5 * joints,
        }
        for name, fake in self.method_fakes.items():
            patcher = patch.object(BiomechanicsJoystickEnv, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        prepare_patcher = patch.object(audit, "prepare_reference_curves")
        self.prepare = prepare_patcher.start()
        self.prepare.return_value = {"closure_diagnostics": [{"mode": "synthetic"}]}
        self.addCleanup(prepare_patcher.stop)

    def make_fake_environment(self, config: EnvConfig) -> BiomechanicsJoystickEnv:
        """Exercise the wrapped methods in the real constructor's stage order."""
        references = self.env._filter_reference_clips(self.references)
        joints = references.qpos_targets
        quaternions = references.root_quat_targets
        roots, _, _ = self.env._align_reference_roots_to_sim_floor(
            joints, references.root_pos_targets, quaternions
        )
        targets = (
            references.ik_target_positions
            + (roots - references.root_pos_targets)[:, :, None, :]
        )
        targets = targets + [0, 0, 0.1]
        joints = self.env._retarget_reference_qpos_to_ik_targets(
            joints,
            targets,
            references.ik_target_names,
            roots,
            quaternions,
            references.frame_counts,
        )
        joints = self.env._apply_reference_stability_prior(joints)
        roots, _, _ = self.env._align_reference_roots_to_sim_floor(
            joints, roots, quaternions
        )
        biomechanics_env.prepare_reference_curves(
            joints,
            roots,
            quaternions,
            np.array([0.01]),
            references.frame_counts,
            np.array([1]),
            np.zeros((1, 3)),
            loop_closure="bounded",
        )
        self.env._bvh_reference_qpos_targets = joints.copy()
        self.env._bvh_reference_sim_root_pos_targets_np = roots + [0.25, 0, 0]
        self.env._bvh_reference_sim_root_quat_targets_np = quaternions.copy()
        return self.env

    def capture(self):
        with patch.object(audit, "make_environment", self.make_fake_environment):
            return audit.capture_retarget_stages(EnvConfig())

    def test_stage_order_and_target_shifts_follow_constructor(self) -> None:
        env, metadata, stages = self.capture()
        self.assertIs(env, self.env)
        self.assertEqual(metadata["names"], self.references.ik_target_names)
        self.assertEqual(
            [stage["name"] for stage in stages],
            [
                "raw_mapped_pose",
                "initial_floor_alignment",
                "normalized_ik_input",
                "ik_solution",
                "stability_prior",
                "final_floor_alignment",
                "prepared_curve_nodes",
            ],
        )
        raw = self.references.ik_target_positions
        for stage, shift in zip(
            stages,
            (
                [0, 0, 0],
                [0, 0, 0.5],
                [0, 0, 0.6],
                [0, 0, 0.6],
                [0, 0, 0.6],
                [0, 0, 1.1],
                [0.25, 0, 1.1],
            ),
            strict=True,
        ):
            np.testing.assert_allclose(stage["targets"], raw + shift)
        np.testing.assert_allclose(
            stages[-1]["joints"], 0.5 * (self.references.qpos_targets + 0.1)
        )
        self.assertEqual(self.prepare.call_count, 2)
        self.assertEqual(self.prepare.call_args.kwargs["loop_closure"], "bounded")
        self.assertEqual(
            metadata["per_clip_curve_gates"],
            [
                {
                    "clip_id": 0,
                    "accepted": True,
                    "closure_diagnostics": [{"mode": "synthetic"}],
                }
            ],
        )

    def test_recorded_arrays_are_detached_from_inputs_and_final_environment(
        self,
    ) -> None:
        _, metadata, stages = self.capture()
        raw_targets = stages[0]["targets"].copy()
        final_joints = stages[-1]["joints"].copy()
        self.references.ik_target_positions.fill(99)
        self.references.frame_counts.fill(99)
        self.env._bvh_reference_qpos_targets.fill(99)
        np.testing.assert_array_equal(stages[0]["targets"], raw_targets)
        np.testing.assert_array_equal(stages[-1]["joints"], final_joints)
        np.testing.assert_array_equal(metadata["counts"], [2])

    def test_only_curve_gate_failure_returns_partial_diagnostic_environment(
        self,
    ) -> None:
        self.prepare.side_effect = ValueError("synthetic loop gate rejection")
        env, metadata, stages = self.capture()
        self.assertIs(env, self.env)
        self.assertEqual(
            metadata["curve_preparation_error"], "synthetic loop gate rejection"
        )
        self.assertEqual(stages[-1]["name"], "final_floor_alignment")
        self.assertNotIn("prepared_curve_nodes", [stage["name"] for stage in stages])
        self.assertFalse(hasattr(env, "_bvh_reference_qpos_targets"))
        self.assertEqual(
            metadata["per_clip_curve_gates"],
            [
                {
                    "clip_id": 0,
                    "accepted": False,
                    "error": "synthetic loop gate rejection",
                }
            ],
        )

    def test_unrelated_constructor_errors_are_not_swallowed(self) -> None:
        with patch.object(
            audit, "make_environment", side_effect=ValueError("missing source marker")
        ):
            with self.assertRaisesRegex(ValueError, "missing source marker"):
                audit.capture_retarget_stages(EnvConfig())

    def test_missing_cartesian_targets_fail_normally(self) -> None:
        self.references.ik_target_positions = None
        with self.assertRaisesRegex(ValueError, "Cartesian IK targets"):
            self.capture()

    def test_wrapped_methods_are_restored_after_success_and_curve_failure(self) -> None:
        original_curve_function = biomechanics_env.prepare_reference_curves
        for failure in (None, ValueError("synthetic loop gate rejection")):
            self.prepare.side_effect = failure
            self.capture()
            for name, fake in self.method_fakes.items():
                self.assertIs(getattr(BiomechanicsJoystickEnv, name), fake)
            self.assertIs(
                biomechanics_env.prepare_reference_curves, original_curve_function
            )


if __name__ == "__main__":
    unittest.main()
