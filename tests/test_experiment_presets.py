"""Tests for the isolated locomotion experiment presets."""

from __future__ import annotations

import unittest
from argparse import Namespace

from train import (
    apply_low_vram_preset,
    apply_reference_tracker_backend,
    apply_reference_tracker_inplace_preset,
    apply_reference_tracker_preset,
    apply_task_forward_bootstrap_preset,
    apply_task_forward_curriculum_preset,
    apply_task_only_baseline_preset,
    apply_task_step_in_place_preset,
    apply_task_weight_transfer_preset,
    debug_run_defaults,
)


def make_args() -> Namespace:
    """Return the CLI fields mutated by the experiment presets."""
    return Namespace(
        arm_actuators=True,
        bare=False,
        command_profile="standard",
        deepmimic_reward_mode="mixed",
        deepmimic_root_velocity_weight_scale=0.5,
        discounting=None,
        domain_randomization=True,
        erfi=True,
        gait_shaping_scale=1.0,
        gait_active_without_command=False,
        gait_period_steps=50,
        gait_warmup_steps=0,
        foot_clearance_target=0.08,
        gait_balance_reward_scale=0.0,
        gait_balance_target_fraction=0.5,
        gait_balance_sigma=0.035,
        procedural_balance_prior_scale=0.0,
        procedural_step_prior_scale=0.0,
        procedural_step_residual_scale=1.0,
        neutral_stance_hip_x=0.18,
        step_gait_scale=0.25,
        step_clearance_target=0.04,
        weight_transfer_scale=0.5,
        weight_transfer_prior_scale=1.0,
        tracking_sigma=0.25,
        forward_progress_reward_scale=1.0,
        overspeed_cost_scale=0.75,
        no_domain_randomization=False,
        no_erfi=False,
        learning_rate=None,
        num_envs=None,
        num_eval_envs=None,
        unroll_length=None,
        batch_size=None,
        num_minibatches=None,
        updates_per_batch=None,
        physics_backend="mjx_warp",
        playground_impl="warp",
        pose_termination=None,
        reference_action_mode="mimickit",
        reference_force_clip_id=None,
        reference_gait="bvh",
        reference_loop_mode="auto",
        reference_replay_target_step=1,
        reference_residual_scale=0.25,
        reference_lock_stance_feet=True,
        reference_root_xy_scale=1.0,
        reference_stability_other_alpha=1.0,
        reference_stability_sagittal_alpha=1.0,
        reference_tracker_backend="jax",
        run_tag=None,
    )


class ExperimentPresetTest(unittest.TestCase):
    """Ensure task learning and motion tracking cannot silently mix."""

    def test_task_only_baseline_disables_reference_and_perturbations(self) -> None:
        args = make_args()

        apply_task_only_baseline_preset(args)

        self.assertEqual(args.command_profile, "forward_slow")
        self.assertEqual(args.reference_gait, "none")
        self.assertFalse(args.arm_actuators)
        self.assertFalse(args.erfi)
        self.assertFalse(args.domain_randomization)
        self.assertTrue(args.bare)
        self.assertEqual(args.run_tag, "v21_task_only_baseline")

    def test_forward_curriculum_requires_nontrivial_forward_commands(self) -> None:
        args = make_args()

        apply_task_forward_curriculum_preset(args)

        self.assertEqual(args.command_profile, "forward")
        self.assertEqual(args.reference_gait, "none")
        self.assertFalse(args.arm_actuators)
        self.assertFalse(args.erfi)
        self.assertFalse(args.domain_randomization)
        self.assertTrue(args.bare)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.run_tag, "v21_forward_curriculum")

    def test_step_in_place_requests_alternating_foot_lift(self) -> None:
        args = make_args()

        apply_task_step_in_place_preset(args)

        self.assertEqual(args.command_profile, "step_in_place")
        self.assertEqual(args.reference_gait, "none")
        self.assertEqual(args.neutral_stance_hip_x, 0.0)
        self.assertTrue(args.gait_active_without_command)
        self.assertEqual(args.gait_shaping_scale, 0.25)
        self.assertEqual(args.gait_period_steps, 100)
        self.assertEqual(args.gait_warmup_steps, 25)
        self.assertEqual(args.foot_clearance_target, 0.04)
        self.assertEqual(args.gait_balance_reward_scale, 0.5)
        self.assertEqual(args.forward_progress_reward_scale, 0.0)
        self.assertEqual(args.overspeed_cost_scale, 0.0)
        self.assertEqual(args.discounting, 0.995)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.run_tag, "v22_step_in_place")

    def test_weight_transfer_precedes_foot_lift(self) -> None:
        args = make_args()

        apply_task_weight_transfer_preset(args)

        self.assertEqual(args.command_profile, "step_in_place")
        self.assertEqual(args.reference_gait, "none")
        self.assertEqual(args.neutral_stance_hip_x, 0.0)
        self.assertTrue(args.gait_active_without_command)
        self.assertEqual(args.gait_period_steps, 100)
        self.assertEqual(args.gait_warmup_steps, 25)
        self.assertEqual(args.gait_shaping_scale, 0.0)
        self.assertEqual(args.gait_balance_reward_scale, 0.5)
        self.assertEqual(args.procedural_balance_prior_scale, 1.0)
        self.assertEqual(args.procedural_step_prior_scale, 0.0)
        self.assertEqual(args.procedural_step_residual_scale, 1.0)
        self.assertEqual(args.forward_progress_reward_scale, 0.0)
        self.assertEqual(args.overspeed_cost_scale, 0.0)
        self.assertEqual(args.discounting, 0.995)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.run_tag, "v22_weight_transfer_guided")

    def test_forward_bootstrap_defers_phase_clock_foot_shaping(self) -> None:
        args = make_args()

        apply_task_forward_bootstrap_preset(args)

        self.assertEqual(args.command_profile, "forward_bridge")
        self.assertEqual(args.reference_gait, "none")
        self.assertEqual(args.gait_shaping_scale, 0.0)
        self.assertEqual(args.tracking_sigma, 0.02)
        self.assertEqual(args.forward_progress_reward_scale, 0.0)
        self.assertEqual(args.overspeed_cost_scale, 2.0)
        self.assertEqual(args.discounting, 0.995)
        self.assertFalse(args.erfi)
        self.assertFalse(args.domain_randomization)
        self.assertTrue(args.bare)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.run_tag, "v21_forward_bootstrap")

    def test_reference_tracker_is_pure_single_clip_experiment(self) -> None:
        args = make_args()

        apply_reference_tracker_preset(args)

        self.assertEqual(args.reference_gait, "bvh")
        self.assertEqual(args.reference_action_mode, "residual")
        self.assertEqual(args.reference_loop_mode, "wrap")
        self.assertEqual(args.deepmimic_reward_mode, "pure")
        self.assertEqual(args.discounting, 0.995)
        self.assertEqual(args.reference_force_clip_id, 1)
        self.assertFalse(args.arm_actuators)
        self.assertFalse(args.erfi)
        self.assertFalse(args.domain_randomization)
        self.assertTrue(args.bare)
        self.assertEqual(args.run_tag, "reference_tracker_0201")

    def test_inplace_tracker_uses_survivable_preflight_variant(self) -> None:
        args = make_args()

        apply_reference_tracker_inplace_preset(args)

        self.assertEqual(args.reference_gait, "bvh")
        self.assertEqual(args.reference_action_mode, "residual")
        self.assertEqual(args.reference_root_xy_scale, 0.0)
        self.assertFalse(args.reference_lock_stance_feet)
        self.assertEqual(args.reference_stability_sagittal_alpha, 0.10)
        self.assertEqual(args.reference_stability_other_alpha, 0.35)
        self.assertEqual(args.reference_residual_scale, 0.10)
        self.assertEqual(args.deepmimic_reward_mode, "pure")
        self.assertEqual(args.discounting, 0.995)
        self.assertEqual(args.run_tag, "reference_tracker_inplace_residual010_0201")

    def test_reference_tracker_horizon_exceeds_known_pd_failure(self) -> None:
        defaults = debug_run_defaults(False, reference_tracker=True)

        self.assertEqual(defaults["episode_length"], 500)

    def test_low_vram_preset_reduces_ppo_working_set(self) -> None:
        args = make_args()

        apply_low_vram_preset(args)

        self.assertEqual(args.num_envs, 512)
        self.assertEqual(args.num_eval_envs, 8)
        self.assertEqual(args.unroll_length, 10)
        self.assertEqual(args.batch_size, 512)
        self.assertEqual(args.num_minibatches, 4)
        self.assertEqual(args.updates_per_batch, 2)

    def test_low_vram_preset_preserves_explicit_overrides(self) -> None:
        args = make_args()
        args.num_envs = 256
        args.batch_size = 128

        apply_low_vram_preset(args)

        self.assertEqual(args.num_envs, 256)
        self.assertEqual(args.batch_size, 128)

    def test_low_vram_preset_uses_smaller_mjx_jax_batch(self) -> None:
        args = make_args()
        apply_reference_tracker_inplace_preset(args)

        apply_low_vram_preset(args)

        self.assertEqual(args.physics_backend, "mjx_jax")
        self.assertEqual(args.num_envs, 128)
        self.assertEqual(args.num_eval_envs, 4)
        self.assertEqual(args.batch_size, 128)

    def test_reference_tracker_backend_can_use_warp(self) -> None:
        args = make_args()
        apply_reference_tracker_inplace_preset(args)
        args.reference_tracker_backend = "warp"

        apply_reference_tracker_backend(args)

        self.assertEqual(args.playground_impl, "warp")
        self.assertEqual(args.physics_backend, "mjx_warp")


if __name__ == "__main__":
    unittest.main()
