"""Keep policy evaluation faithful to the saved project-v2 environment."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import evaluate
import train
from config import EnvConfig, default_biomechanics_env_config


class SavedConfigTests(unittest.TestCase):
    def test_v4_strict_mapping_version_round_trip_and_forwarding(self) -> None:
        original = EnvConfig(
            recovery_physics_version="consistent_v4", reference_loop_closure="bounded"
        )
        saved = json.loads(json.dumps({"env": asdict(original)}))
        restored = evaluate.restore_saved_env_config(saved)
        self.assertEqual(restored, original)
        for factory in (evaluate, train):
            with patch.object(factory, "BiomechanicsJoystickEnv") as constructor:
                factory.make_environment(restored)
            forwarded = constructor.call_args.kwargs["config_overrides"]
            self.assertEqual(forwarded["recovery_physics_version"], "consistent_v4")
            self.assertEqual(forwarded["reference_loop_closure"], "bounded")

    def test_resume_refuses_v3_to_v4_source_mapping_change(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "checkpoints" / "000000000001"
            checkpoint.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"env": {"recovery_physics_version": "consistent_v3"}}),
                encoding="utf-8",
            )
            train.validate_resume_physics_version(
                checkpoint, EnvConfig(recovery_physics_version="consistent_v3")
            )
            with self.assertRaisesRegex(ValueError, "consistent_v3.*consistent_v4"):
                train.validate_resume_physics_version(
                    checkpoint, EnvConfig(recovery_physics_version="consistent_v4")
                )

    def test_v3_normalization_version_round_trip_and_forwarding(self) -> None:
        original = EnvConfig(
            recovery_physics_version="consistent_v3", reference_loop_closure="bounded"
        )
        restored = evaluate.restore_saved_env_config({"env": asdict(original)})
        self.assertEqual(restored, original)
        for factory in (evaluate, train):
            with patch.object(factory, "BiomechanicsJoystickEnv") as constructor:
                factory.make_environment(restored)
            forwarded = constructor.call_args.kwargs["config_overrides"]
            self.assertEqual(forwarded["recovery_physics_version"], "consistent_v3")
            self.assertEqual(forwarded["reference_loop_closure"], "bounded")

    def test_resume_refuses_v2_to_v3_retarget_change(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "checkpoints" / "000000000001"
            checkpoint.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"env": {"recovery_physics_version": "consistent_v2"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "consistent_v2.*consistent_v3"):
                train.validate_resume_physics_version(
                    checkpoint, EnvConfig(recovery_physics_version="consistent_v3")
                )

    def test_resume_rejects_implicit_physics_migration(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_path = run_dir / "checkpoints" / "000000000001"
            checkpoint_path.mkdir(parents=True)
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps({"env": {}}), encoding="utf-8")
            train.validate_resume_physics_version(checkpoint_path, EnvConfig())
            corrected = EnvConfig(recovery_physics_version="consistent_v1")
            with self.assertRaisesRegex(ValueError, "differs"):
                train.validate_resume_physics_version(checkpoint_path, corrected)
            config_path.write_text(
                json.dumps({"env": {"recovery_physics_version": "consistent_v1"}}),
                encoding="utf-8",
            )
            train.validate_resume_physics_version(checkpoint_path, corrected)
            with self.assertRaisesRegex(ValueError, "differs"):
                train.validate_resume_physics_version(checkpoint_path, EnvConfig())

    def test_recovery_resume_requires_config_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Cannot verify"):
                train.validate_resume_physics_version(
                    Path(directory), EnvConfig(recovery_physics_version="consistent_v1")
                )

    def test_resume_rejects_v1_to_v2_interpolation_migration(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_path = run_dir / "checkpoints" / "000000000001"
            checkpoint_path.mkdir(parents=True)
            config_path = run_dir / "config.json"
            config_path.write_text(
                json.dumps({"env": {"recovery_physics_version": "consistent_v1"}}),
                encoding="utf-8",
            )
            continuous = EnvConfig(
                recovery_physics_version="consistent_v2",
                reference_loop_closure="bounded",
            )
            with self.assertRaisesRegex(
                ValueError, "consistent_v1.*differs.*consistent_v2"
            ):
                train.validate_resume_physics_version(checkpoint_path, continuous)
            config_path.write_text(
                json.dumps({"env": asdict(continuous)}), encoding="utf-8"
            )
            train.validate_resume_physics_version(checkpoint_path, continuous)

    def test_saved_action_reset_and_recovery_settings_are_restored(self) -> None:
        original = EnvConfig(
            reference_residual_scale=0.1,
            reference_action_range_scale=0.7,
            reference_stability_sagittal_alpha=0.4,
            reference_forced_clip_id=2,
            bvh_target_observation_steps=(0, 2, 7),
            reset_projection_levels=(1.0, 0.5),
            pose_termination=True,
            recovery_physics_version="consistent_v1",
            reference_motion_speed_scale=0.25,
        )
        saved = asdict(original)
        saved["bvh_target_observation_steps"] = [0, 2, 7]
        restored = evaluate.restore_saved_env_config({"env": saved})
        self.assertEqual(restored, original)

    def test_saved_v2_bounded_closure_survives_evaluator_round_trip(self) -> None:
        original = EnvConfig(
            recovery_physics_version="consistent_v2",
            reference_loop_closure="bounded",
            reference_motion_speed_scale=0.25,
        )
        saved = json.loads(json.dumps({"env": asdict(original)}))
        restored = evaluate.restore_saved_env_config(saved)
        self.assertEqual(restored, original)
        with patch.object(evaluate, "BiomechanicsJoystickEnv") as constructor:
            evaluate.make_environment(restored)
        forwarded = constructor.call_args.kwargs["config_overrides"]
        self.assertEqual(forwarded["recovery_physics_version"], "consistent_v2")
        self.assertEqual(forwarded["reference_loop_closure"], "bounded")
        self.assertEqual(forwarded["reference_motion_speed_scale"], 0.25)

    def test_missing_version_preserves_project_v2_legacy(self) -> None:
        config = evaluate.restore_saved_env_config(
            {"env": {"reference_residual_scale": 0.4}}
        )
        self.assertEqual(config.recovery_physics_version, "legacy")
        self.assertEqual(config.reference_motion_speed_scale, 1.0)
        self.assertEqual(config.reference_loop_closure, "strict")
        self.assertEqual(config.reference_residual_scale, 0.4)
        self.assertEqual(
            default_biomechanics_env_config().recovery_physics_version, "legacy"
        )

    def test_explicit_overrides_win_without_mutating_saved_config(self) -> None:
        saved = {"env": {"xml_path": "saved.xml", "reference_residual_scale": 0.1}}
        restored = evaluate.restore_saved_env_config(saved, xml_path="local.xml")
        self.assertEqual(restored.xml_path, "local.xml")
        self.assertEqual(restored.reference_residual_scale, 0.1)
        self.assertEqual(saved["env"]["xml_path"], "saved.xml")

    def test_incompatible_branch_fields_fail_instead_of_disappearing(self) -> None:
        with self.assertRaisesRegex(ValueError, "original branch"):
            evaluate.restore_saved_env_config({"env": {"unknown_assist_prior": 1.0}})

    def test_malformed_saved_config_is_rejected(self) -> None:
        for saved in ({"env": []}, {"env": {"reset_projection_levels": 1.0}}):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                evaluate.restore_saved_env_config(saved)

    def test_smpl_is_not_silently_evaluated_without_reference(self) -> None:
        self.assertEqual(
            evaluate.infer_reference_gait({"env": {"reference_gait": "smpl"}}), "smpl"
        )
        with self.assertRaises(ValueError):
            evaluate.infer_reference_gait({"env": {"reference_gait": "unknown"}})
        with self.assertRaises(ValueError):
            evaluate.infer_command_profile(
                Path("unused"), {"env": {"command_profile": "unsupported"}}
            )

    def test_evaluator_forwards_all_environment_settings(self) -> None:
        config = EnvConfig(
            recovery_physics_version="consistent_v1", reference_residual_scale=0.1
        )
        with patch.object(evaluate, "BiomechanicsJoystickEnv") as constructor:
            evaluate.make_environment(config)
        forwarded = constructor.call_args.kwargs["config_overrides"]
        excluded = {
            "env_version",
            "playground_impl",
            "accurate_physics",
            "xml_path",
            "init_qpos_file",
        }
        for name, value in asdict(config).items():
            if name not in excluded:
                self.assertEqual(forwarded[name], value, name)
        self.assertEqual(forwarded["impl"], config.playground_impl)
        self.assertFalse(forwarded["enable_erfi"])

    def test_trainer_forwards_version_and_speed(self) -> None:
        config = EnvConfig(
            recovery_physics_version="consistent_v1", reference_motion_speed_scale=0.5
        )
        with patch.object(train, "BiomechanicsJoystickEnv") as constructor:
            train.make_environment(config)
        forwarded = constructor.call_args.kwargs["config_overrides"]
        self.assertEqual(forwarded["recovery_physics_version"], "consistent_v1")
        self.assertEqual(forwarded["reference_motion_speed_scale"], 0.5)

    def test_trainer_forwards_v2_bounded_loop_closure(self) -> None:
        config = EnvConfig(
            recovery_physics_version="consistent_v2",
            reference_loop_closure="bounded",
        )
        with patch.object(train, "BiomechanicsJoystickEnv") as constructor:
            train.make_environment(config)
        forwarded = constructor.call_args.kwargs["config_overrides"]
        self.assertEqual(forwarded["recovery_physics_version"], "consistent_v2")
        self.assertEqual(forwarded["reference_loop_closure"], "bounded")

    def test_bounded_closure_requires_v2_before_model_loading(self) -> None:
        for version in ("legacy", "consistent_v1"):
            with (
                self.subTest(version=version),
                patch("biomechanics_env.build_trainable_scene_xml") as build_xml,
                patch("biomechanics_env.mujoco.MjModel") as model,
            ):
                with self.assertRaisesRegex(ValueError, "requires consistent_v2"):
                    evaluate.BiomechanicsJoystickEnv(
                        config_overrides={
                            "recovery_physics_version": version,
                            "reference_loop_closure": "bounded",
                        }
                    )
                build_xml.assert_not_called()
                model.from_xml_path.assert_not_called()

    def test_bad_versions_and_legacy_speed_changes_fail_before_model_loading(
        self,
    ) -> None:
        for overrides in (
            {"recovery_physics_version": "typo"},
            {"reference_motion_speed_scale": 0.25},
            {
                "recovery_physics_version": "consistent_v1",
                "reference_motion_speed_scale": 0.0,
            },
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                evaluate.BiomechanicsJoystickEnv(config_overrides=overrides)


if __name__ == "__main__":
    unittest.main()
