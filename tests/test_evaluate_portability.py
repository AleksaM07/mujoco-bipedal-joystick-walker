"""Keep checkpoint restoration portable and enforce the saved XML guard."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import evaluate


class EvaluatePortabilityTests(unittest.TestCase):
    def test_checkpoint_network_config_does_not_hide_run_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "checkpoints" / "000000000160"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text(
                json.dumps({"observation_size": [294], "action_size": 18}),
                encoding="utf-8",
            )
            (run_dir / "config.json").write_text(
                json.dumps({"env": {}}), encoding="utf-8"
            )
            xml_path = run_dir / "robot.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")
            manifest_path = run_dir / "xml_manifest.json"
            manifest_path.write_text(
                json.dumps({"xml_sha256": evaluate.file_sha256(xml_path)}),
                encoding="utf-8",
            )

            self.assertEqual(evaluate.find_run_dir(checkpoint), run_dir)
            env = SimpleNamespace(xml_path=xml_path)
            evaluate.validate_xml_manifest(checkpoint, env)
            xml_path.write_text("<mujoco model='changed'/>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checkpoint/XML mismatch"):
                evaluate.validate_xml_manifest(checkpoint, env)

    def test_run_config_and_manifest_can_identify_run_independently(self) -> None:
        for filename, content in (
            ("config.json", {"env": {}}),
            ("xml_manifest.json", {}),
        ):
            with self.subTest(filename=filename), TemporaryDirectory() as directory:
                run_dir = Path(directory)
                checkpoint = run_dir / "checkpoints" / "000000000160"
                checkpoint.mkdir(parents=True)
                (run_dir / filename).write_text(json.dumps(content), encoding="utf-8")
                self.assertEqual(evaluate.find_run_dir(checkpoint), run_dir)

    def test_orphan_network_config_is_not_a_run(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                json.dumps({"observation_size": [294], "action_size": 18}),
                encoding="utf-8",
            )
            self.assertIsNone(evaluate.find_run_dir(checkpoint))

    def test_relative_xml_is_resolved_against_checkout(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            xml_path = project / "custom_models" / "robot.xml"
            xml_path.parent.mkdir()
            xml_path.write_text("<mujoco/>", encoding="utf-8")

            # This guard models Linux, where backslashes are filename characters.
            def portable_path(value: str) -> Path:
                self.assertNotIn("\\", value)
                return Path(value)

            with (
                patch.object(evaluate, "PROJECT_ROOT", project),
                patch.object(evaluate, "GENERATED_MODEL_DIR", project / "generated"),
                patch.object(evaluate, "Path", side_effect=portable_path),
            ):
                for saved in ("custom_models/robot.xml", "custom_models\\robot.xml"):
                    with self.subTest(saved=saved):
                        self.assertEqual(
                            evaluate.resolve_saved_xml_path(saved), str(xml_path)
                        )

    def test_foreign_absolute_xml_rebases_to_generated_model(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            generated = project / "generated_models"
            generated.mkdir()
            xml_path = generated / "robot.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")
            with (
                patch.object(evaluate, "PROJECT_ROOT", project),
                patch.object(evaluate, "GENERATED_MODEL_DIR", generated),
            ):
                for saved in (
                    "Z:\\old-checkout\\generated_models\\robot.xml",
                    "/mnt/c/old-checkout/generated_models/robot.xml",
                    "/home/other-user/old-checkout/generated_models/robot.xml",
                ):
                    with self.subTest(saved=saved):
                        self.assertEqual(
                            evaluate.resolve_saved_xml_path(saved), str(xml_path)
                        )

    def test_existing_absolute_xml_takes_priority_over_filename_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            xml_path = project / "robot.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")
            generated = project / "generated_models"
            generated.mkdir()
            (generated / "robot.xml").write_text("<mujoco model='other'/>")
            with patch.object(evaluate, "GENERATED_MODEL_DIR", generated):
                self.assertEqual(
                    Path(evaluate.resolve_saved_xml_path(str(xml_path))), xml_path
                )

    def test_missing_xml_and_directory_are_not_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            with (
                patch.object(evaluate, "PROJECT_ROOT", project),
                patch.object(evaluate, "GENERATED_MODEL_DIR", project),
            ):
                self.assertIsNone(evaluate.resolve_saved_xml_path("absent.xml"))
                self.assertIsNone(evaluate.resolve_saved_xml_path(str(project)))

    def test_network_without_normalization_passes_observations_through(self) -> None:
        observations = np.array([0.1, -0.2], dtype=np.float32)
        normalizer_params = object()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "ppo_network_config.json").write_text(
                json.dumps(
                    {
                        "observation_size": [2],
                        "action_size": 1,
                        "normalize_observations": False,
                        "network_factory_kwargs": {"activation": "silu"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(evaluate.ppo_networks, "make_ppo_networks") as factory:
                evaluate.load_ppo_networks(checkpoint)
            preprocessor = factory.call_args.kwargs["preprocess_observations_fn"]
            self.assertIs(
                preprocessor(observations, normalizer_params), observations
            )


if __name__ == "__main__":
    unittest.main()
