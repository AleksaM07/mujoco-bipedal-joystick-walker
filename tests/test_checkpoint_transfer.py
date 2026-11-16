"""Verify stand CLI isolation and selective actor/critic checkpoint transfer."""

import functools
import hashlib
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src import mjx_env

import train
from config import EnvConfig, TrainConfig


class InitializationEnvironment:
    """Minimal environment for real Brax network initialization without training."""

    action_size = 1

    def reset(self, rng: jax.Array) -> mjx_env.State:
        del rng
        zero = jp.array(0.0)
        return mjx_env.State(
            data=zero,
            obs=jp.zeros(3),
            reward=zero,
            done=zero,
            metrics={},
            info={},
        )


class CheckpointTransferTests(unittest.TestCase):
    def test_old_training_config_preserves_critic_restore_default(self) -> None:
        old_saved_config = json.loads('{"seed": 7}')
        self.assertTrue(TrainConfig(**old_saved_config).restore_value_fn)
        config = TrainConfig(restore_value_fn=False)
        restored = TrainConfig(**json.loads(json.dumps(asdict(config))))
        self.assertFalse(restored.restore_value_fn)

    def test_cli_keeps_stand_independent_of_motion_paths(self) -> None:
        for flag, expected in (([], True), (["--no-restore-value-fn"], False)):
            arguments = [
                "train.py",
                "--locomotion-skill",
                "stand",
                "--resume-from",
                "saved_checkpoint",
                "--reference-gait-list",
                "missing_motion_list.txt",
                *flag,
            ]
            with (
                self.subTest(flag=flag),
                patch("sys.argv", arguments),
                patch.object(train, "choose_device"),
                patch.object(train.jax.config, "update"),
                patch.object(train, "expand_reference_gait_files") as expand,
                patch.object(train, "run_training") as run,
            ):
                train.main()
            env_config, training_config, _ = run.call_args.args
            self.assertEqual(env_config.reference_gait_file, [])
            self.assertEqual(env_config.locomotion_skill, "stand")
            self.assertEqual(training_config.restore_value_fn, expected)
            self.assertEqual(training_config.resume_from, "saved_checkpoint")
            expand.assert_not_called()

    def test_stand_rejects_walk_only_overfit_preset(self) -> None:
        with (
            patch(
                "sys.argv",
                ["train.py", "--locomotion-skill", "stand", "--one-clip-overfit"],
            ),
            patch("sys.stderr"),
            patch.object(train, "choose_device") as device,
            self.assertRaises(SystemExit) as error,
        ):
            train.main()
        self.assertEqual(error.exception.code, 2)
        device.assert_not_called()

    def test_training_forwards_and_saves_critic_restore_choice(self) -> None:
        for restore_value_fn in (True, False):
            with (
                self.subTest(restore_value_fn=restore_value_fn),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                env_config = EnvConfig(
                    playground_impl="jax",
                    physics_backend="mjx_jax",
                    reference_gait="none",
                )
                train_config = TrainConfig(
                    num_timesteps=1,
                    num_envs=1,
                    batch_size=1,
                    num_evals=1,
                    restore_value_fn=restore_value_fn,
                )
                env = SimpleNamespace(_config=SimpleNamespace(reference_gait="none"))
                with (
                    patch.object(train, "logger"),
                    patch.object(train, "make_environment", return_value=env),
                    patch.object(train, "write_xml_manifest"),
                    patch.object(train, "log_environment_summary"),
                    patch.object(train, "log_eval_environment_summary"),
                    patch.object(train.ppo, "train") as ppo,
                ):
                    result = train.run_training(env_config, train_config, root / "runs")
                self.assertEqual(
                    ppo.call_args.kwargs["restore_value_fn"], restore_value_fn
                )
                saved = json.loads((result / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["train"]["restore_value_fn"], restore_value_fn)

    def test_brax_restores_actor_normalizer_and_optionally_critic(self) -> None:
        train.patch_jax_for_brax_compatibility()
        options = {
            "environment": InitializationEnvironment(),
            "num_timesteps": 0,
            "num_envs": 1,
            "batch_size": 1,
            "num_minibatches": 1,
            "episode_length": 2,
            "seed": 7,
            "use_pmap_on_reset": False,
            "wrap_env_fn": train.wrap_biomechanics_training,
            "network_factory": functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=(4,),
                value_hidden_layer_sizes=(4,),
            ),
        }
        _, fresh, _ = train.ppo.train(**options)
        saved = tuple(jax.tree.map(lambda leaf: leaf + 2, part) for part in fresh)
        for restore_value_fn in (True, False):
            with (
                self.subTest(restore_value_fn=restore_value_fn),
                patch.object(train.ppo.checkpoint, "load", return_value=saved) as load,
            ):
                _, restored, _ = train.ppo.train(
                    **options,
                    restore_checkpoint_path="saved_checkpoint",
                    restore_value_fn=restore_value_fn,
                )
            load.assert_called_once_with("saved_checkpoint")
            expected = (saved[0], saved[1], saved[2] if restore_value_fn else fresh[2])
            self.assertEqual(jax.tree.structure(restored), jax.tree.structure(expected))
            for actual, wanted in zip(
                jax.tree.leaves(restored), jax.tree.leaves(expected)
            ):
                np.testing.assert_array_equal(actual, wanted)

    def test_resume_infers_the_same_xml_for_training_and_evaluation(self) -> None:
        with TemporaryDirectory() as directory:
            environment = SimpleNamespace(
                _config=SimpleNamespace(reference_gait="none")
            )
            with (
                patch.object(train, "logger"),
                patch.object(
                    train,
                    "resolve_resume_checkpoint_path",
                    return_value=Path("checkpoint"),
                ),
                patch.object(train, "validate_resume_physics_version"),
                patch.object(
                    train, "infer_xml_path_from_resume", return_value="archived.xml"
                ),
                patch.object(train, "validate_resume_xml_guard"),
                patch.object(
                    train, "make_environment", return_value=environment
                ) as make,
                patch.object(train, "write_xml_manifest"),
                patch.object(train, "log_environment_summary"),
                patch.object(train, "log_eval_environment_summary"),
                patch.object(train.ppo, "train"),
            ):
                train.run_training(
                    EnvConfig(physics_backend="mjx_jax", reference_gait="none"),
                    TrainConfig(num_timesteps=1, num_envs=1, batch_size=1),
                    Path(directory) / "runs",
                )
            self.assertEqual(make.call_args_list[0].args[0].xml_path, "archived.xml")
            self.assertEqual(make.call_args_list[1].args[0].xml_path, "archived.xml")

    def test_resume_prefers_xml_copied_inside_portable_archive(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "archive"
            checkpoint = run_dir / "model_000001"
            archived_xml = run_dir / "generated_models" / "robot.xml"
            checkpoint.mkdir(parents=True)
            archived_xml.parent.mkdir()
            archived_xml.write_text("<mujoco/>", encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps({"env": {"xml_path": "generated_models/robot.xml"}}),
                encoding="utf-8",
            )

            resolved = train.infer_xml_path_from_resume(checkpoint)

        self.assertEqual(Path(resolved), archived_xml)

    def test_resume_keeps_repo_relative_xml_when_archive_has_no_copy(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            checkpoint = run_dir / "checkpoints" / "000001"
            checkpoint.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"env": {"xml_path": "generated_models/robot.xml"}}),
                encoding="utf-8",
            )

            resolved = train.infer_xml_path_from_resume(checkpoint)

        self.assertEqual(resolved, "generated_models/robot.xml")

    def test_resume_xml_guard_accepts_only_newline_portable_hash_change(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "checkpoints" / "000001"
            checkpoint.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"env": {}}), encoding="utf-8"
            )
            xml_path = run_dir / "robot.xml"
            lf_xml = b"<mujoco>\n  <worldbody/>\n</mujoco>\n"
            xml_path.write_bytes(lf_xml)
            crlf_hash = hashlib.sha256(lf_xml.replace(b"\n", b"\r\n")).hexdigest()
            (run_dir / "xml_manifest.json").write_text(
                json.dumps({"xml_sha256": crlf_hash}), encoding="utf-8"
            )
            environment = SimpleNamespace(xml_path=xml_path)

            train.validate_resume_xml_guard(checkpoint, environment)

            xml_path.write_bytes(lf_xml.replace(b"worldbody", b"asset"))
            with self.assertRaisesRegex(ValueError, "Checkpoint/XML mismatch"):
                train.validate_resume_xml_guard(checkpoint, environment)


if __name__ == "__main__":
    unittest.main()
