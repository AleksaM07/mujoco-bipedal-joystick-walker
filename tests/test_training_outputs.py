"""Protect previous runs and keep reported checkpoint locations usable."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import train
from config import EnvConfig, TrainConfig


class TrainingOutputTests(unittest.TestCase):
    def test_reserves_new_run_and_checkpoint_directories(self) -> None:
        with TemporaryDirectory() as directory:
            requested = Path(directory) / "new_run"
            run_dir, checkpoint_dir = train.reserve_training_directories(
                requested, requested / "checkpoints"
            )
            self.assertEqual(run_dir, requested.resolve())
            self.assertEqual(checkpoint_dir, run_dir / "checkpoints")
            self.assertTrue(checkpoint_dir.is_dir())

    def test_existing_run_is_never_reused_or_modified(self) -> None:
        with TemporaryDirectory() as directory:
            existing = Path(directory) / "existing_run"
            existing.mkdir()
            for name in ("train.log", "config.json"):
                (existing / name).write_text("important result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train.reserve_training_directories(existing, existing / "checkpoints")
            for name in ("train.log", "config.json"):
                self.assertEqual(
                    (existing / name).read_text(encoding="utf-8"), "important result"
                )
            self.assertFalse((existing / "checkpoints").exists())

    def test_custom_checkpoint_destination_must_be_fresh_even_when_empty(self) -> None:
        for has_checkpoint in (False, True):
            with self.subTest(has_checkpoint=has_checkpoint):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    checkpoint_dir = root / "saved_checkpoints"
                    checkpoint_dir.mkdir()
                    if has_checkpoint:
                        (checkpoint_dir / "000000002000").mkdir()
                    with self.assertRaises(FileExistsError):
                        train.reserve_training_directories(
                            root / "new_run", checkpoint_dir
                        )
                    self.assertFalse((root / "new_run").exists())
                    self.assertEqual(
                        (checkpoint_dir / "000000002000").exists(), has_checkpoint
                    )

    def test_checkpoint_destination_cannot_be_run_directory_or_its_parent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for checkpoint_dir in (root / "run", root):
                with self.subTest(checkpoint_dir=checkpoint_dir):
                    with self.assertRaises(ValueError):
                        train.reserve_training_directories(root / "run", checkpoint_dir)
            self.assertFalse((root / "run").exists())

    def test_disabled_checkpoints_do_not_create_checkpoint_directory(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir, checkpoint_dir = train.reserve_training_directories(
                Path(directory) / "run", None
            )
            self.assertIsNone(checkpoint_dir)
            self.assertEqual(list(run_dir.iterdir()), [])

    def test_atomic_creation_rejects_racing_run_without_touching_it(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            original_mkdir = Path.mkdir

            def competing_mkdir(path: Path, *args, **kwargs) -> None:
                if path == run_dir:
                    original_mkdir(path, parents=True, exist_ok=False)
                    (path / "train.log").write_text("other process", encoding="utf-8")
                original_mkdir(path, *args, **kwargs)

            with patch.object(Path, "mkdir", competing_mkdir):
                with self.assertRaises(FileExistsError):
                    train.reserve_training_directories(run_dir, run_dir / "checkpoints")
            self.assertEqual(
                (run_dir / "train.log").read_text(encoding="utf-8"), "other process"
            )

    def run_mocked_training(self, root: Path, checkpoint_out=None, failure=None):
        env_config = EnvConfig(
            playground_impl="jax", physics_backend="mjx_jax", reference_gait="none"
        )
        train_config = TrainConfig(
            num_timesteps=1, num_envs=1, batch_size=1, num_evals=1,
            checkpoint_out=str(checkpoint_out) if checkpoint_out is not None else None,
        )
        env = SimpleNamespace(_config=SimpleNamespace(reference_gait="none"))
        with (
            patch.object(train, "logger") as logger,
            patch.object(train, "make_environment", return_value=env),
            patch.object(train, "write_xml_manifest"),
            patch.object(train, "log_environment_summary"),
            patch.object(train, "log_eval_environment_summary"),
            patch.object(train.ppo, "train", side_effect=failure) as ppo,
        ):
            if failure is not None:
                with self.assertRaises(type(failure)):
                    train.run_training(env_config, train_config, root / "runs")
                return None, logger, ppo
            result = train.run_training(env_config, train_config, root / "runs")
        return result, logger, ppo

    def test_success_logs_real_checkpoint_path_after_rename(self) -> None:
        for external in (False, True):
            with self.subTest(external=external), TemporaryDirectory() as directory:
                root = Path(directory)
                custom = root / "external_checkpoints" if external else None
                result, logger, ppo = self.run_mocked_training(root, custom)
                expected = custom if external else result / "checkpoints"
                self.assertTrue(expected.is_dir())
                final_log = logger.info.call_args.args
                self.assertEqual(final_log[2], expected)
                original = Path(ppo.call_args.kwargs["save_checkpoint_path"])
                self.assertEqual(original.exists(), external)

    def test_control_c_marks_incomplete_ppo_run_without_hiding_interrupt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_mocked_training(root, failure=KeyboardInterrupt())
            runs = list((root / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0].name.endswith("_f"))
            self.assertTrue((runs[0] / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
