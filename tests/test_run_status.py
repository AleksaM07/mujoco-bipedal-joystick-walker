"""A Windows/WSL directory lock must not invalidate completed training."""

import json
from pathlib import Path

import pytest

from train import mark_run_status


def test_completed_run_is_renamed_and_records_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment_running"
    run_dir.mkdir()
    result = mark_run_status(run_dir, "s", final_reward=1.0, best_reward=2.0)
    assert result.name.endswith("_s")
    assert not run_dir.exists()
    saved = json.loads((result / "run_status.json").read_text(encoding="utf-8"))
    assert saved == {"status": "s", "final_reward": 1.0, "best_reward": 2.0}


def test_directory_lock_keeps_completed_run_accessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "experiment_running"
    run_dir.mkdir()

    def locked_rename(self: Path, target: Path) -> Path:
        raise PermissionError("Directory held open by a Windows reader")

    monkeypatch.setattr(Path, "rename", locked_rename)
    result = mark_run_status(run_dir, "s", final_reward=1.0)
    assert result == run_dir
    assert run_dir.exists()
    saved = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert saved["status"] == "s"
    assert saved["final_reward"] == 1.0
