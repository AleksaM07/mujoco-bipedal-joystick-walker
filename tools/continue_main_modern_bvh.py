"""Wait for the GPU pilot, then extend the same experiment to at least 80M."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


def main() -> None:
    """Hand off a completed pilot checkpoint to the unchanged main PPO trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-pid", type=int, default=None)
    parser.add_argument(
        "--pilot-run",
        type=Path,
        default=Path("runs/20261010_1129_bio_main_modern_bvh_pilot_1m_running"),
    )
    parser.add_argument("--total-timesteps", type=int, default=80_000_000)
    parser.add_argument("--num-evals", type=int, default=21)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Launch this handoff inside WSL/Linux, where the pilot runs.")
    if (
        (args.pilot_pid is not None and args.pilot_pid <= 0)
        or args.total_timesteps <= 0
        or args.num_evals < 2
    ):
        parser.error("PID and total steps must be positive; num-evals must be >= 2.")

    project_root = Path(__file__).resolve().parents[1]
    pilot_run = args.pilot_run.resolve()
    if pilot_run.parent != project_root / "runs":
        parser.error("The pilot must be a direct child of this checkout's runs/.")
    prefix = pilot_run.name.removesuffix("_running")
    process_command = (
        Path(f"/proc/{args.pilot_pid}/cmdline") if args.pilot_pid is not None else None
    )
    print(
        f"Queued: wait for pilot PID {args.pilot_pid}; "
        f"target >= {args.total_timesteps:,} cumulative steps.",
        flush=True,
    )
    next_notice = 0.0
    while process_command is not None and process_command.exists():
        try:
            command = process_command.read_bytes().replace(b"\0", b" ")
        except FileNotFoundError:
            break
        if b"train.py" not in command or b"main_modern_bvh_pilot" not in command:
            raise RuntimeError(
                "The PID no longer belongs to our pilot; refusing overlap."
            )
        if time.monotonic() >= next_notice:
            print(
                "Pilot still active; no second GPU trainer will be started.", flush=True
            )
            next_notice = time.monotonic() + 60.0
        time.sleep(10)

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=project_root, text=True
    ).strip()
    if branch != "experiment/main-modern-bvh":
        raise RuntimeError(
            f"Checkout changed to {branch!r}; refusing a different experiment."
        )
    completed_runs = []
    for path in pilot_run.parent.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        status_path = path / "run_status.json"
        completed = path.name.endswith("_s")
        if status_path.exists():
            completed = (
                json.loads(status_path.read_text(encoding="utf-8")).get("status") == "s"
            )
        if completed:
            completed_runs.append(path)
    if len(completed_runs) != 1:
        raise RuntimeError(
            "Pilot did not finish successfully; automatic continuation stopped."
        )
    completed_run = completed_runs[0]
    checkpoints = [
        path
        for path in (completed_run / "checkpoints").iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (
            (path / "config.json").exists()
            or (path / "ppo_network_config.json").exists()
        )
    ]
    if not checkpoints:
        raise RuntimeError("Completed pilot has no usable numeric checkpoint.")
    resume_checkpoint = max(checkpoints, key=lambda path: int(path.name))
    pilot_steps = int(resume_checkpoint.name)
    remaining_steps = args.total_timesteps - pilot_steps
    if remaining_steps <= 0:
        print("The pilot already reached the requested cumulative budget.", flush=True)
        return
    saved = json.loads((completed_run / "config.json").read_text(encoding="utf-8"))
    if saved["env"].get("reference_pipeline") != "modern":
        raise RuntimeError("Checkpoint is not from the modern-BVH pilot.")

    # Import GPU dependencies only after the old trainer has exited.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", ".jax-cache/main-modern-bvh")
    import jax

    from config import EnvConfig, TrainConfig
    from train import choose_device, run_training

    jax.config.update("jax_default_device", choose_device("gpu", allow_cpu=False))
    env_config = EnvConfig(**saved["env"])
    train_config = replace(
        TrainConfig(**saved["train"]),
        num_timesteps=remaining_steps,
        num_evals=args.num_evals,
        # Main's resolver handles a run directory across both checkpoint
        # metadata filenames; it then selects this completed run's latest one.
        resume_from=str(completed_run),
        run_tag="main_modern_bvh_80m_total",
    )
    print(
        f"Pilot completed {pilot_steps:,} steps; continuing for at least "
        f"{remaining_steps:,} more from {resume_checkpoint}.",
        flush=True,
    )
    print(
        "Brax restores policy/value/normalizer, not optimizer state or its step counter.",
        flush=True,
    )
    run_training(env_config, train_config, project_root / "runs")


if __name__ == "__main__":
    main()
