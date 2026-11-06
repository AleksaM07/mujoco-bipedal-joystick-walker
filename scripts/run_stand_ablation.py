"""Run controlled stand learning experiments on one GPU.

Each training/evaluation subprocess releases its GPU memory before the next
starts. All generated artifacts live in a fresh experiment directory in runs/.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RUNS_DIR, EnvConfig, TrainConfig

ARCHIVE = PROJECT_ROOT / "prezentacija/5th_success_final_natural_walk"
VARIANTS = ("transfer", "scratch")
ALL_VARIANTS = (*VARIANTS, "resume")


def build_configs(
    variant: str,
    timesteps: int,
    reward_mode: str = "imitation",
    learning_rate: float = 2e-5,
    resume_from: Path | None = None,
    episode_length: int = 500,
    stand_push_max_fraction: float = 0.0,
) -> tuple[EnvConfig, TrainConfig]:
    """Keep archived physics/PPO settings with explicit initialization and reward."""
    if variant not in ALL_VARIANTS or timesteps <= 0:
        raise ValueError("Expected transfer/scratch/resume and positive timesteps.")
    if (variant == "resume") != (resume_from is not None):
        raise ValueError("Only the resume variant requires --resume-from.")
    if reward_mode not in {"imitation", "balance_v1"}:
        raise ValueError("Expected imitation or balance_v1 stand reward.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if episode_length < 500:
        raise ValueError("Stand training episodes must contain at least 500 steps.")
    if (
        not math.isfinite(stand_push_max_fraction)
        or not 0 <= stand_push_max_fraction <= 0.05
    ):
        raise ValueError("Stand push fraction must be finite and in [0, 0.05].")
    saved = json.loads((ARCHIVE / "config.json").read_text(encoding="utf-8"))
    environment = replace(
        EnvConfig(**saved["env"]),
        locomotion_skill="stand",
        stand_reward_mode=reward_mode,
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=str(
            ARCHIVE
            / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
        ),
        reference_gait_file=[],
        reference_clip_ids=None,
        reference_forced_clip_id=0,
    )
    training = replace(
        TrainConfig(**saved["train"]),
        num_timesteps=timesteps,
        seed=7,
        num_envs=64,
        num_eval_envs=4,
        num_evals=6,
        episode_length=episode_length,
        batch_size=128,
        unroll_length=10,
        num_minibatches=4,
        num_updates_per_batch=2,
        learning_rate=learning_rate,
        training_metrics_steps=50000,
        enable_erfi=False,
        enable_domain_randomization=False,
        stand_push_max_fraction=stand_push_max_fraction,
        bare=True,
        save_checkpoints=True,
        checkpoint_out=None,
        resume_from=(
            str(ARCHIVE / "model_000001024000") if variant == "transfer" else None
        ),
        restore_value_fn=variant != "transfer",
        run_tag=f"stand_{reward_mode}_{variant}_{timesteps}_lr{learning_rate:g}_ep{episode_length}",
        debug_run=False,
    )
    if resume_from is not None:
        checkpoint = resume_from.resolve()
        if not checkpoint.is_relative_to(RUNS_DIR.resolve()):
            raise ValueError("Stand continuation requires a checkpoint under runs/.")
        if (
            checkpoint.parent.name != "checkpoints"
            or not checkpoint.name.isdigit()
            or not (checkpoint / "ppo_network_config.json").is_file()
        ):
            raise ValueError("Expected one saved numeric stand checkpoint directory.")
        source = json.loads(
            (checkpoint.parent.parent / "config.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (checkpoint.parent.parent / "xml_manifest.json").read_text(encoding="utf-8")
        )
        current_hash = sha256(Path(environment.xml_path).read_bytes()).hexdigest()
        if manifest.get("xml_sha256") != current_hash:
            raise ValueError("Resume requires a matching source XML manifest.")
        # Paths differ between Windows and WSL. The training loader separately
        # verifies the checkpoint XML hash before starting physics/training.
        source_env = replace(EnvConfig(**source["env"]), xml_path=environment.xml_path)
        if source_env != environment:
            raise ValueError("Resume requires the same stand environment and reward.")
        source_training, target_training = (
            asdict(TrainConfig(**source["train"])),
            asdict(training),
        )
        for key in (
            "num_timesteps",
            "learning_rate",
            "run_tag",
            "resume_from",
            "restore_value_fn",
            "episode_length",
            "stand_push_max_fraction",
        ):
            source_training.pop(key)
            target_training.pop(key)
        if source_training != target_training:
            raise ValueError("Resume requires matching controlled PPO settings.")
        training = replace(training, resume_from=str(checkpoint), restore_value_fn=True)
    return environment, training


def checkpoint_rank(report: dict[str, Any]) -> tuple[float, ...]:
    """Rank physical completion before dwell, drift/speed, and imitation reward."""
    rollouts = report["rollouts"]

    def worst(name: str) -> float:
        values = [row[name] for row in rollouts]
        return (
            max(values) if all(value is not None for value in values) else float("inf")
        )

    return (
        float(report["passed"]),
        sum(row["steps"] == 500 for row in rollouts),
        sum(row["finite"] and not row["terminated"] for row in rollouts),
        sum(row["final_success"] for row in rollouts),
        min(row["steps"] for row in rollouts),
        -worst("max_drift"),
        -worst("mean_speed"),
        sum(row["reward"] or 0.0 for row in rollouts) / len(rollouts),
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Save the current experiment state after every completed stage."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_command(command: list[str], log_path: Path) -> int:
    """Capture a subprocess log while preserving its exit code."""
    print(f"Starting {log_path.name}; log: {log_path}", flush=True)
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MUJOCO_GL": "egl",
    }
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"Finished {log_path.name}: exit={result.returncode}", flush=True)
    return result.returncode


def run_variant(
    variant: str,
    output: Path,
    timesteps: int,
    reward_mode: str = "imitation",
    learning_rate: float = 2e-5,
    resume_from: Path | None = None,
    episode_length: int = 500,
    stand_push_max_fraction: float = 0.0,
) -> dict[str, Any]:
    """Train one variant, then physically check every saved checkpoint."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        variant,
        "--output",
        str(output / variant),
        "--timesteps",
        str(timesteps),
        "--stand-reward-mode",
        reward_mode,
        "--learning-rate",
        str(learning_rate),
        "--episode-length",
        str(episode_length),
        "--stand-push-max-fraction",
        str(stand_push_max_fraction),
    ]
    result: dict[str, Any] = {"variant": variant, "training_command": command}
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from.resolve())])
    result["training_exit"] = run_command(command, output / f"{variant}_train.log")
    if result["training_exit"] != 0:
        result["status"] = "training_failed"
        return result
    run_configs = list((output / variant).glob("*/config.json"))
    if len(run_configs) != 1:
        raise RuntimeError(
            f"Expected one training run for {variant}, found {run_configs}"
        )
    run_dir = run_configs[0].parent
    result["run_dir"] = str(run_dir)
    gates = output / f"{variant}_gates"
    gate_command = [
        sys.executable,
        str(PROJECT_ROOT / "tools/evaluate_stand.py"),
        "--checkpoints-dir",
        str(run_dir / "checkpoints"),
        "--output-dir",
        str(gates),
        "--device",
        "gpu",
    ]
    result["gate_command"] = gate_command
    result["gate_exit"] = run_command(gate_command, output / f"{variant}_gate.log")
    if result["gate_exit"] not in (0, 1):
        result["status"] = "evaluation_failed"
        return result
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(gates.glob("*.json"))
    ]
    checkpoints = [
        path for path in (run_dir / "checkpoints").iterdir() if path.name.isdigit()
    ]
    if not reports or len(reports) != len(checkpoints):
        raise RuntimeError(f"Incomplete physical checkpoint evaluation for {variant}")
    best = max(reports, key=checkpoint_rank)
    result.update(
        status="evaluated", best_checkpoint=best["checkpoint"], best_gate=best
    )
    return result


def main() -> int:
    """Run requested variants sequentially; a failed physical gate remains a result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--stand-push-max-fraction", type=float, default=0.0)
    parser.add_argument("--robustness-check", action="store_true")
    parser.add_argument("--variants", choices=ALL_VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--stand-reward-mode", choices=("imitation", "balance_v1"), default="imitation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=ALL_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("timesteps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.episode_length < 500:
        parser.error("episode-length must be at least 500")
    if args.robustness_check and len(args.variants) != 1:
        parser.error("--robustness-check requires exactly one variant")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    output = args.output or RUNS_DIR / (
        f"{datetime.now(UTC):%Y%m%d_%H%M%SZ}_stand_{args.stand_reward_mode}_{args.timesteps}_seed7"
    )
    output = output.resolve()
    if not output.is_relative_to(RUNS_DIR.resolve()):
        parser.error("output must be a fresh directory under runs/")
    plan = {}
    for variant in (args.worker,) if args.worker else args.variants:
        try:
            env, training = build_configs(
                variant,
                args.timesteps,
                args.stand_reward_mode,
                args.learning_rate,
                args.resume_from,
                args.episode_length,
                args.stand_push_max_fraction,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        plan[variant] = {"env": asdict(env), "train": asdict(training)}
    if args.dry_run:
        print(json.dumps({"output": str(output), "plan": plan}, indent=2))
        return 0
    if args.worker:
        import jax

        from train import choose_device, configure_stdout_encoding, run_training

        configure_stdout_encoding()
        jax.config.update("jax_default_device", choose_device("gpu", allow_cpu=False))
        env, training = build_configs(
            args.worker,
            args.timesteps,
            args.stand_reward_mode,
            args.learning_rate,
            args.resume_from,
            args.episode_length,
            args.stand_push_max_fraction,
        )
        run_training(env, training, output)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "stage": "training",
        "robustness_requested": args.robustness_check,
        "plan": plan,
        "variants": [],
        "selection": "Physical rollout completion, no termination, final success, drift/speed, then reward.",
    }
    write_report(output / "comparison.json", report)
    for variant in args.variants:
        result = run_variant(
            variant,
            output,
            args.timesteps,
            args.stand_reward_mode,
            args.learning_rate,
            args.resume_from,
            args.episode_length,
            args.stand_push_max_fraction,
        )
        report["variants"].append(result)
        write_report(output / "comparison.json", report)
    evaluated = [row for row in report["variants"] if row["status"] == "evaluated"]
    if evaluated:
        best = max(evaluated, key=lambda row: checkpoint_rank(row["best_gate"]))
        report["best_candidate"] = best["variant"]
        report["stand_gate_passed"] = best["best_gate"]["passed"]
        if not args.skip_video:
            report["stage"] = "video"
            write_report(output / "comparison.json", report)
            video_command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/evaluate_stand.py"),
                "--checkpoint",
                best["best_checkpoint"],
                "--device",
                "gpu",
                "--output-dir",
                str(output / "selected_video_gate"),
                "--video-dir",
                str(output / "videos"),
            ]
            report["video_exit"] = run_command(
                video_command, output / "selected_video.log"
            )
            report["video_complete"] = len(
                list((output / "videos").glob("*/stand_seed_*.mp4"))
            ) == 4 and report["video_exit"] in (0, 1)
        if args.robustness_check:
            checkpoints = sorted(
                path
                for path in (Path(best["run_dir"]) / "checkpoints").iterdir()
                if path.is_dir() and path.name.isdigit()
            )
            robustness_output = output / "robustness"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/evaluate_stand_robustness.py"),
                "--checkpoints",
                *map(str, checkpoints),
                "--output-dir",
                str(robustness_output),
                "--device",
                "gpu",
            ]
            report["stage"] = "robustness"
            report["robustness_report"] = str(robustness_output / "report.json")
            write_report(output / "comparison.json", report)
            report["robustness_exit"] = run_command(command, output / "robustness.log")
            report["robustness_complete"] = report["robustness_exit"] in (0, 1)
            if report["robustness_complete"]:
                try:
                    robust = json.loads(
                        (robustness_output / "report.json").read_text(encoding="utf-8")
                    )
                    expected_cases = {
                        "hold_60s",
                        "push_x_pos",
                        "push_x_neg",
                        "push_y_pos",
                        "push_y_neg",
                    }
                    expected = {
                        (str(path.resolve()), case)
                        for path in checkpoints
                        for case in expected_cases
                    }
                    observed = {
                        (row["checkpoint"], row["probe"]["name"])
                        for row in robust["results"]
                    }
                    report["robustness_complete"] = (
                        robust["status"] == "complete"
                        and observed == expected
                        and len(robust["results"]) == len(expected)
                    )
                    report["robustness_candidates_passed"] = [
                        str(path.resolve())
                        for path in checkpoints
                        if report["robustness_complete"]
                        and all(
                            row["passed"]
                            for row in robust["results"]
                            if row["checkpoint"] == str(path.resolve())
                        )
                    ]
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["robustness_complete"] = False
                    report["robustness_error"] = str(exc)
    report["status"] = (
        "complete" if len(evaluated) == len(args.variants) else "incomplete"
    )
    if report.get("video_complete") is False:
        report["status"] = "incomplete"
    if report.get("robustness_complete") is False:
        report["status"] = "incomplete"
    report["stage"] = "finished"
    write_report(output / "comparison.json", report)
    print(f"Comparison: {output / 'comparison.json'}", flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
