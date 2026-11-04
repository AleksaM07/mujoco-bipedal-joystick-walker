"""Gradually transfer the successful walking policy to full BVH elbow motion.

Each level runs in a fresh process so JAX releases compiled programs and GPU
memory.  A deterministic rollout gate must pass before the next elbow target
strength is allowed to train.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "prezentacija/4th_success_better_arms_stiff_elbows/model_000001003520"
)
ELBOW_LEVELS = (0.25, 0.50, 0.75, 1.00)


def select_checkpoint(run_dir: Path) -> Path:
    """Choose a complete checkpoint by episode length, reward, then step."""
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    candidates: list[tuple[float, float, float, Path]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " - eval | " not in line:
            continue
        values: dict[str, float] = {}
        for part in line.split(" | "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                values[key] = float(value)
            except ValueError:
                continue
        if not {"step", "reward", "episode_length"} <= values.keys():
            continue
        if not all(
            math.isfinite(values[key])
            for key in ("step", "reward", "episode_length")
        ):
            continue
        checkpoint = run_dir / "checkpoints" / f"{int(values['step']):012d}"
        metadata_path = checkpoint / "_CHECKPOINT_METADATA"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("commit_timestamp_nsecs"):
            continue
        candidates.append(
            (
                values["episode_length"],
                values["reward"],
                values["step"],
                checkpoint,
            )
        )
    if not candidates:
        raise RuntimeError(f"No complete evaluated checkpoint found in {run_dir}")
    return max(candidates)[3]


def train_level(args: argparse.Namespace) -> None:
    """Train one fixed elbow-target level from the promoted checkpoint."""
    import jax

    from config import TrainConfig
    from evaluate import find_run_config, restore_saved_env_config
    from train import configure_stdout_encoding, run_training

    configure_stdout_encoding()
    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    saved = find_run_config(args.resume)
    if saved is None:
        raise FileNotFoundError(f"No config.json found for {args.resume}")

    env_config = restore_saved_env_config(saved)
    env_config.deepmimic_reward_mode = "pure"
    env_config.command_profile = "walk"
    env_config.arm_reference_mode = "cmu_elbow_wrist_ik"
    env_config.reference_elbow_flexion_alpha = args.alpha
    env_config.deepmimic_arm_reward_weight = 1.0
    env_config.reference_stability_arm_alpha = 1.0

    train_config = TrainConfig(**saved["train"])
    train_config.num_timesteps = args.timesteps
    train_config.num_envs = args.num_envs
    train_config.num_eval_envs = 4
    train_config.num_evals = 6
    train_config.training_metrics_steps = 50_000
    train_config.learning_rate = args.learning_rate
    train_config.resume_from = str(args.resume)
    train_config.checkpoint_out = None
    train_config.save_checkpoints = True
    train_config.bare = True
    train_config.run_tag = (
        f"elbow_curriculum_a{round(args.alpha * 100):03d}_"
        f"{args.campaign.name}"
    )

    run_dir = run_training(env_config, train_config, PROJECT_ROOT / "runs")
    result = {
        "alpha": args.alpha,
        "resume": str(args.resume),
        "run_dir": str(run_dir),
    }
    (args.campaign / f"level_{args.level}_train.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def audit_level(args: argparse.Namespace) -> None:
    """Audit a trained level in a clean GPU process."""
    import jax

    from evaluate import (
        find_run_config,
        load_ppo_policy,
        make_environment,
        restore_saved_env_config,
        validate_xml_manifest,
    )
    from tools.audit_elbow_transfer import (
        evaluate_promotion_gate,
        measure_elbow_rollouts,
    )

    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    saved = find_run_config(args.resume)
    if saved is None:
        raise FileNotFoundError(f"No config.json found for {args.resume}")
    env_config = restore_saved_env_config(saved)
    if not math.isclose(
        env_config.reference_elbow_flexion_alpha,
        args.alpha,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Checkpoint alpha does not match requested audit level: "
            f"{env_config.reference_elbow_flexion_alpha} != {args.alpha}"
        )
    env = make_environment(env_config)
    validate_xml_manifest(args.resume, env)
    policy = load_ppo_policy(args.resume, deterministic=True)
    records = measure_elbow_rollouts(env, policy, [11, 22, 33, 44], 500)
    report = {
        "checkpoint": str(args.resume),
        "alpha": args.alpha,
        "records": records,
        "gate": evaluate_promotion_gate(records),
    }
    (args.campaign / f"level_{args.level}_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_campaign(args: argparse.Namespace) -> None:
    """Run all levels, stopping immediately if walking retention is lost."""
    if not SOURCE_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Missing source checkpoint: {SOURCE_CHECKPOINT}")
    args.campaign.mkdir(parents=True, exist_ok=False)
    checkpoint = SOURCE_CHECKPOINT
    state: dict[str, object] = {
        "status": "running",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "steps_per_level": args.steps_per_level,
        "num_envs": args.num_envs,
        "learning_rate": args.learning_rate,
        "levels": [],
    }
    status_path = args.campaign / "status.json"

    for level_index, alpha in enumerate(ELBOW_LEVELS, start=1):
        common = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--campaign",
            str(args.campaign),
            "--level",
            str(level_index),
            "--alpha",
            str(alpha),
            "--num-envs",
            str(args.num_envs),
            "--learning-rate",
            str(args.learning_rate),
        ]
        entry: dict[str, object] = {
            "level": level_index,
            "alpha": alpha,
            "resume": str(checkpoint),
            "status": "training",
        }
        levels = state["levels"]
        assert isinstance(levels, list)
        levels.append(entry)
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(
            f"ELBOW CURRICULUM level={level_index}/{len(ELBOW_LEVELS)} "
            f"alpha={alpha:.2f} steps={args.steps_per_level}",
            flush=True,
        )

        subprocess.run(
            [
                *common,
                "--worker",
                "train",
                "--timesteps",
                str(args.steps_per_level),
                "--resume",
                str(checkpoint),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        train_result = json.loads(
            (args.campaign / f"level_{level_index}_train.json").read_text(
                encoding="utf-8"
            )
        )
        run_dir = Path(train_result["run_dir"])
        candidate = select_checkpoint(run_dir)
        entry.update(
            run_dir=str(run_dir),
            checkpoint=str(candidate),
            status="auditing",
        )
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        subprocess.run(
            [
                *common,
                "--worker",
                "audit",
                "--resume",
                str(candidate),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        audit = json.loads(
            (args.campaign / f"level_{level_index}_audit.json").read_text(
                encoding="utf-8"
            )
        )
        gate = audit["gate"]
        entry.update(status="passed" if gate["passed"] else "failed", gate=gate)
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if not gate["passed"]:
            state["status"] = "stopped_on_failed_gate"
            state["last_promoted_checkpoint"] = str(checkpoint)
            status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(
                "Promotion gate failed; stopping without advancing difficulty.",
                flush=True,
            )
            return
        checkpoint = candidate

    state["status"] = "complete"
    state["final_checkpoint"] = str(checkpoint)
    status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"ELBOW CURRICULUM COMPLETE checkpoint={checkpoint}", flush=True)


def main() -> None:
    """Launch the curriculum supervisor or one internal worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-level", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / ("elbow_curriculum_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--worker", choices=["train", "audit"])
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=ELBOW_LEVELS[0])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.steps_per_level <= 0 or args.timesteps <= 0:
        parser.error("Training steps must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        parser.error("Alpha must be in [0, 1].")
    if args.worker:
        if args.resume is None:
            parser.error("Workers require --resume.")
        args.campaign.mkdir(parents=True, exist_ok=True)
        (train_level if args.worker == "train" else audit_level)(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
