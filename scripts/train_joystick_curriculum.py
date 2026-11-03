"""Spend a 30M budget on a walking-to-command curriculum with retention gates.

Each block runs in a fresh process to release JAX executables and GPU memory.
The actor and observation statistics resume; Brax restarts optimizer state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from statistics import fmean
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE = PROJECT_ROOT / "prezentacija/2nd_success_stiff_arms/model_000010035200"
WARMSTART = PROJECT_ROOT / "artifacts/joystick_curriculum_v3_warmstart"
# Keep forward walking within a single clip's plausible range. Reverse, stop
# and strafing require later reference/task work and are deliberately absent.
LEVELS = (
    {"speed_radius": 0.25, "yaw_limit": 0.0, "reward_weight": 0.45},
    {"speed_radius": 0.30, "yaw_limit": 0.15, "reward_weight": 0.50},
    {"speed_radius": 0.35, "yaw_limit": 0.30, "reward_weight": 0.55},
)


def promotion_gate(rows: list[dict], level: int, speed_center: float) -> dict:
    """Require retained walking AND a measured response to paired commands."""
    if not rows or len(rows) % 2:
        raise ValueError("Expected two equal groups of paired command rollouts.")
    count = len(rows) // 2
    low, high = rows[:count], rows[count:]
    if [row["seed"] for row in low] != [row["seed"] for row in high]:
        raise ValueError("Command comparisons must use identical seeds.")
    speed_response = fmean([b["measured"][0] - a["measured"][0]
                           for a, b in zip(low, high, strict=True)])
    yaw_response = fmean([b["measured"][2] - a["measured"][2]
                         for a, b in zip(low, high, strict=True)])
    yaw_limit = LEVELS[level]["yaw_limit"]
    checks = {
        "survival": all(row["length"] == 500 and row["survived"] for row in rows),
        "pose": min(row["pose"] for row in rows) >= 0.95,
        # A cyclic gait has intra-stride oscillations. Gate the achieved mean
        # velocity; retain instantaneous MAE in the report for diagnostics.
        "linear_error": fmean(math.dist(row["measured"][:2], row["command"][:2])
                              for row in rows) <= 0.15,
        "yaw_error": fmean(abs(row["measured"][2] - row["command"][2])
                           for row in rows) <= 0.15,
        "speed_response": speed_response >= 0.15,
        "yaw_response": yaw_limit == 0 or yaw_response >= 0.08,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "speed_response": speed_response, "yaw_response": yaw_response,
            "speed_center": speed_center}


def select_checkpoint(run_dir: Path) -> Path:
    """Choose a completed evaluated checkpoint by survival, then command reward."""
    candidates = []
    for line in (run_dir / "train.log").read_text("utf-8").splitlines():
        if " - eval | " not in line:
            continue
        values = {}
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
        if not all(math.isfinite(values[k]) for k in ("step", "reward", "episode_length")):
            continue
        checkpoint = run_dir / "checkpoints" / f"{int(values['step']):012d}"
        metadata = checkpoint / "_CHECKPOINT_METADATA"
        if not metadata.is_file():
            continue
        if not json.loads(metadata.read_text("utf-8")).get("commit_timestamp_nsecs"):
            continue
        candidates.append((values["episode_length"], values["reward"],
                           values["step"], checkpoint))
    if not candidates:
        raise RuntimeError(f"No completed evaluated checkpoint in {run_dir}")
    return max(candidates)[3]


def train_block(args) -> None:
    """Run one PPO block using the archived physics and observation settings."""
    import jax
    from config import TrainConfig
    from evaluate import find_run_config, restore_saved_env_config
    from train import configure_stdout_encoding, run_training

    configure_stdout_encoding()
    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    saved = find_run_config(SOURCE)
    env = restore_saved_env_config(saved)
    level = LEVELS[args.level]
    env.deepmimic_reward_mode = "command"
    env.command_profile = "joystick_curriculum"
    env.joystick_speed_center = args.speed_center
    env.joystick_speed_radius = level["speed_radius"]
    env.joystick_yaw_limit = level["yaw_limit"]
    env.joystick_reward_weight = level["reward_weight"]
    env.tracking_sigma = 0.06
    env.tracking_yaw_sigma = 0.05
    env.arm_reference_mode = "legacy"
    training = TrainConfig(**saved["train"])
    training.num_timesteps = args.timesteps
    training.num_envs = args.num_envs
    training.num_eval_envs = 4
    training.num_evals = 11
    training.learning_rate = 0.00005
    training.resume_from = str(args.resume)
    training.checkpoint_out = None
    training.run_tag = f"{args.campaign.name}_block{args.block}_level{args.level}"
    result = run_training(env, training, PROJECT_ROOT / "runs")
    (args.campaign / f"block{args.block}_result.json").write_text(
        json.dumps({"run_dir": str(result), "level": args.level}, indent=2),
        encoding="utf-8",
    )


def audit_block(args) -> None:
    """Compare different commands from identical initial reference phases."""
    import jax
    from evaluate import find_run_config, load_ppo_policy, make_environment, restore_saved_env_config
    from tools.check_joystick_transfer import measure_rollouts

    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    env_config = restore_saved_env_config(find_run_config(args.resume))
    env = make_environment(env_config)
    policy = load_ppo_policy(args.resume, deterministic=True)
    level = LEVELS[args.level]
    radius, yaw = level["speed_radius"], level["yaw_limit"]
    # Vary speed and yaw independently; a turning policy must not pass merely
    # by changing speed or exploiting a particular start phase.
    commands = [[args.speed_center - radius, 0.0, 0.0],
                [args.speed_center + radius, 0.0, 0.0]]
    rows = measure_rollouts(env, policy, commands, [11, 22])
    gate = promotion_gate(rows, args.level, args.speed_center)
    yaw_rows = []
    if yaw > 0:
        yaw_rows = measure_rollouts(env, policy,
                                   [[args.speed_center, 0.0, -yaw],
                                    [args.speed_center, 0.0, yaw]], [11, 22])
        pairs = zip(yaw_rows[:2], yaw_rows[2:], strict=True)
        response = sum(b["measured"][2] - a["measured"][2] for a, b in pairs) / 2
        gate["yaw_response"] = response
        gate["checks"]["yaw_response"] = response >= 0.08
        gate["checks"]["turn_survival"] = all(
            row["survived"] and row["length"] == 500 for row in yaw_rows)
        gate["checks"]["turn_pose"] = min(row["pose"] for row in yaw_rows) >= 0.95
        gate["checks"]["turn_error"] = max(
            abs(row["measured"][2] - row["command"][2]) for row in yaw_rows) <= 0.15
        gate["passed"] = all(gate["checks"].values())
    report = {"gate": gate, "speed_rollouts": rows, "turn_rollouts": yaw_rows}
    (args.campaign / f"block{args.block}_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def run_campaign(args) -> None:
    """Run three training blocks; repeat difficulty if retention gates fail."""
    # Keep the supervisor off the GPU so workers have the full VRAM budget.
    subprocess.run(
        [sys.executable, "-m", "tools.prepare_joystick_checkpoint",
         "--source-checkpoint", str(SOURCE), "--output-root", str(WARMSTART)],
        cwd=PROJECT_ROOT, env={**os.environ, "JAX_PLATFORMS": "cpu"}, check=True,
    )
    checkpoint = WARMSTART / "checkpoints/000000000000"
    args.campaign.mkdir(parents=True, exist_ok=False)
    state = {"requested_steps": args.timesteps, "source": str(SOURCE),
             "speed_center": args.speed_center, "levels": LEVELS, "blocks": []}
    status_path = args.campaign / "status.json"
    level = 0
    budget = [args.timesteps // 3] * 3
    budget[-1] += args.timesteps - sum(budget)
    for block, steps in enumerate(budget, start=1):
        common = [sys.executable, "-u", str(Path(__file__).resolve()),
                  "--campaign", str(args.campaign), "--block", str(block),
                  "--level", str(level), "--speed-center", str(args.speed_center),
                  "--num-envs", str(args.num_envs)]
        entry = {"block": block, "level": level, "steps": steps,
                 "resume": str(checkpoint), "status": "training"}
        state["blocks"].append(entry)
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"CURRICULUM block={block}/3 level={level} steps={steps}", flush=True)
        subprocess.run([*common, "--worker", "train", "--timesteps", str(steps),
                        "--resume", str(checkpoint)], cwd=PROJECT_ROOT, check=True)
        result = json.loads((args.campaign / f"block{block}_result.json").read_text("utf-8"))
        previous_checkpoint = checkpoint
        checkpoint = select_checkpoint(Path(result["run_dir"]))
        entry.update(run_dir=result["run_dir"], checkpoint=str(checkpoint), status="auditing")
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        subprocess.run([*common, "--worker", "audit", "--resume", str(checkpoint)],
                       cwd=PROJECT_ROOT, check=True)
        audit = json.loads((args.campaign / f"block{block}_audit.json").read_text("utf-8"))
        entry.update(status="complete", gate=audit["gate"])
        if audit["gate"]["passed"]:
            level = min(level + 1, len(LEVELS) - 1)
        else:
            print("Gate not met: spend the next block at the same difficulty.", flush=True)
            rows = audit["speed_rollouts"] + audit["turn_rollouts"]
            if fmean(row["length"] for row in rows) < 450:
                checkpoint = previous_checkpoint
                entry["rollback_to"] = str(checkpoint)
                print("Walking retention lost: next block restores prior weights.", flush=True)
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    state["status"] = "complete"
    status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    """Launch the campaign, or an internal training/evaluation worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=30_000_000)
    parser.add_argument("--num-envs", type=int, choices=[32, 64, 128, 256], default=64)
    parser.add_argument("--speed-center", type=float, default=1.1)
    parser.add_argument("--campaign", type=Path, default=PROJECT_ROOT / "artifacts" /
                        ("joystick_v3_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--worker", choices=["train", "audit"])
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--level", type=int, choices=range(len(LEVELS)), default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.timesteps < 3 or not math.isfinite(args.speed_center) or args.speed_center <= .15:
        parser.error("Require at least 3 timesteps and forward speed center > 0.15 m/s.")
    if args.worker:
        if args.resume is None:
            parser.error("Workers require --resume.")
        (train_block if args.worker == "train" else audit_block)(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
