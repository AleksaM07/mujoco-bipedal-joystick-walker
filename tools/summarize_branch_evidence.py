"""Collect saved experiment evidence without loading policies or changing runs."""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

NUMBER_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z_0-9]*)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
ENV_KEYS = (
    "xml_path",
    "command_profile",
    "reference_gait",
    "reference_pipeline",
    "reference_action_mode",
    "reference_residual_scale",
    "deepmimic_reward_mode",
    "legacy_action_prior",
    "experiment_preset",
    "physics_backend",
)
TRAIN_KEYS = (
    "num_timesteps",
    "num_envs",
    "num_eval_envs",
    "resume_from",
    "seed",
    "enable_erfi",
    "enable_domain_randomization",
    "no_erfi",
    "no_domain_randomization",
    "bare",
)


def git_output(project_root: Path, *args: str) -> str:
    """Read Git metadata; surface failures instead of inventing provenance."""
    return subprocess.check_output(
        ["git", *args], cwd=project_root, text=True, encoding="utf-8"
    ).strip()


def parse_evaluations(log_path: Path) -> list[dict[str, Any]]:
    """Parse logged evaluations, preserving exact log line references."""
    evaluations = []
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if "eval |" not in line:
            continue
        metrics = {key: float(value) for key, value in NUMBER_PATTERN.findall(line)}
        if "reward" not in metrics or "step" not in metrics:
            continue
        evaluations.append({"log_line": line_number, "timestamp": line[:23], **metrics})
    return evaluations


def collect_run(project_root: Path, config_path: Path) -> tuple[dict, list[dict]]:
    """Summarize one saved run; folder names do not establish completion."""
    run_dir = config_path.parent
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    env = saved.get("env", {})
    training = saved.get("train", {})
    ppo = saved.get("brax_ppo", {})
    log_path = run_dir / "train.log"
    evaluations = parse_evaluations(log_path) if log_path.is_file() else []
    run_name = run_dir.relative_to(project_root).as_posix()
    record: dict[str, Any] = {"run": run_name}
    record.update({f"env_{key}": env.get(key) for key in ENV_KEYS})
    record.update({f"train_{key}": training.get(key) for key in TRAIN_KEYS})
    record["ppo_episode_length"] = ppo.get("episode_length")
    record["ppo_discounting"] = ppo.get("discounting")
    record["ppo_learning_rate"] = ppo.get("learning_rate")
    record["eval_count"] = len(evaluations)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = (
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        if checkpoint_dir.is_dir()
        else []
    )
    record["latest_checkpoint_step"] = max(
        (int(path.name) for path in checkpoints), default=None
    )
    status_path = run_dir / "run_status.json"
    if status_path.is_file():
        record["recorded_status"] = json.loads(
            status_path.read_text(encoding="utf-8")
        ).get("status")
    manifest_path = run_dir / "xml_manifest.json"
    if manifest_path.is_file():
        record["xml_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if evaluations:
        best = max(evaluations, key=lambda item: item["reward"])
        for label, evaluation in (
            ("first", evaluations[0]),
            ("best", best),
            ("last", evaluations[-1]),
        ):
            for key in (
                "step",
                "reward",
                "episode_length",
                "done",
                "terminated",
                "truncated",
                "dm_pose_step",
                "dm_vel_step",
                "dm_root_vel_step",
                "dm_key_step",
                "log_line",
            ):
                record[f"{label}_{key}"] = evaluation.get(key)
    return record, [{"run": run_name, **item} for item in evaluations]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous evidence rows with a stable union of columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_evidence(output_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    """Plot survival within each experiment; raw rewards are not cross-compared."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = (
        ("20260817_0133_classic_200m", "Pure imitation: 200M", 0),
        ("20260817_1051_guided_mixed_aggressive_200m", "Mixed objective: 200M", 0),
        ("20261009_2241_v22_narrow_stand", "Narrow standing: last 10M stage", 0),
        (
            "20261010_0727_v22_weight_transfer_curriculum",
            "Assisted weight transfer: 1M",
            0,
        ),
        ("20261010_1204_bio_main_modern_bvh", "Main + modern BVH: stopped", 1_146_880),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (prefix, title, offset) in zip(axes.flat, cases, strict=False):
        rows = [row for row in evaluations if Path(row["run"]).name.startswith(prefix)]
        if not rows:
            raise ValueError(f"Missing plot evidence for {prefix}")
        axis.plot(
            [(row["step"] + offset) / 1e6 for row in rows],
            [row["episode_length"] / 500 for row in rows],
            marker="o",
            markersize=4,
            color="#2563a6",
        )
        axis.axhline(1.0, color="#718096", linestyle="--", linewidth=0.8)
        axis.set(title=title, ylim=(0, 1.07), ylabel="Mean episode length / 500")
        axis.set_xlabel(
            "Total steps incl. pilot (M)" if offset else "Steps in this stage (M)"
        )
        axis.grid(alpha=0.2)
    axis = axes.flat[-1]
    rows = [
        row
        for row in evaluations
        if Path(row["run"]).name.startswith("20260817_0133_classic_200m")
    ]
    for metric, label, color in (
        ("dm_pose_step", "Pose", "#2563a6"),
        ("dm_vel_step", "Joint velocity", "#56a36c"),
        ("dm_root_vel_step", "Root velocity", "#d67a28"),
        ("dm_key_step", "Foot key points", "#b34d70"),
    ):
        axis.plot(
            [row["step"] / 1e6 for row in rows],
            [row[metric] for row in rows],
            label=label,
            color=color,
            marker="o",
            markersize=3,
        )
    axis.set(
        title="Pure imitation: component scores per step",
        ylim=(0, 1.07),
        xlabel="Steps in this stage (M)",
        ylabel="Logged mean component score",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="center right")
    figure.suptitle(
        "Saved evaluation evidence — one seed; different objectives and sample counts\n"
        "Survival is necessary, but standing for 500 steps is not evidence of walking",
        fontsize=13,
    )
    figure.savefig(output_dir / "training_evidence.png", dpi=170)
    figure.savefig(output_dir / "training_evidence.pdf")
    plt.close(figure)


def main() -> None:
    """Export local branch identities, run summaries, and raw eval metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/branch_diagnosis_20261011"),
    )
    parser.add_argument(
        "--plot", action="store_true", help="Also plot saved evaluations."
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output if args.output.is_absolute() else project_root / args.output
    )
    runs_dir = project_root / "runs"
    config_paths = set(runs_dir.glob("*/config.json"))
    # Archived runs sit one level deeper; do not traverse checkpoint trees.
    config_paths.update(runs_dir.glob("*/*/config.json"))
    records = []
    evaluations = []
    errors = []
    for config_path in sorted(config_paths):
        if config_path.parent.name == "checkpoints":
            continue
        try:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if "env" not in saved or "train" not in saved:
                continue
            record, run_evaluations = collect_run(project_root, config_path)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            errors.append({"path": str(config_path), "error": str(exc)})
            continue
        records.append(record)
        evaluations.extend(run_evaluations)
    branches = git_output(
        project_root,
        "for-each-ref",
        "--format=%(refname:short)|%(objectname)",
        "refs/heads/",
    ).splitlines()
    snapshot = {
        "current_branch": git_output(project_root, "branch", "--show-current"),
        "branches": dict(line.split("|", 1) for line in branches),
        "run_count": len(records),
        "evaluation_count": len(evaluations),
        "read_errors": errors,
        "limitations": [
            "Run configs often lack a Git commit; directory dates are not provenance.",
            "Raw reward values across different objectives are not comparable.",
            "Checkpoint step counters restart for warm-started runs.",
            "Folder suffixes alone do not prove completion or locomotion success.",
            "Reported metrics are saved evaluations, not new rollouts.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "run_inventory.csv", records)
    write_csv(output_dir / "eval_history.csv", evaluations)
    (output_dir / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.plot:
        plot_evidence(output_dir, evaluations)
    print(json.dumps(snapshot, indent=2))
    if errors:
        raise SystemExit("Evidence export has read errors; inspect snapshot.json.")


if __name__ == "__main__":
    main()
