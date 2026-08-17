"""Render control-paradigm comparison videos for one reference clip.

This wrapper exists to compare:
- direct oracle PD targets
- DeepMimic-like zero residual playback
- MimicKit-like oracle-through-action-map playback
- raw MimicKit-like oracle with joint-midpoint / joint-limit bounds
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_TOOL = PROJECT_ROOT / "tools" / "render_bvh_reference_videos.py"


@dataclass(frozen=True)
class ParadigmCase:
    """One control-paradigm video configuration."""

    name: str
    title: str
    control_source: str
    reference_action_mode: str
    reference_action_center: str = "default"
    reference_action_range: str = "reference_targets"
    reference_action_range_scale: float = 1.1
    reference_residual_scale: float = 1.0
    reference_replay_target_step: int = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gait-file", type=Path, action="append", required=True)
    parser.add_argument("--clip-id", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("videos/control_paradigm_probe"))
    parser.add_argument("--start-phase", type=float, default=0.0)
    parser.add_argument("--segment-seconds", type=float, default=0.35)
    parser.add_argument("--reference-speed-scale", type=float, default=0.2)
    parser.add_argument("--trace-dt", type=float, default=0.01)
    parser.add_argument("--video-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--reference-loop-mode",
        choices=["auto", "wrap", "clamp"],
        default="wrap",
    )
    parser.add_argument(
        "--arms-on",
        dest="arm_actuators",
        action="store_true",
        help="Use the arm-actuated model variant.",
    )
    parser.add_argument(
        "--arms-off",
        dest="arm_actuators",
        action="store_false",
        help="Use the no-arms model variant.",
    )
    parser.set_defaults(arm_actuators=False)
    parser.add_argument(
        "--reference-lock-stance-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_cases() -> list[ParadigmCase]:
    """Return the current set of comparison paradigms."""
    return [
        ParadigmCase(
            name="case_a_direct_reference",
            title="CASE A | DIRECT REFERENCE ORACLE",
            control_source="direct_reference",
            reference_action_mode="residual",
            reference_action_range="reference_targets",
        ),
        ParadigmCase(
            name="case_b_deepmimic_zero_residual",
            title="CASE B | DEEPMIMIC-LIKE ZERO RESIDUAL",
            control_source="zero_policy",
            reference_action_mode="residual",
            reference_action_range="action_scale",
            reference_action_range_scale=1.0,
            reference_residual_scale=1.0,
            reference_replay_target_step=0,
        ),
        ParadigmCase(
            name="case_c_mimickit_oracle_targets",
            title="CASE C | MIMICKIT-LIKE ORACLE TARGETS",
            control_source="encoded_reference",
            reference_action_mode="mimickit",
            reference_action_center="default",
            reference_action_range="reference_targets",
            reference_action_range_scale=1.1,
        ),
        ParadigmCase(
            name="case_d_mimickit_oracle_raw",
            title="CASE D | RAW MIMICKIT ORACLE LIMITS",
            control_source="encoded_reference",
            reference_action_mode="mimickit",
            reference_action_center="joint_midpoint",
            reference_action_range="joint_limits",
            reference_action_range_scale=1.0,
        ),
    ]


def run_case(args: argparse.Namespace, case: ParadigmCase) -> Path:
    """Invoke the renderer for one paradigm case."""
    case_dir = args.out_dir / case.name
    command = [
        sys.executable,
        str(RENDER_TOOL),
        "--out-dir",
        str(case_dir),
        "--reference-loop-mode",
        args.reference_loop_mode,
        "--clip-id",
        str(args.clip_id),
        "--mode",
        "compare",
        "--start-phase",
        str(args.start_phase),
        "--segment-seconds",
        str(args.segment_seconds),
        "--reference-speed-scale",
        str(args.reference_speed_scale),
        "--trace-dt",
        str(args.trace_dt),
        "--video-seconds",
        str(args.video_seconds),
        "--fps",
        str(args.fps),
        "--control-source",
        case.control_source,
        "--reference-action-mode",
        case.reference_action_mode,
        "--reference-action-center",
        case.reference_action_center,
        "--reference-action-range",
        case.reference_action_range,
        "--reference-action-range-scale",
        str(case.reference_action_range_scale),
        "--reference-residual-scale",
        str(case.reference_residual_scale),
        "--reference-replay-target-step",
        str(case.reference_replay_target_step),
        "--overlay-title",
        case.title,
    ]
    if args.arm_actuators:
        command.append("--arms-on")
    else:
        command.append("--arms-off")
    if args.reference_lock_stance_feet:
        command.append("--reference-lock-stance-feet")
    else:
        command.append("--no-reference-lock-stance-feet")
    for path in args.reference_gait_file:
        command.extend(["--reference-gait-file", str(path)])

    print(f"\n=== Running {case.name} ===")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return case_dir


def summarize_case(case_dir: Path) -> dict[str, str]:
    """Load the single-row manifest produced for one case."""
    manifest_path = case_dir / "manifest_compare.csv"
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one manifest row in {manifest_path}.")
    return rows[0]


def main() -> None:
    """Run all paradigm videos and write a small summary CSV."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    for case in build_cases():
        case_dir = run_case(args, case)
        row = summarize_case(case_dir)
        summary_rows.append(
            {
                "case_name": case.name,
                "title": case.title,
                "case_dir": str(case_dir),
                "file": row.get("file", ""),
                "pd_fail_reason": row.get("pd_fail_reason", ""),
                "pd_fail_time_s": row.get("pd_fail_time_s", ""),
                "control_source": row.get("control_source", ""),
                "reference_action_mode": row.get("reference_action_mode", ""),
                "reference_action_center": row.get("reference_action_center", ""),
                "reference_action_range": row.get("reference_action_range", ""),
                "reference_action_range_scale": row.get("reference_action_range_scale", ""),
                "reference_residual_scale": row.get("reference_residual_scale", ""),
                "reference_replay_target_step": row.get("reference_replay_target_step", ""),
            }
        )

    summary_path = args.out_dir / "paradigm_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
