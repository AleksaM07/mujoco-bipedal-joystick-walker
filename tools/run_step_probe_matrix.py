"""Run a small BVH step-probe diagnostic matrix and summarize the results.

This wrapper exists so we can compare the same short clip segment under
different root/gravity conditions without hand-editing long shell commands.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_TOOL = PROJECT_ROOT / "tools" / "render_bvh_reference_videos.py"


@dataclass(frozen=True)
class ProbeCase:
    """One step-probe render configuration."""

    name: str
    gravity_scale: float
    pin_root_to_reference: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the step-probe matrix runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        action="append",
        required=True,
        help="BVH file(s) passed through to the render tool.",
    )
    parser.add_argument("--clip-id", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("videos/step_probe_matrix"))
    parser.add_argument("--start-phase", type=float, default=0.0)
    parser.add_argument("--segment-seconds", type=float, default=0.35)
    parser.add_argument("--reference-speed-scale", type=float, default=0.20)
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
        help="Run the matrix with arm actuators enabled.",
    )
    parser.add_argument(
        "--arms-off",
        dest="arm_actuators",
        action="store_false",
        help="Run the matrix with arm actuators disabled.",
    )
    parser.set_defaults(arm_actuators=False)
    parser.add_argument(
        "--reference-lock-stance-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through stance-foot locking to the renderer.",
    )
    return parser.parse_args()


def build_cases() -> list[ProbeCase]:
    """Return the two core tests for the current debug phase."""
    return [
        ProbeCase(
            name="gravity1_rootpinned",
            gravity_scale=1.0,
            pin_root_to_reference=True,
        ),
        ProbeCase(
            name="gravity1_rootfree",
            gravity_scale=1.0,
            pin_root_to_reference=False,
        ),
    ]


def run_case(args: argparse.Namespace, case: ProbeCase) -> Path:
    """Invoke the render tool for one case and return its output directory."""
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
        "--gravity-scale",
        str(case.gravity_scale),
    ]
    if case.pin_root_to_reference:
        command.append("--pin-root-to-reference")
    else:
        command.append("--no-pin-root-to-reference")
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


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one manifest row in {manifest_path}.")
    return rows[0]


def _load_trace(trace_path: Path) -> list[dict[str, str]]:
    with trace_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _float_series(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        values.append(float(raw))
    return values


def summarize_case(case_dir: Path) -> dict[str, object]:
    """Summarize one completed case from manifest + detail trace."""
    manifest = _load_manifest(case_dir / "manifest_compare.csv")
    rows = _load_trace(case_dir / "trace_detail_compare.csv")
    rmse = _float_series(rows, "pd_pose_rmse")
    force_ratio = _float_series(rows, "pd_max_actuator_force_ratio")
    torso_up = _float_series(rows, "pd_torso_up")
    root_x = _float_series(rows, "pd_root_x")
    root_y = _float_series(rows, "pd_root_y")
    root_z = _float_series(rows, "pd_root_z")
    kin_x = _float_series(rows, "kin_root_x")
    kin_y = _float_series(rows, "kin_root_y")
    kin_z = _float_series(rows, "kin_root_z")
    root_err = [
        math.sqrt((px - kx) ** 2 + (py - ky) ** 2 + (pz - kz) ** 2)
        for px, py, pz, kx, ky, kz in zip(root_x, root_y, root_z, kin_x, kin_y, kin_z)
    ]
    return {
        "case_dir": case_dir,
        "pd_fail_reason": manifest.get("pd_fail_reason", ""),
        "pd_fail_time_s": manifest.get("pd_fail_time_s", ""),
        "rmse_mean": statistics.mean(rmse) if rmse else float("nan"),
        "rmse_max": max(rmse) if rmse else float("nan"),
        "force_mean": statistics.mean(force_ratio) if force_ratio else float("nan"),
        "force_max": max(force_ratio) if force_ratio else float("nan"),
        "torso_min": min(torso_up) if torso_up else float("nan"),
        "root_err_mean": statistics.mean(root_err) if root_err else float("nan"),
        "root_err_max": max(root_err) if root_err else float("nan"),
        "trace_rows": len(rows),
    }


def print_summary(case_name: str, summary: dict[str, object]) -> None:
    """Print a compact human-readable case summary."""
    print(f"\n--- {case_name} ---")
    print(f"dir={summary['case_dir']}")
    print(
        "fail_reason={fail} fail_time_s={time} rows={rows}".format(
            fail=summary["pd_fail_reason"] or "none",
            time=summary["pd_fail_time_s"] or "n/a",
            rows=summary["trace_rows"],
        )
    )
    print(
        "rmse_mean={:.6f} rmse_max={:.6f} force_mean={:.6f} force_max={:.6f}".format(
            float(summary["rmse_mean"]),
            float(summary["rmse_max"]),
            float(summary["force_mean"]),
            float(summary["force_max"]),
        )
    )
    print(
        "root_err_mean={:.6f} root_err_max={:.6f} torso_min={:.6f}".format(
            float(summary["root_err_mean"]),
            float(summary["root_err_max"]),
            float(summary["torso_min"]),
        )
    )


def main() -> None:
    """Run the configured matrix and print summaries."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    for case in cases:
        case_dir = run_case(args, case)
        summary = summarize_case(case_dir)
        print_summary(case.name, summary)


if __name__ == "__main__":
    main()
