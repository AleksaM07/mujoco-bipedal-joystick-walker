"""Rank existing CMU BVH files by measured stand/walk endpoint dynamics.

This reads the repository's existing walking manifest.  It does not create a
new motion manifest or classify clips from filenames.  Ranking uses only root
translation recorded in each BVH, so shortlisted clips still require visual
and retargeting review before training.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bvh_reference import BVH_POSITION_SCALE, _global_joint_positions, _parse_bvh

DEFAULT_MANIFEST = PROJECT_ROOT / "BVH_walking_animation/walking_filter_manifest.txt"
FOOT_CONTACT_HEIGHT_METERS = 0.05


def manifest_paths(manifest: Path) -> list[Path]:
    """Return existing KEEP paths from the repository's walking manifest."""
    root = manifest.parent
    paths: list[Path] = []
    in_keep = False
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[KEEP]":
            in_keep = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_keep = False
        if in_keep and line:
            path = root / Path(line.replace("\\", "/"))
            if path.is_file():
                paths.append(path)
    if not paths:
        raise ValueError(f"No existing BVH files found in {manifest}")
    return paths


def load_root_motion(path: Path) -> tuple[np.ndarray, float]:
    """Load root XYZ positions and frame time from a conventional BVH file."""
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        motion_index = lines.index("MOTION")
    except ValueError as exc:
        raise ValueError(f"Missing MOTION section: {path}") from exc
    if motion_index + 3 >= len(lines):
        raise ValueError(f"Incomplete MOTION header: {path}")
    frame_count = int(lines[motion_index + 1].split(":", 1)[1])
    frame_time = float(lines[motion_index + 2].split(":", 1)[1])
    if frame_count < 3 or not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError(f"Invalid BVH timeline: {path}")
    frames = np.asarray(
        [
            [float(value) for value in line.split()[:3]]
            for line in lines[motion_index + 3 : motion_index + 3 + frame_count]
        ],
        dtype=np.float64,
    )
    if frames.shape != (frame_count, 3) or not np.isfinite(frames).all():
        raise ValueError(f"Invalid BVH root frames: {path}")
    return frames, frame_time


def analyze_clip(path: Path, root: Path, window_seconds: float) -> dict[str, Any]:
    """Measure endpoint and middle horizontal root speeds for one clip."""
    positions, frame_time = load_root_motion(path)
    horizontal = positions[:, [0, 2]] * BVH_POSITION_SCALE
    speed = np.linalg.norm(np.diff(horizontal, axis=0), axis=1) / frame_time
    window = min(len(speed) // 3, max(2, round(window_seconds / frame_time)))
    if window < 2:
        raise ValueError(f"Clip too short for endpoint windows: {path}")
    middle_start = max(window, len(speed) // 4)
    middle_stop = min(len(speed) - window, 3 * len(speed) // 4)
    middle = speed[middle_start:middle_stop]
    if not len(middle):
        raise ValueError(f"Clip has no middle speed window: {path}")
    start_speed = float(np.median(speed[:window]))
    end_speed = float(np.median(speed[-window:]))
    moving_speed = float(np.percentile(middle, 75))
    displacement = float(np.linalg.norm(horizontal[-1] - horizontal[0]))
    start_contrast = moving_speed - start_speed
    stop_contrast = moving_speed - end_speed
    root_start_score = start_contrast + 0.10 * min(displacement, 5.0)
    root_stop_score = stop_contrast + 0.10 * min(displacement, 5.0)
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "seconds": float((len(positions) - 1) * frame_time),
        "frames": len(positions),
        "frame_time": frame_time,
        "start_speed_m_s": start_speed,
        "middle_speed_p75_m_s": moving_speed,
        "end_speed_m_s": end_speed,
        "displacement_m": displacement,
        "start_contrast_m_s": start_contrast,
        "stop_contrast_m_s": stop_contrast,
        "root_start_score": root_start_score,
        "root_stop_score": root_stop_score,
        "root_start_stop_score": min(root_start_score, root_stop_score),
    }


def add_foot_metrics(
    row: dict[str, Any],
    path: Path,
    window_seconds: float,
) -> dict[str, Any]:
    """Add endpoint foot motion and double-support metrics to one shortlist row."""
    bvh = _parse_bvh(path)
    joint_positions = _global_joint_positions(bvh)
    left_name = next(
        name
        for name in ("LeftToeBase", "LeftToe", "LeftFoot")
        if name in joint_positions
    )
    right_name = next(
        name
        for name in ("RightToeBase", "RightToe", "RightFoot")
        if name in joint_positions
    )
    feet = np.stack([joint_positions[left_name], joint_positions[right_name]], axis=1)
    feet *= BVH_POSITION_SCALE
    foot_speed = (
        np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=2) / bvh.frame_time
    )
    window = min(
        len(foot_speed) // 3,
        max(2, round(window_seconds / bvh.frame_time)),
    )
    ground_height = float(np.min(feet[:, :, 1]))
    double_support = np.all(
        feet[:, :, 1] <= ground_height + FOOT_CONTACT_HEIGHT_METERS,
        axis=1,
    )
    start_foot_speed = float(np.max(np.median(foot_speed[:window], axis=0)))
    end_foot_speed = float(np.max(np.median(foot_speed[-window:], axis=0)))
    start_double_support = float(np.mean(double_support[:window]))
    end_double_support = float(np.mean(double_support[-window:]))
    start_score = (
        row["root_start_score"] - 0.25 * start_foot_speed + 0.25 * start_double_support
    )
    stop_score = (
        row["root_stop_score"] - 0.25 * end_foot_speed + 0.25 * end_double_support
    )
    return row | {
        "start_foot_speed_max_m_s": start_foot_speed,
        "end_foot_speed_max_m_s": end_foot_speed,
        "start_double_support_fraction": start_double_support,
        "end_double_support_fraction": end_double_support,
        "start_score": start_score,
        "stop_score": stop_score,
        "start_stop_score": min(start_score, stop_score),
    }


def top_rows(rows: list[dict[str, Any]], key: str, count: int) -> list[dict[str, Any]]:
    """Return plausible moving clips ranked by one endpoint score."""
    plausible = [
        row
        for row in rows
        if row["seconds"] >= 1.5
        and row["displacement_m"] >= 0.5
        and row["middle_speed_p75_m_s"] >= 0.35
    ]
    return sorted(plausible, key=lambda row: row[key], reverse=True)[:count]


def main() -> None:
    """Scan and print a compact transition-candidate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--window-seconds", type=float, default=0.35)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.top < 1:
        raise ValueError("window-seconds and top must be positive")
    manifest = args.manifest.resolve(strict=True)
    paths = manifest_paths(manifest)
    rows = [analyze_clip(path, manifest.parent, args.window_seconds) for path in paths]
    preselect_count = max(20, args.top * 2)
    preselected_paths = {
        row["path"]
        for key in (
            "root_start_score",
            "root_stop_score",
            "root_start_stop_score",
        )
        for row in sorted(rows, key=lambda item: item[key], reverse=True)[
            :preselect_count
        ]
    }
    selected_rows = [
        add_foot_metrics(
            row,
            manifest.parent / row["path"],
            args.window_seconds,
        )
        for row in rows
        if row["path"] in preselected_paths
    ]
    report = {
        "scope": "existing_manifest_transition_shortlist_with_endpoint_feet",
        "manifest": str(manifest),
        "clips_scanned": len(rows),
        "clips_foot_scanned": len(selected_rows),
        "window_seconds": args.window_seconds,
        "bvh_position_scale": BVH_POSITION_SCALE,
        "foot_contact_height_meters": FOOT_CONTACT_HEIGHT_METERS,
        "limitation": (
            "Root motion was scanned for every available manifest clip; foot FK was "
            "computed for shortlisted endpoints. Candidates still require visual "
            "retargeting and physics validation."
        ),
        "start_candidates": top_rows(selected_rows, "start_score", args.top),
        "stop_candidates": top_rows(selected_rows, "stop_score", args.top),
        "start_stop_candidates": top_rows(
            selected_rows,
            "start_stop_score",
            args.top,
        ),
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
