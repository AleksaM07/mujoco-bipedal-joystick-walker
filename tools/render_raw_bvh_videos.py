"""Render original BVH skeleton segments without MuJoCo retargeting."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bvh_reference import (  # noqa: E402
    MotionSegment,
    _candidate_step_csv_paths,
    _global_joint_positions,
    _long_cycle_segments,
    _motion_segments_for_bvh,
    _parse_bvh,
    _promote_stride_cycles,
    _read_step_segments,
    expand_motion_paths,
)


SKELETON_LINKS = (
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Neck"),
    ("Neck", "Head"),
    ("Spine1", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToeBase"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("raw_bvh_provera"))
    parser.add_argument("--reference-gait-file", type=Path, action="append")
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--max-clips-per-source", type=int, default=8)
    parser.add_argument(
        "--segments",
        choices=["steps", "stride", "long", "training"],
        default="stride",
        help=(
            "steps renders Marina's half-step cuts; stride renders same-foot "
            "gait cycles; long renders longer same-foot cycles; training "
            "renders the current loader candidates."
        ),
    )
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import mediapy as media

        media.write_video(path, frames, fps=fps)
        return
    except Exception as exc:
        print(f"mediapy video writer failed, trying imageio | {exc}", flush=True)

    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frames), fps=fps)
        return
    except Exception as exc:
        raise RuntimeError(
            "Install mediapy or imageio to write videos: "
            "pip install mediapy imageio imageio-ffmpeg"
        ) from exc


def step_segments_for_bvh(path: Path, frame_count: int) -> tuple[MotionSegment, ...]:
    for csv_path in _candidate_step_csv_paths(path):
        segments = _read_step_segments(csv_path, frame_count)
        if segments:
            return segments
    return ()


def selected_segments(
    path: Path,
    bvh,
    mode: str,
) -> tuple[tuple[str, MotionSegment], ...]:
    steps = step_segments_for_bvh(path, bvh.frames)
    if mode == "steps":
        return tuple(("STEP", segment) for segment in steps)
    if mode == "stride":
        return tuple(
            ("STRIDE", segment)
            for segment in _promote_stride_cycles(steps, bvh.frames)
        )
    if mode == "long":
        return tuple(
            ("LONG", segment)
            for segment in _long_cycle_segments(steps, bvh.frames)
        )
    return tuple(
        ("TRAINING", segment)
        for segment in _motion_segments_for_bvh(path, bvh)
    )


def draw_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
    ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
    radius = max(0, thickness // 2)
    height, width = image.shape[:2]
    for x, y in zip(xs, ys, strict=True):
        xmin = max(0, x - radius)
        xmax = min(width, x + radius + 1)
        ymin = max(0, y - radius)
        ymax = min(height, y + radius + 1)
        image[ymin:ymax, xmin:xmax] = color


def draw_point(
    image: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 4,
) -> None:
    x, y = point
    height, width = image.shape[:2]
    xmin = max(0, x - radius)
    xmax = min(width, x + radius + 1)
    ymin = max(0, y - radius)
    ymax = min(height, y + radius + 1)
    image[ymin:ymax, xmin:xmax] = color


def project_points(
    frame_positions: dict[str, np.ndarray],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    axes: tuple[int, int],
    viewport: tuple[int, int, int, int],
) -> dict[str, tuple[int, int]]:
    x0, y0, width, height = viewport
    axis_min = bounds_min[list(axes)]
    axis_max = bounds_max[list(axes)]
    span = np.maximum(axis_max - axis_min, 1e-6)
    scale = min((width - 80) / span[0], (height - 80) / span[1])
    center = 0.5 * (axis_min + axis_max)
    viewport_center = np.array([x0 + width * 0.5, y0 + height * 0.5])

    projected: dict[str, tuple[int, int]] = {}
    for name, position in frame_positions.items():
        value = position[list(axes)]
        pixel = viewport_center + np.array(
            [
                (value[0] - center[0]) * scale,
                -(value[1] - center[1]) * scale,
            ]
        )
        projected[name] = (int(round(pixel[0])), int(round(pixel[1])))
    return projected


def render_segment(
    path: Path,
    positions: dict[str, np.ndarray],
    kind: str,
    segment: MotionSegment,
    out_dir: Path,
    fps: int,
    width: int,
    height: int,
) -> dict[str, object]:
    frame_indices = range(segment.start_frame, segment.end_frame)
    joint_names = tuple(positions)
    segment_positions = np.stack(
        [positions[name][segment.start_frame : segment.end_frame] for name in joint_names],
        axis=1,
    )
    bounds_min = segment_positions.reshape(-1, 3).min(axis=0)
    bounds_max = segment_positions.reshape(-1, 3).max(axis=0)
    margin = np.maximum((bounds_max - bounds_min) * 0.08, 1.0)
    bounds_min -= margin
    bounds_max += margin

    frames: list[np.ndarray] = []
    side_view = (0, 0, width // 2, height)
    front_view = (width // 2, 0, width - width // 2, height)
    for frame_id in frame_indices:
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        frame_positions = {name: positions[name][frame_id] for name in joint_names}
        side = project_points(frame_positions, bounds_min, bounds_max, (2, 1), side_view)
        front = project_points(frame_positions, bounds_min, bounds_max, (0, 1), front_view)

        for projected in (side, front):
            for parent, child in SKELETON_LINKS:
                if parent not in projected or child not in projected:
                    continue
                is_left = child.startswith("Left") or parent.startswith("Left")
                is_right = child.startswith("Right") or parent.startswith("Right")
                color = (40, 100, 220) if is_left else (220, 80, 60) if is_right else (35, 35, 35)
                draw_line(image, projected[parent], projected[child], color, thickness=3)
            for name in ("Hips", "Head", "LeftFoot", "RightFoot", "LeftHand", "RightHand"):
                if name in projected:
                    draw_point(image, projected[name], (20, 20, 20), radius=4)

        image[:, width // 2 - 1 : width // 2 + 1] = (180, 180, 180)
        frames.append(image)

    duration = len(frames) / max(fps, 1)
    duration_label = f"{duration:.2f}".replace(".", "p")
    output_name = (
        f"raw_{safe_name(kind)}_{safe_name(path.stem)}_"
        f"f{segment.start_frame:04d}-{segment.end_frame:04d}_"
        f"frames{len(frames):03d}_dur{duration_label}s.mp4"
    )
    output_path = out_dir / output_name
    write_video(output_path, frames, fps=fps)
    return {
        "file": output_path.name,
        "source_path": path.as_posix(),
        "kind": kind,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "frame_count": len(frames),
        "duration_s": duration,
        "support_foot": segment.support_foot,
    }


def main() -> None:
    args = parse_args()
    if not args.reference_gait_file:
        raise ValueError("Dodaj bar jedan --reference-gait-file BVH fajl ili folder.")

    paths = list(expand_motion_paths(tuple(args.reference_gait_file)))
    if args.source_limit is not None:
        paths = paths[: max(0, args.source_limit)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in paths:
        bvh = _parse_bvh(path)
        positions = _global_joint_positions(bvh)
        segments = selected_segments(path, bvh, args.segments)
        if args.max_clips_per_source is not None:
            segments = segments[: max(0, args.max_clips_per_source)]
        print(f"{path}: rendering {len(segments)} raw {args.segments} clips")
        for kind, segment in segments:
            row = render_segment(
                path,
                positions,
                kind,
                segment,
                args.out_dir,
                args.fps,
                args.width,
                args.height,
            )
            rows.append(row)
            print(f"  {row['file']}", flush=True)

    if rows:
        manifest_path = args.out_dir / "manifest_raw_bvh.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
