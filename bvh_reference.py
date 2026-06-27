import argparse
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    BVH_INDEX_PATTERN,
    BVH_ROOT,
    BVH_TIER1_EXCLUDE,
    BVH_TIER2_HINTS,
    BVH_UNEVEN_HINTS,
    PROJECT_ROOT,
)


@dataclass(frozen=True)
class BvhReference:
    """Retargetovana BVH referenca u redosledu MuJoCo aktuatora."""

    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    frame_time: float
    source_path: Path


@dataclass(frozen=True)
class BvhReferenceBatch:
    """Padded BVH clips ready for JAX/MJX static-shape reference tracking.

    Raw BVH clips have different frame counts. The env samples clips by index,
    so the references are padded into dense tensors and `frame_counts` keeps
    the real per-clip length.
    """

    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    frame_times: np.ndarray
    frame_counts: np.ndarray
    source_paths: tuple[Path, ...]


def load_bvh_references(
    paths: tuple[str | Path, ...],
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> BvhReferenceBatch:
    """Ucita vise BVH referenci i pad-uje ih u jedan staticki tensor."""
    if not paths:
        raise ValueError("Potrebna je bar jedna BVH putanja.")

    references = [
        load_bvh_reference(
            path,
            actuator_joint_names,
            default_ctrl,
            lower_limits,
            upper_limits,
        )
        for path in paths
    ]
    max_frames = max(reference.qpos_targets.shape[0] for reference in references)
    action_size = references[0].qpos_targets.shape[1]
    targets = np.zeros((len(references), max_frames, action_size), dtype=np.float32)
    velocities = np.zeros(
        (len(references), max_frames, action_size),
        dtype=np.float32,
    )
    frame_times = np.zeros(len(references), dtype=np.float32)
    frame_counts = np.zeros(len(references), dtype=np.int32)

    for index, reference in enumerate(references):
        frame_count = reference.qpos_targets.shape[0]
        targets[index, :frame_count] = reference.qpos_targets
        targets[index, frame_count:] = reference.qpos_targets[-1]
        velocities[index, :frame_count] = reference.qvel_targets
        velocities[index, frame_count:] = 0.0
        frame_times[index] = reference.frame_time
        frame_counts[index] = frame_count

    return BvhReferenceBatch(
        qpos_targets=targets,
        qvel_targets=velocities,
        frame_times=frame_times,
        frame_counts=frame_counts,
        source_paths=tuple(reference.source_path for reference in references),
    )


def load_bvh_reference(
    path: str | Path,
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> BvhReference:
    """Ucita CMU-style BVH i retargetuje osnovni hod na nas model."""
    source_path = Path(path).expanduser()
    bvh = _parse_bvh(source_path)
    targets = np.tile(np.asarray(default_ctrl, dtype=np.float32), (bvh.frames, 1))

    def assign_target(joint_name: str, values: np.ndarray) -> None:
        if joint_name not in actuator_joint_names:
            return
        index = actuator_joint_names.index(joint_name)
        targets[:, index] = np.clip(values, lower_limits[index], upper_limits[index])

    # CMU BVH koristi pozitivnu X rotaciju kolena za fleksiju, dok nas model
    # koristi negativan knee_z. Zato kolena mapiramo posebno kao fleksiju.
    assign_target(
        "left_hip_x",
        default_ctrl[actuator_joint_names.index("left_hip_x")]
        + 0.55 * _centered_rotation(bvh, "LeftHip", "Xrotation"),
    )
    assign_target(
        "right_hip_x",
        default_ctrl[actuator_joint_names.index("right_hip_x")]
        + 0.55 * _centered_rotation(bvh, "RightHip", "Xrotation"),
    )
    assign_target(
        "left_knee_z",
        default_ctrl[actuator_joint_names.index("left_knee_z")]
        - 0.75 * _positive_flexion(bvh, "LeftKnee", "Xrotation"),
    )
    assign_target(
        "right_knee_z",
        default_ctrl[actuator_joint_names.index("right_knee_z")]
        - 0.75 * _positive_flexion(bvh, "RightKnee", "Xrotation"),
    )
    assign_target(
        "left_ankle_y",
        default_ctrl[actuator_joint_names.index("left_ankle_y")]
        + 0.35 * _centered_rotation(bvh, "LeftAnkle", "Xrotation"),
    )
    assign_target(
        "right_ankle_y",
        default_ctrl[actuator_joint_names.index("right_ankle_y")]
        + 0.35 * _centered_rotation(bvh, "RightAnkle", "Xrotation"),
    )

    return BvhReference(
        qpos_targets=targets.astype(np.float32),
        qvel_targets=_target_velocities(targets, bvh.frame_time),
        frame_time=bvh.frame_time,
        source_path=source_path,
    )


def _target_velocities(targets: np.ndarray, frame_time: float) -> np.ndarray:
    """Izracuna reference joint brzine iz retargetovanih poza."""
    if targets.shape[0] < 2:
        return np.zeros_like(targets, dtype=np.float32)
    return np.gradient(targets, frame_time, axis=0).astype(np.float32)


@dataclass(frozen=True)
class ParsedBvh:
    """Minimalni BVH podaci potrebni za retargeting."""

    channels: tuple[tuple[str, str], ...]
    motion: np.ndarray
    frames: int
    frame_time: float


def _parse_bvh(path: Path) -> ParsedBvh:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    channels: list[tuple[str, str]] = []
    stack: list[str | None] = []
    motion_index = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "MOTION":
            motion_index = index
            break

        parts = stripped.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"}:
            stack.append(parts[1])
        elif parts[0] == "End":
            stack.append(None)
        elif parts[0] == "}":
            if stack:
                stack.pop()
        elif parts[0] == "CHANNELS" and stack:
            joint_name = stack[-1]
            if joint_name is None:
                continue
            for channel_name in parts[2:]:
                channels.append((joint_name, channel_name))

    if motion_index is None:
        raise ValueError(f"{path} nema MOTION sekciju.")

    frames = int(lines[motion_index + 1].split(":", 1)[1])
    frame_time = float(lines[motion_index + 2].split(":", 1)[1])
    motion_lines = lines[motion_index + 3 : motion_index + 3 + frames]
    motion = np.asarray(
        [[float(value) for value in line.split()] for line in motion_lines],
        dtype=np.float32,
    )

    if motion.shape != (frames, len(channels)):
        raise ValueError(
            f"{path} ima motion shape {motion.shape}, "
            f"ali channels={len(channels)} i frames={frames}."
        )

    return ParsedBvh(
        channels=tuple(channels),
        motion=motion,
        frames=frames,
        frame_time=frame_time,
    )


def _rotation_degrees(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    try:
        channel_index = bvh.channels.index((joint_name, channel_name))
    except ValueError as exc:
        raise ValueError(
            f"BVH nema kanal {joint_name}.{channel_name}."
        ) from exc
    return bvh.motion[:, channel_index]


def _centered_rotation(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    degrees = _rotation_degrees(bvh, joint_name, channel_name)
    centered = degrees - np.mean(degrees)
    return np.deg2rad(centered)


def _positive_flexion(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    degrees = _rotation_degrees(bvh, joint_name, channel_name)
    baseline = np.percentile(degrees, 5.0)
    return np.deg2rad(np.maximum(degrees - baseline, 0.0))


def build_walk_tiers_main() -> None:
    """Build BVH walking tier lists from the CMU index text files."""
    args = parse_walk_tier_args()
    if not BVH_ROOT.exists():
        raise FileNotFoundError(f"BVH folder not found: {BVH_ROOT}")

    descriptions = read_bvh_descriptions()
    existing_bvh = sorted(BVH_ROOT.rglob("*.bvh"))
    if not existing_bvh:
        raise ValueError(f"No BVH files found under {BVH_ROOT}")

    buckets = build_tier_buckets(existing_bvh, descriptions)

    if not args.dry_run:
        for filename, entries in buckets.items():
            write_reference_list(BVH_ROOT / filename, entries)
        write_walk_tier_summary(BVH_ROOT / "walk_tiers_summary.md", buckets)

    action = "checked" if args.dry_run else "wrote"
    print(f"{action} BVH walking tier lists")
    for filename, entries in buckets.items():
        print(f"{filename}: {len(entries)}")

    if args.check_duplicates:
        duplicate_groups = find_duplicate_bvh_files(existing_bvh)
        print_duplicate_report(duplicate_groups)


def parse_walk_tier_args() -> argparse.Namespace:
    """Parse BVH walking tier CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build BVH walking tiers and optionally audit exact duplicates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify files and print counts without writing tier lists.",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Print exact duplicate BVH files grouped by SHA-256 hash.",
    )
    return parser.parse_args()


def build_tier_buckets(
    bvh_paths: list[Path],
    descriptions: dict[str, str],
) -> dict[str, list[tuple[str, str]]]:
    """Classify BVH paths into curriculum tiers."""
    buckets = {
        "tier1_forward_walk.txt": [],
        "tier2_walk_variations.txt": [],
        "tier3_style_or_complex_walks.txt": [],
        "uneven_terrain_walks.txt": [],
    }

    for bvh_path in bvh_paths:
        description = descriptions.get(bvh_path.stem, "")
        bucket = classify_bvh_description(description)
        relative_path = bvh_path.relative_to(PROJECT_ROOT).as_posix()
        buckets[bucket].append((relative_path, description))
    return buckets


def read_bvh_descriptions() -> dict[str, str]:
    """Read motion id descriptions from every bundled CMU text index."""
    descriptions: dict[str, str] = {}
    for index_path in BVH_ROOT.rglob("cmu-mocap-index-text.txt"):
        for line in index_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            match = BVH_INDEX_PATTERN.match(line)
            if not match:
                continue
            motion_id, description = match.groups()
            descriptions.setdefault(motion_id, description.strip())
    return descriptions


def classify_bvh_description(description: str) -> str:
    """Classify a walking clip into curriculum tiers."""
    normalized = description.lower()
    words = set(re.findall(r"[a-z]+", normalized))

    if words & BVH_UNEVEN_HINTS:
        return "uneven_terrain_walks.txt"
    if is_tier1_forward_walk(normalized, words):
        return "tier1_forward_walk.txt"
    if words & BVH_TIER2_HINTS:
        return "tier2_walk_variations.txt"
    return "tier3_style_or_complex_walks.txt"


def is_tier1_forward_walk(description: str, words: set[str]) -> bool:
    """Return True for plain forward walking references."""
    if words & BVH_TIER1_EXCLUDE:
        return False
    return description in {"walk", "normal walk"} or "normal walk" in description


def write_reference_list(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write one path-per-line list with descriptions as comments."""
    lines = [
        "# One BVH path per non-comment line.",
        "# Description is kept above each path for review.",
    ]
    for relative_path, description in entries:
        lines.append(f"# {description}")
        lines.append(relative_path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_walk_tier_summary(
    path: Path,
    buckets: dict[str, list[tuple[str, str]]],
) -> None:
    """Write a short human-readable summary of the tiers."""
    lines = [
        "# BVH Walking Tiers",
        "",
        "Generated from `cmu-mocap-index-text.txt` descriptions.",
        "",
        "Recommended curriculum:",
        "",
        "- Start with `tier1_forward_walk.txt` only.",
        "- After stable walking, resume with tier1 + tier2.",
        "- Keep tier3 and uneven terrain for later robustness experiments.",
        "- Do not switch tiers automatically inside the env; use separate runs.",
        "",
    ]
    for filename, entries in buckets.items():
        lines.append(f"## {filename}")
        lines.append("")
        lines.append(f"Count: {len(entries)}")
        lines.append("")
        for relative_path, description in entries[:20]:
            lines.append(f"- `{relative_path}` - {description}")
        if len(entries) > 20:
            lines.append(f"- ... {len(entries) - 20} more")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def find_duplicate_bvh_files(paths: list[Path]) -> list[list[Path]]:
    """Find exact duplicate BVH files by hashing file contents."""
    paths_by_hash: defaultdict[str, list[Path]] = defaultdict(list)
    for path in paths:
        paths_by_hash[file_sha256(path)].append(path)
    return [group for group in paths_by_hash.values() if len(group) > 1]


def file_sha256(path: Path) -> str:
    """Return a SHA-256 hash for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_duplicate_report(duplicate_groups: list[list[Path]]) -> None:
    """Print duplicate BVH groups in repo-relative form."""
    if not duplicate_groups:
        print("duplicate BVH files: none")
        return

    duplicate_file_count = sum(len(group) for group in duplicate_groups)
    extra_copy_count = duplicate_file_count - len(duplicate_groups)
    print(
        "duplicate BVH files: "
        f"{len(duplicate_groups)} groups, {extra_copy_count} removable copies"
    )
    for group_index, group in enumerate(duplicate_groups, start=1):
        print(f"duplicate group {group_index}:")
        for path in group:
            print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    build_walk_tiers_main()
