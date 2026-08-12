from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


INDEX_LINE_RE = re.compile(r"^(?P<clip_id>\d+_\d+)\t(?P<description>.+?)\s*$")
STRICT_WALK_DESCRIPTIONS = {
    "walk",
    "walking",
    "slow walk",
    "walk slow",
    "slow walking",
}


@dataclass(frozen=True)
class ClipDecision:
    clip_id: str
    relative_path: str
    description: str
    keep: bool
    reason: str


def parse_index(index_path: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        match = INDEX_LINE_RE.match(line)
        if match:
            descriptions[match.group("clip_id")] = match.group("description").strip()
    return descriptions


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def should_keep(description: str) -> tuple[bool, str]:
    text = normalize(description)
    if text in STRICT_WALK_DESCRIPTIONS:
        return True, "strict_flat_walk_description"
    return False, "non_strict_walk_description"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_decisions(root: Path, descriptions: dict[str, str]) -> list[ClipDecision]:
    decisions: list[ClipDecision] = []
    kept_hashes: dict[str, str] = {}

    for file_path in sorted(root.rglob("*_poses.npz")):
        clip_id = file_path.stem.removesuffix("_poses")
        description = descriptions.get(clip_id, "")

        if not description:
            keep = False
            reason = "missing_index_description"
        else:
            keep, reason = should_keep(description)

        if keep:
            file_hash = file_sha256(file_path)
            existing_clip = kept_hashes.get(file_hash)
            if existing_clip is None:
                kept_hashes[file_hash] = clip_id
            else:
                keep = False
                reason = f"duplicate_of_{existing_clip}"

        decisions.append(
            ClipDecision(
                clip_id=clip_id,
                relative_path=str(file_path.relative_to(root)),
                description=description,
                keep=keep,
                reason=reason,
            )
        )

    return decisions


def write_reports(root: Path, decisions: list[ClipDecision]) -> None:
    report_dir = root / "_walking_filter_report"
    report_dir.mkdir(exist_ok=True)

    keep_lines: list[str] = []
    remove_lines: list[str] = []
    for item in decisions:
        line = f"{item.clip_id}\t{item.relative_path}\t{item.reason}\t{item.description}"
        if item.keep:
            keep_lines.append(line)
        else:
            remove_lines.append(line)

    (report_dir / "keep_manifest.txt").write_text(
        "\n".join(keep_lines) + ("\n" if keep_lines else ""),
        encoding="utf-8",
    )
    (report_dir / "remove_manifest.txt").write_text(
        "\n".join(remove_lines) + ("\n" if remove_lines else ""),
        encoding="utf-8",
    )

    summary = {
        "root": str(root),
        "total": len(decisions),
        "keep": sum(item.keep for item in decisions),
        "remove": sum(not item.keep for item in decisions),
        "missing_description": sum(item.reason == "missing_index_description" for item in decisions),
        "duplicates_removed": sum(item.reason.startswith("duplicate_of_") for item in decisions),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def delete_removed(root: Path, decisions: list[ClipDecision]) -> int:
    removed = 0
    for item in decisions:
        if item.keep:
            continue
        target = root / item.relative_path
        if target.exists():
            target.unlink()
            removed += 1
    return removed


def remove_empty_directories(root: Path) -> int:
    removed = 0
    for directory in sorted(root.rglob("*"), reverse=True):
        if directory.is_dir() and directory.name != "_walking_filter_report":
            if not any(directory.iterdir()):
                directory.rmdir()
                removed += 1
    return removed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only flat-ground walk and slow-walk CMU SMPL+H clips."
    )
    parser.add_argument("root", type=Path, help="Path to the CMU_SMPL+H-G directory.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("BVH_walking_animation/cmuconvert-max-01-09/cmu-mocap-index-text.txt"),
        help="Path to the local CMU mocap index text file.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete clips that are not classified as strict flat walking.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    root = args.root.resolve()
    index_path = args.index.resolve()

    descriptions = parse_index(index_path)
    decisions = collect_decisions(root, descriptions)
    write_reports(root, decisions)

    kept = sum(item.keep for item in decisions)
    removed = sum(not item.keep for item in decisions)
    print(f"total={len(decisions)} keep={kept} remove={removed}")

    if args.delete:
        deleted_files = delete_removed(root, decisions)
        deleted_dirs = remove_empty_directories(root)
        print(f"deleted_files={deleted_files} deleted_empty_dirs={deleted_dirs}")


if __name__ == "__main__":
    main()
