"""Build tiered BVH walking reference lists from the CMU index text files."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BVH_ROOT = PROJECT_ROOT / "BVH_walking_animation"
INDEX_PATTERN = re.compile(r"^\s*(\d{2,3}_\d{2})\s+(.+?)\s*$")

TIER1_EXCLUDE = {
    "back",
    "backward",
    "backwards",
    "bent",
    "bouncy",
    "carry",
    "carries",
    "crouch",
    "crouched",
    "digital",
    "duck",
    "figure",
    "hobble",
    "jump",
    "ladder",
    "lean",
    "left",
    "limp",
    "march",
    "navigate",
    "obstacle",
    "right",
    "run",
    "side",
    "sideway",
    "sideways",
    "stairs",
    "stop",
    "style",
    "turn",
    "uneven",
    "veer",
    "weird",
    "with",
    "zigzag",
}
TIER2_HINTS = {
    "brisk",
    "fast",
    "forward",
    "jog",
    "left",
    "right",
    "run",
    "slow",
    "start",
    "stop",
    "stride",
    "turn",
    "veer",
}
UNEVEN_HINTS = {"stair", "stairs", "terrain", "uneven"}


def main() -> None:
    descriptions = read_descriptions()
    existing_bvh = sorted(BVH_ROOT.rglob("*.bvh"))
    buckets = {
        "tier1_forward_walk.txt": [],
        "tier2_walk_variations.txt": [],
        "tier3_style_or_complex_walks.txt": [],
        "uneven_terrain_walks.txt": [],
    }

    for bvh_path in existing_bvh:
        description = descriptions.get(bvh_path.stem, "")
        bucket = classify_description(description)
        relative_path = bvh_path.relative_to(PROJECT_ROOT).as_posix()
        buckets[bucket].append((relative_path, description))

    for filename, entries in buckets.items():
        write_list(BVH_ROOT / filename, entries)

    write_summary(BVH_ROOT / "walk_tiers_summary.md", buckets)
    print("wrote BVH walking tier lists")
    for filename, entries in buckets.items():
        print(f"{filename}: {len(entries)}")


def read_descriptions() -> dict[str, str]:
    """Read motion id descriptions from every bundled CMU text index."""
    descriptions: dict[str, str] = {}
    for index_path in BVH_ROOT.rglob("cmu-mocap-index-text.txt"):
        for line in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = INDEX_PATTERN.match(line)
            if not match:
                continue
            motion_id, description = match.groups()
            descriptions.setdefault(motion_id, description.strip())
    return descriptions


def classify_description(description: str) -> str:
    """Classify a walking clip into curriculum tiers."""
    normalized = description.lower()
    words = set(re.findall(r"[a-z]+", normalized))

    if words & UNEVEN_HINTS:
        return "uneven_terrain_walks.txt"
    if is_tier1_forward_walk(normalized, words):
        return "tier1_forward_walk.txt"
    if words & TIER2_HINTS:
        return "tier2_walk_variations.txt"
    return "tier3_style_or_complex_walks.txt"


def is_tier1_forward_walk(description: str, words: set[str]) -> bool:
    """Return True for plain forward walking references."""
    if words & TIER1_EXCLUDE:
        return False
    return description in {"walk", "normal walk"} or "normal walk" in description


def write_list(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write one path-per-line list with descriptions as comments."""
    lines = [
        "# One BVH path per non-comment line.",
        "# Description is kept above each path for review.",
    ]
    for relative_path, description in entries:
        lines.append(f"# {description}")
        lines.append(relative_path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, buckets: dict[str, list[tuple[str, str]]]) -> None:
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


if __name__ == "__main__":
    main()
