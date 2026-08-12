from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
KEEP_MANIFEST = PROJECT_ROOT / "CMU_SMPL+H-G" / "_walking_filter_report" / "keep_manifest.txt"
BVH_INDEX_ROOT = PROJECT_ROOT / "BVH_walking_animation" / "cmuconvert-max-01-09"
OUTPUT_LIST = PROJECT_ROOT / "BVH_walking_animation" / "cmu_strict_flat_walk_96.txt"


def main() -> None:
    lines = KEEP_MANIFEST.read_text(encoding="utf-8").splitlines()
    clip_ids: list[str] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 1:
            clip_ids.append(parts[0].strip())

    bvh_paths: list[str] = []
    missing: list[str] = []
    for clip_id in clip_ids:
        subject, motion = clip_id.split("_", maxsplit=1)
        relative_path = Path(subject) / f"{subject}_{motion}.bvh"
        candidate = BVH_INDEX_ROOT / relative_path
        if candidate.exists():
            bvh_paths.append(candidate.relative_to(PROJECT_ROOT).as_posix())
        else:
            missing.append(clip_id)

    OUTPUT_LIST.write_text(
        "\n".join(bvh_paths) + ("\n" if bvh_paths else ""),
        encoding="utf-8",
    )

    print(f"keep_manifest={len(clip_ids)} bvh_found={len(bvh_paths)} missing={len(missing)}")
    if missing:
        print("missing_clip_ids=" + ",".join(missing))


if __name__ == "__main__":
    main()
