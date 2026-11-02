"""Verify the frozen first-walk payload without importing JAX or changing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "prezentacija" / "1st_success"


def verify_archive(archive: Path, require_tracked: bool = False) -> int:
    """Check every manifest payload and optionally its inclusion in Git's index."""
    archive = archive.resolve()
    manifest = json.loads((archive / "SHA256_MANIFEST.json").read_text("utf-8"))
    errors = []
    for relative, expected in manifest["files"].items():
        path = (archive / relative).resolve()
        if not path.is_relative_to(archive):
            errors.append(f"Unsafe manifest path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"Missing file: {relative}")
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            errors.append(f"Content mismatch: {relative}")
        if require_tracked:
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                errors.append(f"Not in Git index: {relative}")
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"OK: {len(manifest['files'])} payload files match SHA-256 and size.")
    if require_tracked:
        print("OK: every payload file is present in the Git index.")
    return 0


def main() -> None:
    """Run a read-only archive integrity check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args()
    raise SystemExit(verify_archive(args.archive, args.require_tracked))


if __name__ == "__main__":
    main()
