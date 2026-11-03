"""Launch the frozen 26-DOF walking policy with its archived XML and BVH."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT_ROOT / "prezentacija" / "2nd_success_stiff_arms"


def main() -> None:
    """Forward viewer/recording flags using the visually approved checkpoint."""
    arguments = sys.argv[1:]
    print_only = "--print-command" in arguments
    arguments = [value for value in arguments if value != "--print-command"]
    pinned_flags = {"--checkpoint", "--xml-path", "--reference-gait-file"}
    if any(value.split("=", 1)[0] in pinned_flags for value in arguments):
        raise SystemExit("Use evaluate.py directly to choose another model or reference.")
    checkpoint = ARCHIVE / "model_000010035200"
    xml = (
        ARCHIVE / "generated_models"
        / "human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
    )
    bvh = ARCHIVE / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
    for path in (checkpoint, xml, bvh, ARCHIVE / "config.json"):
        if not path.exists():
            raise SystemExit(f"Missing second-success archive asset: {path}")
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "evaluate.py"),
        "--checkpoint", str(checkpoint),
        "--device", "cpu",
        "--xml-path", str(xml),
        "--reference-gait-file", str(bvh),
        *arguments,
    ]
    if print_only:
        print(subprocess.list2cmdline(command))
        return
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
