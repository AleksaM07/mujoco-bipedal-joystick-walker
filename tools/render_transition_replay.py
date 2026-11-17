"""Render a saved transition replay to the project's central videos folder."""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import render_host_replay_frames, write_video
from skills.policy_registry import FINAL_WALKING
from tools.evaluate_skill_transitions import load_walk_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if not args.name.endswith(".mp4") or Path(args.name).name != args.name:
        raise ValueError("--name must be one plain .mp4 filename")

    replay = np.load(args.replay.resolve())
    snapshots = list(zip(replay["qpos"], replay["qvel"], strict=True))
    walk_config = load_walk_config(FINAL_WALKING.checkpoint)
    if walk_config.xml_path is None:
        raise ValueError("FINAL_WALKING XML is unresolved")
    model = mujoco.MjModel.from_xml_path(walk_config.xml_path)
    frames = render_host_replay_frames(
        model,
        snapshots,
        args.width,
        args.height,
        layout="quad",
    )
    output = PROJECT_ROOT / "videos" / "final_skill_transitions" / args.name
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(output, frames, fps=args.fps)
    print(output)


if __name__ == "__main__":
    main()
