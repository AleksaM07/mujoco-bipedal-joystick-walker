"""Render a saved stand robustness replay from several fixed cameras."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import render_host_replay_frames, write_video
from tools.check_mimickit_model import DEFAULT_XML


def main() -> None:
    """Load sparse qpos/qvel states and write a replay video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("single", "quad"), default="quad")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every Nth saved snapshot (the final snapshot is always kept).",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Video speed relative to simulation time.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.width < 320 or args.height < 240:
        raise ValueError("Video dimensions are too small.")
    if args.stride < 1:
        raise ValueError("Stride must be at least 1.")
    if args.playback_speed <= 0:
        raise ValueError("Playback speed must be positive.")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve(strict=True)))
    with np.load(args.replay.resolve(strict=True), allow_pickle=False) as replay:
        qpos = replay["qpos"].copy()
        qvel = replay["qvel"].copy()
        steps = replay["steps"].astype(np.int64)
        dt = float(replay["dt"])
    if (
        qpos.ndim != 2
        or qvel.ndim != 2
        or qpos.shape != (len(steps), model.nq)
        or qvel.shape != (len(steps), model.nv)
    ):
        raise ValueError(
            f"Replay shape mismatch: qpos={qpos.shape}, qvel={qvel.shape}, "
            f"steps={steps.shape}, model=({model.nq}, {model.nv})"
        )
    if len(steps) < 2 or dt <= 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Replay steps and dt must define an increasing timeline.")

    selected_indices = np.arange(0, len(steps), args.stride)
    if selected_indices[-1] != len(steps) - 1:
        selected_indices = np.append(selected_indices, len(steps) - 1)
    qpos = qpos[selected_indices]
    qvel = qvel[selected_indices]
    steps = steps[selected_indices]
    snapshot_interval = float(np.median(np.diff(steps))) * dt
    fps = max(1, round(args.playback_speed / snapshot_interval))
    snapshots = list(zip(qpos, qvel, strict=True))
    frames = render_host_replay_frames(
        model,
        snapshots,
        args.width,
        args.height,
        layout=args.layout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(args.output, frames, fps=fps)
    metadata = {
        "replay": str(args.replay.resolve()),
        "xml": str(args.xml.resolve()),
        "output": str(args.output.resolve()),
        "frames": len(frames),
        "fps": fps,
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "simulation_seconds": float(steps[-1] * dt),
        "stride": args.stride,
        "playback_speed": args.playback_speed,
        "layout": args.layout,
        "width": args.width,
        "height": args.height,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
