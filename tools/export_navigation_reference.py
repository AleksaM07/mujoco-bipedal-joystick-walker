"""Export the frozen walk's prepared reference through the existing environment."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from evaluate import make_environment

ARCHIVE = PROJECT_ROOT / "prezentacija/5th_success_final_natural_walk"


def main() -> None:
    """Save prepared reference knots without training or loading policy weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved_config = ARCHIVE / "config.json"
    xml = (
        ARCHIVE
        / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
    )
    bvh = ARCHIVE / "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
    config = replace(
        EnvConfig(**json.loads(saved_config.read_text(encoding="utf-8"))["env"]),
        xml_path=str(xml),
        reference_gait_file=(str(bvh),),
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
    )
    env = make_environment(config)
    clip = int(config.reference_forced_clip_id)
    count = int(np.asarray(env._bvh_reference_frame_counts)[clip])
    qpos = np.tile(env._init_q_np, (count, 1))
    qpos[:, :3] = env._bvh_reference_sim_root_pos_targets_np[clip, :count]
    qpos[:, 3:7] = env._bvh_reference_sim_root_quat_targets_np[clip, :count]
    qpos[:, np.asarray(env._actuator_qpos_indices)] = np.asarray(
        env._bvh_reference_qpos_targets
    )[clip, :count]
    if not np.isfinite(qpos).all() or count < 2:
        raise ValueError("Prepared reference is empty or non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scope": "prepared_reference_knots_not_policy_rollout",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "bvh_sha256": hashlib.sha256(bvh.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(saved_config.read_bytes()).hexdigest(),
        "clip": clip,
        "frames": count,
        "frame_dt": float(np.asarray(env._bvh_reference_frame_times)[clip]),
        "loop_mode": int(np.asarray(env._bvh_reference_loop_modes)[clip]),
        "passive_joints": "fixed at the existing environment initial pose",
        "limitation": (
            "MimicKit interpolation/velocities differ from the current Hermite "
            "reference and need separate validation"
        ),
    }
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, qpos=qpos, metadata=json.dumps(metadata))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
