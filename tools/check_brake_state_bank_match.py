"""Compare a supervisor handoff state with the closest walking reset-bank state."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.policy_registry import FINAL_WALKING
from tools.evaluate_skill_transitions import load_walk_config, physical_match_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("state_bank", type=Path)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--lead-min", type=float, default=0.2)
    parser.add_argument("--lead-max", type=float, default=0.8)
    parser.add_argument("--blend-seconds", type=float, default=0.8)
    parser.add_argument("--stop-phase", type=float, default=70.0 / 133.0)
    args = parser.parse_args()

    replay = np.load(args.replay.resolve())
    bank = np.load(args.state_bank.resolve())
    replay_matches = np.flatnonzero(replay["steps"] == args.step)
    if replay_matches.size != 1:
        raise ValueError(f"Replay does not contain exactly one step {args.step}")
    replay_index = int(replay_matches[0])

    config = load_walk_config(FINAL_WALKING.checkpoint)
    if config.xml_path is None:
        raise ValueError("FINAL_WALKING XML is unresolved")
    model = mujoco.MjModel.from_xml_path(config.xml_path)
    joint_ids = model.actuator_trnid[:, 0]
    qpos_indices = model.jnt_qposadr[joint_ids]
    dof_indices = model.jnt_dofadr[joint_ids]

    # FINAL_WALKING uses one selected clip, so one duration applies to the bank.
    source_start = 79
    source_end = 213
    frame_time = 0.0083333
    duration = (source_end - source_start - 1) * frame_time
    stop_time = duration * args.stop_phase
    phase_time = np.mod(bank["motion_time"], duration)
    first_stop_delta = np.mod(stop_time - phase_time, duration)
    extra_cycles = np.maximum(
        np.ceil((args.blend_seconds - first_stop_delta) / duration), 0.0
    )
    lead = first_stop_delta + extra_cycles * duration - args.blend_seconds
    eligible = np.flatnonzero((lead >= args.lead_min) & (lead <= args.lead_max))
    if eligible.size == 0:
        raise ValueError("No reset-bank states are in the configured lead window")

    candidates = [
        physical_match_score(
            replay["qpos"][replay_index],
            replay["qvel"][replay_index],
            bank["qpos"][index],
            bank["qvel"][index],
            qpos_indices,
            dof_indices,
        )
        for index in eligible
    ]
    local_index = min(range(len(candidates)), key=lambda index: candidates[index]["total"])
    bank_index = int(eligible[local_index])
    print(
        json.dumps(
            {
                "replay_step": args.step,
                "eligible_bank_states": int(eligible.size),
                "best_bank_index": bank_index,
                "lead_seconds": float(lead[bank_index]),
                "physical_match": candidates[local_index],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
