"""Export deterministic stand-policy controls for cross-engine replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np
from brax.training import checkpoint as brax_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (
    choose_device,
    make_environment,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
)
from tools.evaluate_stand import load_stand_config, make_rollout_step


def main() -> None:
    """Run the saved actor once and save its applied actuator targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    config = load_stand_config(checkpoint)

    with jax.default_device(choose_device(args.device)):
        env = make_environment(config)
        validate_xml_manifest(checkpoint, env)
        validate_action_compatibility(checkpoint, env.action_size)
        reset = jax.jit(env.reset)
        advance = make_rollout_step(env, checkpoint)
        params = brax_checkpoint.load(str(checkpoint))
        key = jax.random.PRNGKey(args.seed)
        state = reset(key)
        validate_observation_compatibility(checkpoint, state.obs)
        initial_qpos = np.asarray(state.data.qpos)
        initial_qvel = np.asarray(state.data.qvel)
        initial_xpos = np.asarray(state.data.xpos)
        initial_xquat = np.asarray(state.data.xquat)
        initial_xmat = np.asarray(state.data.xmat)
        initial_cvel = np.asarray(state.data.cvel)
        controls = []
        qpos = []
        qvel = []
        xpos = []
        xquat = []
        xmat = []
        cvel = []
        samples = []
        for _ in range(args.steps):
            key, action_key = jax.random.split(key)
            state, sample = advance(state, action_key, params)
            host_sample = {
                name: float(value) for name, value in jax.device_get(sample).items()
            }
            controls.append(np.asarray(state.data.ctrl))
            qpos.append(np.asarray(state.data.qpos))
            qvel.append(np.asarray(state.data.qvel))
            xpos.append(np.asarray(state.data.xpos))
            xquat.append(np.asarray(state.data.xquat))
            xmat.append(np.asarray(state.data.xmat))
            cvel.append(np.asarray(state.data.cvel))
            samples.append(host_sample)
            if host_sample["physical_failure"] or host_sample["done"]:
                break

    metadata = {
        "scope": "saved_MJX_policy_control_trace_for_open_loop_engine_replay",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata_sha256": hashlib.sha256(
            (checkpoint / "_CHECKPOINT_METADATA").read_bytes()
        ).hexdigest(),
        "xml_path": str(Path(env.xml_path).resolve()),
        "xml_sha256": hashlib.sha256(Path(env.xml_path).read_bytes()).hexdigest(),
        "seed": args.seed,
        "requested_steps": args.steps,
        "recorded_steps": len(controls),
        "control_dt": float(env.dt),
        "simulation_dt": float(env.mj_model.opt.timestep),
        "passed_source_policy_rollout": bool(
            len(controls) == args.steps
            and all(
                not sample["physical_failure"] and not sample["done"]
                for sample in samples
            )
        ),
        "limitation": (
            "Controls are replayed open-loop in another engine; this validates "
            "control/physics transfer, not closed-loop observation transfer."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata=json.dumps(metadata),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        initial_xpos=initial_xpos,
        initial_xquat=initial_xquat,
        initial_xmat=initial_xmat,
        initial_cvel=initial_cvel,
        controls=np.asarray(controls),
        source_qpos=np.asarray(qpos),
        source_qvel=np.asarray(qvel),
        source_xpos=np.asarray(xpos),
        source_xquat=np.asarray(xquat),
        source_xmat=np.asarray(xmat),
        source_cvel=np.asarray(cvel),
    )
    print(json.dumps(metadata, indent=2))
    raise SystemExit(0 if metadata["passed_source_policy_rollout"] else 1)


if __name__ == "__main__":
    main()
