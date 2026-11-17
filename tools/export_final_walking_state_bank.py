"""Export real FINAL_WALKING rollout states for bridge-policy reset sampling."""

import argparse
import hashlib
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (
    load_ppo_policy,
    make_environment,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
)
from skills.policy_registry import FINAL_WALKING
from tools.evaluate_skill_transitions import load_walk_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    assets_dir = (PROJECT_ROOT / "skills" / "assets").resolve()
    if output.parent != assets_dir or output.suffix != ".npz":
        raise ValueError("Output must be one NPZ file directly under skills/assets/")
    if output.exists():
        raise FileExistsError(output)
    if args.steps <= args.warmup_steps or args.max_states < 1:
        raise ValueError("Need steps > warmup_steps and a positive max_states")

    device = jax.devices(args.device)[0]
    config = load_walk_config(FINAL_WALKING.checkpoint)
    with jax.default_device(device):
        env = make_environment(config)
        validate_xml_manifest(FINAL_WALKING.checkpoint, env)
        validate_action_compatibility(FINAL_WALKING.checkpoint, env.action_size)
        policy = load_ppo_policy(FINAL_WALKING.checkpoint, deterministic=True)

        @jax.jit
        def rollout(seed: jax.Array) -> dict[str, jax.Array]:
            reset_key, policy_key = jax.random.split(seed)
            state = env.reset(reset_key)

            def step_once(carry, unused):
                current, key = carry
                key, action_key = jax.random.split(key)
                action, _ = policy(current.obs, action_key)
                next_state = env.step(current, action)
                motion_time = next_state.info["bvh_reference_time_offset"] + (
                    next_state.info["motion_step"].astype(jp.float32) * env.dt
                )
                sample = {
                    "qpos": next_state.data.qpos,
                    "qvel": next_state.data.qvel,
                    "ctrl": next_state.data.ctrl,
                    "last_action": next_state.info["last_action"],
                    "motion_time": motion_time,
                    "clip_id": next_state.info["bvh_reference_clip_id"],
                    "done": next_state.done,
                }
                return (next_state, key), sample

            (_, _), samples = jax.lax.scan(
                step_once,
                (state, policy_key),
                xs=None,
                length=args.steps,
            )
            return samples

        initial = env.reset(jax.random.PRNGKey(args.seeds[0]))
        validate_observation_compatibility(FINAL_WALKING.checkpoint, initial.obs)
        batches = [
            jax.device_get(rollout(jax.random.PRNGKey(seed))) for seed in args.seeds
        ]

    arrays: dict[str, np.ndarray] = {}
    for name in ("qpos", "qvel", "ctrl", "last_action", "motion_time", "clip_id"):
        arrays[name] = np.concatenate(
            [np.asarray(batch[name])[args.warmup_steps :] for batch in batches],
            axis=0,
        )
    done = np.concatenate(
        [np.asarray(batch["done"])[args.warmup_steps :] for batch in batches]
    ).astype(bool)
    keep = np.flatnonzero(~done)
    if keep.size < args.max_states:
        raise ValueError(
            f"Only {keep.size} nonterminal walking states, need {args.max_states}."
        )
    selected = keep[np.linspace(0, keep.size - 1, args.max_states).round().astype(int)]
    arrays = {name: values[selected] for name, values in arrays.items()}

    xml_path = Path(config.xml_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        dt=np.asarray(env.dt, dtype=np.float32),
        xml_sha256=np.asarray(file_sha256(xml_path)),
        source_checkpoint=np.asarray(FINAL_WALKING.checkpoint_relative_path),
        source_policy=np.asarray(FINAL_WALKING.label),
        seeds=np.asarray(args.seeds, dtype=np.int32),
    )
    print(
        f"wrote {output} | states={args.max_states} | "
        f"qpos={arrays['qpos'].shape} | qvel={arrays['qvel'].shape}"
    )


if __name__ == "__main__":
    main()
