"""Check stationary reference consistency and free-root PD holding, without PPO."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig  # noqa: E402
from evaluate import (  # noqa: E402
    load_ppo_policy,
    make_environment,
    validate_action_compatibility,
    validate_observation_compatibility,
)
from skills.stand import StandEnv  # noqa: E402
from skills.stand_metrics import StandThresholds, required_dwell_steps  # noqa: E402

ARCHIVE = PROJECT_ROOT / "prezentacija/5th_success_final_natural_walk"


def stand_config() -> EnvConfig:
    """Pin the successful morphology and settings, selecting a static skill."""
    values = json.loads((ARCHIVE / "config.json").read_text(encoding="utf-8"))["env"]
    return replace(
        EnvConfig(**values),
        locomotion_skill="stand",
        playground_impl="jax",
        physics_backend="mjx_jax",
        warp_num_worlds=1,
        xml_path=str(
            ARCHIVE
            / "generated_models"
            / "human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
        ),
        reference_forced_clip_id=0,
        reference_clip_ids=None,
        reference_gait_file=(),
    )


def run_pd_hold(env: StandEnv, seconds: float) -> dict[str, object]:
    """Hold the exact target with production position actuators and a free root."""
    model = env.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env._init_q_np
    data.qvel[:] = 0
    data.ctrl[:] = np.asarray(env._default_ctrl)
    mujoco.mj_forward(model, data)
    thresholds = StandThresholds()
    root_xy = data.qpos[:2].copy()
    root_height = data.qpos[2]
    foot_ids = set(int(value) for value in env._foot_geom_ids_np)
    steps = int(np.ceil(seconds / env.dt))
    substeps = int(round(env.dt / model.opt.timestep))
    dwell_steps = 0
    max_dwell_steps = 0
    min_height = root_height
    min_up = 1.0
    max_speed = 0.0
    max_distance = 0.0
    fell = False
    completed = 0
    initial_foot_heights = [
        float(env._geom_min_z(data, geom_id)) for geom_id in sorted(foot_ids)
    ]
    for step in range(steps):
        mujoco.mj_step(model, data, nstep=substeps)
        # mj_step positions and velocities are staggered; refresh for metrics.
        mujoco.mj_forward(model, data)
        up = float(data.xmat[env._torso_body_id].reshape(3, 3)[2, 1])
        speed = float(np.linalg.norm(data.qvel[:2]))
        distance = float(np.linalg.norm(data.qpos[:2] - root_xy))
        finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        touching = set()
        for contact in data.contact[: data.ncon]:
            pair = set(int(value) for value in contact.geom)
            if env._floor_geom_id in pair and contact.dist <= 0:
                touching.update(pair & foot_ids)
        fell = bool(
            not finite
            or up < 0.25
            or data.qpos[2]
            < float(env._standing_height()) * env.MIN_STANDING_HEIGHT_RATIO
        )
        eligible = (
            not fell
            and up >= thresholds.min_torso_up
            and data.qpos[2] >= root_height * thresholds.min_height_ratio
            and speed < thresholds.max_speed
            and distance < thresholds.max_distance
            and touching == foot_ids
        )
        dwell_steps = dwell_steps + 1 if eligible else 0
        max_dwell_steps = max(max_dwell_steps, dwell_steps)
        min_height = min(min_height, float(data.qpos[2]))
        min_up = min(min_up, up)
        max_speed = max(max_speed, speed)
        max_distance = max(max_distance, distance)
        completed = step + 1
        if fell:
            break
    return {
        "controller": "constant_neutral_PD_targets_no_policy",
        "full_gravity": model.opt.gravity.tolist(),
        "root_free": bool(model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE),
        "failure_torso_up_threshold": 0.25,
        "steps": completed,
        "seconds": completed * env.dt,
        "fell": fell,
        "initial_foot_heights": initial_foot_heights,
        "min_height": min_height,
        "min_torso_up": min_up,
        "max_speed": max_speed,
        "max_distance": max_distance,
        "max_dwell_seconds": max_dwell_steps * env.dt,
        "final_dwell_seconds": dwell_steps * env.dt,
        "passed": bool(
            not fell
            and completed == steps
            and dwell_steps >= required_dwell_steps(env.dt, thresholds)
        ),
    }


def main() -> None:
    """Write a reproducible feasibility report; optionally smoke-test MJX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--mjx-steps", type=int, default=0)
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Probe the frozen walking actor on stand targets; this does not train it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs/skills/stand_preflight.json",
    )
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 1 or args.mjx_steps < 0:
        parser.error("seconds must be finite and >= 1; mjx-steps must be >= 0")
    if args.policy and not args.mjx_steps:
        parser.error("--policy requires --mjx-steps")
    config = stand_config()
    env = make_environment(config)
    print(
        "Stand environment ready; checking reference and free-root PD hold.", flush=True
    )
    query = jax.jit(env._query_bvh_reference_clip)
    reference = query(jp.array(0), jp.array(0.0))
    later = query(jp.array(0), jp.array(100.0))
    target = np.asarray(reference["qpos"])
    reference_checks = {
        "stationary": all(
            np.allclose(reference[name], later[name])
            for name in ("qpos", "root_pos", "root_quat", "key_pos")
        ),
        "zero_velocities": all(
            np.allclose(reference[name], 0)
            for name in ("qvel", "root_vel", "root_angvel", "sim_root_vel")
        ),
        "within_control_limits": bool(
            np.all(
                (target >= env._actuator_ctrl_lower_limits_np)
                & (target <= env._actuator_ctrl_upper_limits_np)
            )
        ),
    }
    saved_environment = asdict(config)
    saved_environment["xml_path"] = (
        Path(config.xml_path).relative_to(PROJECT_ROOT).as_posix()
    )
    report = {
        "environment": saved_environment,
        "xml_sha256": hashlib.sha256(Path(config.xml_path).read_bytes()).hexdigest(),
        "thresholds": asdict(StandThresholds()),
        "actions": env.action_size,
        "control_dt": env.dt,
        "simulation_dt": float(env.mj_model.opt.timestep),
        "reference_checks": reference_checks,
        "pd_hold": run_pd_hold(env, args.seconds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["pd_hold"], indent=2), flush=True)
    if args.mjx_steps:
        print("Compiling MJX reset/step smoke test (no training).", flush=True)
        state = jax.jit(env.reset)(jax.random.PRNGKey(11))
        step = jax.jit(env.step)
        policy = None
        if args.policy:
            checkpoint = ARCHIVE / "model_000001024000"
            validate_action_compatibility(checkpoint, env.action_size)
            validate_observation_compatibility(checkpoint, state.obs)
            policy = jax.jit(load_ppo_policy(checkpoint, deterministic=True))
        completed = 0
        max_dwell = 0.0
        for index in range(args.mjx_steps):
            action = jp.zeros(env.action_size)
            if policy is not None:
                action, _ = policy(state.obs, jax.random.PRNGKey(11 + index))
            state = step(state, action)
            completed = index + 1
            max_dwell = max(max_dwell, float(state.metrics["stand_dwell_seconds"]))
            if float(state.done):
                break
        jax.block_until_ready(state)
        report["mjx_smoke"] = {
            "controller": "frozen_walking_actor" if policy else "zero_residual",
            "steps": completed,
            "requested_steps": args.mjx_steps,
            "max_dwell_seconds": max_dwell,
            "observation_sizes": {
                key: list(value.shape) for key, value in state.obs.items()
            },
            "finite": bool(
                all(np.isfinite(value).all() for value in state.obs.values())
            ),
            "done": float(state.done),
            "metrics": {
                key: float(value)
                for key, value in state.metrics.items()
                if key.startswith("stand_")
            },
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"environment", "thresholds"}
            },
            indent=2,
        )
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
