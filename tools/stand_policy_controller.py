"""Serve the saved JAX stand actor and observation pipeline over JSON lines."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (
    choose_device,
    load_ppo_policy,
    make_environment,
    validate_action_compatibility,
    validate_observation_compatibility,
    validate_xml_manifest,
)
from tools.evaluate_stand import load_stand_config


class StandPolicyController:
    """Keep JAX policy state while physics runs in a separate process."""

    def __init__(self, checkpoint: Path, seed: int, device: str):
        self.checkpoint = checkpoint.resolve(strict=True)
        self.seed = seed
        self.device = choose_device(device)
        with jax.default_device(self.device):
            config = load_stand_config(self.checkpoint)
            self.env = make_environment(config)
            validate_xml_manifest(self.checkpoint, self.env)
            validate_action_compatibility(self.checkpoint, self.env.action_size)
            self.policy = load_ppo_policy(self.checkpoint, deterministic=True)
            self.key = jax.random.PRNGKey(seed)
            state = jax.jit(self.env.reset)(self.key)
            validate_observation_compatibility(self.checkpoint, state.obs)
            self.initial_state = state
            self.data_template = state.data
            self.info = state.info
            self.initial_qpos = np.asarray(state.data.qpos)
            self.initial_qvel = np.asarray(state.data.qvel)
            self._step = jax.jit(self._controller_step)

    def _controller_step(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        body_xpos: jax.Array,
        body_xquat: jax.Array,
        body_xmat: jax.Array,
        body_cvel: jax.Array,
        info: dict[str, Any],
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, Any], jax.Array, jax.Array]:
        """Rebuild actor observations and apply the production action mapping."""
        data = self.data_template.replace(
            qpos=qpos,
            qvel=qvel,
            xpos=body_xpos,
            xquat=body_xquat,
            xmat=body_xmat,
            cvel=body_cvel,
        )
        obs = self.env._get_obs(data, info)
        key, action_key = jax.random.split(key)
        action, _ = self.policy(obs, action_key)
        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        policy_action = jp.clip(action, -1.0, 1.0)
        previous_action = info["last_action"]
        smoothed_action = (
            self.env._config.action_smoothing * policy_action
            + (1.0 - self.env._config.action_smoothing) * previous_action
        )
        motor_targets = self.env._clip_motor_targets(
            self.env._policy_action_to_motor_targets(smoothed_action, info)
        )

        updated = dict(info)
        updated["rng"], _, command_key = jax.random.split(info["rng"], 3)
        should_resample = (~self.env._is_pure_reference_imitation()) & (
            info["command_step"] > self.env._config.command_resample_steps
        )
        updated["command"] = jp.where(
            should_resample,
            self.env.sample_command(command_key),
            info["command"],
        )
        updated["last_action"] = smoothed_action
        updated["motion_step"] = info["motion_step"] + jp.array(
            1, dtype=info["motion_step"].dtype
        )
        updated["command_step"] = jp.where(
            should_resample,
            jp.array(0, dtype=info["command_step"].dtype),
            info["command_step"] + jp.array(1, dtype=info["command_step"].dtype),
        )
        updated["gait_step"] = jp.mod(
            info["gait_step"] + jp.array(1, dtype=info["gait_step"].dtype),
            self.env.GAIT_PERIOD_STEPS,
        )
        policy_obs = obs["state"] if isinstance(obs, dict) else obs
        return motor_targets, action, updated, key, policy_obs

    def control(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        body_xpos: np.ndarray,
        body_xquat: np.ndarray,
        body_xmat: np.ndarray,
        body_cvel: np.ndarray,
    ) -> dict[str, Any]:
        """Return one production control target and advance controller state."""
        if (
            qpos.shape != self.initial_qpos.shape
            or qvel.shape != self.initial_qvel.shape
        ):
            raise ValueError(
                f"State shape mismatch: qpos={qpos.shape}, qvel={qvel.shape}"
            )
        if (
            body_xpos.shape != self.data_template.xpos.shape
            or body_xquat.shape != self.data_template.xquat.shape
            or body_xmat.shape != self.data_template.xmat.shape
            or body_cvel.shape != self.data_template.cvel.shape
        ):
            raise ValueError(
                "Body derived-state shape mismatch: "
                f"xpos={body_xpos.shape}, xquat={body_xquat.shape}, "
                f"xmat={body_xmat.shape}, "
                f"cvel={body_cvel.shape}"
            )
        with jax.default_device(self.device):
            targets, action, self.info, self.key, obs = self._step(
                jp.asarray(qpos),
                jp.asarray(qvel),
                jp.asarray(body_xpos),
                jp.asarray(body_xquat),
                jp.asarray(body_xmat),
                jp.asarray(body_cvel),
                self.info,
                self.key,
            )
            targets, action, obs = jax.device_get((targets, action, obs))
        return {
            "targets": np.asarray(targets).tolist(),
            "action_l2": float(np.linalg.norm(action)),
            "action_clip_fraction": float(np.mean(np.abs(action) > 1.0)),
            "observation_l2": float(np.linalg.norm(obs)),
        }


def validate_trace(
    controller: StandPolicyController, trace_path: Path
) -> dict[str, Any]:
    """Prove that split observation/control code reproduces a saved MJX rollout."""
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {
            "initial_xpos",
            "initial_xquat",
            "initial_xmat",
            "initial_cvel",
            "source_xpos",
            "source_xquat",
            "source_xmat",
            "source_cvel",
        }
        missing = required - set(trace.files)
        if missing:
            raise ValueError(
                "Trace predates exact derived-state transfer; re-export it with "
                "tools/export_stand_policy_trace.py. Missing arrays: "
                f"{sorted(missing)}"
            )
        qpos = trace["initial_qpos"].copy()
        qvel = trace["initial_qvel"].copy()
        body_xpos = trace["initial_xpos"].copy()
        body_xquat = trace["initial_xquat"].copy()
        body_xmat = trace["initial_xmat"].copy()
        body_cvel = trace["initial_cvel"].copy()
        controls = trace["controls"].copy()
        source_qpos = trace["source_qpos"].copy()
        source_qvel = trace["source_qvel"].copy()
        source_xpos = trace["source_xpos"].copy()
        source_xquat = trace["source_xquat"].copy()
        source_xmat = trace["source_xmat"].copy()
        source_cvel = trace["source_cvel"].copy()
    errors = []
    component_indices = []
    for index, expected in enumerate(controls):
        result = controller.control(
            qpos,
            qvel,
            body_xpos,
            body_xquat,
            body_xmat,
            body_cvel,
        )
        difference = np.abs(np.asarray(result["targets"]) - expected)
        errors.append(float(np.max(difference)))
        component_indices.append(int(np.argmax(difference)))
        qpos = source_qpos[index]
        qvel = source_qvel[index]
        body_xpos = source_xpos[index]
        body_xquat = source_xquat[index]
        body_xmat = source_xmat[index]
        body_cvel = source_cvel[index]
    max_step = int(np.argmax(errors))
    return {
        "trace": str(trace_path),
        "steps": len(errors),
        "first_control_component_error": errors[0],
        "max_control_component_error": max(errors),
        "max_error_step": max_step,
        "max_error_actuator_index": component_indices[max_step],
        "first_ten_step_errors": errors[:10],
        "steps_over_1e-6": int(np.count_nonzero(np.asarray(errors) >= 1e-6)),
        "passed": bool(max(errors) < 1e-6),
    }


def diagnose_reconstruction(
    controller: StandPolicyController, steps: int
) -> dict[str, Any]:
    """Compare post-step MJX observations with qpos/qvel reconstruction."""
    if steps < 1:
        raise ValueError("steps must be positive")

    def advance(state: Any, action_key: jax.Array) -> Any:
        action, _ = controller.policy(state.obs, action_key)
        return controller.env.step(state, action)

    def reconstruct(
        state: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        data = controller.data_template.replace(
            qpos=state.data.qpos,
            qvel=state.data.qvel,
        )
        data = mjx.forward(controller.env._mjx_model, data)
        obs = controller.env._get_obs(data, state.info)
        actor_obs = obs["state"] if isinstance(obs, dict) else obs
        bridged_data = controller.data_template.replace(
            qpos=state.data.qpos,
            qvel=state.data.qvel,
            xpos=state.data.xpos,
            xquat=state.data.xquat,
            xmat=state.data.xmat,
            cvel=state.data.cvel,
        )
        bridged_obs = controller.env._get_obs(bridged_data, state.info)
        bridged_actor_obs = (
            bridged_obs["state"] if isinstance(bridged_obs, dict) else bridged_obs
        )
        exact_obs = controller.env._get_obs(state.data, state.info)
        exact_actor_obs = (
            exact_obs["state"] if isinstance(exact_obs, dict) else exact_obs
        )
        return actor_obs, bridged_actor_obs, exact_actor_obs, data.cvel, data.xmat

    with jax.default_device(controller.device):
        advance_jit = jax.jit(advance)
        reconstruct_jit = jax.jit(reconstruct)
        state = controller.initial_state
        key = jax.random.PRNGKey(controller.seed)
        rows = []
        for step in range(steps):
            key, action_key = jax.random.split(key)
            state = advance_jit(state, action_key)
            (
                reconstructed_obs,
                bridged_obs,
                exact_recomputed_obs,
                reconstructed_cvel,
                reconstructed_xmat,
            ) = reconstruct_jit(state)
            canonical_obs = (
                state.obs["state"] if isinstance(state.obs, dict) else state.obs
            )
            (
                canonical_obs,
                reconstructed_obs,
                bridged_obs,
                exact_recomputed_obs,
                canonical_cvel,
                reconstructed_cvel,
                canonical_xmat,
                reconstructed_xmat,
            ) = jax.device_get(
                (
                    canonical_obs,
                    reconstructed_obs,
                    bridged_obs,
                    exact_recomputed_obs,
                    state.data.cvel,
                    reconstructed_cvel,
                    state.data.xmat,
                    reconstructed_xmat,
                )
            )
            obs_difference = np.abs(canonical_obs - reconstructed_obs)
            bridged_obs_difference = np.abs(canonical_obs - bridged_obs)
            exact_recomputed_difference = np.abs(canonical_obs - exact_recomputed_obs)
            rows.append(
                {
                    "step": step + 1,
                    "observation_max_abs_error": float(np.max(obs_difference)),
                    "observation_l2_error": float(np.linalg.norm(obs_difference)),
                    "observation_max_error_index": int(np.argmax(obs_difference)),
                    "bridged_observation_max_abs_error": float(
                        np.max(bridged_obs_difference)
                    ),
                    "bridged_observation_max_error_index": int(
                        np.argmax(bridged_obs_difference)
                    ),
                    "exact_recomputed_observation_max_abs_error": float(
                        np.max(exact_recomputed_difference)
                    ),
                    "cvel_max_abs_error": float(
                        np.max(np.abs(canonical_cvel - reconstructed_cvel))
                    ),
                    "xmat_max_abs_error": float(
                        np.max(np.abs(canonical_xmat - reconstructed_xmat))
                    ),
                }
            )

    worst = max(rows, key=lambda row: row["observation_max_abs_error"])
    return {
        "scope": "MJX_post_step_observation_vs_qpos_qvel_reconstruction",
        "steps": len(rows),
        "maximum_observation_abs_error": worst["observation_max_abs_error"],
        "maximum_observation_l2_error": max(
            row["observation_l2_error"] for row in rows
        ),
        "maximum_error_step": worst["step"],
        "maximum_error_observation_index": worst["observation_max_error_index"],
        "maximum_cvel_abs_error": max(row["cvel_max_abs_error"] for row in rows),
        "maximum_xmat_abs_error": max(row["xmat_max_abs_error"] for row in rows),
        "maximum_bridged_observation_abs_error": max(
            row["bridged_observation_max_abs_error"] for row in rows
        ),
        "first_ten_steps": rows[:10],
        "passed": bool(
            max(row["bridged_observation_max_abs_error"] for row in rows) < 1e-6
        ),
    }


def serve(controller: StandPolicyController) -> None:
    """Read states from stdin and emit one prefixed JSON response per line."""
    ready = {
        "checkpoint": str(controller.checkpoint),
        "initial_qpos": controller.initial_qpos.tolist(),
        "initial_qvel": controller.initial_qvel.tolist(),
        "control_dt": float(controller.env.dt),
        "simulation_dt": float(controller.env.mj_model.opt.timestep),
        "action_size": int(controller.env.action_size),
    }
    print("READY " + json.dumps(ready), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            print("CLOSED {}", flush=True)
            return
        result = controller.control(
            np.asarray(request["qpos"], dtype=np.float32),
            np.asarray(request["qvel"], dtype=np.float32),
            np.asarray(request["body_xpos"], dtype=np.float32),
            np.asarray(request["body_xquat"], dtype=np.float32),
            np.asarray(request["body_xmat"], dtype=np.float32),
            np.asarray(request["body_cvel"], dtype=np.float32),
        )
        print("CONTROL " + json.dumps(result), flush=True)


def main() -> None:
    """Run trace validation or the persistent controller protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--validate-trace", type=Path)
    mode.add_argument("--diagnose-reconstruction", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    controller = StandPolicyController(args.checkpoint, args.seed, args.device)
    if args.serve:
        serve(controller)
        return
    if args.output is None:
        parser.error("diagnostic modes require --output")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.diagnose_reconstruction:
        report = diagnose_reconstruction(controller, args.steps)
    else:
        report = validate_trace(controller, args.validate_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
