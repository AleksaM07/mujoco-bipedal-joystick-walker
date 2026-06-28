"""Compare the best policies stored under ``runs/successful``.

The module is intentionally notebook-friendly: ``run_analysis`` performs the
rollouts, exports tidy CSV files, and returns the resulting DataFrames.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import mujoco
import numpy as np
import pandas as pd
from mujoco_playground import locomotion

import barkley_legacy_walking as berkeley_eval
import evaluate as biomechanics_eval
from config import EnvConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "successful"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"

EVAL_MARKER = "eval |"
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


@dataclass(frozen=True)
class Scenario:
    """One fixed joystick command used during policy comparison."""

    name: str
    command_x: float
    command_y: float
    command_yaw: float

    @property
    def command(self) -> np.ndarray:
        """Return the scenario command as a float32 vector."""
        return np.asarray(
            [self.command_x, self.command_y, self.command_yaw],
            dtype=np.float32,
        )


DEFAULT_SCENARIOS = (
    Scenario("stand", 0.0, 0.0, 0.0),
    Scenario("forward_slow", 0.15, 0.0, 0.0),
    Scenario("forward", 0.35, 0.0, 0.0),
    Scenario("lateral", 0.0, 0.20, 0.0),
    Scenario("turn", 0.0, 0.0, 0.35),
    Scenario("diagonal", 0.25, 0.15, 0.25),
)


@dataclass
class LoadedPolicy:
    """Runtime objects and metadata needed for one policy rollout."""

    policy_id: str
    run_name: str
    policy_type: str
    checkpoint_path: Path
    checkpoint_step: int
    training_reward: float | None
    command_profile: str
    env: Any
    policy: Callable
    reset_fn: Callable
    step_fn: Callable


@dataclass(frozen=True)
class ModelMetadata:
    """MuJoCo indices and stable column labels used during extraction."""

    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_qpos_indices: np.ndarray
    actuator_dof_indices: np.ndarray
    joint_names: tuple[str, ...]
    joint_qpos_indices: np.ndarray
    joint_dof_indices: np.ndarray
    torso_body_id: int
    head_body_id: int
    left_foot_body_id: int
    right_foot_body_id: int
    floor_geom_id: int
    left_foot_geom_ids: np.ndarray
    right_foot_geom_ids: np.ndarray


def _safe_name(value: str) -> str:
    """Convert a MuJoCo or metric name into a CSV-safe suffix."""
    normalized = SAFE_NAME_PATTERN.sub("_", value.strip()).strip("_").lower()
    return normalized or "unnamed"


def _parse_scalar(value: str) -> float | str | None:
    """Parse one log value without losing non-numeric diagnostics."""
    stripped = value.strip()
    if stripped.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped


def parse_training_history(run_dir: Path) -> list[dict[str, Any]]:
    """Parse all evaluation rows from a run's ``train.log``."""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if EVAL_MARKER not in line:
            continue
        payload = line.split(EVAL_MARKER, maxsplit=1)[1]
        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "timestamp": line[:23],
        }
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.strip().split("=", maxsplit=1)
            row[_safe_name(key)] = _parse_scalar(value)
        if "step" not in row:
            continue
        row["step"] = int(float(row["step"]))
        rows.append(row)
    return rows


def _checkpoint_dirs(run_dir: Path) -> list[Path]:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return []
    return sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


def _policy_type(run_name: str) -> str:
    return "berkeley" if run_name.startswith("ppo_Berkeley") else "biomechanics"


def _short_policy_id(index: int, run_name: str, run_config: dict[str, Any]) -> str:
    env_config = run_config.get("env", {})
    if _policy_type(run_name) == "berkeley":
        suffix = "berkeley_flat"
    else:
        reference = env_config.get("reference_gait") or "no_ref"
        profile = env_config.get("command_profile") or "standard"
        xml_path = str(env_config.get("xml_path") or "")
        xml_match = re.search(r"trainfast_v(\d+)", xml_path)
        xml_version = f"v{xml_match.group(1)}" if xml_match else "auto_xml"
        suffix = f"{xml_version}_{profile}_{reference}"
    return f"P{index:02d}_{suffix}"


def discover_best_checkpoints(
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the highest-reward available checkpoint from every run.

    Selection only compares rewards inside a run. Reward scales differ across
    environment generations, so training reward is not used for cross-policy
    ranking.
    """
    history_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    for index, run_dir in enumerate(run_dirs, start=1):
        config_path = run_dir / "config.json"
        run_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        run_history = parse_training_history(run_dir)
        history_rows.extend(run_history)

        checkpoints = _checkpoint_dirs(run_dir)
        if not checkpoints:
            continue
        checkpoint_by_step = {int(path.name): path for path in checkpoints}
        eligible_history = [
            row
            for row in run_history
            if row.get("reward") is not None
            and np.isfinite(float(row["reward"]))
            and int(row["step"]) in checkpoint_by_step
        ]
        if eligible_history:
            best_eval = max(eligible_history, key=lambda row: float(row["reward"]))
            checkpoint_step = int(best_eval["step"])
            training_reward = float(best_eval["reward"])
            selection_reason = "max_logged_eval_reward"
        else:
            checkpoint_step = max(checkpoint_by_step)
            training_reward = None
            selection_reason = "latest_checkpoint_fallback"

        env_config = run_config.get("env", {})
        selected_rows.append(
            {
                "policy_id": _short_policy_id(index, run_dir.name, run_config),
                "run_name": run_dir.name,
                "policy_type": _policy_type(run_dir.name),
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": str(checkpoint_by_step[checkpoint_step]),
                "training_reward": training_reward,
                "selection_reason": selection_reason,
                "env_version": env_config.get("env_version", "standard"),
                "command_profile": env_config.get("command_profile", "standard"),
                "reference_gait": env_config.get("reference_gait", "none"),
                "xml_path_saved": env_config.get("xml_path"),
                "legacy_action_prior_saved": env_config.get(
                    "legacy_action_prior"
                ),
            }
        )

    selected = pd.DataFrame(selected_rows).sort_values("policy_id")
    history = pd.DataFrame(history_rows)
    if not history.empty:
        history = history.sort_values(["run_name", "step"])
    return selected.reset_index(drop=True), history.reset_index(drop=True)


def _load_biomechanics_policy(row: pd.Series) -> LoadedPolicy:
    checkpoint_path = Path(row["checkpoint_path"])
    run_config = biomechanics_eval.find_run_config(checkpoint_path)
    checkpoint_observation_size = (
        biomechanics_eval.read_checkpoint_observation_size(checkpoint_path)
    )
    checkpoint_action_size = biomechanics_eval.read_checkpoint_action_size(
        checkpoint_path
    )
    policy_observation_dict = biomechanics_eval.checkpoint_uses_dict_observation(
        checkpoint_path
    )

    env_version = biomechanics_eval.run_env_value(
        run_config,
        "env_version",
        "standard",
    )
    playground_impl = biomechanics_eval.run_env_value(
        run_config,
        "playground_impl",
        "jax",
    )
    command_profile = biomechanics_eval.infer_command_profile(
        checkpoint_path,
        run_config,
    )
    reference_gait = biomechanics_eval.infer_reference_gait(run_config)
    reference_gait_file = biomechanics_eval.run_env_value(
        run_config,
        "reference_gait_file",
    )
    reference_target_observation = bool(
        biomechanics_eval.run_env_value(
            run_config,
            "reference_target_observation",
            False,
        )
    )
    saved_legacy = biomechanics_eval.run_env_value(
        run_config,
        "legacy_action_prior",
        None,
    )
    legacy_action_prior = True if saved_legacy is None else bool(saved_legacy)

    env_config = EnvConfig(
        env_version=env_version,
        playground_impl=playground_impl,
        command_profile=command_profile,
        reference_gait=reference_gait,
        reference_gait_file=reference_gait_file,
        reference_target_observation=reference_target_observation,
        policy_observation_size=checkpoint_observation_size,
        policy_observation_dict=policy_observation_dict,
        xml_path=biomechanics_eval.infer_xml_path(checkpoint_path, run_config),
        legacy_action_prior=legacy_action_prior,
        action_smoothing=float(
            biomechanics_eval.run_env_value(
                run_config,
                "action_smoothing",
                0.5,
            )
        ),
        init_qpos_file=biomechanics_eval.run_env_value(
            run_config,
            "init_qpos_file",
        ),
        accurate_physics=bool(
            biomechanics_eval.run_env_value(
                run_config,
                "accurate_physics",
                True,
            )
        ),
    )
    env = biomechanics_eval.make_environment(env_config)
    biomechanics_eval.validate_action_compatibility(
        checkpoint_path,
        env.action_size,
    )
    policy = biomechanics_eval.load_ppo_policy(
        checkpoint_path,
        deterministic=True,
    )
    return LoadedPolicy(
        policy_id=row["policy_id"],
        run_name=row["run_name"],
        policy_type="biomechanics",
        checkpoint_path=checkpoint_path,
        checkpoint_step=int(row["checkpoint_step"]),
        training_reward=_optional_float(row.get("training_reward")),
        command_profile=command_profile,
        env=env,
        policy=policy,
        reset_fn=biomechanics_eval.reset_state,
        step_fn=biomechanics_eval.simulation_step,
    )


def _load_berkeley_policy(row: pd.Series) -> LoadedPolicy:
    checkpoint_path = Path(row["checkpoint_path"])
    run_config = berkeley_eval.find_run_config(checkpoint_path) or {}
    saved_env = run_config.get("env", {})
    env_version = saved_env.get("env_version", "standard")
    playground_impl = saved_env.get("playground_impl", "jax")
    env_name = (
        saved_env.get("playground_flat_env", "BerkeleyHumanoidJoystickFlatTerrain")
        if env_version == "standard"
        else saved_env.get(
            "playground_hardcore_env",
            "BerkeleyHumanoidJoystickRoughTerrain",
        )
    )
    berkeley_eval.patch_jax_for_brax_compatibility()
    env = locomotion.load(
        env_name,
        config_overrides={"impl": playground_impl},
    )
    policy = berkeley_eval.load_ppo_policy(
        checkpoint_path,
        deterministic=True,
    )
    return LoadedPolicy(
        policy_id=row["policy_id"],
        run_name=row["run_name"],
        policy_type="berkeley",
        checkpoint_path=checkpoint_path,
        checkpoint_step=int(row["checkpoint_step"]),
        training_reward=_optional_float(row.get("training_reward")),
        command_profile="standard",
        env=env,
        policy=policy,
        reset_fn=berkeley_eval.reset_legacy_state,
        step_fn=berkeley_eval.legacy_simulation_step,
    )


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_policy(row: pd.Series) -> LoadedPolicy:
    """Load one selected policy and its matching historical environment."""
    if row["policy_type"] == "berkeley":
        return _load_berkeley_policy(row)
    return _load_biomechanics_policy(row)


def _object_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name or f"id_{object_id}"


def _find_named_id(
    names: tuple[str, ...],
    exact_candidates: tuple[str, ...],
    token_candidates: tuple[tuple[str, ...], ...],
) -> int:
    lowered = tuple(name.lower() for name in names)
    for candidate in exact_candidates:
        if candidate.lower() in lowered:
            return lowered.index(candidate.lower())
    for tokens in token_candidates:
        for index, name in enumerate(lowered):
            if all(token in name for token in tokens):
                return index
    return -1


def build_model_metadata(model: mujoco.MjModel) -> ModelMetadata:
    """Build reusable MuJoCo index mappings for wide trajectory extraction."""
    body_names = tuple(
        _object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    )
    geom_names = tuple(
        _object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
    )

    torso_body_id = _find_named_id(
        body_names,
        ("thorax", "torso"),
        (("torso",), ("trunk",), ("chest",)),
    )
    head_body_id = _find_named_id(body_names, ("head",), (("head",),))
    left_foot_body_id = _find_named_id(
        body_names,
        ("left_foot", "foot_left"),
        (("left", "foot"), ("left", "ankle")),
    )
    right_foot_body_id = _find_named_id(
        body_names,
        ("right_foot", "foot_right"),
        (("right", "foot"), ("right", "ankle")),
    )
    floor_geom_id = _find_named_id(
        geom_names,
        ("floor", "ground"),
        (("floor",), ("ground",), ("terrain",)),
    )

    def foot_geom_ids(side: str, foot_body_id: int) -> np.ndarray:
        ids = [
            geom_id
            for geom_id, name in enumerate(geom_names)
            if side in name.lower()
            and any(token in name.lower() for token in ("foot", "ankle", "toe"))
        ]
        if foot_body_id >= 0:
            ids.extend(
                geom_id
                for geom_id, body_id in enumerate(model.geom_bodyid)
                if int(body_id) == foot_body_id
            )
        return np.asarray(sorted(set(ids)), dtype=np.int32)

    actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    actuator_joint_names = tuple(
        _object_name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in actuator_joint_ids
    )
    actuator_names = tuple(
        _object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    actuator_qpos_indices = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in actuator_joint_ids],
        dtype=np.int32,
    )
    actuator_dof_indices = np.asarray(
        [model.jnt_dofadr[joint_id] for joint_id in actuator_joint_ids],
        dtype=np.int32,
    )

    joint_names: list[str] = []
    joint_qpos_indices: list[int] = []
    joint_dof_indices: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        joint_names.append(
            _object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        )
        joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_indices.append(int(model.jnt_dofadr[joint_id]))

    return ModelMetadata(
        actuator_names=actuator_names,
        actuator_joint_names=actuator_joint_names,
        actuator_qpos_indices=actuator_qpos_indices,
        actuator_dof_indices=actuator_dof_indices,
        joint_names=tuple(joint_names),
        joint_qpos_indices=np.asarray(joint_qpos_indices, dtype=np.int32),
        joint_dof_indices=np.asarray(joint_dof_indices, dtype=np.int32),
        torso_body_id=torso_body_id,
        head_body_id=head_body_id,
        left_foot_body_id=left_foot_body_id,
        right_foot_body_id=right_foot_body_id,
        floor_geom_id=floor_geom_id,
        left_foot_geom_ids=foot_geom_ids("left", left_foot_body_id),
        right_foot_geom_ids=foot_geom_ids("right", right_foot_body_id),
    )


def _quat_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Convert a MuJoCo ``wxyz`` quaternion to roll, pitch, and yaw."""
    w, x, y, z = quaternion
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(sin_pitch))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def _body_position(data: Any, body_id: int) -> np.ndarray:
    if body_id < 0:
        return np.full(3, np.nan)
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def _body_up(data: Any, body_id: int, axis: int) -> float:
    if body_id < 0:
        return float("nan")
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return float(rotation[2, axis])


def _actual_foot_contact(
    data: Any,
    foot_geom_ids: np.ndarray,
    floor_geom_id: int,
) -> float:
    if floor_geom_id < 0 or foot_geom_ids.size == 0:
        return float("nan")
    contact_geom = np.asarray(data.contact.geom)
    contact_dist = np.asarray(data.contact.dist)
    if contact_geom.size == 0:
        return 0.0
    geom_a = contact_geom[:, 0]
    geom_b = contact_geom[:, 1]
    foot_a = np.isin(geom_a, foot_geom_ids) & (geom_b == floor_geom_id)
    foot_b = np.isin(geom_b, foot_geom_ids) & (geom_a == floor_geom_id)
    return float(np.any((foot_a | foot_b) & (contact_dist <= 0.01)))


def _scalar_metrics(prefix: str, values: Any) -> dict[str, float]:
    """Flatten scalar JAX values from state.metrics/state.info."""
    result: dict[str, float] = {}
    if not isinstance(values, dict):
        return result
    for key, value in values.items():
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.size != 1 or not np.issubdtype(array.dtype, np.number):
            continue
        result[f"{prefix}_{_safe_name(str(key))}"] = float(array.reshape(-1)[0])
    return result


def _scenario_is_in_distribution(command_profile: str, scenario: Scenario) -> bool:
    if command_profile in {"forward", "forward_slow", "walk"}:
        return (
            scenario.command_x >= 0.0
            and scenario.command_y == 0.0
            and scenario.command_yaw == 0.0
        )
    return True


def extract_step_row(
    loaded: LoadedPolicy,
    metadata: ModelMetadata,
    state: Any,
    action: Any,
    scenario: Scenario,
    seed: int,
    trial_id: str,
    episode_id: int,
    episode_step: int,
    scenario_step: int,
    previous_action: np.ndarray | None,
) -> dict[str, Any]:
    """Extract a wide, lossless-enough row from one post-step MJX state."""
    data = state.data
    model = loaded.env.mj_model
    dt = float(loaded.env.dt)
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    action_array = np.asarray(action, dtype=np.float64).reshape(-1)
    ctrl = np.asarray(data.ctrl, dtype=np.float64).reshape(-1)
    qfrc_actuator = np.asarray(data.qfrc_actuator, dtype=np.float64)

    torso_axis = 1 if loaded.policy_type == "biomechanics" else 2
    torso_up = _body_up(data, metadata.torso_body_id, torso_axis)
    head_up = _body_up(data, metadata.head_body_id, torso_axis)
    torso_rotation = (
        np.asarray(data.xmat[metadata.torso_body_id], dtype=np.float64).reshape(3, 3)
        if metadata.torso_body_id >= 0
        else np.eye(3)
    )
    local_linear_velocity = torso_rotation.T @ qvel[:3]
    local_angular_velocity = torso_rotation.T @ qvel[3:6]
    if loaded.policy_type == "biomechanics":
        measured_command = np.asarray(
            [
                local_linear_velocity[0],
                local_linear_velocity[2],
                local_angular_velocity[1],
            ]
        )
    else:
        measured_command = np.asarray(
            [
                local_linear_velocity[0],
                local_linear_velocity[1],
                local_angular_velocity[2],
            ]
        )
    command_error = measured_command - scenario.command

    joint_positions = qpos[metadata.actuator_qpos_indices]
    joint_velocities = qvel[metadata.actuator_dof_indices]
    actuator_torques = qfrc_actuator[metadata.actuator_dof_indices]
    motor_target_error = ctrl - joint_positions
    actuator_power = actuator_torques * joint_velocities
    action_delta = (
        np.zeros_like(action_array)
        if previous_action is None
        else action_array - previous_action
    )
    roll, pitch, yaw = _quat_to_euler(qpos[3:7])
    left_foot_position = _body_position(data, metadata.left_foot_body_id)
    right_foot_position = _body_position(data, metadata.right_foot_body_id)

    row: dict[str, Any] = {
        "policy_id": loaded.policy_id,
        "run_name": loaded.run_name,
        "policy_type": loaded.policy_type,
        "checkpoint_step": loaded.checkpoint_step,
        "checkpoint_path": str(loaded.checkpoint_path),
        "training_reward": loaded.training_reward,
        "command_profile": loaded.command_profile,
        "scenario": scenario.name,
        "scenario_in_distribution": _scenario_is_in_distribution(
            loaded.command_profile,
            scenario,
        ),
        "seed": seed,
        "trial_id": trial_id,
        "episode_id": episode_id,
        "episode_step": episode_step,
        "scenario_step": scenario_step,
        "time_s": (scenario_step + 1) * dt,
        "episode_time_s": (episode_step + 1) * dt,
        "dt": dt,
        "reward": float(np.asarray(state.reward)),
        "done": int(bool(np.asarray(state.done))),
        "command_x": scenario.command_x,
        "command_y": scenario.command_y,
        "command_yaw": scenario.command_yaw,
        "measured_command_x": measured_command[0],
        "measured_command_y": measured_command[1],
        "measured_command_yaw": measured_command[2],
        "command_error_x": command_error[0],
        "command_error_y": command_error[1],
        "command_error_yaw": command_error[2],
        "command_error_norm": float(np.linalg.norm(command_error)),
        "command_failure": int(np.linalg.norm(command_error) > 0.25),
        "root_x": qpos[0],
        "root_y": qpos[1],
        "root_z": qpos[2],
        "root_quat_w": qpos[3],
        "root_quat_x": qpos[4],
        "root_quat_y": qpos[5],
        "root_quat_z": qpos[6],
        "root_roll": roll,
        "root_pitch": pitch,
        "root_yaw": yaw,
        "root_vx": qvel[0],
        "root_vy": qvel[1],
        "root_vz": qvel[2],
        "root_wx": qvel[3],
        "root_wy": qvel[4],
        "root_wz": qvel[5],
        "local_vx": local_linear_velocity[0],
        "local_vy": local_linear_velocity[1],
        "local_vz": local_linear_velocity[2],
        "local_wx": local_angular_velocity[0],
        "local_wy": local_angular_velocity[1],
        "local_wz": local_angular_velocity[2],
        "horizontal_speed": float(np.linalg.norm(qvel[:2])),
        "speed_3d": float(np.linalg.norm(qvel[:3])),
        "angular_speed": float(np.linalg.norm(qvel[3:6])),
        "torso_up": torso_up,
        "head_up": head_up,
        "tilt_angle_rad": math.acos(float(np.clip(torso_up, -1.0, 1.0))),
        "left_foot_x": left_foot_position[0],
        "left_foot_y": left_foot_position[1],
        "left_foot_z": left_foot_position[2],
        "right_foot_x": right_foot_position[0],
        "right_foot_y": right_foot_position[1],
        "right_foot_z": right_foot_position[2],
        "left_foot_contact": _actual_foot_contact(
            data,
            metadata.left_foot_geom_ids,
            metadata.floor_geom_id,
        ),
        "right_foot_contact": _actual_foot_contact(
            data,
            metadata.right_foot_geom_ids,
            metadata.floor_geom_id,
        ),
        "action_norm": float(np.linalg.norm(action_array)),
        "action_mean_abs": float(np.mean(np.abs(action_array))),
        "action_max_abs": float(np.max(np.abs(action_array))),
        "action_saturation_rate": float(np.mean(np.abs(action_array) >= 0.95)),
        "action_delta_norm": float(np.linalg.norm(action_delta)),
        "action_rate_norm": float(np.linalg.norm(action_delta) / dt),
        "motor_target_error_norm": float(np.linalg.norm(motor_target_error)),
        "actuator_torque_norm": float(np.linalg.norm(actuator_torques)),
        "actuator_effort_sq": float(np.sum(np.square(actuator_torques))),
        "mechanical_power_signed": float(np.sum(actuator_power)),
        "mechanical_power_abs": float(np.sum(np.abs(actuator_power))),
        "external_contact_force_norm": float(
            np.linalg.norm(np.asarray(data.cfrc_ext, dtype=np.float64))
        ),
    }

    if hasattr(data, "subtree_com"):
        center_of_mass = np.asarray(data.subtree_com[0], dtype=np.float64)
        row.update(
            {
                "center_of_mass_x": center_of_mass[0],
                "center_of_mass_y": center_of_mass[1],
                "center_of_mass_z": center_of_mass[2],
            }
        )

    row.update(_scalar_metrics("env_metric", state.metrics))
    row.update(_scalar_metrics("info", state.info))

    for index, value in enumerate(qpos):
        row[f"qpos_{index:02d}"] = value
    for index, value in enumerate(qvel):
        row[f"qvel_{index:02d}"] = value
    for name, qpos_index, dof_index in zip(
        metadata.joint_names,
        metadata.joint_qpos_indices,
        metadata.joint_dof_indices,
        strict=True,
    ):
        suffix = _safe_name(name)
        row[f"joint_pos_{suffix}"] = qpos[qpos_index]
        row[f"joint_vel_{suffix}"] = qvel[dof_index]
    for index, name in enumerate(metadata.actuator_joint_names):
        suffix = _safe_name(name)
        row[f"action_{suffix}"] = action_array[index]
        row[f"motor_target_{suffix}"] = ctrl[index]
        row[f"motor_error_{suffix}"] = motor_target_error[index]
        row[f"torque_{suffix}"] = actuator_torques[index]
        row[f"power_{suffix}"] = actuator_power[index]
    return row


def _add_derived_step_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """Add finite-difference trajectory, acceleration, jerk, and slip metrics."""
    steps = steps.sort_values(
        ["policy_id", "scenario", "seed", "episode_id", "episode_step"]
    ).reset_index(drop=True)
    episode_keys = ["policy_id", "scenario", "seed", "episode_id"]
    grouped = steps.groupby(episode_keys, sort=False)

    for axis in ("x", "y", "z"):
        delta = grouped[f"root_{axis}"].diff().fillna(0.0)
        steps[f"root_delta_{axis}"] = delta
    steps["path_increment_2d"] = np.hypot(
        steps["root_delta_x"],
        steps["root_delta_y"],
    )
    steps["path_increment_3d"] = np.sqrt(
        steps["root_delta_x"] ** 2
        + steps["root_delta_y"] ** 2
        + steps["root_delta_z"] ** 2
    )
    steps["path_length_2d"] = steps.groupby(episode_keys, sort=False)[
        "path_increment_2d"
    ].cumsum()

    for axis in ("x", "y", "z"):
        velocity_column = f"root_v{axis}"
        acceleration_column = f"root_a{axis}"
        acceleration = grouped[velocity_column].diff().fillna(0.0) / steps["dt"]
        steps[acceleration_column] = acceleration
        steps[f"root_jerk_{axis}"] = (
            steps.groupby(episode_keys, sort=False)[acceleration_column]
            .diff()
            .fillna(0.0)
            / steps["dt"]
        )
    steps["acceleration_norm"] = np.sqrt(
        steps["root_ax"] ** 2
        + steps["root_ay"] ** 2
        + steps["root_az"] ** 2
    )
    steps["jerk_norm"] = np.sqrt(
        steps["root_jerk_x"] ** 2
        + steps["root_jerk_y"] ** 2
        + steps["root_jerk_z"] ** 2
    )

    for side in ("left", "right"):
        foot_dx = grouped[f"{side}_foot_x"].diff().fillna(0.0)
        foot_dy = grouped[f"{side}_foot_y"].diff().fillna(0.0)
        foot_speed = np.hypot(foot_dx, foot_dy) / steps["dt"]
        steps[f"{side}_foot_horizontal_speed"] = foot_speed
        contact = steps[f"{side}_foot_contact"].fillna(0.0)
        steps[f"{side}_foot_slip_speed"] = foot_speed * contact
    steps["foot_slip_speed"] = (
        steps["left_foot_slip_speed"] + steps["right_foot_slip_speed"]
    )
    return steps


def rollout_policy(
    loaded: LoadedPolicy,
    scenarios: tuple[Scenario, ...],
    seeds: tuple[int, ...],
    steps_per_scenario: int,
) -> pd.DataFrame:
    """Run deterministic headless trials and return one row per control step."""
    metadata = build_model_metadata(loaded.env.mj_model)
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        command = scenario.command
        if loaded.policy_type == "biomechanics":
            biomechanics_eval.clip_command(command)
        else:
            berkeley_eval.clip_command(command)
        for seed in seeds:
            trial_id = f"{loaded.policy_id}:{scenario.name}:seed{seed}"
            rng = jax.random.PRNGKey(seed)
            state = loaded.reset_fn(loaded.env, rng, command)
            episode_id = 0
            episode_step = 0
            previous_action: np.ndarray | None = None

            for scenario_step in range(steps_per_scenario):
                rng, action_key, reset_key = jax.random.split(rng, 3)
                state, action = loaded.step_fn(
                    loaded.env,
                    loaded.policy,
                    state,
                    action_key,
                    command,
                )
                jax.block_until_ready(action)
                rows.append(
                    extract_step_row(
                        loaded=loaded,
                        metadata=metadata,
                        state=state,
                        action=action,
                        scenario=scenario,
                        seed=seed,
                        trial_id=trial_id,
                        episode_id=episode_id,
                        episode_step=episode_step,
                        scenario_step=scenario_step,
                        previous_action=previous_action,
                    )
                )
                previous_action = np.asarray(action, dtype=np.float64).reshape(-1)
                episode_step += 1

                if bool(np.asarray(state.done)):
                    episode_id += 1
                    episode_step = 0
                    previous_action = None
                    state = loaded.reset_fn(loaded.env, reset_key, command)

    return _add_derived_step_metrics(pd.DataFrame(rows))


def build_episode_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate step-level data into episode-level locomotion metrics."""
    keys = [
        "policy_id",
        "run_name",
        "policy_type",
        "checkpoint_step",
        "scenario",
        "scenario_in_distribution",
        "seed",
        "trial_id",
        "episode_id",
    ]

    def summarize(group: pd.DataFrame) -> pd.Series:
        dx = group["root_x"].iloc[-1] - group["root_x"].iloc[0]
        dy = group["root_y"].iloc[-1] - group["root_y"].iloc[0]
        displacement = float(np.hypot(dx, dy))
        path_length = float(group["path_increment_2d"].sum())
        return pd.Series(
            {
                "steps": len(group),
                "duration_s": group["dt"].sum(),
                "terminated": int(group["done"].max()),
                "total_reward": group["reward"].sum(),
                "mean_reward": group["reward"].mean(),
                "reward_std": group["reward"].std(ddof=0),
                "displacement_x": dx,
                "displacement_y": dy,
                "displacement_2d": displacement,
                "path_length_2d": path_length,
                "trajectory_straightness": displacement / max(path_length, 1e-9),
                "mean_horizontal_speed": group["horizontal_speed"].mean(),
                "max_horizontal_speed": group["horizontal_speed"].max(),
                "tracking_rmse": np.sqrt(
                    np.mean(np.square(group["command_error_norm"]))
                ),
                "tracking_mae_x": group["command_error_x"].abs().mean(),
                "tracking_mae_y": group["command_error_y"].abs().mean(),
                "tracking_mae_yaw": group["command_error_yaw"].abs().mean(),
                "command_failure_rate": group["command_failure"].mean(),
                "mean_root_height": group["root_z"].mean(),
                "min_root_height": group["root_z"].min(),
                "root_height_std": group["root_z"].std(ddof=0),
                "mean_torso_up": group["torso_up"].mean(),
                "min_torso_up": group["torso_up"].min(),
                "mean_tilt_angle_rad": group["tilt_angle_rad"].mean(),
                "mean_action_norm": group["action_norm"].mean(),
                "mean_action_rate_norm": group["action_rate_norm"].mean(),
                "mean_action_saturation_rate": group[
                    "action_saturation_rate"
                ].mean(),
                "mean_motor_target_error": group[
                    "motor_target_error_norm"
                ].mean(),
                "mean_actuator_torque_norm": group[
                    "actuator_torque_norm"
                ].mean(),
                "mean_mechanical_power_abs": group[
                    "mechanical_power_abs"
                ].mean(),
                "mean_acceleration_norm": group["acceleration_norm"].mean(),
                "mean_jerk_norm": group["jerk_norm"].mean(),
                "mean_foot_slip_speed": group["foot_slip_speed"].mean(),
                "external_contact_force_rms": np.sqrt(
                    np.mean(np.square(group["external_contact_force_norm"]))
                ),
            }
        )

    return (
        steps.groupby(keys, dropna=False, sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )


def build_trial_metrics(
    steps: pd.DataFrame,
    steps_per_scenario: int,
) -> pd.DataFrame:
    """Aggregate complete fixed-length trials, including repeated falls."""
    keys = [
        "policy_id",
        "run_name",
        "policy_type",
        "checkpoint_step",
        "scenario",
        "scenario_in_distribution",
        "seed",
        "trial_id",
    ]

    def summarize(group: pd.DataFrame) -> pd.Series:
        done_steps = group.loc[group["done"] == 1, "scenario_step"]
        first_fall_step = (
            int(done_steps.iloc[0]) + 1 if not done_steps.empty else steps_per_scenario
        )
        return pd.Series(
            {
                "steps": len(group),
                "falls": int(group["done"].sum()),
                "first_fall_step": first_fall_step,
                "first_fall_time_s": first_fall_step * group["dt"].iloc[0],
                "survival_fraction": first_fall_step / steps_per_scenario,
                "total_reward": group["reward"].sum(),
                "mean_reward": group["reward"].mean(),
                "tracking_rmse": np.sqrt(
                    np.mean(np.square(group["command_error_norm"]))
                ),
                "command_failure_rate": group["command_failure"].mean(),
                "mean_torso_up": group["torso_up"].mean(),
                "min_torso_up": group["torso_up"].min(),
                "mean_root_height": group["root_z"].mean(),
                "min_root_height": group["root_z"].min(),
                "mean_horizontal_speed": group["horizontal_speed"].mean(),
                "mean_action_rate_norm": group["action_rate_norm"].mean(),
                "mean_mechanical_power_abs": group[
                    "mechanical_power_abs"
                ].mean(),
                "mean_foot_slip_speed": group["foot_slip_speed"].mean(),
            }
        )

    return (
        steps.groupby(keys, dropna=False, sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )


def _policy_summary_group(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "steps": len(group),
            "trials": group["trial_id"].nunique(),
            "falls": int(group["done"].sum()),
            "fall_rate_per_1000_steps": 1000.0 * group["done"].sum() / len(group),
            "mean_reward": group["reward"].mean(),
            "reward_std": group["reward"].std(ddof=0),
            "tracking_rmse": np.sqrt(
                np.mean(np.square(group["command_error_norm"]))
            ),
            "tracking_failure_rate": group["command_failure"].mean(),
            "mean_horizontal_speed": group["horizontal_speed"].mean(),
            "mean_root_height": group["root_z"].mean(),
            "min_root_height": group["root_z"].min(),
            "mean_torso_up": group["torso_up"].mean(),
            "min_torso_up": group["torso_up"].min(),
            "mean_tilt_angle_rad": group["tilt_angle_rad"].mean(),
            "mean_action_norm": group["action_norm"].mean(),
            "mean_action_rate_norm": group["action_rate_norm"].mean(),
            "mean_action_saturation_rate": group["action_saturation_rate"].mean(),
            "mean_motor_target_error": group["motor_target_error_norm"].mean(),
            "mean_actuator_torque_norm": group["actuator_torque_norm"].mean(),
            "mean_mechanical_power_abs": group["mechanical_power_abs"].mean(),
            "mean_acceleration_norm": group["acceleration_norm"].mean(),
            "mean_jerk_norm": group["jerk_norm"].mean(),
            "mean_foot_slip_speed": group["foot_slip_speed"].mean(),
            "external_contact_force_rms": np.sqrt(
                np.mean(np.square(group["external_contact_force_norm"]))
            ),
        }
    )


def build_policy_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """Build per-scenario plus fair in-distribution policy comparisons."""
    identity = ["policy_id", "run_name", "policy_type", "checkpoint_step"]
    scenario_rows = (
        steps.groupby(identity + ["scenario"], sort=False)
        .apply(_policy_summary_group, include_groups=False)
        .reset_index()
    )

    in_distribution = steps.loc[steps["scenario_in_distribution"]].copy()
    in_distribution["scenario"] = "IN_DISTRIBUTION"
    in_distribution_rows = (
        in_distribution.groupby(identity + ["scenario"], sort=False)
        .apply(_policy_summary_group, include_groups=False)
        .reset_index()
    )
    policy_metrics = pd.concat(
        [scenario_rows, in_distribution_rows],
        ignore_index=True,
    )

    trial_survival = (
        in_distribution.groupby(identity + ["trial_id"], sort=False)
        .apply(
            lambda group: 1.0
            if not group["done"].any()
            else (group.loc[group["done"] == 1, "scenario_step"].iloc[0] + 1)
            / len(group),
            include_groups=False,
        )
        .groupby(identity)
        .mean()
        .rename("mean_first_fall_survival_fraction")
        .reset_index()
    )
    policy_metrics = policy_metrics.merge(trial_survival, on=identity, how="left")
    policy_metrics = _add_composite_score(policy_metrics)
    return policy_metrics.sort_values(
        ["scenario", "composite_score"],
        ascending=[True, False],
    )


def _min_max_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    finite = series.replace([np.inf, -np.inf], np.nan)
    minimum = finite.min()
    maximum = finite.max()
    if pd.isna(minimum) or pd.isna(maximum) or math.isclose(minimum, maximum):
        score = pd.Series(1.0, index=series.index)
    else:
        score = (finite - minimum) / (maximum - minimum)
    return score if higher_is_better else 1.0 - score


def _add_composite_score(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent multi-objective score within each scenario."""
    scored_groups: list[pd.DataFrame] = []
    for _, group in policy_metrics.groupby("scenario", sort=False):
        group = group.copy()
        group["score_stability"] = _min_max_score(
            group["mean_first_fall_survival_fraction"],
            higher_is_better=True,
        )
        group["score_tracking"] = _min_max_score(
            group["tracking_rmse"],
            higher_is_better=False,
        )
        group["score_upright"] = _min_max_score(
            group["mean_torso_up"],
            higher_is_better=True,
        )
        group["score_smoothness"] = _min_max_score(
            group["mean_action_rate_norm"],
            higher_is_better=False,
        )
        group["composite_score"] = (
            0.40 * group["score_stability"]
            + 0.30 * group["score_tracking"]
            + 0.20 * group["score_upright"]
            + 0.10 * group["score_smoothness"]
        )
        scored_groups.append(group)
    return pd.concat(scored_groups, ignore_index=True)


def build_actuator_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """Create a long-form actuator comparison table from wide step columns."""
    action_columns = [column for column in steps if column.startswith("action_")]
    excluded = {
        "action_norm",
        "action_mean_abs",
        "action_max_abs",
        "action_saturation_rate",
        "action_delta_norm",
        "action_rate_norm",
    }
    action_columns = [column for column in action_columns if column not in excluded]
    identity = ["policy_id", "run_name", "scenario"]
    rows: list[dict[str, Any]] = []
    for action_column in action_columns:
        actuator = action_column.removeprefix("action_")
        motor_error_column = f"motor_error_{actuator}"
        torque_column = f"torque_{actuator}"
        power_column = f"power_{actuator}"
        if motor_error_column not in steps:
            continue
        for keys, group in steps.groupby(identity, sort=False):
            rows.append(
                {
                    "policy_id": keys[0],
                    "run_name": keys[1],
                    "scenario": keys[2],
                    "actuator": actuator,
                    "mean_action": group[action_column].mean(),
                    "mean_abs_action": group[action_column].abs().mean(),
                    "std_action": group[action_column].std(ddof=0),
                    "saturation_rate": (group[action_column].abs() >= 0.95).mean(),
                    "motor_error_rmse": np.sqrt(
                        np.mean(np.square(group[motor_error_column]))
                    ),
                    "torque_rms": np.sqrt(
                        np.mean(np.square(group[torque_column]))
                    ),
                    "mean_abs_power": group[power_column].abs().mean(),
                }
            )
    return pd.DataFrame(rows)


def export_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Write every analysis table as a UTF-8 CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def load_exported_tables(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Load previously exported analysis CSVs."""
    table_names = (
        "selected_checkpoints",
        "training_history",
        "rollout_steps",
        "episode_metrics",
        "trial_metrics",
        "policy_metrics",
        "actuator_metrics",
    )
    return {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in table_names
        if (output_dir / f"{name}.csv").exists()
    }


def run_analysis(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    seeds: tuple[int, ...] = (7, 19),
    steps_per_scenario: int = 500,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run or load the complete best-checkpoint walking analysis."""
    expected_csv = output_dir / "policy_metrics.csv"
    if expected_csv.exists() and not force:
        print(f"Loading cached analysis from {output_dir}")
        return load_exported_tables(output_dir)

    selected, history = discover_best_checkpoints(runs_dir)
    all_steps: list[pd.DataFrame] = []
    for index, row in selected.iterrows():
        print(
            f"[{index + 1}/{len(selected)}] loading {row['policy_id']} "
            f"checkpoint={int(row['checkpoint_step'])}"
        )
        loaded = load_policy(row)
        policy_steps = rollout_policy(
            loaded,
            scenarios=scenarios,
            seeds=seeds,
            steps_per_scenario=steps_per_scenario,
        )
        all_steps.append(policy_steps)
        print(
            f"[{index + 1}/{len(selected)}] done {row['policy_id']} "
            f"rows={len(policy_steps)} falls={int(policy_steps['done'].sum())}"
        )

    steps = pd.concat(all_steps, ignore_index=True, sort=False)
    tables = {
        "selected_checkpoints": selected,
        "training_history": history,
        "rollout_steps": steps,
        "episode_metrics": build_episode_metrics(steps),
        "trial_metrics": build_trial_metrics(steps, steps_per_scenario),
        "policy_metrics": build_policy_metrics(steps),
        "actuator_metrics": build_actuator_metrics(steps),
    }
    export_tables(tables, output_dir)
    print(f"Exported {len(tables)} CSV files to {output_dir}")
    return tables
