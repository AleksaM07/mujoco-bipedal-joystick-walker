import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from bvh_reference import load_bvh_references
from biomechanics_model import (
    HumanSpec,
    LEG_ACTUATED_JOINTS,
    TRUNK_ACTUATED_JOINTS,
    build_trainable_scene_xml,
)
from config import PROJECT_ROOT
from mujoco_playground._src import mjx_env


QPOS_NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class BiomechanicsEnvConfig:
    env_version: str = "standard"
    impl: str = "jax"
    ctrl_dt: float = 0.02
    sim_dt: float = 0.005
    episode_length: int = 1000
    action_scale: float = 0.5
    action_smoothing: float = 0.5
    command_profile: str = "standard"
    reference_gait: str = "none"
    reference_gait_file: str | list[str] | None = None
    reference_target_observation: bool = False
    reference_phase_randomization: bool = False
    reference_state_init: bool = False
    xml_path: str | None = None
    legacy_action_prior: bool = False
    command_resample_steps: int = 500
    tracking_sigma: float = 0.25
    action_noise_std: float = 0.03
    episode_bias_std: float = 0.02
    rfi_torque_limit: float = 2.0
    rao_torque_limit: float = 2.0
    enable_erfi: bool = True
    init_qpos_file: str | None = None


def default_config() -> config_dict.ConfigDict:
    """Vraca config kompatibilan sa MuJoCo Playground MjxEnv bazom."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        action_scale=0.5,
        action_smoothing=0.5,
        command_profile="standard",
        reference_gait="none",
        reference_gait_file=None,
        reference_target_observation=False,
        reference_phase_randomization=False,
        reference_state_init=False,
        xml_path=None,
        legacy_action_prior=False,
        command_resample_steps=500,
        tracking_sigma=0.25,
        action_noise_std=0.03,
        episode_bias_std=0.02,
        rfi_torque_limit=2.0,
        rao_torque_limit=2.0,
        enable_erfi=True,
        init_qpos_file=None,
        impl="jax",
    )


def load_qpos_from_mjdata_file(path: str | Path, expected_size: int) -> np.ndarray:
    """Ucita QPOS blok iz MJDATA-style tekst fajla."""
    qpos_path = resolve_project_path(path)
    text = qpos_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    start_index = 0
    for index, line in enumerate(lines):
        if line.strip().upper() == "QPOS":
            start_index = index + 1
            break

    qpos_lines = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if qpos_lines and stripped.isalpha():
            break
        qpos_lines.append(line)

    values = [
        float(value)
        for value in QPOS_NUMBER_PATTERN.findall("\n".join(qpos_lines))
    ]
    if len(values) != expected_size:
        raise ValueError(
            f"{qpos_path} ima {len(values)} QPOS vrednosti, "
            f"a model ocekuje nq={expected_size}."
        )
    return np.asarray(values, dtype=np.float64)


def resolve_project_path(path: str | Path) -> Path:
    """Vrati apsolutnu putanju, uz podrsku za repo-relative fajlove."""
    resolved = Path(path).expanduser()
    if resolved.is_absolute() or resolved.exists():
        return resolved
    return PROJECT_ROOT / resolved


class BiomechanicsJoystickEnv(mjx_env.MjxEnv):
    """Joystick locomotion env za humanoida iz `mujoco-biomechanics`."""

    WORLD_GRAVITY = jp.array([0.0, 0.0, -1.0])
    FOOT_SOLE_GEOMS = ("left_foot_sole", "right_foot_sole")
    ILLEGAL_CONTACT_BODY_NAMES = (
        "left_thigh",
        "right_thigh",
        "left_shank",
        "right_shank",
    )
    FATAL_CONTACT_BODY_NAMES = ("pelvis",)
    FOOT_CONTACT_PRELOAD = 0.005
    FOOT_CONTACT_HEIGHT = 0.095
    FOOT_CONTACT_DISTANCE = 0.01
    ILLEGAL_CONTACT_DISTANCE = 0.01
    FORWARD_SLOW_COMMAND_RANGE = (0.02, 0.12)
    FORWARD_SLOW_ZERO_COMMAND_PROBABILITY = 0.25
    FORWARD_COMMAND_RANGE = (0.15, 0.35)
    STEER_X_COMMAND_RANGE = (0.05, 0.60)
    STEER_Y_COMMAND_RANGE = (-0.20, 0.20)
    STEER_YAW_COMMAND_RANGE = (-0.35, 0.35)
    STEER_ZERO_COMMAND_PROBABILITY = 0.05
    STANDARD_EASY_X_COMMAND_RANGE = (-0.25, 0.65)
    STANDARD_EASY_Y_COMMAND_RANGE = (-0.25, 0.25)
    STANDARD_EASY_YAW_COMMAND_RANGE = (-0.45, 0.45)
    STANDARD_EASY_ZERO_COMMAND_PROBABILITY = 0.10
    STANDARD_X_COMMAND_RANGE = (-0.6, 0.8)
    STANDARD_Y_COMMAND_RANGE = (-0.45, 0.45)
    STANDARD_YAW_COMMAND_RANGE = (-0.8, 0.8)
    ZERO_COMMAND_PROBABILITY = 0.0
    STANDARD_ZERO_COMMAND_PROBABILITY = 0.1
    NEUTRAL_JOINT_POSE = {
        "left_hip_x": 0.18,
        "left_hip_y": 0.0,
        "left_hip_z": 0.03,
        "left_knee_z": -0.15,
        "left_ankle_y": 0.05,
        "left_ankle_z": 0.05,
        "right_hip_x": -0.18,
        "right_hip_y": 0.0,
        "right_hip_z": 0.03,
        "right_knee_z": -0.15,
        "right_ankle_y": -0.05,
        "right_ankle_z": 0.05,
    }
    TRUNK_ACTION_SCALE = {
        "abdomen_x": 0.08,
        "abdomen_y": 0.06,
        "abdomen_z": 0.08,
        "pelvis_x": 0.05,
        "pelvis_y": 0.04,
        "pelvis_z": 0.05,
    }
    LEG_ACTION_SCALE = {
        "left_hip_x": 0.35,
        "right_hip_x": 0.35,
        "left_hip_y": 0.12,
        "right_hip_y": 0.12,
        "left_hip_z": 0.14,
        "right_hip_z": 0.14,
        "left_knee_z": 0.55,
        "right_knee_z": 0.55,
        "left_ankle_y": 0.24,
        "right_ankle_y": 0.24,
        "left_ankle_z": 0.08,
        "right_ankle_z": 0.08,
    }
    POSTURE_STD_STANDING = {
        "trunk": 0.06,
        "hip_stride": 0.08,
        "knee": 0.10,
        "ankle_pitch": 0.07,
        "hip_lateral": 0.06,
        "ankle_lateral": 0.05,
    }
    POSTURE_STD_WALKING = {
        "trunk": 0.09,
        "hip_stride": 0.38,
        "knee": 0.55,
        "ankle_pitch": 0.24,
        "hip_lateral": 0.14,
        "ankle_lateral": 0.08,
    }
    INIT_TRUNK_NOISE = 0.005
    INIT_LEG_NOISE = 0.02
    HEIGHT_PENALTY_START_RATIO = 0.9
    MIN_STANDING_HEIGHT_RATIO = 0.6
    ALIVE_REWARD_SCALE = 0.05
    ACTION_COST_SCALE = 0.01
    ACTION_RATE_COST_SCALE = 0.005
    BASE_HEIGHT_REWARD_SCALE = 0.6
    POSTURE_REWARD_SCALE = 0.02
    VARIABLE_POSTURE_REWARD_SCALE = 0.45
    VARIABLE_POSTURE_COST_SCALE = 0.08
    TRUNK_POSTURE_COST_SCALE = 0.25
    VELOCITY_TRACKING_REWARD_SCALE = 1.5
    FORWARD_PROGRESS_REWARD_SCALE = 1.0
    UPRIGHT_REWARD_SCALE = 0.3
    HEAD_UP_REWARD_SCALE = 0.1
    OVERSPEED_COST_SCALE = 0.75
    OVERSPEED_COST_CAP = 2.0
    VERTICAL_VELOCITY_COST_SCALE = 0.05
    ANGULAR_VELOCITY_COST_SCALE = 0.02
    GAIT_PERIOD_STEPS = 50
    FOOT_CLEARANCE_TARGET = 0.08
    FOOT_CLEARANCE_REWARD_SCALE = 1.1
    STANCE_CONTACT_REWARD_SCALE = 0.3
    FOOT_SLIP_FREE_SPEED = 0.03
    FOOT_SLIP_COST_SCALE = 0.12
    FOOT_SLIP_COST_CAP = 2.0
    SWING_FOOT_DRAG_COST_SCALE = 0.12
    SWING_CONTACT_COST_SCALE = 0.10
    SWING_CLEARANCE_DEFICIT_COST_SCALE = 0.12
    DOUBLE_CONTACT_COST_SCALE = 0.08
    DOUBLE_SUPPORT_DRAG_COST_SCALE = 0.04
    LOCOMOTION_QUALITY_TRACKING_FLOOR = 0.25
    LOCOMOTION_QUALITY_CLEARANCE_WEIGHT = 0.45
    LOCOMOTION_QUALITY_SINGLE_SUPPORT_WEIGHT = 0.55
    REFERENCE_GAIT_REWARD_SCALE = 0.35
    REFERENCE_GAIT_ERROR_SCALE = 8.0
    REFERENCE_VELOCITY_REWARD_SCALE = 0.15
    REFERENCE_VELOCITY_ERROR_SCALE = 0.25
    REFERENCE_FOOT_REWARD_SCALE = 0.45
    REFERENCE_FOOT_ERROR_SCALE = 35.0
    REFERENCE_ROOT_REWARD_SCALE = 0.35
    REFERENCE_ROOT_HEIGHT_ERROR_SCALE = 35.0
    REFERENCE_ROOT_VELOCITY_ERROR_SCALE = 8.0
    CONTACT_FORCE_COST_SCALE = 1e-4
    CONTACT_FORCE_COST_CLIP = 1000.0
    SOFT_JOINT_LIMIT_FACTOR = 0.9
    SOFT_JOINT_LIMIT_COST_SCALE = 2.0
    ILLEGAL_CONTACT_COST_SCALE = 2.0
    STUCK_COMMAND_THRESHOLD = 0.10
    STUCK_VELOCITY_THRESHOLD = 0.05
    STUCK_PENALTY = 1.0
    LOW_HEIGHT_COST_SCALE = 8.0
    REWARD_MIN = 0.0
    REWARD_MAX = 3.0
    FALL_REWARD = -8.0
    BVH_MIMIC_REWARD_SCALE = 2.4
    BVH_MIMIC_POSE_WEIGHT = 0.35
    BVH_MIMIC_VELOCITY_WEIGHT = 0.15
    BVH_MIMIC_FOOT_WEIGHT = 0.25
    BVH_MIMIC_ROOT_WEIGHT = 0.20
    BVH_MIMIC_CONTACT_WEIGHT = 0.05
    BVH_TASK_TRACKING_SCALE = 0.25
    BVH_TASK_PROGRESS_SCALE = 0.20
    BVH_STABILITY_REWARD_SCALE = 0.35
    BVH_STUCK_PENALTY_SCALE = 0.25
    BVH_LOCOMOTION_GATE_FLOOR = 0.2
    BVH_BOOTSTRAP_GAIT_SCALE = 1.1
    BVH_BOOTSTRAP_PROGRESS_SCALE = 0.45
    BVH_REGULARIZATION_COST_CAP = 1.0
    BVH_REG_SWING_DRAG_WEIGHT = 0.08
    BVH_REG_SWING_CONTACT_WEIGHT = 0.08
    BVH_REG_CLEARANCE_DEFICIT_WEIGHT = 0.06
    BVH_REG_DOUBLE_CONTACT_WEIGHT = 0.08
    BVH_REG_DOUBLE_SUPPORT_DRAG_WEIGHT = 0.06

    def __init__(
        self,
        env_version: str = "standard",
        human_spec: HumanSpec = HumanSpec(),
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict | None = None,
    ) -> None:
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)
        configured_xml_path = self._config.get("xml_path", None)
        if configured_xml_path:
            self._xml_path = resolve_project_path(configured_xml_path)
        else:
            self._xml_path = build_trainable_scene_xml(env_version, human_spec)
        self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._mj_model.opt.timestep = self._sim_dt
        init_q = np.array(self._mj_model.keyframe("a-pose").qpos, copy=True)
        init_q = self._build_initial_qpos(init_q)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._init_q_np = np.array(init_q, copy=True)
        self._init_q = jp.array(init_q)
        self._default_qpos = self._init_q[7:]
        self._actuator_qpos_indices_np = np.array([
            self._mj_model.jnt_qposadr[joint_id]
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ], dtype=np.int32)
        self._actuator_qpos_indices = jp.array(self._actuator_qpos_indices_np)
        self._actuator_dof_indices_np = np.array([
            self._mj_model.jnt_dofadr[joint_id]
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ], dtype=np.int32)
        self._actuator_dof_indices = jp.array(self._actuator_dof_indices_np)
        actuator_joint_names = tuple(
            mujoco.mj_id2name(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        )
        expected_joint_names = TRUNK_ACTUATED_JOINTS + LEG_ACTUATED_JOINTS
        if actuator_joint_names != expected_joint_names:
            raise RuntimeError(
                "Neocekivan redosled aktuatora u generated XML-u: "
                f"{actuator_joint_names}"
            )
        self._actuator_joint_names = actuator_joint_names
        self._trunk_actuator_mask = jp.array([
            joint_name in TRUNK_ACTUATED_JOINTS
            for joint_name in actuator_joint_names
        ])
        self._action_scale = jp.array([
            self._joint_action_scale(joint_name)
            for joint_name in actuator_joint_names
        ])
        posture_std_pairs = [
            self._variable_posture_std_pair(joint_name)
            for joint_name in actuator_joint_names
        ]
        self._posture_std_standing = jp.array([
            std_pair[0] for std_pair in posture_std_pairs
        ])
        self._posture_std_walking = jp.array([
            std_pair[1] for std_pair in posture_std_pairs
        ])
        self._init_actuator_noise = jp.array([
            self.INIT_TRUNK_NOISE
            if joint_name in TRUNK_ACTUATED_JOINTS
            else self.INIT_LEG_NOISE
            for joint_name in actuator_joint_names
        ])
        self._torque_injection_scale = jp.array([
            0.2 if joint_name in TRUNK_ACTUATED_JOINTS else 1.0
            for joint_name in actuator_joint_names
        ])
        self._reference_gait_mask = jp.array([
            joint_name
            in {
                "left_hip_x",
                "right_hip_x",
                "left_knee_z",
                "right_knee_z",
                "left_ankle_y",
                "right_ankle_y",
            }
            for joint_name in actuator_joint_names
        ])
        self._reference_gait_sin_offsets = jp.array([
            {
                "left_hip_x": 0.22,
                "right_hip_x": -0.22,
                "left_ankle_y": -0.12,
                "right_ankle_y": 0.12,
            }.get(joint_name, 0.0)
            for joint_name in actuator_joint_names
        ])
        self._reference_gait_pos_sin_offsets = jp.array([
            {"left_knee_z": -0.45}.get(joint_name, 0.0)
            for joint_name in actuator_joint_names
        ])
        self._reference_gait_neg_sin_offsets = jp.array([
            {"right_knee_z": -0.45}.get(joint_name, 0.0)
            for joint_name in actuator_joint_names
        ])
        self._left_knee_actuator_index = actuator_joint_names.index("left_knee_z")
        self._right_knee_actuator_index = actuator_joint_names.index("right_knee_z")
        self._actuator_qpos_lower_limits_np = np.array([
            self._mj_model.jnt_range[joint_id, 0]
            if self._mj_model.jnt_limited[joint_id]
            else -np.inf
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ])
        self._actuator_qpos_upper_limits_np = np.array([
            self._mj_model.jnt_range[joint_id, 1]
            if self._mj_model.jnt_limited[joint_id]
            else np.inf
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ])
        self._actuator_qpos_lower_limits = jp.array(
            self._actuator_qpos_lower_limits_np
        )
        self._actuator_qpos_upper_limits = jp.array(
            self._actuator_qpos_upper_limits_np
        )
        actuator_limit_mid = 0.5 * (
            self._actuator_qpos_lower_limits_np
            + self._actuator_qpos_upper_limits_np
        )
        actuator_limit_half_width = 0.5 * (
            self._actuator_qpos_upper_limits_np
            - self._actuator_qpos_lower_limits_np
        )
        soft_half_width = actuator_limit_half_width * self.SOFT_JOINT_LIMIT_FACTOR
        self._soft_actuator_qpos_lower_limits = jp.array(
            actuator_limit_mid - soft_half_width
        )
        self._soft_actuator_qpos_upper_limits = jp.array(
            actuator_limit_mid + soft_half_width
        )
        self._default_ctrl_np = self._init_q_np[self._actuator_qpos_indices_np]
        self._default_ctrl = jp.array(self._default_ctrl_np)
        self._torso_body_id = self._mj_model.body("thorax").id
        self._head_body_id = self._mj_model.body("head").id
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._left_foot_sole_geom_id = self._mj_model.geom("left_foot_sole").id
        self._right_foot_sole_geom_id = self._mj_model.geom("right_foot_sole").id
        self._foot_geom_ids = jp.array([
            self._left_foot_sole_geom_id,
            self._right_foot_sole_geom_id,
        ])
        self._illegal_contact_geom_ids = jp.array(
            self._geom_ids_for_body_names(self.ILLEGAL_CONTACT_BODY_NAMES),
            dtype=jp.int32,
        )
        self._fatal_contact_geom_ids = jp.array(
            self._geom_ids_for_body_names(self.FATAL_CONTACT_BODY_NAMES),
            dtype=jp.int32,
        )
        self._bvh_reference_qpos_targets = jp.expand_dims(
            jp.expand_dims(self._default_ctrl, axis=0),
            axis=0,
        )
        self._bvh_reference_qvel_targets = jp.zeros_like(
            self._bvh_reference_qpos_targets
        )
        self._bvh_reference_frame_times = jp.array([self.dt], dtype=jp.float32)
        self._bvh_reference_frame_counts = jp.array([1], dtype=jp.int32)
        self._bvh_reference_root_height_offsets = jp.zeros((1, 1), dtype=jp.float32)
        self._bvh_reference_root_velocity_factors = jp.ones(
            (1, 1),
            dtype=jp.float32,
        )
        default_foot_targets = self._build_reference_foot_position_targets(
            self._default_ctrl_np[None, None, :].astype(np.float32),
        )
        self._bvh_reference_foot_pos_targets = jp.array(default_foot_targets)
        (
            default_init_qpos,
            default_init_qvel,
        ) = self._build_reference_init_state_targets(
            self._default_ctrl_np[None, None, :].astype(np.float32),
            np.zeros((1, 1, self.action_size), dtype=np.float32),
        )
        self._bvh_reference_init_qpos_targets = jp.array(default_init_qpos)
        self._bvh_reference_init_qvel_targets = jp.array(default_init_qvel)
        self._bvh_reference_clip_count = 1
        self._configure_bvh_reference()
        self._n_substeps = int(round(self._ctrl_dt / self._sim_dt))

    def _geom_ids_for_body_names(self, body_names: tuple[str, ...]) -> np.ndarray:
        """Vrati sve geom id-jeve koji pripadaju zadatim body imenima."""
        body_ids = []
        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if body_id >= 0:
                body_ids.append(body_id)

        return np.asarray(
            [
                geom_id
                for geom_id, body_id in enumerate(self._mj_model.geom_bodyid)
                if int(body_id) in body_ids
            ],
            dtype=np.int32,
        )

    def _joint_action_scale(self, joint_name: str) -> float:
        """Unitree-style action prior: stride joints move more than twist joints."""
        global_scale = float(self._config.action_scale) / 0.5
        if joint_name in self.TRUNK_ACTION_SCALE:
            return self.TRUNK_ACTION_SCALE[joint_name] * global_scale
        if self._config.get("legacy_action_prior", False):
            return float(self._config.action_scale)
        if joint_name in self.LEG_ACTION_SCALE:
            return self.LEG_ACTION_SCALE[joint_name] * global_scale
        return float(self._config.action_scale)

    @classmethod
    def _variable_posture_std_pair(cls, joint_name: str) -> tuple[float, float]:
        """Vrati standing/walking toleranciju za promenljivi posture prior."""
        if joint_name in TRUNK_ACTUATED_JOINTS:
            category = "trunk"
        elif joint_name in {"left_hip_x", "right_hip_x"}:
            category = "hip_stride"
        elif joint_name in {"left_knee_z", "right_knee_z"}:
            category = "knee"
        elif joint_name in {"left_ankle_y", "right_ankle_y"}:
            category = "ankle_pitch"
        elif joint_name in {"left_ankle_z", "right_ankle_z"}:
            category = "ankle_lateral"
        else:
            category = "hip_lateral"
        return (
            cls.POSTURE_STD_STANDING[category],
            cls.POSTURE_STD_WALKING[category],
        )

    def _configure_bvh_reference(self) -> None:
        """Ucita BVH referencu ako je trazena u config-u."""
        if self._config.get("reference_gait", "none") != "bvh":
            return
        reference_gait_files = self._reference_gait_files()
        if not reference_gait_files:
            raise ValueError(
                "reference_gait='bvh' trazi bar jedan --reference-gait-file."
            )

        references = load_bvh_references(
            tuple(resolve_project_path(path) for path in reference_gait_files),
            self._actuator_joint_names,
            np.asarray(self._default_ctrl, dtype=np.float32),
            self._actuator_qpos_lower_limits_np,
            self._actuator_qpos_upper_limits_np,
        )
        self._bvh_reference_qpos_targets = jp.array(references.qpos_targets)
        self._bvh_reference_qvel_targets = jp.array(references.qvel_targets)
        self._bvh_reference_frame_times = jp.array(references.frame_times)
        self._bvh_reference_frame_counts = jp.array(references.frame_counts)
        self._bvh_reference_root_height_offsets = jp.array(
            references.root_height_offsets
        )
        self._bvh_reference_root_velocity_factors = jp.array(
            references.root_forward_velocity_factors
        )
        self._bvh_reference_foot_pos_targets = jp.array(
            self._build_reference_foot_position_targets(references.qpos_targets)
        )
        init_qpos, init_qvel = self._build_reference_init_state_targets(
            references.qpos_targets,
            references.qvel_targets,
        )
        self._bvh_reference_init_qpos_targets = jp.array(init_qpos)
        self._bvh_reference_init_qvel_targets = jp.array(init_qvel)
        self._bvh_reference_clip_count = len(references.source_paths)

    def _build_reference_foot_position_targets(
        self,
        qpos_targets: np.ndarray,
    ) -> np.ndarray:
        """Precompute trunk-relative foot targets from reference joint poses."""
        foot_targets = np.zeros(
            (*qpos_targets.shape[:2], 2, 3),
            dtype=np.float32,
        )
        data = mujoco.MjData(self._mj_model)
        qpos = np.array(self._init_q_np, copy=True)
        qvel = np.zeros(self._mj_model.nv, dtype=np.float64)
        foot_geom_ids = np.array(
            [self._left_foot_sole_geom_id, self._right_foot_sole_geom_id],
            dtype=np.int32,
        )
        for clip_index in range(qpos_targets.shape[0]):
            for frame_index in range(qpos_targets.shape[1]):
                qpos[:] = self._init_q_np
                qpos[self._actuator_qpos_indices_np] = qpos_targets[
                    clip_index,
                    frame_index,
                ]
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(self._mj_model, data)
                foot_targets[clip_index, frame_index] = (
                    self._relative_foot_positions_np(data, foot_geom_ids)
                )
        return foot_targets

    def _relative_foot_positions_np(
        self,
        data: mujoco.MjData,
        foot_geom_ids: np.ndarray,
    ) -> np.ndarray:
        """Vrati pozicije stopala u lokalnom frame-u toraksa."""
        torso_pos = data.xpos[self._torso_body_id]
        torso_xmat = data.xmat[self._torso_body_id].reshape((3, 3))
        foot_positions = data.geom_xpos[foot_geom_ids]
        return (foot_positions - torso_pos) @ torso_xmat

    def _build_reference_init_state_targets(
        self,
        qpos_targets: np.ndarray,
        qvel_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute DRLoco-style random-state init targets from BVH frames."""
        qpos_init = np.zeros(
            (*qpos_targets.shape[:2], self._mj_model.nq),
            dtype=np.float32,
        )
        qvel_init = np.zeros(
            (*qpos_targets.shape[:2], self._mj_model.nv),
            dtype=np.float32,
        )
        data = mujoco.MjData(self._mj_model)
        for clip_index in range(qpos_targets.shape[0]):
            for frame_index in range(qpos_targets.shape[1]):
                qpos = np.array(self._init_q_np, copy=True)
                qpos[self._actuator_qpos_indices_np] = qpos_targets[
                    clip_index,
                    frame_index,
                ]
                qpos = self._place_feet_on_floor_with_data(qpos, data)
                qvel = np.zeros(self._mj_model.nv, dtype=np.float32)
                qvel[self._actuator_dof_indices_np] = qvel_targets[
                    clip_index,
                    frame_index,
                ]
                qpos_init[clip_index, frame_index] = qpos.astype(np.float32)
                qvel_init[clip_index, frame_index] = qvel
        return qpos_init, qvel_init

    def _reference_gait_files(self) -> tuple[str, ...]:
        """Vrati BVH fajlove iz config-a kao tuple stringova."""
        reference_gait_file = self._config.get("reference_gait_file", None)
        if reference_gait_file is None:
            return ()
        if isinstance(reference_gait_file, str):
            return tuple(
                path.strip()
                for path in reference_gait_file.split(";")
                if path.strip()
            )
        return tuple(str(path) for path in reference_gait_file)

    def _build_initial_qpos(self, fallback_qpos: np.ndarray) -> np.ndarray:
        """Napravi pocetni qpos iz fajla ili iz stabilnije standing-home poze."""
        init_qpos_file = self._config.get("init_qpos_file", None)
        if init_qpos_file:
            qpos = load_qpos_from_mjdata_file(init_qpos_file, self._mj_model.nq)
            return self._prepare_loaded_qpos(qpos)
        return self._apply_locomotion_neutral_pose(fallback_qpos)

    def _prepare_loaded_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Sanitizuje ucitani qpos i postavi stopala blizu poda."""
        qpos = np.asarray(qpos, dtype=np.float64).copy()
        self._normalize_root_quaternion(qpos)
        self._clip_limited_joints(qpos)
        return self._place_feet_on_floor(qpos)

    def _normalize_root_quaternion(self, qpos: np.ndarray) -> None:
        """Normalizuj free-joint quaternion iz eksternog MJDATA fajla."""
        quat_norm = np.linalg.norm(qpos[3:7])
        if not np.isfinite(quat_norm) or quat_norm < 1e-8:
            raise ValueError("QPOS root quaternion nije validan.")
        qpos[3:7] /= quat_norm

    def _clip_limited_joints(self, qpos: np.ndarray) -> None:
        """Drzi ucitani qpos unutar MuJoCo joint limita."""
        for joint_id in range(self._mj_model.njnt):
            if self._mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if not self._mj_model.jnt_limited[joint_id]:
                continue
            qpos_id = self._mj_model.jnt_qposadr[joint_id]
            lower, upper = self._mj_model.jnt_range[joint_id]
            qpos[qpos_id] = np.clip(qpos[qpos_id], lower, upper)

    def _place_feet_on_floor(self, qpos: np.ndarray) -> np.ndarray:
        """Pomeraj root Z tako da najnizi djon pocne sa malim preload kontaktom."""
        data = mujoco.MjData(self._mj_model)
        return self._place_feet_on_floor_with_data(qpos, data)

    def _place_feet_on_floor_with_data(
        self,
        qpos: np.ndarray,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """Varijanta za batch precompute bez alociranja MjData po frejmu."""
        data.qpos[:] = qpos
        mujoco.mj_forward(self._mj_model, data)
        qpos[2] -= self._minimum_geom_z(data) + self.FOOT_CONTACT_PRELOAD
        return qpos

    def _apply_locomotion_neutral_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Centira akcije oko stabilnije stojece poze, ne oko krute A-poze."""
        for joint_name, joint_value in self.NEUTRAL_JOINT_POSE.items():
            joint_id = mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            qpos[self._mj_model.jnt_qposadr[joint_id]] = joint_value

        return self._place_feet_on_floor(qpos)

    def _minimum_geom_z(self, data: mujoco.MjData) -> float:
        """Vraca najnizu world-Z tacku djonova u trenutnoj pozi."""
        min_z = np.inf
        for geom_name in self.FOOT_SOLE_GEOMS:
            geom_id = mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            min_z = min(min_z, self._geom_min_z(data, geom_id))
        return float(min_z)

    def _geom_min_z(self, data: mujoco.MjData, geom_id: int) -> float:
        """Proceni donju Z tacku za foot geom, uz ispravan capsule tretman."""
        geom_pos = data.geom_xpos[geom_id]
        geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self._mj_model.geom_size[geom_id]
        geom_type = self._mj_model.geom_type[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            min_z = np.inf
            for signs in product((-1.0, 1.0), repeat=3):
                corner = geom_pos + geom_xmat @ (geom_size * np.array(signs))
                min_z = min(min_z, corner[2])
            return float(min_z)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_pos[2] - geom_size[0])

        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = geom_size[0]
            half_length = geom_size[1]
            axis = geom_xmat[:, 2]
            endpoint_a = geom_pos + axis * half_length
            endpoint_b = geom_pos - axis * half_length
            return float(min(endpoint_a[2], endpoint_b[2]) - radius)

        # Conservative fallback for simple geoms that may appear in generated feet.
        return float(geom_pos[2] - np.max(geom_size))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resetuje human u pocetnu pozu i uzorkuje joystick komandu."""
        (
            rng,
            command_key,
            pose_key,
            erfi_key,
            bias_key,
            gait_key,
            bvh_key,
            frame_key,
        ) = jax.random.split(rng, 8)
        bvh_reference_clip_id = jax.random.randint(
            bvh_key,
            shape=(),
            minval=0,
            maxval=int(self._bvh_reference_clip_count),
        )
        frame_count = self._bvh_reference_frame_counts[bvh_reference_clip_id]
        sampled_frame_offset = jp.floor(
            jax.random.uniform(frame_key, shape=()) * frame_count.astype(jp.float32)
        ).astype(jp.int32)
        sampled_frame_offset = jp.minimum(
            sampled_frame_offset,
            frame_count - jp.array(1, dtype=jp.int32),
        )
        if (
            self._config.get("reference_phase_randomization", False)
            or self._config.get("reference_state_init", False)
        ):
            bvh_reference_frame_offset = sampled_frame_offset
        else:
            bvh_reference_frame_offset = jp.array(0, dtype=jp.int32)
        command = self.sample_command(command_key)
        if (
            self._config.get("reference_gait", "none") == "bvh"
            and self._config.get("reference_state_init", False)
        ):
            qpos = self._bvh_reference_init_qpos_targets[
                bvh_reference_clip_id,
                bvh_reference_frame_offset,
            ]
            qvel = self._bvh_reference_init_qvel_targets[
                bvh_reference_clip_id,
                bvh_reference_frame_offset,
            ]
            qvel = qvel.at[0].set(
                self._get_bvh_reference_root_velocity_target_by_frame(
                    bvh_reference_clip_id,
                    bvh_reference_frame_offset,
                    command,
                )
            )
            ctrl = qpos[self._actuator_qpos_indices]
            initial_action = jp.clip(
                (ctrl - self._default_ctrl) / jp.maximum(self._action_scale, 1e-6),
                -1.0,
                1.0,
            )
        else:
            qpos = self._init_q
            qpos = qpos.at[self._actuator_qpos_indices].add(
                jax.random.uniform(
                    pose_key,
                    shape=(self.action_size,),
                    minval=-1.0,
                    maxval=1.0,
                ) * self._init_actuator_noise
            )
            qvel = jp.zeros(self._mjx_model.nv)
            ctrl = self._default_ctrl
            initial_action = jp.zeros(self.action_size)
        data = mjx_env.make_data(self._mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self._mjx_model, data)
        info = {
            "rng": rng,
            "command": command,
            "last_action": initial_action,
            "episode_torque_offset": self.sample_episode_torque_offset(bias_key),
            "use_rfi": jax.random.bernoulli(erfi_key, p=0.5),
            "step": jp.array(0),
            "gait_step": jax.random.randint(
                gait_key,
                shape=(),
                minval=0,
                maxval=int(self.GAIT_PERIOD_STEPS),
            ),
            "bvh_reference_clip_id": bvh_reference_clip_id,
            "bvh_reference_frame_offset": bvh_reference_frame_offset,
            "last_foot_xy": self._foot_xy(data),
        }
        obs = self._get_obs(data, info)
        metrics = {
            "reward": jp.array(0.0),
            "reward_raw": jp.array(0.0),
            "reward_clip_low": jp.array(0.0),
            "reward_clip_high": jp.array(0.0),
            "reward_done_override": jp.array(0.0),
            "reward_nonterminal": jp.array(0.0),
            "tracking_lin_vel": jp.array(0.0),
            "forward_vel": jp.array(0.0),
            "command_norm": jp.array(0.0),
            "command_progress": jp.array(0.0),
            "torso_up": jp.array(1.0),
            "head_up": jp.array(1.0),
            "height": qpos[2],
            "foot_slip": jp.array(0.0),
            "swing_drag": jp.array(0.0),
            "swing_clearance": jp.array(0.0),
            "swing_clearance_deficit": jp.array(0.0),
            "locomotion_quality": jp.array(0.0),
            "gated_tracking": jp.array(0.0),
            "gated_progress": jp.array(0.0),
            "swing_contact": jp.array(0.0),
            "stance_contact": jp.array(0.0),
            "double_contact": jp.array(0.0),
            "double_support_drag": jp.array(0.0),
            "gait_cost_scale": jp.array(0.0),
            "task_reward": jp.array(0.0),
            "zombie_sine_reward": jp.array(0.0),
            "bvh_mimic_reward": jp.array(0.0),
            "bvh_mimic_core": jp.array(0.0),
            "bvh_stability_reward": jp.array(0.0),
            "bvh_joystick_reward": jp.array(0.0),
            "bvh_regularization_cost": jp.array(0.0),
            "bvh_regularization_raw": jp.array(0.0),
            "bvh_locomotion_gate": jp.array(0.0),
            "bvh_bootstrap_reward": jp.array(0.0),
            "base_height": jp.array(1.0),
            "variable_posture": jp.array(1.0),
            "gait_reward": jp.array(0.0),
            "reference_gait": jp.array(0.0),
            "reference_velocity": jp.array(0.0),
            "reference_foot": jp.array(0.0),
            "reference_root": jp.array(0.0),
            "contact_force": jp.array(0.0),
            "joint_limit": jp.array(0.0),
            "illegal_contact": jp.array(0.0),
            "foot_slip_cost": jp.array(0.0),
            "swing_drag_cost": jp.array(0.0),
            "swing_contact_cost": jp.array(0.0),
            "clearance_deficit_cost": jp.array(0.0),
            "double_contact_cost": jp.array(0.0),
            "double_support_drag_cost": jp.array(0.0),
            "overspeed_cost": jp.array(0.0),
            "height_cost": jp.array(0.0),
            "done_low_height": jp.array(0.0),
            "done_tipped": jp.array(0.0),
            "done_illegal_contact": jp.array(0.0),
            "done_invalid": jp.array(0.0),
            "done": jp.array(0.0),
        }
        return mjx_env.State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Izvrsi jedan locomotion korak za zadatu akciju politike."""
        info = dict(state.info)
        info["rng"], rfi_key, command_key = jax.random.split(info["rng"], 3)

        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        policy_action = jp.clip(action, -1.0, 1.0)
        previous_action = info["last_action"]
        smoothed_action = (
            self._config.action_smoothing * policy_action
            + (1.0 - self._config.action_smoothing) * previous_action
        )
        erfi_torque = self.sample_erfi_torque(
            rfi_key,
            info["episode_torque_offset"],
            info["use_rfi"],
        )
        data_with_erfi = self.apply_joint_torque_injection(
            state.data,
            erfi_torque,
        )
        motor_targets = self._default_ctrl + (smoothed_action * self._action_scale)
        data = mjx_env.step(
            self._mjx_model,
            data_with_erfi,
            motor_targets,
            self._n_substeps,
        )

        should_resample = info["step"] > self._config.command_resample_steps
        info["command"] = jp.where(
            should_resample,
            self.sample_command(command_key),
            info["command"],
        )
        info["last_action"] = smoothed_action
        info["step"] = jp.where(should_resample, 0, info["step"] + 1)
        info["gait_step"] = jp.mod(
            info["gait_step"] + jp.array(1, dtype=info["gait_step"].dtype),
            self.GAIT_PERIOD_STEPS,
        )

        reward_components = self._get_reward_components(
            data,
            smoothed_action,
            previous_action,
            info,
        )
        reward = reward_components["reward"]
        done = reward_components["done"]
        obs = self._get_obs(data, info)
        metrics = dict(state.metrics)
        metrics["reward"] = reward
        metrics["reward_raw"] = reward_components["reward_raw"]
        metrics["reward_clip_low"] = reward_components["reward_clip_low"]
        metrics["reward_clip_high"] = reward_components["reward_clip_high"]
        metrics["reward_done_override"] = reward_components["reward_done_override"]
        metrics["reward_nonterminal"] = reward_components["reward_nonterminal"]
        metrics["tracking_lin_vel"] = reward_components["tracking_lin_vel"]
        metrics["forward_vel"] = reward_components["forward_vel"]
        metrics["command_norm"] = reward_components["command_norm"]
        metrics["command_progress"] = reward_components["command_progress"]
        metrics["torso_up"] = reward_components["torso_up"]
        metrics["head_up"] = reward_components["head_up"]
        metrics["height"] = reward_components["height"]
        metrics["foot_slip"] = reward_components["foot_slip"]
        metrics["swing_drag"] = reward_components["swing_drag"]
        metrics["swing_clearance"] = reward_components["swing_clearance"]
        metrics["swing_clearance_deficit"] = reward_components[
            "swing_clearance_deficit"
        ]
        metrics["locomotion_quality"] = reward_components["locomotion_quality"]
        metrics["gated_tracking"] = reward_components["gated_tracking"]
        metrics["gated_progress"] = reward_components["gated_progress"]
        metrics["swing_contact"] = reward_components["swing_contact"]
        metrics["stance_contact"] = reward_components["stance_contact"]
        metrics["double_contact"] = reward_components["double_contact"]
        metrics["double_support_drag"] = reward_components["double_support_drag"]
        metrics["gait_cost_scale"] = reward_components["gait_cost_scale"]
        metrics["task_reward"] = reward_components["task_reward"]
        metrics["zombie_sine_reward"] = reward_components["zombie_sine_reward"]
        metrics["bvh_mimic_reward"] = reward_components["bvh_mimic_reward"]
        metrics["bvh_mimic_core"] = reward_components["bvh_mimic_core"]
        metrics["bvh_stability_reward"] = reward_components["bvh_stability_reward"]
        metrics["bvh_joystick_reward"] = reward_components["bvh_joystick_reward"]
        metrics["bvh_regularization_cost"] = reward_components[
            "bvh_regularization_cost"
        ]
        metrics["bvh_regularization_raw"] = reward_components[
            "bvh_regularization_raw"
        ]
        metrics["bvh_locomotion_gate"] = reward_components["bvh_locomotion_gate"]
        metrics["bvh_bootstrap_reward"] = reward_components["bvh_bootstrap_reward"]
        metrics["base_height"] = reward_components["base_height"]
        metrics["variable_posture"] = reward_components["variable_posture"]
        metrics["gait_reward"] = reward_components["gait_reward"]
        metrics["reference_gait"] = reward_components["reference_gait"]
        metrics["reference_velocity"] = reward_components["reference_velocity"]
        metrics["reference_foot"] = reward_components["reference_foot"]
        metrics["reference_root"] = reward_components["reference_root"]
        metrics["contact_force"] = reward_components["contact_force"]
        metrics["joint_limit"] = reward_components["joint_limit"]
        metrics["illegal_contact"] = reward_components["illegal_contact"]
        metrics["foot_slip_cost"] = reward_components["foot_slip_cost"]
        metrics["swing_drag_cost"] = reward_components["swing_drag_cost"]
        metrics["swing_contact_cost"] = reward_components["swing_contact_cost"]
        metrics["clearance_deficit_cost"] = reward_components[
            "clearance_deficit_cost"
        ]
        metrics["double_contact_cost"] = reward_components["double_contact_cost"]
        metrics["double_support_drag_cost"] = reward_components[
            "double_support_drag_cost"
        ]
        metrics["overspeed_cost"] = reward_components["overspeed_cost"]
        metrics["height_cost"] = reward_components["height_cost"]
        metrics["done_low_height"] = reward_components["done_low_height"].astype(
            reward.dtype
        )
        metrics["done_tipped"] = reward_components["done_tipped"].astype(reward.dtype)
        metrics["done_illegal_contact"] = reward_components[
            "done_illegal_contact"
        ].astype(reward.dtype)
        metrics["done_invalid"] = reward_components["done_invalid"].astype(
            reward.dtype
        )
        metrics["done"] = done.astype(reward.dtype)
        info["last_foot_xy"] = self._foot_xy(data)
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(reward.dtype),
            metrics=metrics,
            info=info,
        )

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Uzorkuje ciljnu brzinu: napred/nazad, levo/desno, yaw."""
        if self._config.command_profile in ("forward_slow", "forward", "walk"):
            x_key, zero_key = jax.random.split(rng, 2)
            command_range = (
                self.FORWARD_SLOW_COMMAND_RANGE
                if self._config.command_profile == "forward_slow"
                else self.FORWARD_COMMAND_RANGE
            )
            zero_probability = (
                self.FORWARD_SLOW_ZERO_COMMAND_PROBABILITY
                if self._config.command_profile == "forward_slow"
                else self.ZERO_COMMAND_PROBABILITY
            )
            command = jp.array([
                jax.random.uniform(
                    x_key,
                    minval=command_range[0],
                    maxval=command_range[1],
                ),
                0.0,
                0.0,
            ])
            return jp.where(
                jax.random.bernoulli(zero_key, p=zero_probability),
                jp.zeros(3),
                command,
            )

        command_ranges = {
            "steer": (
                self.STEER_X_COMMAND_RANGE,
                self.STEER_Y_COMMAND_RANGE,
                self.STEER_YAW_COMMAND_RANGE,
                self.STEER_ZERO_COMMAND_PROBABILITY,
            ),
            "standard_easy": (
                self.STANDARD_EASY_X_COMMAND_RANGE,
                self.STANDARD_EASY_Y_COMMAND_RANGE,
                self.STANDARD_EASY_YAW_COMMAND_RANGE,
                self.STANDARD_EASY_ZERO_COMMAND_PROBABILITY,
            ),
            "standard": (
                self.STANDARD_X_COMMAND_RANGE,
                self.STANDARD_Y_COMMAND_RANGE,
                self.STANDARD_YAW_COMMAND_RANGE,
                self.STANDARD_ZERO_COMMAND_PROBABILITY,
            ),
        }
        if self._config.command_profile not in command_ranges:
            raise ValueError(
                "command_profile mora biti 'forward_slow', 'forward', 'walk', "
                "'steer', 'standard_easy' ili 'standard'."
            )

        x_range, y_range, yaw_range, zero_probability = command_ranges[
            self._config.command_profile
        ]
        x_key, y_key, yaw_key, zero_key = jax.random.split(rng, 4)
        command = jp.array([
            jax.random.uniform(
                x_key,
                minval=x_range[0],
                maxval=x_range[1],
            ),
            jax.random.uniform(
                y_key,
                minval=y_range[0],
                maxval=y_range[1],
            ),
            jax.random.uniform(
                yaw_key,
                minval=yaw_range[0],
                maxval=yaw_range[1],
            ),
        ])
        return jp.where(
            jax.random.bernoulli(zero_key, p=zero_probability),
            jp.zeros(3),
            command,
        )

    def sample_episode_torque_offset(self, rng: jax.Array) -> jax.Array:
        """Uzorkuje RAO: konstantan torque offset za celu epizodu."""
        return jax.random.uniform(
            rng,
            shape=(self.action_size,),
            minval=-self._config.rao_torque_limit,
            maxval=self._config.rao_torque_limit,
        )

    def sample_erfi_torque(
        self,
        rng: jax.Array,
        episode_offset: jax.Array,
        use_rfi: jax.Array,
    ) -> jax.Array:
        """Implementira ERFI-50: pola epizoda RFI, pola RAO."""
        if not self._config.enable_erfi:
            return jp.zeros(self.action_size)

        rfi_torque = jax.random.uniform(
            rng,
            shape=(self.action_size,),
            minval=-self._config.rfi_torque_limit,
            maxval=self._config.rfi_torque_limit,
        )
        return jp.where(use_rfi, rfi_torque, episode_offset) * (
            self._torque_injection_scale
        )

    def apply_joint_torque_injection(
        self,
        data: mjx.Data,
        joint_torque: jax.Array,
    ) -> mjx.Data:
        """Upise RFI/RAO torques u qfrc_applied na kontrolisane DoF-ove."""
        qfrc_applied = jp.zeros_like(data.qfrc_applied)
        qfrc_applied = qfrc_applied.at[self._actuator_dof_indices].set(joint_torque)
        return data.replace(qfrc_applied=qfrc_applied)

    def _get_obs(self, data: mjx.Data, info: dict) -> dict[str, jax.Array]:
        """Sastavi policy i privileged critic observation."""
        local_linvel = self._local_root_linvel(data)
        local_angvel = self._local_root_angvel(data)
        projected_gravity = self._projected_gravity(data)
        joint_pos = data.qpos[7:] - self._default_qpos
        joint_vel = data.qvel[6:]
        state_parts = [
            local_linvel,
            local_angvel,
            projected_gravity,
            info["command"],
            self._get_gait_phase_obs(info),
            joint_pos,
            joint_vel,
            info["last_action"],
        ]
        if self._config.get("reference_target_observation", False):
            state_parts.insert(5, self._get_reference_target_delta(info))
        state_obs = jp.concatenate(state_parts)
        state_obs = jp.nan_to_num(state_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        privileged_obs = jp.concatenate([
            state_obs,
            data.qpos[:3],
            data.qvel[:6],
            self._gymnasium_privileged_obs(data),
            data.qfrc_actuator[self._actuator_dof_indices],
            self._foot_positions(data).reshape(-1),
            self._foot_contact(data),
            self._action_scale,
        ])
        privileged_obs = jp.nan_to_num(
            privileged_obs,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        return {
            "state": state_obs,
            "privileged_state": privileged_obs,
        }

    def _gymnasium_privileged_obs(self, data: mjx.Data) -> jax.Array:
        """Gymnasium Humanoid-style physical signals for the critic only."""
        return jp.concatenate([
            self._bounded_obs(data.cinert.reshape(-1), scale=0.01),
            self._bounded_obs(data.cvel.reshape(-1), scale=0.1),
            self._bounded_obs(data.cfrc_ext.reshape(-1), scale=0.01),
        ])

    def _bounded_obs(
        self,
        values: jax.Array,
        scale: float = 1.0,
        limit: float = 10.0,
    ) -> jax.Array:
        """Skaliraj i ogranici velike physics vrednosti pre critic inputa."""
        values = jp.nan_to_num(values * scale, nan=0.0, posinf=limit, neginf=-limit)
        return jp.clip(values, -limit, limit)

    def _get_gait_phase_obs(self, info: dict) -> jax.Array:
        """Dodaje clock signal politici za lakse uskladjivanje nogu."""
        phase_angle = self._get_gait_phase_angle(info)
        return jp.array([jp.sin(phase_angle), jp.cos(phase_angle)])

    def _get_reference_target_delta(self, info: dict) -> jax.Array:
        """Policy vidi trenutni reference target, inace uci skriveni zadatak."""
        if self._config.get("reference_gait", "none") == "none":
            return jp.zeros(self.action_size)
        target_qpos = self._get_reference_gait_target(info)
        return jp.where(
            self._reference_gait_mask,
            target_qpos - self._default_ctrl,
            0.0,
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> jax.Array:
        """Nagrada za joystick hod: prati command vektor, stabilnost je uslov."""
        return self._get_reward_components(
            data,
            action,
            previous_action,
            info,
        )["reward"]

    def _get_reward_components(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> dict[str, jax.Array]:
        """Izracunaj reward i eval metrike jednim prolazom kroz hot path."""
        local_linvel = self._local_root_linvel(data)
        local_angvel = self._local_root_angvel(data)
        measured_command = jp.array([
            local_linvel[0],
            local_linvel[2],
            local_angvel[1],
        ])
        command_norm = jp.linalg.norm(info["command"])
        measured_norm = jp.linalg.norm(measured_command)
        command_active = command_norm > self.STUCK_COMMAND_THRESHOLD
        tracking_error = jp.sum(jp.square(info["command"] - measured_command))
        tracking = jp.exp(-tracking_error / self._config.tracking_sigma)
        command_norm_sq = jp.sum(jp.square(info["command"]))
        progress_active = command_norm_sq > self.STUCK_COMMAND_THRESHOLD**2
        alignment = jp.dot(measured_command, info["command"]) / jp.maximum(
            command_norm_sq,
            1e-6,
        )
        command_progress = jp.where(
            progress_active,
            jp.clip(alignment, 0.0, 1.0) * tracking,
            0.0,
        )
        overspeed_denominator = jp.maximum(command_norm, 0.05)
        overspeed = jp.maximum(measured_norm - 1.25 * command_norm, 0.0)
        overspeed_cost = jp.minimum(
            self.OVERSPEED_COST_SCALE * jp.square(
                overspeed / overspeed_denominator
            ),
            self.OVERSPEED_COST_CAP,
        )
        idle_motion_cost = jp.where(
            command_active,
            0.0,
            0.25 * jp.square(measured_norm),
        )
        commanded_axis = info["command"] / jp.maximum(command_norm, 1e-6)
        velocity_along_command = jp.dot(measured_command, commanded_axis)
        stuck_penalty = jp.where(
            command_active & (velocity_along_command < self.STUCK_VELOCITY_THRESHOLD),
            self.STUCK_PENALTY,
            0.0,
        )
        torso_up = self._torso_up(data)
        head_up = self._head_up(data)
        upright = jp.clip(torso_up, 0.0, 1.0)
        clipped_head_up = jp.clip(head_up, 0.0, 1.0)
        low_height = jp.maximum(
            0.0,
            self.HEIGHT_PENALTY_START_RATIO * self._init_q[2] - data.qpos[2],
        )
        action_cost = self.ACTION_COST_SCALE * jp.sum(jp.square(action))
        action_rate_cost = self.ACTION_RATE_COST_SCALE * jp.sum(
            jp.square(action - previous_action)
        )
        vertical_velocity_cost = (
            self.VERTICAL_VELOCITY_COST_SCALE * jp.square(data.qvel[2])
        )
        angular_velocity_cost = self.ANGULAR_VELOCITY_COST_SCALE * (
            jp.square(local_angvel[0]) + jp.square(local_angvel[2])
        )
        actuator_position_error = (
            data.qpos[self._actuator_qpos_indices] - self._default_ctrl
        )
        posture_error = jp.mean(jp.square(actuator_position_error))
        posture_reward = self.POSTURE_REWARD_SCALE * jp.exp(-posture_error)
        variable_posture_error = self._get_variable_posture_error(data, info)
        variable_posture = jp.exp(-0.5 * variable_posture_error)
        variable_posture_reward = (
            self.VARIABLE_POSTURE_REWARD_SCALE * variable_posture
        )
        variable_posture_cost = (
            self.VARIABLE_POSTURE_COST_SCALE
            * jp.maximum(variable_posture_error - 1.0, 0.0)
        )
        if self._config.get("legacy_action_prior", False):
            variable_posture_reward = jp.array(0.0)
            variable_posture_cost = jp.array(0.0)
        trunk_error = jp.sum(
            jp.square(jp.where(self._trunk_actuator_mask, actuator_position_error, 0.0))
        )
        trunk_posture_cost = self.TRUNK_POSTURE_COST_SCALE * trunk_error
        base_height = self._get_base_height_reward(data)
        base_height_reward = self.BASE_HEIGHT_REWARD_SCALE * base_height
        gait_reward = self._get_gait_reward(data, info)
        swing_clearance = self._get_swing_clearance(data, info)
        swing_clearance_deficit = self._get_swing_clearance_deficit_cost(
            data,
            info,
        )
        locomotion_quality = self._get_locomotion_quality(data, info)
        swing_contact = self._get_swing_contact(data, info)
        stance_contact = self._get_stance_contact(data, info)
        double_contact = self._get_double_contact(data)
        gated_tracking = jp.where(
            command_active,
            tracking
            * (
                self.LOCOMOTION_QUALITY_TRACKING_FLOOR
                + (1.0 - self.LOCOMOTION_QUALITY_TRACKING_FLOOR)
                * locomotion_quality
            ),
            tracking,
        )
        gated_progress = command_progress * locomotion_quality
        gait_cost_scale = jp.where(
            command_active,
            jp.clip(measured_norm / jp.maximum(command_norm, 0.10), 0.0, 1.0),
            0.0,
        )
        swing_clearance_deficit_cost = (
            self.SWING_CLEARANCE_DEFICIT_COST_SCALE
            * gait_cost_scale
            * swing_clearance_deficit
        )
        reference_gait = self._get_reference_gait_reward(data, info)
        reference_gait_reward = (
            self.REFERENCE_GAIT_REWARD_SCALE * reference_gait
        )
        reference_velocity = self._get_reference_velocity_reward(data, info)
        reference_velocity_reward = (
            self.REFERENCE_VELOCITY_REWARD_SCALE * reference_velocity
        )
        reference_foot = self._get_reference_foot_reward(data, info)
        reference_foot_reward = (
            self.REFERENCE_FOOT_REWARD_SCALE * reference_foot
        )
        reference_root = self._get_reference_root_reward(data, info)
        reference_root_reward = (
            self.REFERENCE_ROOT_REWARD_SCALE * reference_root
        )
        contact_force_cost = self._get_contact_force_cost(data)
        joint_limit = self._get_soft_joint_limit_violation(data)
        joint_limit_cost = self.SOFT_JOINT_LIMIT_COST_SCALE * joint_limit
        illegal_contact = self._get_illegal_contact(data)
        illegal_contact_cost = self.ILLEGAL_CONTACT_COST_SCALE * illegal_contact
        foot_slip = self._get_foot_slip_cost(
            data,
            info,
        )
        foot_slip_cost = (
            self.FOOT_SLIP_COST_SCALE
            * jp.minimum(foot_slip, self.FOOT_SLIP_COST_CAP)
        )
        swing_drag = self._get_swing_foot_drag_cost(data, info)
        swing_drag_cost = (
            self.SWING_FOOT_DRAG_COST_SCALE * gait_cost_scale * swing_drag
        )
        swing_contact_cost = jp.where(
            command_active,
            self.SWING_CONTACT_COST_SCALE * gait_cost_scale * swing_contact,
            0.0,
        )
        double_support_drag = self._get_double_support_drag_cost(
            data,
            info,
            measured_command,
            command_norm,
        )
        double_support_drag_cost = (
            self.DOUBLE_SUPPORT_DRAG_COST_SCALE * double_support_drag
        )
        double_contact_cost = jp.where(
            command_active,
            self.DOUBLE_CONTACT_COST_SCALE * gait_cost_scale * double_contact,
            0.0,
        )
        velocity_tracking_scale = self.VELOCITY_TRACKING_REWARD_SCALE
        forward_progress_scale = self.FORWARD_PROGRESS_REWARD_SCALE
        if self._config.command_profile == "forward_slow":
            velocity_tracking_scale = 0.8
            forward_progress_scale = 0.25

        height_cost = self.LOW_HEIGHT_COST_SCALE * jp.square(low_height)
        reward_terms = {
            "command_progress_raw": forward_progress_scale * command_progress,
            "gated_tracking": velocity_tracking_scale * gated_tracking,
            "gated_progress": forward_progress_scale * gated_progress,
            "upright": self.UPRIGHT_REWARD_SCALE * upright,
            "head_up": self.HEAD_UP_REWARD_SCALE * clipped_head_up,
            "base_height": base_height_reward,
            "posture": posture_reward,
            "variable_posture": variable_posture_reward,
            "gait": gait_reward,
            "reference_pose": reference_gait_reward,
            "reference_velocity": reference_velocity_reward,
            "reference_foot": reference_foot_reward,
            "reference_root": reference_root_reward,
            "stuck": stuck_penalty,
            "action": action_cost,
            "action_rate": action_rate_cost,
            "trunk_posture": trunk_posture_cost,
            "variable_posture_cost": variable_posture_cost,
            "contact_force": contact_force_cost,
            "joint_limit": joint_limit_cost,
            "illegal_contact": illegal_contact_cost,
            "foot_slip": foot_slip_cost,
            "swing_drag": swing_drag_cost,
            "swing_contact": swing_contact_cost,
            "clearance_deficit": swing_clearance_deficit_cost,
            "double_contact": double_contact_cost,
            "double_support_drag": double_support_drag_cost,
            "height": height_cost,
            "overspeed": overspeed_cost,
            "idle_motion": idle_motion_cost,
            "vertical_velocity": vertical_velocity_cost,
            "angular_velocity": angular_velocity_cost,
        }
        task_reward = self._get_task_velocity_reward(reward_terms)
        zombie_sine_reward = self._get_zombie_walking_trajectory_sine_reward(
            reward_terms
        )
        bvh_mimic_components = self._get_bvh_mimic_components(
            reference_gait,
            reference_velocity,
            reference_foot,
            reference_root,
            locomotion_quality,
            reward_terms,
        )
        bvh_mimic_reward = bvh_mimic_components["reward"]
        reference_gait_type = self._config.get("reference_gait", "none")
        if reference_gait_type == "bvh":
            reward_raw = bvh_mimic_reward
        elif reference_gait_type == "sine":
            reward_raw = zombie_sine_reward
        else:
            reward_raw = task_reward
        reward_clip_low = reward_raw < self.REWARD_MIN
        reward_clip_high = reward_raw > self.REWARD_MAX
        reward_nonterminal = jp.clip(reward_raw, self.REWARD_MIN, self.REWARD_MAX)
        reward_nonterminal = jp.nan_to_num(
            reward_nonterminal,
            nan=0.0,
            posinf=self.REWARD_MAX,
            neginf=self.REWARD_MIN,
        )
        too_low, tipped_over, invalid = self._get_done_reasons_from_torso(
            data,
            torso_up,
        )
        done_illegal_contact = self._get_fatal_illegal_contact(data)
        done = too_low | tipped_over | invalid | done_illegal_contact
        reward_done_override = done.astype(reward_nonterminal.dtype)
        reward = jp.where(done, self.FALL_REWARD, reward_nonterminal)
        return {
            "reward": reward,
            "reward_raw": reward_raw,
            "reward_clip_low": reward_clip_low.astype(reward.dtype),
            "reward_clip_high": reward_clip_high.astype(reward.dtype),
            "reward_done_override": reward_done_override,
            "reward_nonterminal": reward_nonterminal,
            "tracking_lin_vel": tracking,
            "forward_vel": local_linvel[0],
            "command_norm": command_norm,
            "command_progress": command_progress,
            "torso_up": torso_up,
            "head_up": head_up,
            "height": data.qpos[2],
            "foot_slip": foot_slip,
            "swing_drag": swing_drag,
            "swing_clearance": swing_clearance,
            "swing_clearance_deficit": swing_clearance_deficit,
            "locomotion_quality": locomotion_quality,
            "gated_tracking": gated_tracking,
            "gated_progress": gated_progress,
            "swing_contact": swing_contact,
            "stance_contact": stance_contact,
            "double_contact": double_contact,
            "double_support_drag": double_support_drag,
            "gait_cost_scale": gait_cost_scale,
            "task_reward": task_reward,
            "zombie_sine_reward": zombie_sine_reward,
            "bvh_mimic_reward": bvh_mimic_reward,
            "bvh_mimic_core": bvh_mimic_components["mimic_core"],
            "bvh_stability_reward": bvh_mimic_components["stability"],
            "bvh_joystick_reward": bvh_mimic_components["joystick"],
            "bvh_regularization_cost": bvh_mimic_components["regularization"],
            "bvh_regularization_raw": bvh_mimic_components["regularization_raw"],
            "bvh_locomotion_gate": bvh_mimic_components["locomotion_gate"],
            "bvh_bootstrap_reward": bvh_mimic_components["bootstrap"],
            "base_height": base_height,
            "variable_posture": variable_posture,
            "gait_reward": gait_reward,
            "reference_gait": reference_gait,
            "reference_velocity": reference_velocity,
            "reference_foot": reference_foot,
            "reference_root": reference_root,
            "contact_force": contact_force_cost,
            "joint_limit": joint_limit,
            "illegal_contact": illegal_contact,
            "foot_slip_cost": foot_slip_cost,
            "swing_drag_cost": swing_drag_cost,
            "swing_contact_cost": swing_contact_cost,
            "clearance_deficit_cost": swing_clearance_deficit_cost,
            "double_contact_cost": double_contact_cost,
            "double_support_drag_cost": double_support_drag_cost,
            "overspeed_cost": overspeed_cost,
            "height_cost": height_cost,
            "done_low_height": too_low,
            "done_tipped": tipped_over,
            "done_illegal_contact": done_illegal_contact,
            "done_invalid": invalid,
            "done": done,
        }

    def _get_task_velocity_reward(
        self,
        terms: dict[str, jax.Array],
    ) -> jax.Array:
        """Standardni joystick/velocity reward bez explicit motion imitation-a."""
        return (
            self.ALIVE_REWARD_SCALE
            + terms["gated_tracking"]
            + terms["gated_progress"]
            + terms["upright"]
            + terms["head_up"]
            + terms["base_height"]
            + terms["posture"]
            + terms["variable_posture"]
            - terms["stuck"]
            - terms["action"]
            - terms["action_rate"]
            - terms["trunk_posture"]
            - terms["variable_posture_cost"]
            - terms["contact_force"]
            - terms["joint_limit"]
            - terms["illegal_contact"]
            - terms["foot_slip"]
            - terms["swing_drag"]
            - terms["swing_contact"]
            - terms["clearance_deficit"]
            - terms["double_contact"]
            - terms["double_support_drag"]
            - terms["height"]
            - terms["overspeed"]
            - terms["idle_motion"]
            - terms["vertical_velocity"]
            - terms["angular_velocity"]
        )

    def _get_zombie_walking_trajectory_sine_reward(
        self,
        terms: dict[str, jax.Array],
    ) -> jax.Array:
        """Proceduralni sine/zombie-walking gait reward, odvojen od BVH mimic-a."""
        return (
            self.ALIVE_REWARD_SCALE
            + terms["gated_tracking"]
            + terms["gated_progress"]
            + terms["upright"]
            + terms["head_up"]
            + terms["base_height"]
            + terms["posture"]
            + terms["variable_posture"]
            + terms["gait"]
            + terms["reference_pose"]
            - terms["stuck"]
            - terms["action"]
            - terms["action_rate"]
            - terms["trunk_posture"]
            - terms["variable_posture_cost"]
            - terms["contact_force"]
            - terms["joint_limit"]
            - terms["illegal_contact"]
            - terms["foot_slip"]
            - terms["swing_drag"]
            - terms["swing_contact"]
            - terms["clearance_deficit"]
            - terms["double_contact"]
            - terms["double_support_drag"]
            - terms["height"]
            - terms["overspeed"]
            - terms["idle_motion"]
            - terms["vertical_velocity"]
            - terms["angular_velocity"]
        )

    def _get_bvh_mimic_reward(
        self,
        pose_reward: jax.Array,
        velocity_reward: jax.Array,
        foot_reward: jax.Array,
        root_reward: jax.Array,
        contact_reward: jax.Array,
        terms: dict[str, jax.Array],
    ) -> jax.Array:
        """Compatibility helper: return only the scalar BVH mimic reward."""
        return self._get_bvh_mimic_components(
            pose_reward,
            velocity_reward,
            foot_reward,
            root_reward,
            contact_reward,
            terms,
        )["reward"]

    def _get_bvh_mimic_components(
        self,
        pose_reward: jax.Array,
        velocity_reward: jax.Array,
        foot_reward: jax.Array,
        root_reward: jax.Array,
        contact_reward: jax.Array,
        terms: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """DeepMimic/LocoMuJoCo-style BVH reward: mimic first, task second."""
        locomotion_gate = (
            self.BVH_LOCOMOTION_GATE_FLOOR
            + (1.0 - self.BVH_LOCOMOTION_GATE_FLOOR) * contact_reward
        )
        pose_like_reward = (
            self.BVH_MIMIC_POSE_WEIGHT * pose_reward
            + self.BVH_MIMIC_VELOCITY_WEIGHT * velocity_reward
            + self.BVH_MIMIC_FOOT_WEIGHT * foot_reward
            + self.BVH_MIMIC_ROOT_WEIGHT * root_reward
        )
        mimic_reward = (
            pose_like_reward
            + self.BVH_MIMIC_CONTACT_WEIGHT * contact_reward
        )
        stability_reward = self.BVH_STABILITY_REWARD_SCALE * (
            0.5 * terms["upright"]
            + 0.2 * terms["head_up"]
            + 0.3 * terms["base_height"]
        )
        joystick_reward = (
            self.BVH_TASK_TRACKING_SCALE * terms["gated_tracking"]
            + self.BVH_TASK_PROGRESS_SCALE * terms["gated_progress"]
        )
        bootstrap_reward = (
            self.BVH_BOOTSTRAP_GAIT_SCALE * terms["gait"]
            + self.BVH_BOOTSTRAP_PROGRESS_SCALE * terms["command_progress_raw"]
        )
        regularization_raw = (
            0.50 * terms["action"]
            + 0.50 * terms["action_rate"]
            + 0.35 * terms["trunk_posture"]
            + 0.35 * terms["variable_posture_cost"]
            + 0.25 * jp.tanh(terms["contact_force"])
            + 0.35 * jp.tanh(terms["joint_limit"])
            + 0.50 * terms["illegal_contact"]
            + 0.25 * jp.tanh(terms["foot_slip"])
            + self.BVH_REG_SWING_DRAG_WEIGHT * terms["swing_drag"]
            + self.BVH_REG_SWING_CONTACT_WEIGHT * terms["swing_contact"]
            + self.BVH_REG_CLEARANCE_DEFICIT_WEIGHT * terms["clearance_deficit"]
            + self.BVH_REG_DOUBLE_CONTACT_WEIGHT * terms["double_contact"]
            + (
                self.BVH_REG_DOUBLE_SUPPORT_DRAG_WEIGHT
                * jp.tanh(terms["double_support_drag"])
            )
            + 0.35 * jp.tanh(terms["height"])
            + 0.50 * terms["idle_motion"]
            + 0.35 * terms["vertical_velocity"]
            + 0.35 * terms["angular_velocity"]
        )
        regularization_cost = jp.minimum(
            regularization_raw,
            self.BVH_REGULARIZATION_COST_CAP,
        )
        reward = (
            self.ALIVE_REWARD_SCALE
            + self.BVH_MIMIC_REWARD_SCALE * mimic_reward
            + stability_reward
            + joystick_reward
            + bootstrap_reward
            - self.BVH_STUCK_PENALTY_SCALE * terms["stuck"]
            - regularization_cost
        )
        return {
            "reward": reward,
            "mimic_core": mimic_reward,
            "stability": stability_reward,
            "joystick": joystick_reward,
            "regularization": regularization_cost,
            "regularization_raw": regularization_raw,
            "locomotion_gate": locomotion_gate,
            "bootstrap": bootstrap_reward,
        }

    def _get_done(self, data: mjx.Data) -> jax.Array:
        """Zavrsi epizodu ako human padne ili numerika ode u NaN."""
        too_low, tipped_over, invalid = self._get_done_reasons(data)
        return too_low | tipped_over | invalid | self._get_fatal_illegal_contact(data)

    def _get_done_reasons(self, data: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Vrati pojedinacne termination razloge za debug logove."""
        return self._get_done_reasons_from_torso(data, self._torso_up(data))

    def _get_done_reasons_from_torso(
        self,
        data: mjx.Data,
        torso_up: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Varijanta koja koristi vec izracunat torso-up signal."""
        too_low = data.qpos[2] < self.MIN_STANDING_HEIGHT_RATIO * self._init_q[2]
        tipped_over = torso_up < 0.25
        invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return too_low, tipped_over, invalid

    def _get_gait_phase_angle(self, info: dict) -> jax.Array:
        """Periodican signal koji govori politici koja noga treba da bude swing."""
        if self._config.get("reference_gait", "none") == "bvh":
            return self._get_bvh_reference_phase_angle(info)
        phase = jp.mod(
            info["gait_step"].astype(jp.float32),
            self.GAIT_PERIOD_STEPS,
        ) / self.GAIT_PERIOD_STEPS
        return 2.0 * jp.pi * phase

    def _get_bvh_reference_phase_angle(self, info: dict) -> jax.Array:
        """Phase signal izveden iz aktivnog BVH clip-a."""
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        frame_count = self._bvh_reference_frame_counts[clip_id]
        frame_time = self._bvh_reference_frame_times[clip_id]
        frame_float = (
            info["step"].astype(jp.float32)
            * jp.array(self.dt, dtype=jp.float32)
            / frame_time
        )
        phase = jp.mod(frame_float, frame_count.astype(jp.float32)) / (
            frame_count.astype(jp.float32)
        )
        return 2.0 * jp.pi * phase

    def _get_gait_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Nagradi swing clearance i stance contact u fazi koraka."""
        left_swing = self._is_left_swing(info)
        foot_heights = self._foot_heights(data)
        foot_contact = self._foot_contact(data)
        stance_contact = jp.where(left_swing, foot_contact[1], foot_contact[0])
        swing_contact = jp.where(left_swing, foot_contact[0], foot_contact[1])
        single_support = stance_contact * (1.0 - swing_contact)
        relative_clearance = self._get_swing_clearance(data, info)
        clearance_reward = jp.clip(
            relative_clearance / self.FOOT_CLEARANCE_TARGET,
            0.0,
            1.0,
        )
        command_active = jp.linalg.norm(info["command"]) > self.STUCK_COMMAND_THRESHOLD
        return jp.where(
            command_active,
            self.FOOT_CLEARANCE_REWARD_SCALE * clearance_reward
            + self.STANCE_CONTACT_REWARD_SCALE * single_support,
            0.0,
        )

    def _get_stance_contact(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kontakt stance noge u aktivnoj gait fazi."""
        left_swing = self._is_left_swing(info)
        foot_contact = self._foot_contact(data)
        return jp.where(left_swing, foot_contact[1], foot_contact[0])

    def _get_swing_contact(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kontakt swing noge; za pravi korak ovo treba cesto da bude nula."""
        left_swing = self._is_left_swing(info)
        foot_contact = self._foot_contact(data)
        return jp.where(left_swing, foot_contact[0], foot_contact[1])

    def _get_double_contact(self, data: mjx.Data) -> jax.Array:
        """Da li su oba stopala istovremeno u kontaktu sa podom."""
        foot_contact = self._foot_contact(data)
        return foot_contact[0] * foot_contact[1]

    def _get_locomotion_quality(self, data: mjx.Data, info: dict) -> jax.Array:
        """Meki lift-then-step gate koji ne ubija signal pre prvog koraka."""
        command_active = jp.linalg.norm(info["command"]) > self.STUCK_COMMAND_THRESHOLD
        stance_contact = self._get_stance_contact(data, info)
        swing_contact = self._get_swing_contact(data, info)
        single_support = stance_contact * (1.0 - swing_contact)
        clearance_quality = jp.clip(
            self._get_swing_clearance(data, info) / self.FOOT_CLEARANCE_TARGET,
            0.0,
            1.0,
        )
        quality = (
            self.LOCOMOTION_QUALITY_CLEARANCE_WEIGHT * clearance_quality
            + self.LOCOMOTION_QUALITY_SINGLE_SUPPORT_WEIGHT * single_support
        )
        return jp.where(command_active, quality, 1.0)

    def _get_double_support_drag_cost(
        self,
        data: mjx.Data,
        info: dict,
        measured_command: jax.Array,
        command_norm: jax.Array,
    ) -> jax.Array:
        """Kazni command tracking dok obe noge klize/vuku pod."""
        command_active = command_norm > self.STUCK_COMMAND_THRESHOLD
        commanded_axis = info["command"] / jp.maximum(command_norm, 1e-6)
        velocity_along_command = jp.dot(measured_command, commanded_axis)
        normalized_speed = jp.maximum(velocity_along_command, 0.0) / jp.maximum(
            command_norm,
            0.05,
        )
        return jp.where(
            command_active,
            self._get_double_contact(data) * jp.square(normalized_speed),
            0.0,
        )

    def _get_swing_foot_drag_cost(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kazni kada swing noga ostane zalepljena za pod umesto da se podigne."""
        left_swing = self._is_left_swing(info)
        foot_contact = self._foot_contact(data)
        swing_contact = jp.where(left_swing, foot_contact[0], foot_contact[1])
        command_active = jp.linalg.norm(info["command"]) > self.STUCK_COMMAND_THRESHOLD
        return jp.where(command_active, swing_contact, 0.0)

    def _get_swing_clearance_deficit_cost(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Kazni swing fazu ako stopalo nije stvarno odignuto od stance noge."""
        clearance = self._get_swing_clearance(data, info)
        deficit = jp.maximum(self.FOOT_CLEARANCE_TARGET - clearance, 0.0)
        normalized_deficit = deficit / self.FOOT_CLEARANCE_TARGET
        command_active = jp.linalg.norm(info["command"]) > self.STUCK_COMMAND_THRESHOLD
        return jp.where(command_active, jp.square(normalized_deficit), 0.0)

    def _get_swing_clearance(self, data: mjx.Data, info: dict) -> jax.Array:
        """Relativna visina swing stopala iznad stance stopala."""
        left_swing = self._is_left_swing(info)
        foot_heights = self._foot_heights(data)
        swing_foot_z = jp.where(left_swing, foot_heights[0], foot_heights[1])
        stance_foot_z = jp.where(left_swing, foot_heights[1], foot_heights[0])
        return swing_foot_z - stance_foot_z

    def _is_left_swing(self, info: dict) -> jax.Array:
        """Za BVH koristi referentnu fleksiju kolena, ne nezavisni sine clock."""
        if self._config.get("reference_gait", "none") == "bvh":
            target_qpos = self._get_bvh_reference_gait_target(info)
            left_knee = target_qpos[self._left_knee_actuator_index]
            right_knee = target_qpos[self._right_knee_actuator_index]
            return left_knee < right_knee
        phase_angle = self._get_gait_phase_angle(info)
        return jp.sin(phase_angle) > 0.0

    def _get_reference_gait_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Nagradi blizinu rucno dizajniranoj sinusoidalnoj gait putanji."""
        reference_gait = self._config.get("reference_gait", "none")
        if reference_gait == "none":
            return jp.array(0.0)
        if reference_gait not in ("sine", "bvh"):
            raise ValueError("reference_gait mora biti 'none', 'sine' ili 'bvh'.")

        target_qpos = self._get_reference_gait_target(info)
        current_qpos = data.qpos[self._actuator_qpos_indices]
        qpos_error = jp.where(
            self._reference_gait_mask,
            current_qpos - target_qpos,
            0.0,
        )
        active_joint_count = jp.maximum(jp.sum(self._reference_gait_mask), 1.0)
        pose_error = jp.sum(jp.square(qpos_error)) / active_joint_count
        pose_reward = jp.exp(-self.REFERENCE_GAIT_ERROR_SCALE * pose_error)
        if reference_gait == "bvh":
            return pose_reward
        command_active = jp.linalg.norm(info["command"]) > self.STUCK_COMMAND_THRESHOLD
        return jp.where(command_active, pose_reward, 0.0)

    def _get_reference_velocity_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Nagradi BVH-like joint brzine, ne samo staticku pozu."""
        if self._config.get("reference_gait", "none") != "bvh":
            return jp.array(0.0)
        target_qvel = self._get_bvh_reference_velocity_target(info)
        current_qvel = data.qvel[self._actuator_dof_indices]
        qvel_error = jp.where(
            self._reference_gait_mask,
            current_qvel - target_qvel,
            0.0,
        )
        active_joint_count = jp.maximum(jp.sum(self._reference_gait_mask), 1.0)
        velocity_error = jp.sum(jp.square(qvel_error)) / active_joint_count
        velocity_reward = jp.exp(
            -self.REFERENCE_VELOCITY_ERROR_SCALE * velocity_error
        )
        return velocity_reward

    def _get_reference_foot_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """DeepMimic-style target za relativne pozicije stopala."""
        if self._config.get("reference_gait", "none") != "bvh":
            return jp.array(0.0)
        target_foot_positions = self._get_bvh_reference_foot_position_target(info)
        current_foot_positions = self._relative_foot_positions(data)
        foot_error = jp.mean(
            jp.square(current_foot_positions - target_foot_positions)
        )
        foot_reward = jp.exp(-self.REFERENCE_FOOT_ERROR_SCALE * foot_error)
        return foot_reward

    def _get_reference_root_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Root/COM-style target: pelvis height plus command-scaled BVH speed."""
        if self._config.get("reference_gait", "none") != "bvh":
            return jp.array(0.0)
        target_height = self._get_bvh_reference_root_height_target(info)
        height_error = jp.square(data.qpos[2] - target_height)
        height_reward = jp.exp(
            -self.REFERENCE_ROOT_HEIGHT_ERROR_SCALE * height_error
        )
        target_forward_velocity = self._get_bvh_reference_root_velocity_target(info)
        forward_velocity_error = jp.square(
            self._measured_command(data)[0] - target_forward_velocity
        )
        velocity_reward = jp.exp(
            -self.REFERENCE_ROOT_VELOCITY_ERROR_SCALE * forward_velocity_error
        )
        root_reward = 0.5 * (height_reward + velocity_reward)
        return root_reward

    def _get_reference_gait_target(self, info: dict) -> jax.Array:
        """Vrati target pozu za aktivni reference gait izvor."""
        if self._config.get("reference_gait", "none") == "bvh":
            return self._get_bvh_reference_gait_target(info)
        return self._get_sine_reference_gait_target(info)

    def _get_sine_reference_gait_target(self, info: dict) -> jax.Array:
        """Vrati ciljnu cyclic pozu nogu za trenutnu gait fazu."""
        phase_angle = self._get_gait_phase_angle(info)
        sin_phase = jp.sin(phase_angle)
        offset = (
            self._reference_gait_sin_offsets * sin_phase
            + self._reference_gait_pos_sin_offsets * jp.maximum(sin_phase, 0.0)
            + self._reference_gait_neg_sin_offsets * jp.maximum(-sin_phase, 0.0)
        )
        target_qpos = self._default_ctrl + offset
        return jp.clip(
            target_qpos,
            self._actuator_qpos_lower_limits,
            self._actuator_qpos_upper_limits,
        )

    def _get_bvh_reference_gait_target(self, info: dict) -> jax.Array:
        """Vrati retargetovanu BVH pozu najblizu trenutnom env vremenu."""
        clip_id, frame_index = self._get_bvh_reference_frame(info)
        return self._bvh_reference_qpos_targets[clip_id, frame_index]

    def _get_bvh_reference_velocity_target(self, info: dict) -> jax.Array:
        """Vrati retargetovanu BVH joint brzinu za trenutni frame."""
        clip_id, frame_index = self._get_bvh_reference_frame(info)
        return self._bvh_reference_qvel_targets[clip_id, frame_index]

    def _get_bvh_reference_foot_position_target(self, info: dict) -> jax.Array:
        """Vrati trunk-relative target pozicije stopala za trenutni BVH frame."""
        clip_id, frame_index = self._get_bvh_reference_frame(info)
        return self._bvh_reference_foot_pos_targets[clip_id, frame_index]

    def _get_bvh_reference_root_height_target(self, info: dict) -> jax.Array:
        """Vrati root height target iz BVH vertikalne oscilacije."""
        clip_id, frame_index = self._get_bvh_reference_frame(info)
        return (
            self._init_q[2]
            + self._bvh_reference_root_height_offsets[clip_id, frame_index]
        )

    def _get_bvh_reference_root_velocity_target(self, info: dict) -> jax.Array:
        """Vrati command-scaled root forward velocity za trenutni BVH frame."""
        clip_id, frame_index = self._get_bvh_reference_frame(info)
        return self._get_bvh_reference_root_velocity_target_by_frame(
            clip_id,
            frame_index,
            info["command"],
        )

    def _get_bvh_reference_root_velocity_target_by_frame(
        self,
        clip_id: jax.Array,
        frame_index: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        """Skalira BVH speed profil na trenutnu joystick forward komandu."""
        speed_factor = self._bvh_reference_root_velocity_factors[
            clip_id,
            frame_index,
        ]
        return command[0] * speed_factor

    def _get_bvh_reference_frame(self, info: dict) -> tuple[jax.Array, jax.Array]:
        """Vrati aktivni BVH clip i frame indeks."""
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        frame_count = self._bvh_reference_frame_counts[clip_id]
        frame_time = self._bvh_reference_frame_times[clip_id]
        frame_float = (
            info["bvh_reference_frame_offset"].astype(jp.float32)
            + (
                info["step"].astype(jp.float32)
                * jp.array(self.dt, dtype=jp.float32)
                / frame_time
            )
        )
        frame_index = jp.mod(jp.floor(frame_float).astype(jp.int32), frame_count)
        return clip_id, frame_index

    def _get_variable_posture_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Reward za Unitree-style pose prior sa tolerancijom koja zavisi od hoda."""
        return jp.exp(-0.5 * self._get_variable_posture_error(data, info))

    def _get_variable_posture_error(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Normalizovana greska poze: kruto dok stoji, labavije dok hoda."""
        actuator_position_error = (
            data.qpos[self._actuator_qpos_indices] - self._default_ctrl
        )
        posture_std = self._get_variable_posture_std(info)
        normalized_error = actuator_position_error / jp.maximum(posture_std, 1e-6)
        return jp.mean(jp.square(normalized_error))

    def _get_variable_posture_std(self, info: dict) -> jax.Array:
        """Interpolira standing/walking joint tolerancije iz trenutnog command-a."""
        command_speed = (
            jp.linalg.norm(info["command"][:2])
            + 0.25 * jp.abs(info["command"][2])
        )
        walking_alpha = jp.clip((command_speed - 0.05) / 0.35, 0.0, 1.0)
        return (
            (1.0 - walking_alpha) * self._posture_std_standing
            + walking_alpha * self._posture_std_walking
        )

    def _get_base_height_reward(self, data: mjx.Data) -> jax.Array:
        """Mala gusta nagrada za drzanje root visine blizu pocetne stojece poze."""
        height_error = (data.qpos[2] - self._init_q[2]) / 0.15
        return jp.exp(-jp.square(height_error))

    def _foot_positions(self, data: mjx.Data) -> jax.Array:
        """World pozicije oba djona, redosled: levo, desno."""
        return data.geom_xpos[self._foot_geom_ids]

    def _relative_foot_positions(self, data: mjx.Data) -> jax.Array:
        """Pozicije stopala u lokalnom frame-u toraksa, za mimic reward."""
        torso_pos = data.xpos[self._torso_body_id]
        torso_xmat = self._body_xmat(data)
        return (self._foot_positions(data) - torso_pos) @ torso_xmat

    def _foot_xy(self, data: mjx.Data) -> jax.Array:
        """World XY pozicije oba djona za slip procenu."""
        return self._foot_positions(data)[:, :2]

    def _foot_heights(self, data: mjx.Data) -> jax.Array:
        """World Z visine centara oba foot-sole geom-a."""
        return self._foot_positions(data)[:, 2]

    def _foot_contact(self, data: mjx.Data) -> jax.Array:
        """Kontakt stopala sa podom iz MuJoCo contact parova, ne samo iz visine."""
        contact = self._floor_contact_mask(
            data,
            self._foot_geom_ids,
            self.FOOT_CONTACT_DISTANCE,
        )
        return contact.astype(jp.float32)

    def _floor_contact_mask(
        self,
        data: mjx.Data,
        geom_ids: jax.Array,
        distance_threshold: float,
    ) -> jax.Array:
        """Vrati bool masku koja kaze koji zadati geom-ovi diraju terrain."""
        contact_geom = data.contact.geom
        geom_a = contact_geom[:, 0]
        geom_b = contact_geom[:, 1]
        valid_contact = data.contact.dist <= distance_threshold
        geom_matches_a = geom_a[None, :] == geom_ids[:, None]
        geom_matches_b = geom_b[None, :] == geom_ids[:, None]
        floor_contact = (
            (geom_matches_a & (geom_b[None, :] == self._floor_geom_id))
            | (geom_matches_b & (geom_a[None, :] == self._floor_geom_id))
        )
        return jp.any(floor_contact & valid_contact[None, :], axis=1)

    def _get_illegal_contact(self, data: mjx.Data) -> jax.Array:
        """Unitree-style hip/knee/shank floor contact penalty signal."""
        contact = self._floor_contact_mask(
            data,
            self._illegal_contact_geom_ids,
            self.ILLEGAL_CONTACT_DISTANCE,
        )
        return jp.any(contact).astype(jp.float32)

    def _get_fatal_illegal_contact(self, data: mjx.Data) -> jax.Array:
        """Pelvis floor contact zavrsava epizodu, kao Unitree pelvis termination."""
        contact = self._floor_contact_mask(
            data,
            self._fatal_contact_geom_ids,
            self.ILLEGAL_CONTACT_DISTANCE,
        )
        return jp.any(contact)

    def _get_soft_joint_limit_violation(self, data: mjx.Data) -> jax.Array:
        """Penalizuj ulazak u spoljasnjih 10% actuator joint opsega."""
        qpos = data.qpos[self._actuator_qpos_indices]
        lower_violation = jp.maximum(
            self._soft_actuator_qpos_lower_limits - qpos,
            0.0,
        )
        upper_violation = jp.maximum(
            qpos - self._soft_actuator_qpos_upper_limits,
            0.0,
        )
        return jp.sum(lower_violation + upper_violation)

    def _get_foot_slip_cost(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kazni horizontalno klizanje stopala dok je stopalo u kontaktu."""
        foot_velocity_xy = (self._foot_xy(data) - info["last_foot_xy"]) / self.dt
        slip_speed = jp.linalg.norm(foot_velocity_xy, axis=1)
        slip_speed = jp.maximum(slip_speed - self.FOOT_SLIP_FREE_SPEED, 0.0)
        return jp.sum(jp.square(slip_speed) * self._foot_contact(data))

    def _get_contact_force_cost(self, data: mjx.Data) -> jax.Array:
        """Gymnasium-style mala kazna za prevelike spoljne kontakt sile."""
        force_cost = jp.sum(jp.square(data.cfrc_ext))
        force_cost = jp.clip(force_cost, 0.0, self.CONTACT_FORCE_COST_CLIP)
        return self.CONTACT_FORCE_COST_SCALE * force_cost

    def _body_xmat(self, data: mjx.Data) -> jax.Array:
        """Vraca 3x3 rotaciju toraksa iz body-local u world frame."""
        return data.xmat[self._torso_body_id].reshape((3, 3))

    def _head_xmat(self, data: mjx.Data) -> jax.Array:
        """Vraca 3x3 rotaciju glave iz body-local u world frame."""
        return data.xmat[self._head_body_id].reshape((3, 3))

    def _local_root_linvel(self, data: mjx.Data) -> jax.Array:
        """Root linearna brzina izrazena u lokalnom frame-u toraksa."""
        return self._body_xmat(data).T @ data.qvel[:3]

    def _local_root_angvel(self, data: mjx.Data) -> jax.Array:
        """Root angularna brzina izrazena u lokalnom frame-u toraksa."""
        return self._body_xmat(data).T @ data.qvel[3:6]

    def _projected_gravity(self, data: mjx.Data) -> jax.Array:
        """Gravitacija u lokalnom frame-u; policy iz toga vidi nagib tela."""
        return self._body_xmat(data).T @ self.WORLD_GRAVITY

    def _torso_up(self, data: mjx.Data) -> jax.Array:
        """Koliko je anatomska vertikalna osa toraksa poravnata sa world Z."""
        return self._body_xmat(data)[2, 1]

    def _head_up(self, data: mjx.Data) -> jax.Array:
        """Koliko je glava uspravna u odnosu na world Z."""
        return self._head_xmat(data)[2, 1]

    def _get_tracking_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        """Prati joystick u anatomskim osama: X napred, Z lateralno, Y yaw."""
        error = jp.sum(jp.square(info["command"] - self._measured_command(data)))
        return jp.exp(-error / self._config.tracking_sigma)

    def _measured_command(self, data: mjx.Data) -> jax.Array:
        """Izmeri trenutnu brzinu u istim osama kao joystick command."""
        local_linvel = self._local_root_linvel(data)
        local_angvel = self._local_root_angvel(data)
        return jp.array([
            local_linvel[0],
            local_linvel[2],
            local_angvel[1],
        ])

    def _get_command_progress(self, data: mjx.Data, info: dict) -> jax.Array:
        """Nagradi napredak duz trazenog joystick vektora, za sve smerove."""
        command = info["command"]
        command_norm_sq = jp.sum(jp.square(command))
        command_active = command_norm_sq > self.STUCK_COMMAND_THRESHOLD**2
        alignment = jp.dot(self._measured_command(data), command) / jp.maximum(
            command_norm_sq,
            1e-6,
        )
        return jp.where(
            command_active,
            jp.clip(alignment, 0.0, 1.0) * self._get_tracking_reward(data, info),
            0.0,
        )

    @property
    def xml_path(self) -> str:
        return str(self._xml_path)

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def n_substeps(self) -> int:
        return self._n_substeps

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def randomize_single_model(model, key):
    """Randomizuje numericke parametre za jedan paralelni env."""
    height_key, mass_key, friction_key = jax.random.split(key, 3)

    height_scale = jax.random.uniform(
        height_key,
        minval=1.55 / 1.80,
        maxval=1.95 / 1.80,
    )
    mass_scale = jax.random.uniform(
        mass_key,
        minval=55.0 / 75.0,
        maxval=95.0 / 75.0,
    )

    body_pos = model.body_pos.at[1:].set(model.body_pos[1:] * height_scale)
    body_geom_mask = model.geom_bodyid > 0
    geom_pos = jp.where(
        body_geom_mask[:, None],
        model.geom_pos * height_scale,
        model.geom_pos,
    )
    geom_size = jp.where(
        body_geom_mask[:, None],
        model.geom_size * height_scale,
        model.geom_size,
    )
    site_pos = model.site_pos.at[:].set(model.site_pos * height_scale)
    qpos0 = model.qpos0.at[2].set(model.qpos0[2] * height_scale)

    body_mass = model.body_mass.at[1:].set(model.body_mass[1:] * mass_scale)
    body_inertia = model.body_inertia.at[1:].set(
        model.body_inertia[1:] * mass_scale * height_scale**2
    )
    geom_friction = model.geom_friction.at[0, 0].set(
        jax.random.uniform(friction_key, minval=0.5, maxval=1.1)
    )
    return (
        body_pos,
        geom_pos,
        geom_size,
        site_pos,
        qpos0,
        body_mass,
        body_inertia,
        geom_friction,
    )


# Pre-jit the vmap for domain randomization
_randomize_vmap = jax.jit(jax.vmap(randomize_single_model, in_axes=(None, 0)))


def domain_randomize(model, rng):
    """Randomizuje human velicinu i masu direktno u MJX model arrays (optimizovano sa JIT).

    Ovo je deo koji sprecava generisanje novog XML-a po epizodi: topologija
    ostaje ista, a Brax/MJX dobija razlicite numericke modele po env-u.
    """
    randomized = _randomize_vmap(model, rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({
        "body_pos": 0,
        "geom_pos": 0,
        "geom_size": 0,
        "site_pos": 0,
        "qpos0": 0,
        "body_mass": 0,
        "body_inertia": 0,
        "geom_friction": 0,
    })

    randomized_model = model.tree_replace({
        "body_pos": randomized[0],
        "geom_pos": randomized[1],
        "geom_size": randomized[2],
        "site_pos": randomized[3],
        "qpos0": randomized[4],
        "body_mass": randomized[5],
        "body_inertia": randomized[6],
        "geom_friction": randomized[7],
    })
    return randomized_model, in_axes
