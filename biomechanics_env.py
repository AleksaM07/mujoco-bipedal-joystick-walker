import re
from dataclasses import replace
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from bvh_reference import BvhReferenceBatch, LoopMode, load_bvh_references
from biomechanics_model import (
    HumanSpec,
    LEG_ACTUATED_JOINTS,
    TRUNK_ACTUATED_JOINTS,
    build_trainable_scene_xml,
)
from config import (
    ARM_ACTUATED_JOINTS,
    DEFAULT_BVH_REFERENCE_LIST,
    default_biomechanics_env_config,
    resolve_project_path,
)
from phase1_backends import make_data_kwargs, put_model_for_backend, select_data
from smpl_reference import load_smpl_references


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
        for value in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            "\n".join(qpos_lines),
        )
    ]
    if len(values) != expected_size:
        raise ValueError(
            f"{qpos_path} ima {len(values)} QPOS vrednosti, "
            f"a model ocekuje nq={expected_size}."
        )
    return np.asarray(values, dtype=np.float64)

class BiomechanicsJoystickEnv(mjx_env.MjxEnv):
    """Joystick locomotion env za humanoida iz `mujoco-biomechanics`."""

    WORLD_GRAVITY = jp.array([0.0, 0.0, -1.0])

    # REF: BIOHUMANOID-CONTACT-THRESHOLDS
    # TYPE: MODEL_CALIBRATED
    FOOT_SOLE_GEOMS = ("left_foot_sole", "right_foot_sole")
    FOOT_CONTACT_PRELOAD = 0.005
    # Metatarsal key sites in the generated no-arms XML sit at about 0.047 m in
    # the floor-aligned neutral pose.  A higher IK floor silently asks the BVH
    # retargeter to keep stance feet airborne, which breaks the contact oracle.
    FOOT_CONTACT_HEIGHT = 0.05
    FOOT_CONTACT_DISTANCE = 0.01
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
    ARM_ACTION_SCALE = {
        "left_shoulder_x": 0.35,
        "left_shoulder_y": 0.22,
        "left_shoulder_z": 0.35,
        "left_elbow_z": 0.35,
        "right_shoulder_x": 0.35,
        "right_shoulder_y": 0.22,
        "right_shoulder_z": 0.35,
        "right_elbow_z": 0.35,
    }
    POSTURE_STD_STANDING = {
        "trunk": 0.06,
        "hip_stride": 0.08,
        "knee": 0.10,
        "ankle_pitch": 0.07,
        "hip_lateral": 0.06,
        "ankle_lateral": 0.05,
        "arm": 0.18,
    }
    POSTURE_STD_WALKING = {
        "trunk": 0.09,
        "hip_stride": 0.38,
        "knee": 0.55,
        "ankle_pitch": 0.24,
        "hip_lateral": 0.14,
        "ankle_lateral": 0.08,
        "arm": 0.55,
    }
    INIT_TRUNK_NOISE = 0.005
    INIT_LEG_NOISE = 0.02
    # REF: BIOHUMANOID-FALL-HEIGHT
    # TYPE: MODEL_CALIBRATED
    HEIGHT_PENALTY_START_RATIO = 0.9
    MIN_STANDING_HEIGHT_RATIO = 0.30
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
    VERTICAL_VELOCITY_COST_SCALE = 0.05
    ANGULAR_VELOCITY_COST_SCALE = 0.02
    GAIT_PERIOD_STEPS = 50
    FOOT_CLEARANCE_TARGET = 0.08
    FOOT_CLEARANCE_REWARD_SCALE = 1.1
    STANCE_CONTACT_REWARD_SCALE = 0.3
    FOOT_SLIP_FREE_SPEED = 0.03
    FOOT_SLIP_COST_SCALE = 1.0
    SWING_FOOT_DRAG_COST_SCALE = 2.0
    SWING_CLEARANCE_DEFICIT_COST_SCALE = 1.5
    # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
    # TYPE: REFERENCE_CODE_DERIVED
    # The BVH/DeepMimic term is already a complete weighted imitation reward.
    REFERENCE_GAIT_REWARD_SCALE = 1.0
    REFERENCE_GAIT_ERROR_SCALE = 8.0
    REFERENCE_VELOCITY_REWARD_SCALE = 0.0
    REFERENCE_VELOCITY_ERROR_SCALE = 0.25
    # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
    # TYPE: REFERENCE_CODE_DERIVED
    # Values mirror MimicKit data/envs/deepmimic_humanoid_env.yaml.
    DEEPMIMIC_POSE_WEIGHT = 0.50
    DEEPMIMIC_VELOCITY_WEIGHT = 0.10
    DEEPMIMIC_ROOT_POSE_WEIGHT = 0.15
    DEEPMIMIC_ROOT_VELOCITY_WEIGHT = 0.10
    DEEPMIMIC_KEY_POSITION_WEIGHT = 0.15
    DEEPMIMIC_POSE_SCALE = 0.25
    DEEPMIMIC_VELOCITY_SCALE = 0.01
    DEEPMIMIC_ROOT_POSE_SCALE = 5.0
    DEEPMIMIC_ROOT_VELOCITY_SCALE = 1.0
    DEEPMIMIC_KEY_POSITION_SCALE = 10.0
    # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
    # TYPE: REFERENCE_CODE_DERIVED
    # Mirrors MimicKit deepmimic_humanoid_env.yaml pose_termination_dist.
    POSE_TERMINATION_DIST = 1.0
    RESET_SAMPLE_ATTEMPTS = 8
    RESET_PROJECTION_LEVELS = (1.0, 0.7, 0.45, 0.25)
    REFERENCE_ROOT_HEIGHT_MAX_SPEED = 0.75
    REFERENCE_RETARGET_ROOT_XY_SCALE = 0.35
    REFERENCE_RETARGET_VELOCITY_SCALE = 0.25
    CONTACT_FORCE_COST_SCALE = 1e-4
    CONTACT_FORCE_COST_CLIP = 1000.0
    STUCK_COMMAND_THRESHOLD = 0.10
    STUCK_VELOCITY_THRESHOLD = 0.05
    STUCK_PENALTY = 1.0
    LOW_HEIGHT_COST_SCALE = 20.0
    REWARD_MIN = -5.0
    REWARD_MAX = 3.0
    FALL_REWARD = -25.0

    def __init__(
        self,
        env_version: str = "standard",
        human_spec: HumanSpec = HumanSpec(),
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict | None = None,
    ) -> None:
        if config is None:
            config = default_biomechanics_env_config()
        super().__init__(config, config_overrides)
        configured_xml_path = self._config.get("xml_path", None)
        if configured_xml_path:
            self._xml_path = resolve_project_path(configured_xml_path)
        else:
            self._xml_path = build_trainable_scene_xml(
                env_version,
                human_spec,
                arm_actuators=bool(self._config.get("arm_actuators", False)),
            )
        self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._mj_model.opt.timestep = self._sim_dt
        init_q = np.array(self._mj_model.keyframe("a-pose").qpos, copy=True)
        init_q = self._build_initial_qpos(init_q)
        # REF: BULLET-WARP-BACKEND
        # TYPE: REFERENCE_CODE_DERIVED
        self._physics_backend = self._config.get(
            "physics_backend",
            "mjx_warp" if self._config.impl == "warp" else "mjx_jax",
        )
        self._data_kwargs = make_data_kwargs(
            self._physics_backend,
            self._config.get("warp_naconmax", None),
            self._config.get("warp_njmax", None),
        )
        self._mjx_model = put_model_for_backend(
            self._mj_model,
            self._physics_backend,
            self._config.get("warp_graph_mode", "warp"),
        )
        self._init_q_np = np.asarray(init_q, dtype=np.float64)
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
        supported_joint_orders = (
            TRUNK_ACTUATED_JOINTS + LEG_ACTUATED_JOINTS + ARM_ACTUATED_JOINTS,
            TRUNK_ACTUATED_JOINTS + LEG_ACTUATED_JOINTS,
            LEG_ACTUATED_JOINTS,
        )
        if actuator_joint_names not in supported_joint_orders:
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
        self._reset_pose_projection_weights = jp.array([
            self._reset_pose_projection_weight(joint_name)
            for joint_name in actuator_joint_names
        ])
        self._reset_velocity_projection_weights = jp.array([
            self._reset_velocity_projection_weight(joint_name)
            for joint_name in actuator_joint_names
        ])
        self._torque_injection_scale = jp.array([
            0.2 if joint_name in TRUNK_ACTUATED_JOINTS else 1.0
            for joint_name in actuator_joint_names
        ])
        if self._config.get("reference_gait", "none") in ("bvh", "smpl"):
            # DeepMimic/MimicKit references cover the whole controlled DOF set.
            # For our BVH bridge this is also safer: unmasked joints still get
            # explicit default/retarget targets instead of silently drifting.
            reference_gait_names = set(actuator_joint_names)
        else:
            reference_gait_names = {
                "left_hip_x",
                "right_hip_x",
                "left_knee_z",
                "right_knee_z",
                "left_ankle_y",
                "right_ankle_y",
            }
        self._reference_gait_mask = jp.array([
            joint_name in reference_gait_names
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
        self._actuator_ctrl_lower_limits_np = np.array([
            self._mj_model.actuator_ctrlrange[actuator_id, 0]
            if self._mj_model.actuator_ctrllimited[actuator_id]
            else self._actuator_qpos_lower_limits_np[actuator_id]
            for actuator_id in range(self._mj_model.nu)
        ])
        self._actuator_ctrl_upper_limits_np = np.array([
            self._mj_model.actuator_ctrlrange[actuator_id, 1]
            if self._mj_model.actuator_ctrllimited[actuator_id]
            else self._actuator_qpos_upper_limits_np[actuator_id]
            for actuator_id in range(self._mj_model.nu)
        ])
        self._actuator_ctrl_lower_limits = jp.array(
            self._actuator_ctrl_lower_limits_np
        )
        self._actuator_ctrl_upper_limits = jp.array(
            self._actuator_ctrl_upper_limits_np
        )
        self._default_ctrl = self._init_q[self._actuator_qpos_indices]
        self._configure_mimickit_action_bounds()
        self._n_substeps = int(round(self._ctrl_dt / self._sim_dt))
        self._torso_body_id = self._mj_model.body("thorax").id
        self._head_body_id = self._mj_model.body("head").id
        try:
            self._reference_anchor_body_id = self._mj_model.body("pelvis").id
        except KeyError:
            self._reference_anchor_body_id = self._torso_body_id
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._left_foot_sole_geom_id = self._mj_model.geom("left_foot_sole").id
        self._right_foot_sole_geom_id = self._mj_model.geom("right_foot_sole").id
        self._foot_geom_ids_np = np.array([
            self._left_foot_sole_geom_id,
            self._right_foot_sole_geom_id,
        ], dtype=np.int32)
        self._foot_geom_ids = jp.array(self._foot_geom_ids_np)
        self._foot_geom_sizes = jp.array(
            self._mj_model.geom_size[self._foot_geom_ids_np],
            dtype=jp.float32,
        )
        self._foot_geom_types = jp.array(
            self._mj_model.geom_type[self._foot_geom_ids_np],
            dtype=jp.int32,
        )
        self._deepmimic_body_ids_np = np.arange(1, self._mj_model.nbody)
        self._deepmimic_body_ids = jp.array(self._deepmimic_body_ids_np)
        anchor_local_indices = np.where(
            self._deepmimic_body_ids_np == self._reference_anchor_body_id
        )[0]
        self._reference_anchor_body_local_index = int(
            anchor_local_indices[0] if anchor_local_indices.size else 0
        )
        body_mass = np.asarray(self._mj_model.body_mass)[self._deepmimic_body_ids_np]
        self._deepmimic_body_weights_np = body_mass / max(float(body_mass.sum()), 1e-6)
        self._deepmimic_body_weights = jp.array(self._deepmimic_body_weights_np)
        (
            self._deepmimic_key_body_ids_np,
            self._deepmimic_key_site_ids_np,
            self._deepmimic_key_is_site_np,
        ) = self._resolve_deepmimic_key_marker_ids()
        self._deepmimic_key_body_ids = jp.array(self._deepmimic_key_body_ids_np)
        self._deepmimic_key_site_ids = jp.array(self._deepmimic_key_site_ids_np)
        self._deepmimic_key_is_site = jp.array(self._deepmimic_key_is_site_np)
        self._deepmimic_key_count = len(self._deepmimic_key_is_site_np)
        self._bvh_target_observation_steps = tuple(
            int(step)
            for step in self._config.get(
                "bvh_target_observation_steps",
                (0, 1, 2, 3),
            )
        )
        if not self._bvh_target_observation_steps:
            self._bvh_target_observation_steps = (0,)
        self._reference_target_observation_mode = "deepmimic"
        self._bvh_reference_qpos_targets = jp.expand_dims(
            jp.expand_dims(self._default_ctrl, axis=0),
            axis=0,
        )
        self._bvh_reference_qvel_targets = jp.zeros_like(
            self._bvh_reference_qpos_targets
        )
        self._bvh_reference_frame_times = jp.array([self.dt], dtype=jp.float32)
        self._bvh_reference_frame_counts = jp.array([1], dtype=jp.int32)
        self._bvh_reference_motion_lengths = jp.array([self.dt], dtype=jp.float32)
        self._bvh_reference_loop_modes = jp.array(
            [int(LoopMode.CLAMP)],
            dtype=jp.int32,
        )
        self._bvh_reference_weights = jp.array([1.0], dtype=jp.float32)
        self._bvh_reference_clip_count = 1
        self._bvh_reference_sim_root_pos_targets_np = np.expand_dims(
            np.expand_dims(np.asarray(self._init_q_np[:3], dtype=np.float32), axis=0),
            axis=0,
        )
        self._bvh_reference_sim_root_quat_targets_np = np.expand_dims(
            np.expand_dims(np.asarray(self._init_q_np[3:7], dtype=np.float32), axis=0),
            axis=0,
        )
        self._bvh_reference_sim_root_vel_targets_np = np.zeros((1, 1, 3), dtype=np.float32)
        self._bvh_reference_sim_root_angvel_targets_np = np.zeros(
            (1, 1, 3),
            dtype=np.float32,
        )
        self._bvh_reference_reset_root_pos_targets = jp.array(
            self._bvh_reference_sim_root_pos_targets_np
        )
        self._bvh_reference_reset_root_quat_targets = jp.array(
            self._bvh_reference_sim_root_quat_targets_np
        )
        self._bvh_reference_reset_root_vel_targets = jp.array(
            self._bvh_reference_sim_root_vel_targets_np
        )
        self._bvh_reference_reset_root_angvel_targets = jp.array(
            self._bvh_reference_sim_root_angvel_targets_np
        )
        self._bvh_reference_root_pos_targets_np = self._bvh_reference_sim_root_pos_targets_np
        self._bvh_reference_root_quat_targets_np = self._bvh_reference_sim_root_quat_targets_np
        self._bvh_reference_root_vel_targets_np = self._bvh_reference_sim_root_vel_targets_np
        self._bvh_reference_root_angvel_targets_np = self._bvh_reference_sim_root_angvel_targets_np
        self._bvh_reference_wrap_deltas_np = np.zeros((1, 3), dtype=np.float32)
        self._configure_default_deepmimic_reference()
        self._cache_standing_reference()
        self._configure_bvh_reference()
        self._refresh_reset_data_template()
        self._configure_policy_observation_layout()

    def _refresh_reset_data_template(self) -> None:
        """Allocate reset Data outside vmap/jit using the host MuJoCo model."""
        # REF: BULLET-WARP-BACKEND
        # TYPE: REFERENCE_CODE_DERIVED
        self._reset_data_template = mjx.make_data(
            self._mj_model,
            **self._data_kwargs,
        ).replace(ctrl=self._default_ctrl)

    def _fresh_reset_data(self) -> mjx.Data:
        """Return pristine reset Data without allocating inside JAX transforms."""
        return self._reset_data_template.replace(ctrl=self._default_ctrl)

    def _configure_policy_observation_layout(self) -> None:
        """Reconstruct the policy observation layout saved in a checkpoint."""
        self._include_gait_phase_observation = True
        self._include_reference_target_observation = bool(
            self._config.get("reference_target_observation", False)
        )

        expected_size = self._config.get("policy_observation_size", None)
        if expected_size is None:
            return

        base_size = (
            12
            + (self._mj_model.nq - 7)
            + (self._mj_model.nv - 6)
            + self.action_size
        )
        optional_size = int(expected_size) - base_size
        layouts = {
            0: (False, False, "deepmimic"),
            2: (True, False, "deepmimic"),
            self.action_size: (False, True, "legacy"),
            self.action_size + 2: (True, True, "legacy"),
        }
        deepmimic_target_size = self._reference_target_observation_size()
        layouts[deepmimic_target_size] = (False, True, "deepmimic")
        layouts[deepmimic_target_size + 2] = (True, True, "deepmimic")
        if optional_size not in layouts:
            raise ValueError(
                "Checkpoint observation layout ne odgovara izabranom XML-u: "
                f"checkpoint={expected_size}, osnovni_env={base_size}, "
                f"action_size={self.action_size}."
            )

        (
            self._include_gait_phase_observation,
            self._include_reference_target_observation,
            self._reference_target_observation_mode,
        ) = layouts[optional_size]

    def _joint_action_scale(self, joint_name: str) -> float:
        """Unitree-style action prior: stride joints move more than twist joints."""
        global_scale = float(self._config.action_scale) / 0.5
        if joint_name in self.TRUNK_ACTION_SCALE:
            return self.TRUNK_ACTION_SCALE[joint_name] * global_scale
        if self._config.get("legacy_action_prior", False):
            return float(self._config.action_scale)
        if joint_name in self.LEG_ACTION_SCALE:
            return self.LEG_ACTION_SCALE[joint_name] * global_scale
        if joint_name in self.ARM_ACTION_SCALE:
            return self.ARM_ACTION_SCALE[joint_name] * global_scale
        return float(self._config.action_scale)

    @classmethod
    def _reset_pose_projection_weight(cls, joint_name: str) -> float:
        """How much of the retargeted joint offset to preserve at reset time."""
        if joint_name in TRUNK_ACTUATED_JOINTS:
            return 0.0
        if joint_name in {"left_hip_x", "right_hip_x", "left_knee_z", "right_knee_z"}:
            return 1.0
        if joint_name in {"left_ankle_y", "right_ankle_y"}:
            return 0.9
        if joint_name in {"left_hip_z", "right_hip_z"}:
            return 0.25
        if joint_name in {"left_ankle_z", "right_ankle_z"}:
            return 0.15
        if joint_name in {"left_hip_y", "right_hip_y"}:
            return 0.10
        if joint_name in ARM_ACTUATED_JOINTS:
            return 0.20
        return 0.0

    @classmethod
    def _reset_velocity_projection_weight(cls, joint_name: str) -> float:
        """How much retargeted joint velocity to preserve at reset time."""
        if joint_name in TRUNK_ACTUATED_JOINTS:
            return 0.0
        if joint_name in {"left_hip_x", "right_hip_x", "left_knee_z", "right_knee_z"}:
            return 0.15
        if joint_name in {"left_ankle_y", "right_ankle_y"}:
            return 0.10
        if joint_name in ARM_ACTUATED_JOINTS:
            return 0.05
        return 0.0

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
        elif joint_name in ARM_ACTUATED_JOINTS:
            category = "arm"
        else:
            category = "hip_lateral"
        return (
            cls.POSTURE_STD_STANDING[category],
            cls.POSTURE_STD_WALKING[category],
        )

    def _configure_mimickit_action_bounds(
        self,
        reference_qpos_targets: np.ndarray | None = None,
    ) -> None:
        """Build the normalized policy-action to PD-target map.

        MimicKit actions are absolute PD targets. For retargeted SMPL/BVH, the
        action domain must cover the loaded reference envelope; otherwise tanh
        policies hit +-1 while still being unable to command the rewarded pose.
        """
        finite_action_bounds = (
            np.isfinite(self._actuator_ctrl_lower_limits_np)
            & np.isfinite(self._actuator_ctrl_upper_limits_np)
        )
        default_ctrl_np = np.asarray(self._default_ctrl, dtype=np.float32)
        action_midpoint_np = np.where(
            finite_action_bounds,
            0.5
            * (
                self._actuator_ctrl_lower_limits_np
                + self._actuator_ctrl_upper_limits_np
            ),
            default_ctrl_np,
        )
        action_center_mode = self._config.get("reference_action_center", "default")
        if action_center_mode == "joint_midpoint":
            action_center_np = action_midpoint_np.astype(np.float32)
        else:
            action_center_np = default_ctrl_np

        ctrl_half_range_np = np.where(
            finite_action_bounds,
            np.maximum(
                self._actuator_ctrl_upper_limits_np - action_center_np,
                action_center_np - self._actuator_ctrl_lower_limits_np,
            ),
            np.asarray(self._action_scale, dtype=np.float32),
        ).astype(np.float32)
        action_range_mode = self._config.get("reference_action_range", "action_scale")
        if action_range_mode == "joint_limits":
            action_half_range_np = 1.4 * ctrl_half_range_np
        elif action_range_mode == "reference_targets" and reference_qpos_targets is not None:
            reference_delta_np = np.max(
                np.abs(np.asarray(reference_qpos_targets, dtype=np.float32) - action_center_np),
                axis=(0, 1),
            )
            action_half_range_np = np.maximum(
                np.asarray(self._action_scale, dtype=np.float32),
                reference_delta_np,
            )
        else:
            action_half_range_np = np.asarray(self._action_scale, dtype=np.float32)
        action_half_range_np = (
            float(self._config.get("reference_action_range_scale", 1.0))
            * action_half_range_np
        )
        action_half_range_np = np.minimum(action_half_range_np, ctrl_half_range_np)
        action_half_range_np = np.maximum(action_half_range_np, 1e-4)
        self._mimickit_action_center = jp.array(action_center_np)
        self._mimickit_action_half_range = jp.array(action_half_range_np)
        self._mimickit_action_lower_limits = jp.array(
            np.maximum(
                action_center_np - action_half_range_np,
                self._actuator_ctrl_lower_limits_np,
            )
        )
        self._mimickit_action_upper_limits = jp.array(
            np.minimum(
                action_center_np + action_half_range_np,
                self._actuator_ctrl_upper_limits_np,
            )
        )

    def _configure_bvh_reference(self) -> None:
        """Ucita BVH referencu ako je trazena u config-u."""
        reference_gait = self._config.get("reference_gait", "none")
        if reference_gait not in ("bvh", "smpl"):
            return
        reference_gait_files = self._reference_gait_files()

        loader = load_smpl_references if reference_gait == "smpl" else load_bvh_references
        references = loader(
            tuple(resolve_project_path(path) for path in reference_gait_files),
            self._actuator_joint_names,
            np.asarray(self._default_ctrl, dtype=np.float32),
            self._actuator_ctrl_lower_limits_np,
            self._actuator_ctrl_upper_limits_np,
            initial_root_pos=np.asarray(self._init_q_np[:3], dtype=np.float32),
            initial_root_quat=np.asarray(self._init_q_np[3:7], dtype=np.float32),
        )
        references = self._filter_reference_clips(references)
        references = self._apply_reference_loop_mode_override(references)
        self._bvh_reference_frame_times = jp.array(references.frame_times)
        self._bvh_reference_frame_counts = jp.array(references.frame_counts)
        self._bvh_reference_motion_lengths = jp.array(
            np.maximum(
                references.frame_times * np.maximum(references.frame_counts - 1, 1),
                1e-6,
            ),
            dtype=jp.float32,
        )
        self._bvh_reference_clip_count = len(references.source_paths)
        self._bvh_reference_loop_modes = jp.array(references.loop_modes)
        self._bvh_reference_weights = jp.array(
            self._resolve_reference_clip_weights(references.weights),
            dtype=jp.float32,
        )
        self._bvh_reference_source_paths = tuple(
            str(path) for path in references.source_paths
        )
        self._bvh_reference_source_start_frames = np.asarray(
            references.source_start_frames,
            dtype=np.int32,
        )
        self._bvh_reference_source_end_frames = np.asarray(
            references.source_end_frames,
            dtype=np.int32,
        )
        self._bvh_reference_support_feet = tuple(references.support_feet)
        self._bvh_reference_diagnostic_summaries = tuple(
            getattr(references, "diagnostic_summaries", ())
        )
        root_angvel_targets = np.asarray(
            references.root_angvel_targets,
            dtype=np.float32,
        )
        (
            sim_root_pos_targets,
            sim_root_vel_targets,
            sim_wrap_deltas,
        ) = self._align_reference_roots_to_sim_floor(
            references.qpos_targets,
            references.root_pos_targets,
            references.root_quat_targets,
            references.frame_times,
            references.frame_counts,
            references.loop_modes,
        )
        qpos_targets = np.asarray(references.qpos_targets, dtype=np.float32)
        qvel_targets = np.asarray(references.qvel_targets, dtype=np.float32)
        ik_target_positions = getattr(references, "ik_target_positions", None)
        if ik_target_positions is not None:
            root_delta = sim_root_pos_targets - references.root_pos_targets
            aligned_ik_targets = (
                np.asarray(ik_target_positions, dtype=np.float32)
                + root_delta[:, :, None, :]
            )
            aligned_ik_targets = self._normalize_ik_foot_target_heights(
                aligned_ik_targets,
                tuple(getattr(references, "ik_target_names", ())),
                references.frame_counts,
            )
            qpos_targets = self._retarget_reference_qpos_to_ik_targets(
                qpos_targets,
                aligned_ik_targets,
                tuple(getattr(references, "ik_target_names", ())),
                sim_root_pos_targets,
                references.root_quat_targets,
                references.frame_counts,
            )
            qpos_targets = self._apply_reference_stability_prior(qpos_targets)
            sim_root_pos_targets = self._scale_reference_root_xy_motion(
                sim_root_pos_targets,
                references.frame_counts,
            )
            qvel_targets = self._reference_qvel_from_qpos_targets(
                qpos_targets,
                references.frame_times,
                references.frame_counts,
            )
            velocity_scale = float(self.REFERENCE_RETARGET_VELOCITY_SCALE)
            qvel_targets *= velocity_scale
            (
                sim_root_pos_targets,
                sim_root_vel_targets,
                sim_wrap_deltas,
            ) = self._align_reference_roots_to_sim_floor(
                qpos_targets,
                sim_root_pos_targets,
                references.root_quat_targets,
                references.frame_times,
                references.frame_counts,
                references.loop_modes,
            )
            sim_root_vel_targets *= velocity_scale
            root_angvel_targets *= velocity_scale
        else:
            qpos_targets = self._apply_reference_stability_prior(qpos_targets)
            qvel_targets = self._reference_qvel_from_qpos_targets(
                qpos_targets,
                references.frame_times,
                references.frame_counts,
            )
            sim_root_pos_targets = self._scale_reference_root_xy_motion(
                sim_root_pos_targets,
                references.frame_counts,
            )
            (
                sim_root_pos_targets,
                sim_root_vel_targets,
                sim_wrap_deltas,
            ) = self._align_reference_roots_to_sim_floor(
                qpos_targets,
                sim_root_pos_targets,
                references.root_quat_targets,
                references.frame_times,
                references.frame_counts,
                references.loop_modes,
            )
        self._bvh_reference_qpos_targets = jp.array(qpos_targets)
        self._bvh_reference_qvel_targets = jp.array(qvel_targets)
        self._configure_mimickit_action_bounds(qpos_targets)
        self._bvh_reference_sim_root_pos_targets_np = sim_root_pos_targets
        self._bvh_reference_sim_root_quat_targets_np = references.root_quat_targets
        self._bvh_reference_sim_root_vel_targets_np = sim_root_vel_targets
        self._bvh_reference_sim_root_angvel_targets_np = root_angvel_targets
        self._bvh_reference_wrap_deltas_np = sim_wrap_deltas
        self._bvh_reference_reset_root_pos_targets = jp.array(
            sim_root_pos_targets
        )
        self._bvh_reference_reset_root_quat_targets = jp.array(
            references.root_quat_targets
        )
        self._bvh_reference_reset_root_vel_targets = jp.array(
            sim_root_vel_targets
        )
        self._bvh_reference_reset_root_angvel_targets = jp.array(
            root_angvel_targets
        )
        self._configure_deepmimic_reference_from_qpos(
            qpos_targets,
            qvel_targets,
            sim_root_pos_targets,
            references.root_quat_targets,
            sim_root_vel_targets,
            root_angvel_targets,
        )
        self._infer_missing_reference_support_feet()

    def _resolve_reference_clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Apply optional single-clip sampling override and normalize weights."""
        resolved = np.asarray(weights, dtype=np.float32).copy()
        forced_clip_id = self._config.get("reference_forced_clip_id", None)
        if forced_clip_id is not None:
            clip_id = int(forced_clip_id)
            if clip_id < 0 or clip_id >= resolved.shape[0]:
                raise ValueError(
                    "reference_forced_clip_id out of range: "
                    f"{clip_id} for {resolved.shape[0]} loaded clips."
                )
            resolved[:] = 0.0
            resolved[clip_id] = 1.0
            return resolved

        weight_sum = float(resolved.sum())
        if weight_sum > 0.0:
            return resolved / weight_sum
        resolved[:] = 1.0 / max(float(resolved.shape[0]), 1.0)
        return resolved

    def _filter_reference_clips(
        self,
        references: BvhReferenceBatch,
    ) -> BvhReferenceBatch:
        """Filter out pathological short clips from the training reference batch."""
        min_motion_length = float(self._config.get("reference_min_motion_length", 0.0))
        if min_motion_length <= 0.0:
            return references

        frame_counts = np.asarray(references.frame_counts, dtype=np.int32)
        motion_lengths = np.asarray(references.frame_times, dtype=np.float32) * np.maximum(
            frame_counts - 1,
            1,
        )
        keep = motion_lengths >= min_motion_length
        if not np.any(keep):
            return references

        weights = np.asarray(references.weights[keep], dtype=np.float32)
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            weights /= weight_sum
        else:
            weights[:] = 1.0 / float(weights.shape[0])

        keep_indices = np.flatnonzero(keep)
        diagnostic_summaries = tuple(
            references.diagnostic_summaries[index]
            for index in keep_indices
        ) if len(references.diagnostic_summaries) == len(frame_counts) else (
            references.diagnostic_summaries
        )
        ik_target_positions = (
            references.ik_target_positions[keep]
            if references.ik_target_positions is not None
            else None
        )
        return replace(
            references,
            qpos_targets=references.qpos_targets[keep],
            qvel_targets=references.qvel_targets[keep],
            root_pos_targets=references.root_pos_targets[keep],
            root_quat_targets=references.root_quat_targets[keep],
            root_vel_targets=references.root_vel_targets[keep],
            root_angvel_targets=references.root_angvel_targets[keep],
            wrap_deltas=references.wrap_deltas[keep],
            frame_times=references.frame_times[keep],
            frame_counts=references.frame_counts[keep],
            loop_modes=references.loop_modes[keep],
            weights=weights,
            source_paths=tuple(references.source_paths[index] for index in keep_indices),
            source_start_frames=references.source_start_frames[keep],
            source_end_frames=references.source_end_frames[keep],
            support_feet=tuple(references.support_feet[index] for index in keep_indices),
            diagnostic_summaries=diagnostic_summaries,
            ik_target_positions=ik_target_positions,
        )

    def _apply_reference_loop_mode_override(
        self,
        references: BvhReferenceBatch,
    ) -> BvhReferenceBatch:
        """Optionally force finite/looping playback for inspected references."""
        mode = str(self._config.get("reference_loop_mode", "auto")).lower()
        if mode == "auto":
            return references
        if mode not in ("wrap", "clamp"):
            raise ValueError(
                "reference_loop_mode mora biti 'auto', 'wrap' ili 'clamp', "
                f"dobijeno: {mode!r}."
            )

        loop_value = int(LoopMode.WRAP if mode == "wrap" else LoopMode.CLAMP)
        loop_modes = np.full_like(
            np.asarray(references.loop_modes, dtype=np.int32),
            loop_value,
        )
        return replace(references, loop_modes=loop_modes)

    def _configure_default_deepmimic_reference(self) -> None:
        """Build a one-frame standing reference for non-BVH runs."""
        qpos_targets = np.asarray(self._bvh_reference_qpos_targets, dtype=np.float32)
        qvel_targets = np.asarray(self._bvh_reference_qvel_targets, dtype=np.float32)
        self._configure_deepmimic_reference_from_qpos(qpos_targets, qvel_targets)

    def _cache_standing_reference(self) -> None:
        """Keep a coherent one-frame standing reference for reset fallback."""
        self._standing_reference_qpos_targets = jp.array(self._bvh_reference_qpos_targets)
        self._standing_reference_qvel_targets = jp.array(self._bvh_reference_qvel_targets)
        self._standing_reference_key_pos_targets = jp.array(self._bvh_reference_key_pos_targets)
        self._standing_reference_key_rel_local_targets = jp.array(
            self._bvh_reference_key_rel_local_targets
        )
        self._standing_reference_root_pos_targets = jp.array(self._bvh_reference_root_pos_targets)
        self._standing_reference_root_quat_targets = jp.array(self._bvh_reference_root_quat_targets)
        self._standing_reference_root_vel_targets = jp.array(self._bvh_reference_root_vel_targets)
        self._standing_reference_root_angvel_targets = jp.array(
            self._bvh_reference_root_angvel_targets
        )
        self._standing_reference_sim_root_pos_targets = jp.array(
            self._bvh_reference_sim_root_pos_targets_np
        )
        self._standing_reference_sim_root_quat_targets = jp.array(
            self._bvh_reference_sim_root_quat_targets_np
        )
        self._standing_reference_sim_root_vel_targets = jp.array(
            self._bvh_reference_sim_root_vel_targets_np
        )
        self._standing_reference_sim_root_angvel_targets = jp.array(
            self._bvh_reference_sim_root_angvel_targets_np
        )
        self._standing_reference_reset_root_pos_targets = jp.array(
            self._bvh_reference_reset_root_pos_targets
        )
        self._standing_reference_reset_root_quat_targets = jp.array(
            self._bvh_reference_reset_root_quat_targets
        )
        self._standing_reference_reset_root_vel_targets = jp.array(
            self._bvh_reference_reset_root_vel_targets
        )
        self._standing_reference_reset_root_angvel_targets = jp.array(
            self._bvh_reference_reset_root_angvel_targets
        )

    def _infer_missing_reference_support_feet(self) -> None:
        """Backfill missing clip-level support-foot tags from key-marker heights."""
        support_feet = list(getattr(self, "_bvh_reference_support_feet", ()))
        if not support_feet or self._deepmimic_key_count < 2:
            return
        key_pos = np.asarray(self._bvh_reference_key_pos_targets, dtype=np.float32)
        frame_counts = np.asarray(self._bvh_reference_frame_counts, dtype=np.int32)
        for clip_id, current_tag in enumerate(support_feet):
            if current_tag:
                continue
            frame_count = int(frame_counts[clip_id]) if clip_id < frame_counts.shape[0] else 0
            if frame_count <= 0:
                continue
            window = max(1, min(frame_count, 30))
            right_heights = key_pos[clip_id, :window, 0, 2]
            left_heights = key_pos[clip_id, :window, 1, 2]
            left_mean = float(np.mean(left_heights))
            right_mean = float(np.mean(right_heights))
            if abs(left_mean - right_mean) < 0.015:
                continue
            support_feet[clip_id] = "left_foot" if left_mean < right_mean else "right_foot"
        self._bvh_reference_support_feet = tuple(support_feet)
        self._standing_reference_wrap_deltas = jp.zeros((1, 3), dtype=jp.float32)

    def _retarget_reference_qpos_to_ik_targets(
        self,
        qpos_targets: np.ndarray,
        ik_target_positions: np.ndarray,
        ik_target_names: tuple[str, ...],
        root_pos_targets: np.ndarray,
        root_quat_targets: np.ndarray,
        frame_counts: np.ndarray,
    ) -> np.ndarray:
        """Fit reference actuator poses to SMPL marker targets in this XML space.

        This is the missing DeepMimic/MimicKit-style retarget step: SMPL marker
        positions are solved through the actual MuJoCo skeleton and joint limits
        once during reference loading, instead of expecting raw SMPL local angles
        to be valid controls for a different XML.
        """
        marker_specs = self._resolve_host_ik_markers(ik_target_names)
        if not marker_specs:
            return np.asarray(qpos_targets, dtype=np.float32)

        fitted = np.asarray(qpos_targets, dtype=np.float32).copy()
        seed_targets = np.asarray(qpos_targets, dtype=np.float64)
        ik_target_positions = np.asarray(ik_target_positions, dtype=np.float64)
        root_pos_targets = np.asarray(root_pos_targets, dtype=np.float64)
        root_quat_targets = np.asarray(root_quat_targets, dtype=np.float64)
        frame_counts = np.asarray(frame_counts, dtype=np.int32)

        data = mujoco.MjData(self._mj_model)
        action_size = seed_targets.shape[-1]
        lower = self._actuator_ctrl_lower_limits_np.astype(np.float64)
        upper = self._actuator_ctrl_upper_limits_np.astype(np.float64)
        ik_update_mask = np.array(
            [
                joint_name in LEG_ACTUATED_JOINTS
                or joint_name in ARM_ACTUATED_JOINTS
                for joint_name in self._actuator_joint_names
            ],
            dtype=np.float64,
        )
        pose_reg = 0.20
        smooth_reg = 0.08
        damping = 1e-5
        max_update = 0.12
        max_iters = 10

        for clip_id in range(seed_targets.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            previous_qpos: np.ndarray | None = None
            for frame_id in range(frame_count):
                seed = seed_targets[clip_id, frame_id].copy()
                qpos = seed.copy() if previous_qpos is None else (
                    0.75 * seed + 0.25 * previous_qpos
                )
                qpos = np.clip(qpos, lower, upper)
                for _ in range(max_iters):
                    full_qpos = self._init_q_np.copy()
                    full_qpos[:3] = root_pos_targets[clip_id, frame_id]
                    full_qpos[3:7] = root_quat_targets[clip_id, frame_id]
                    full_qpos[self._actuator_qpos_indices_np] = qpos
                    data.qpos[:] = full_qpos
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(self._mj_model, data)

                    residuals: list[np.ndarray] = []
                    jacobian_rows: list[np.ndarray] = []
                    for target_index, marker in enumerate(marker_specs):
                        target_pos = ik_target_positions[
                            clip_id,
                            frame_id,
                            target_index,
                        ].copy()
                        if marker["is_foot"]:
                            target_pos[2] = max(
                                target_pos[2],
                                float(self.FOOT_CONTACT_HEIGHT),
                            )
                        current_pos = self._host_marker_position(data, marker)
                        marker_weight = float(marker["weight"])
                        sqrt_weight = np.sqrt(marker_weight)
                        residuals.append(sqrt_weight * (target_pos - current_pos))
                        marker_jac = self._host_marker_jacobian(data, marker)
                        jacobian_rows.append(
                            sqrt_weight * marker_jac * ik_update_mask[None, :]
                        )

                    sqrt_pose_reg = np.sqrt(pose_reg)
                    residuals.append(sqrt_pose_reg * (seed - qpos))
                    jacobian_rows.append(sqrt_pose_reg * np.eye(action_size))
                    if previous_qpos is not None:
                        sqrt_smooth_reg = np.sqrt(smooth_reg)
                        residuals.append(sqrt_smooth_reg * (previous_qpos - qpos))
                        jacobian_rows.append(sqrt_smooth_reg * np.eye(action_size))

                    residual = np.concatenate(residuals)
                    jacobian = np.vstack(jacobian_rows)
                    system = jacobian.T @ jacobian
                    rhs = jacobian.T @ residual
                    system += damping * np.eye(action_size)
                    try:
                        update = np.linalg.solve(system, rhs)
                    except np.linalg.LinAlgError:
                        update = np.linalg.lstsq(system, rhs, rcond=None)[0]
                    update = np.clip(update, -max_update, max_update)
                    qpos = np.clip(qpos + update, lower, upper)
                    if float(np.linalg.norm(update)) < 1e-4:
                        break

                fitted[clip_id, frame_id] = qpos.astype(np.float32)
                previous_qpos = qpos.copy()
            fitted[clip_id, frame_count:] = fitted[clip_id, frame_count - 1]

        return fitted

    def _normalize_ik_foot_target_heights(
        self,
        ik_target_positions: np.ndarray,
        ik_target_names: tuple[str, ...],
        frame_counts: np.ndarray,
    ) -> np.ndarray:
        """Shift SMPL foot marker Z so each frame has a stance contact target."""
        foot_indices = [
            index
            for index, marker_name in enumerate(ik_target_names)
            if "metatarsal" in marker_name or "foot" in marker_name
        ]
        if not foot_indices:
            return ik_target_positions

        normalized = np.asarray(ik_target_positions, dtype=np.float32).copy()
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        stance_site_height = self._standing_marker_height(
            ik_target_names[foot_indices[0]]
        )
        for clip_id in range(normalized.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            foot_z = normalized[clip_id, :frame_count, foot_indices, 2]
            per_frame_offset = stance_site_height - np.min(foot_z, axis=1)
            normalized[clip_id, :frame_count, foot_indices, 2] += (
                per_frame_offset[:, None]
            )
            normalized[clip_id, frame_count:] = normalized[clip_id, frame_count - 1]
        return normalized

    def _standing_marker_height(self, marker_name: str) -> float:
        """Return marker height in the simulator's neutral floor-aligned pose."""
        data = mujoco.MjData(self._mj_model)
        data.qpos[:] = self._init_q_np
        data.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, data)
        try:
            return float(data.site_xpos[self._mj_model.site(marker_name).id, 2])
        except KeyError:
            try:
                return float(data.xpos[self._mj_model.body(marker_name).id, 2])
            except KeyError:
                return float(self.FOOT_CONTACT_HEIGHT)

    def _apply_reference_stability_prior(
        self,
        qpos_targets: np.ndarray,
    ) -> np.ndarray:
        """Blend raw SMPL targets toward a dynamically safer locomotion pose.

        Full-amplitude SMPL joint targets can place this XML's COM far outside
        the support foot.  Keep the gait signal, but damp the raw pose enough
        that feed-forward reference playback starts from load-bearing contacts.
        """
        qpos_targets = np.asarray(qpos_targets, dtype=np.float32)
        neutral = np.asarray(self._default_ctrl, dtype=np.float32)
        sagittal_motion_joints = {
            "left_hip_x",
            "right_hip_x",
            "left_hip_z",
            "right_hip_z",
            "left_knee_z",
            "right_knee_z",
            "left_ankle_y",
            "right_ankle_y",
        }
        alpha = np.array(
            [
                float(
                    self._config.get(
                        "reference_stability_sagittal_alpha",
                        0.40,
                    )
                )
                if joint_name in sagittal_motion_joints
                else float(
                    self._config.get(
                        "reference_stability_arm_alpha",
                        0.75,
                    )
                )
                if joint_name in ARM_ACTUATED_JOINTS
                else float(
                    self._config.get(
                        "reference_stability_other_alpha",
                        0.18,
                    )
                )
                for joint_name in self._actuator_joint_names
            ],
            dtype=np.float32,
        )
        return neutral[None, None, :] + alpha[None, None, :] * (
            qpos_targets - neutral[None, None, :]
        )

    def _resolve_host_ik_markers(
        self,
        marker_names: tuple[str, ...],
    ) -> list[dict[str, object]]:
        """Resolve marker names to host MuJoCo site/body ids for IK."""
        specs: list[dict[str, object]] = []
        for marker_name in marker_names:
            is_foot = "metatarsal" in marker_name or "foot" in marker_name
            weight = 4.0 if is_foot else 1.2
            if "pelvis" in marker_name:
                weight = 2.0
            elif "hand" in marker_name:
                weight = 0.15
            elif "head" in marker_name:
                weight = 0.7
            try:
                specs.append(
                    {
                        "name": marker_name,
                        "kind": "site",
                        "id": self._mj_model.site(marker_name).id,
                        "weight": weight,
                        "is_foot": is_foot,
                    }
                )
                continue
            except KeyError:
                pass
            try:
                specs.append(
                    {
                        "name": marker_name,
                        "kind": "body",
                        "id": self._mj_model.body(marker_name).id,
                        "weight": weight,
                        "is_foot": is_foot,
                    }
                )
            except KeyError:
                continue
        return specs

    def _host_marker_position(
        self,
        data: mujoco.MjData,
        marker: dict[str, object],
    ) -> np.ndarray:
        """Return one host marker position for a resolved IK spec."""
        marker_id = int(marker["id"])
        if marker["kind"] == "site":
            return np.asarray(data.site_xpos[marker_id], dtype=np.float64)
        return np.asarray(data.xpos[marker_id], dtype=np.float64)

    def _host_marker_jacobian(
        self,
        data: mujoco.MjData,
        marker: dict[str, object],
    ) -> np.ndarray:
        """Return marker position Jacobian columns for actuated hinge DOFs."""
        jacp = np.zeros((3, self._mj_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._mj_model.nv), dtype=np.float64)
        marker_id = int(marker["id"])
        if marker["kind"] == "site":
            mujoco.mj_jacSite(self._mj_model, data, jacp, jacr, marker_id)
        else:
            mujoco.mj_jacBody(self._mj_model, data, jacp, jacr, marker_id)
        return jacp[:, self._actuator_dof_indices_np]

    def _reference_qvel_from_qpos_targets(
        self,
        qpos_targets: np.ndarray,
        frame_times: np.ndarray,
        frame_counts: np.ndarray,
    ) -> np.ndarray:
        """Differentiate padded actuator qpos targets clip-by-clip."""
        qpos_targets = np.asarray(qpos_targets, dtype=np.float32)
        frame_times = np.asarray(frame_times, dtype=np.float32)
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        qvel_targets = np.zeros_like(qpos_targets, dtype=np.float32)
        for clip_id in range(qpos_targets.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 1:
                continue
            dt = max(float(frame_times[clip_id]), 1e-6)
            qvel_targets[clip_id, :frame_count] = np.gradient(
                qpos_targets[clip_id, :frame_count],
                dt,
                axis=0,
            ).astype(np.float32)
            qvel_targets[clip_id, frame_count:] = qvel_targets[
                clip_id,
                frame_count - 1,
            ]
        return qvel_targets

    def _scale_reference_root_xy_motion(
        self,
        root_pos_targets: np.ndarray,
        frame_counts: np.ndarray,
    ) -> np.ndarray:
        """Keep SMPL root travel consistent with the damped retargeted legs.

        After IK we intentionally blend joint targets toward a stable locomotion
        prior.  Leaving the original full-speed SMPL chest/root translation in
        place creates a treadmill reference: the ghost root moves forward while
        the physically simulated feet remain planted.  Scale only horizontal
        displacement from each clip's first frame; Z is recomputed by the normal
        floor-alignment pass.
        """
        scaled = np.asarray(root_pos_targets, dtype=np.float32).copy()
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        root_xy_scale = float(
            self._config.get(
                "reference_root_xy_scale",
                self.REFERENCE_RETARGET_ROOT_XY_SCALE,
            )
        )
        for clip_id in range(scaled.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            origin_xy = scaled[clip_id, 0, :2].copy()
            scaled[clip_id, :frame_count, :2] = (
                origin_xy
                + root_xy_scale * (scaled[clip_id, :frame_count, :2] - origin_xy)
            )
            scaled[clip_id, frame_count:, :2] = scaled[clip_id, frame_count - 1, :2]
        return scaled

    def _align_reference_roots_to_sim_floor(
        self,
        qpos_targets: np.ndarray,
        root_pos_targets: np.ndarray,
        root_quat_targets: np.ndarray,
        frame_times: np.ndarray,
        frame_counts: np.ndarray,
        loop_modes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bake reference root Z into the same floor-aligned MuJoCo space as reset.

        SMPL/BVH root translations can be globally plausible while still placing
        this XML's foot geoms above or below the MuJoCo floor. MimicKit avoids
        that by storing motions in the simulator character space. This pass does
        the lightweight equivalent for our current retargeted actuator targets:
        run host MuJoCo FK per frame, shift the root so the lowest foot center is
        at the same contact-ready height used by reset, then derive root velocity
        from the baked root trajectory.
        """
        aligned_root_pos = np.asarray(root_pos_targets, dtype=np.float32).copy()
        root_quat_targets = np.asarray(root_quat_targets, dtype=np.float32)
        qpos_targets = np.asarray(qpos_targets, dtype=np.float32)
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        frame_times = np.asarray(frame_times, dtype=np.float32)
        loop_modes = np.asarray(loop_modes, dtype=np.int32)

        data = mujoco.MjData(self._mj_model)
        standing_root_height = float(np.asarray(self._mjx_model.qpos0[2]))
        minimum_reset_height = (
            max(
                standing_root_height * 0.82,
                standing_root_height * float(self.MIN_STANDING_HEIGHT_RATIO) + 0.05,
            )
        )
        desired_lowest_height = -float(self.FOOT_CONTACT_PRELOAD)

        clip_count, max_frames, _ = qpos_targets.shape
        for clip_id in range(clip_count):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            for frame_id in range(max_frames):
                source_frame = min(frame_id, frame_count - 1)
                full_qpos = self._init_q_np.copy()
                full_qpos[:3] = aligned_root_pos[clip_id, source_frame]
                full_qpos[3:7] = root_quat_targets[clip_id, source_frame]
                full_qpos[self._actuator_qpos_indices_np] = qpos_targets[
                    clip_id,
                    source_frame,
                ]
                data.qpos[:] = full_qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(self._mj_model, data)

                foot_heights = np.array(
                    [
                        self._geom_min_z(data, int(geom_id))
                        for geom_id in self._foot_geom_ids_np
                    ],
                    dtype=np.float32,
                )
                z_offset = desired_lowest_height - float(np.min(foot_heights))
                aligned_z = max(float(full_qpos[2]) + z_offset, minimum_reset_height)
                aligned_root_pos[clip_id, frame_id] = aligned_root_pos[
                    clip_id,
                    source_frame,
                ]
                aligned_root_pos[clip_id, frame_id, 2] = aligned_z

        aligned_root_vel = np.zeros_like(aligned_root_pos, dtype=np.float32)
        wrap_deltas = np.zeros((clip_count, 3), dtype=np.float32)
        for clip_id in range(clip_count):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            dt = max(float(frame_times[clip_id]), 1e-6)
            max_root_z_step = float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED) * dt
            root_z = aligned_root_pos[clip_id, :frame_count, 2].copy()
            for frame_id in range(1, frame_count):
                delta_z = np.clip(
                    root_z[frame_id] - root_z[frame_id - 1],
                    -max_root_z_step,
                    max_root_z_step,
                )
                root_z[frame_id] = root_z[frame_id - 1] + delta_z
            for frame_id in range(frame_count - 2, -1, -1):
                delta_z = np.clip(
                    root_z[frame_id] - root_z[frame_id + 1],
                    -max_root_z_step,
                    max_root_z_step,
                )
                root_z[frame_id] = root_z[frame_id + 1] + delta_z
            root_z = np.maximum(root_z, minimum_reset_height)
            aligned_root_pos[clip_id, :frame_count, 2] = root_z
            aligned_root_pos[clip_id, frame_count:, 2] = root_z[-1]
            if bool(self._config.get("reference_lock_stance_feet", True)):
                aligned_root_pos[clip_id] = self._lock_clip_root_xy_to_stance_feet(
                    qpos_targets[clip_id],
                    aligned_root_pos[clip_id],
                    root_quat_targets[clip_id],
                    frame_count,
                )
            if frame_count > 1:
                aligned_root_vel[clip_id, :frame_count] = np.gradient(
                    aligned_root_pos[clip_id, :frame_count],
                    dt,
                    axis=0,
                ).astype(np.float32)
                aligned_root_vel[clip_id, :frame_count, 2] = np.clip(
                    aligned_root_vel[clip_id, :frame_count, 2],
                    -float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED),
                    float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED),
                )
                aligned_root_vel[clip_id, frame_count:] = aligned_root_vel[
                    clip_id,
                    frame_count - 1,
                ]
            if loop_modes[clip_id] == int(LoopMode.WRAP):
                wrap_deltas[clip_id] = (
                    aligned_root_pos[clip_id, frame_count - 1]
                    - aligned_root_pos[clip_id, 0]
                )
                wrap_deltas[clip_id, 2] = 0.0

        return aligned_root_pos, aligned_root_vel, wrap_deltas

    def _lock_clip_root_xy_to_stance_feet(
        self,
        qpos_targets: np.ndarray,
        root_pos_targets: np.ndarray,
        root_quat_targets: np.ndarray,
        frame_count: int,
    ) -> np.ndarray:
        """Reduce kinematic foot skating by anchoring the lowest contact foot."""
        locked_root = np.asarray(root_pos_targets, dtype=np.float32).copy()
        if frame_count <= 1:
            return locked_root

        data = mujoco.MjData(self._mj_model)
        contact_height = float(self._config.get("reference_foot_lock_height", 0.035))
        xy_offset = np.zeros(2, dtype=np.float32)
        active_foot: int | None = None
        active_anchor = np.zeros(2, dtype=np.float32)

        for frame_id in range(frame_count):
            full_qpos = self._init_q_np.copy()
            full_qpos[:3] = locked_root[frame_id]
            full_qpos[:2] += xy_offset
            full_qpos[3:7] = root_quat_targets[frame_id]
            full_qpos[self._actuator_qpos_indices_np] = qpos_targets[frame_id]
            data.qpos[:] = full_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(self._mj_model, data)

            foot_min_z = np.array(
                [
                    self._geom_min_z(data, int(geom_id))
                    for geom_id in self._foot_geom_ids_np
                ],
                dtype=np.float32,
            )
            contact_feet = foot_min_z <= contact_height
            if not np.any(contact_feet):
                active_foot = None
                locked_root[frame_id, :2] += xy_offset
                continue

            foot_xy = np.asarray(
                data.geom_xpos[self._foot_geom_ids_np, :2],
                dtype=np.float32,
            )
            if active_foot is None or not bool(contact_feet[active_foot]):
                active_foot = int(np.argmin(np.where(contact_feet, foot_min_z, np.inf)))
                active_anchor = foot_xy[active_foot].copy()
            else:
                xy_offset += active_anchor - foot_xy[active_foot]
                full_qpos[:2] = locked_root[frame_id, :2] + xy_offset
                data.qpos[:] = full_qpos
                mujoco.mj_forward(self._mj_model, data)
                foot_xy = np.asarray(
                    data.geom_xpos[self._foot_geom_ids_np, :2],
                    dtype=np.float32,
                )
                active_anchor = foot_xy[active_foot].copy()

            locked_root[frame_id, :2] = full_qpos[:2]

        locked_root[frame_count:] = locked_root[frame_count - 1]
        return locked_root

    def _configure_deepmimic_reference_from_qpos(
        self,
        qpos_targets: np.ndarray,
        qvel_targets: np.ndarray,
        root_pos_targets: np.ndarray | None = None,
        root_quat_targets: np.ndarray | None = None,
        root_vel_targets: np.ndarray | None = None,
        root_angvel_targets: np.ndarray | None = None,
    ) -> None:
        """Precompute DeepMimic-style FK targets from retargeted joint targets.

        The old BVH loader still supplies actuator-space poses because that is
        the only retargeted representation this repo already has.  From here on
        imitation is feature-based: body orientations, body velocities,
        end-effectors, root state, and COM, matching the DeepMimic reward shape.
        """
        # REF: PROJECT-BVH-FK-RETARGETING
        # TYPE: MODEL_CALIBRATED
        clip_count, max_frames, _ = qpos_targets.shape
        sim_root_pos_targets = (
            root_pos_targets
            if root_pos_targets is not None
            else self._bvh_reference_sim_root_pos_targets_np
        )
        sim_root_quat_targets = (
            root_quat_targets
            if root_quat_targets is not None
            else self._bvh_reference_sim_root_quat_targets_np
        )
        sim_root_vel_targets = (
            root_vel_targets
            if root_vel_targets is not None
            else self._bvh_reference_sim_root_vel_targets_np
        )
        sim_root_angvel_targets = (
            root_angvel_targets
            if root_angvel_targets is not None
            else self._bvh_reference_sim_root_angvel_targets_np
        )
        body_count = len(self._deepmimic_body_ids_np)
        body_pos = np.zeros((clip_count, max_frames, body_count, 3), dtype=np.float32)
        body_quat = np.zeros((clip_count, max_frames, body_count, 4), dtype=np.float32)
        body_cvel = np.zeros((clip_count, max_frames, body_count, 6), dtype=np.float32)
        key_body_count = self._deepmimic_key_count
        key_pos = np.zeros((clip_count, max_frames, key_body_count, 3), dtype=np.float32)
        key_rel_local = np.zeros(
            (clip_count, max_frames, key_body_count, 3),
            dtype=np.float32,
        )
        root_pos = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
        root_quat = np.zeros((clip_count, max_frames, 4), dtype=np.float32)
        root_vel = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
        root_angvel = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
        com = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
        com_vel = np.zeros((clip_count, max_frames, 3), dtype=np.float32)

        data = mujoco.MjData(self._mj_model)
        for clip_id in range(clip_count):
            previous_com = None
            for frame_id in range(max_frames):
                full_qpos = self._init_q_np.copy()
                full_qvel = np.zeros(self._mj_model.nv, dtype=np.float64)
                full_qpos[:3] = sim_root_pos_targets[clip_id, frame_id]
                full_qpos[3:7] = sim_root_quat_targets[clip_id, frame_id]
                full_qvel[:3] = sim_root_vel_targets[clip_id, frame_id]
                full_qvel[3:6] = sim_root_angvel_targets[clip_id, frame_id]
                full_qpos[self._actuator_qpos_indices_np] = qpos_targets[
                    clip_id,
                    frame_id,
                ]
                full_qvel[self._actuator_dof_indices_np] = qvel_targets[
                    clip_id,
                    frame_id,
                ]
                data.qpos[:] = full_qpos
                data.qvel[:] = full_qvel
                mujoco.mj_forward(self._mj_model, data)

                body_pos[clip_id, frame_id] = data.xpos[self._deepmimic_body_ids_np]
                body_quat[clip_id, frame_id] = data.xquat[self._deepmimic_body_ids_np]
                body_cvel[clip_id, frame_id] = data.cvel[self._deepmimic_body_ids_np]
                key_pos[clip_id, frame_id] = self._key_marker_positions_host(data)
                anchor_pos = body_pos[
                    clip_id,
                    frame_id,
                    self._reference_anchor_body_local_index,
                ]
                anchor_quat = body_quat[
                    clip_id,
                    frame_id,
                    self._reference_anchor_body_local_index,
                ]
                anchor_cvel = body_cvel[
                    clip_id,
                    frame_id,
                    self._reference_anchor_body_local_index,
                ]
                heading = np.asarray(
                    self._heading_world_to_local_from_quat(
                        jp.array(anchor_quat, dtype=jp.float32)
                    )
                )
                key_rel_local[clip_id, frame_id] = (
                    key_pos[clip_id, frame_id] - anchor_pos
                ) @ heading.T
                root_pos[clip_id, frame_id] = anchor_pos
                root_quat[clip_id, frame_id] = anchor_quat
                root_vel[clip_id, frame_id] = anchor_cvel[3:6]
                root_angvel[clip_id, frame_id] = anchor_cvel[:3]
                current_com = np.average(
                    data.xpos[self._deepmimic_body_ids_np],
                    axis=0,
                    weights=self._deepmimic_body_weights_np,
                )
                com[clip_id, frame_id] = current_com
                if previous_com is None:
                    com_vel[clip_id, frame_id] = 0.0
                else:
                    frame_time = float(np.asarray(self._bvh_reference_frame_times)[clip_id])
                    com_vel[clip_id, frame_id] = (current_com - previous_com) / frame_time
                previous_com = current_com.copy()

        self._bvh_reference_body_pos_targets = jp.array(body_pos)
        self._bvh_reference_body_quat_targets = jp.array(body_quat)
        self._bvh_reference_body_cvel_targets = jp.array(body_cvel)
        self._bvh_reference_key_pos_targets = jp.array(key_pos)
        self._bvh_reference_key_rel_local_targets = jp.array(key_rel_local)
        self._bvh_reference_root_pos_targets = jp.array(root_pos)
        self._bvh_reference_root_quat_targets = jp.array(root_quat)
        self._bvh_reference_root_vel_targets = jp.array(root_vel)
        self._bvh_reference_root_angvel_targets = jp.array(root_angvel)
        self._bvh_reference_sim_root_pos_targets = jp.array(sim_root_pos_targets)
        self._bvh_reference_sim_root_quat_targets = jp.array(sim_root_quat_targets)
        self._bvh_reference_sim_root_vel_targets = jp.array(sim_root_vel_targets)
        self._bvh_reference_sim_root_angvel_targets = jp.array(sim_root_angvel_targets)
        self._bvh_reference_wrap_deltas = jp.array(self._bvh_reference_wrap_deltas_np)
        self._bvh_reference_com_targets = jp.array(com)
        self._bvh_reference_com_vel_targets = jp.array(com_vel)

    def _reference_gait_files(self) -> tuple[str, ...]:
        """Vrati BVH fajlove iz config-a kao tuple stringova."""
        reference_gait_file = self._config.get("reference_gait_file", None)
        if reference_gait_file is None:
            return (DEFAULT_BVH_REFERENCE_LIST.as_posix(),)
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
        """Resetuje humana u pocetnu pozu i uzorkuje joystick komandu."""
        rng, command_key, erfi_key, bias_key, gait_key, bvh_key = (
            jax.random.split(rng, 6)
        )
        pure_imitation = self._is_pure_reference_imitation()
        fallback_qpos = self._init_q.at[2].set(self._standing_height())
        fallback_qvel = jp.zeros(self._mjx_model.nv)
        fallback_data = self._fresh_reset_data().replace(
            qpos=fallback_qpos,
            qvel=fallback_qvel,
            ctrl=self._default_ctrl,
        )
        fallback_data = mjx.forward(self._mjx_model, fallback_data)
        data = fallback_data
        qpos = fallback_qpos
        ctrl = self._default_ctrl
        last_action = jp.zeros(self.action_size)
        bvh_clip_id = jp.array(0, dtype=jp.int32)
        bvh_time_offset = jp.array(0.0, dtype=jp.float32)
        sampled_valid = jp.array(False)
        rejected_count = jp.array(0.0, dtype=jp.float32)
        rejected_low_count = jp.array(0.0, dtype=jp.float32)
        rejected_tipped_count = jp.array(0.0, dtype=jp.float32)
        rejected_invalid_count = jp.array(0.0, dtype=jp.float32)
        exact_motion_count = jp.array(0.0, dtype=jp.float32)
        exact_rejected_count = jp.array(0.0, dtype=jp.float32)

        reset_sample_attempts = int(
            self._config.get("reset_sample_attempts", self.RESET_SAMPLE_ATTEMPTS)
        )
        reset_projection_levels = tuple(
            float(level)
            for level in self._config.get(
                "reset_projection_levels",
                self.RESET_PROJECTION_LEVELS,
            )
        )

        for _ in range(reset_sample_attempts):
            bvh_key, clip_key, sample_time_key, pose_key = jax.random.split(
                bvh_key,
                4,
            )
            candidate_clip_id = self._sample_weighted_bvh_clip_id(clip_key)
            candidate_time_offset = self._sample_bvh_motion_time(
                sample_time_key,
                candidate_clip_id,
            )
            (
                candidate_qpos,
                candidate_qvel,
                candidate_ctrl,
                candidate_last_action,
            ) = self._sample_initial_state_from_reference(
                candidate_clip_id,
                candidate_time_offset,
                exact=True,
            )
            candidate_data = self._fresh_reset_data().replace(
                qpos=candidate_qpos,
                qvel=candidate_qvel,
                ctrl=candidate_ctrl,
            )
            candidate_data = mjx.forward(self._mjx_model, candidate_data)
            (
                candidate_low,
                candidate_tipped,
                candidate_invalid,
            ) = self._get_done_reasons(candidate_data)
            candidate_terminal = self._get_done(candidate_data)
            candidate_rejected = (~sampled_valid) & candidate_terminal
            take_candidate = (~sampled_valid) & (~candidate_terminal)
            rejected_count = rejected_count + candidate_rejected.astype(jp.float32)
            rejected_low_count = rejected_low_count + (
                candidate_rejected & candidate_low
            ).astype(jp.float32)
            rejected_tipped_count = rejected_tipped_count + (
                candidate_rejected & candidate_tipped
            ).astype(jp.float32)
            rejected_invalid_count = rejected_invalid_count + (
                candidate_rejected & candidate_invalid
            ).astype(jp.float32)
            exact_motion_count = exact_motion_count + take_candidate.astype(jp.float32)
            exact_rejected_count = exact_rejected_count + candidate_rejected.astype(
                jp.float32
            )
            data = select_data(
                data,
                take_candidate,
                candidate_data,
                self._physics_backend,
            )
            qpos = jp.where(take_candidate, candidate_qpos, qpos)
            ctrl = jp.where(take_candidate, candidate_ctrl, ctrl)
            last_action = jp.where(
                take_candidate,
                candidate_last_action,
                last_action,
            )
            bvh_clip_id = jp.where(take_candidate, candidate_clip_id, bvh_clip_id)
            bvh_time_offset = jp.where(
                take_candidate,
                candidate_time_offset,
                bvh_time_offset,
            )
            sampled_valid = sampled_valid | (~candidate_terminal)
            for projection_level in reset_projection_levels:
                (
                    candidate_qpos,
                    candidate_qvel,
                    candidate_ctrl,
                    candidate_last_action,
                ) = self._sample_initial_state_from_reference(
                    candidate_clip_id,
                    candidate_time_offset,
                    projection_level=projection_level,
                )
                pose_noise = (
                    jax.random.uniform(
                        pose_key,
                        shape=(self.action_size,),
                        minval=-1.0,
                        maxval=1.0,
                    )
                    * self._reference_reset_noise_scale(projection_level)
                )
                candidate_qpos = candidate_qpos.at[self._actuator_qpos_indices].add(
                    pose_noise
                )
                candidate_data = self._fresh_reset_data().replace(
                    qpos=candidate_qpos,
                    qvel=candidate_qvel,
                    ctrl=candidate_ctrl,
                )
                candidate_data = mjx.forward(self._mjx_model, candidate_data)
                (
                    candidate_low,
                    candidate_tipped,
                    candidate_invalid,
                ) = self._get_done_reasons(candidate_data)
                candidate_terminal = self._get_done(candidate_data)
                candidate_rejected = (~sampled_valid) & candidate_terminal
                take_candidate = (~sampled_valid) & (~candidate_terminal)
                rejected_count = rejected_count + candidate_rejected.astype(jp.float32)
                rejected_low_count = rejected_low_count + (
                    candidate_rejected & candidate_low
                ).astype(jp.float32)
                rejected_tipped_count = rejected_tipped_count + (
                    candidate_rejected & candidate_tipped
                ).astype(jp.float32)
                rejected_invalid_count = rejected_invalid_count + (
                    candidate_rejected & candidate_invalid
                ).astype(jp.float32)
                data = select_data(
                    data,
                    take_candidate,
                    candidate_data,
                    self._physics_backend,
                )
                qpos = jp.where(take_candidate, candidate_qpos, qpos)
                ctrl = jp.where(take_candidate, candidate_ctrl, ctrl)
                last_action = jp.where(
                    take_candidate,
                    candidate_last_action,
                    last_action,
                )
                bvh_clip_id = jp.where(take_candidate, candidate_clip_id, bvh_clip_id)
                bvh_time_offset = jp.where(
                    take_candidate,
                    candidate_time_offset,
                    bvh_time_offset,
                )
                sampled_valid = sampled_valid | (~candidate_terminal)

        command = jp.where(
            pure_imitation,
            jp.zeros(3, dtype=jp.float32),
            self.sample_command(command_key),
        )
        info = {
            "rng": rng,
            "command": command,
            "last_action": last_action,
            "episode_torque_offset": self.sample_episode_torque_offset(bias_key),
            "use_rfi": jax.random.bernoulli(erfi_key, p=0.5),
            "motion_step": jp.array(0, dtype=jp.int32),
            "command_step": jp.array(0, dtype=jp.int32),
            "gait_step": jax.random.randint(
                gait_key,
                shape=(),
                minval=0,
                maxval=int(self.GAIT_PERIOD_STEPS),
            ),
            "bvh_reference_clip_id": bvh_clip_id,
            "bvh_reference_time_offset": bvh_time_offset,
            "reference_fallback_standing": ~sampled_valid,
            "truncation": jp.array(0.0, dtype=jp.float32),
            "init_motion_count": sampled_valid.astype(jp.float32),
            "init_fallback_count": (~sampled_valid).astype(jp.float32),
            "init_rejected_count": rejected_count,
            "init_rejected_low_count": rejected_low_count,
            "init_rejected_tipped_count": rejected_tipped_count,
            "init_rejected_invalid_count": rejected_invalid_count,
            "init_exact_count": exact_motion_count,
            "init_exact_rejected_count": exact_rejected_count,
            "warp_world_id": jp.array(0, dtype=jp.int32),
            "last_foot_xy": self._foot_xy(data),
        }
        obs = self._get_obs(data, info)
        initial_reference_gait = self._get_reference_gait_reward(data, info)
        initial_reference_velocity = self._get_reference_velocity_reward(data, info)
        if self._config.get("reference_gait", "none") in ("bvh", "smpl"):
            initial_deepmimic = self._get_bvh_deepmimic_reward(data, info)
        else:
            initial_deepmimic = {
                "pose": jp.array(0.0),
                "velocity": jp.array(0.0),
                "end_effector": jp.array(0.0),
                "root": jp.array(0.0),
                "com": jp.array(0.0),
                "total_no_root_velocity": jp.array(0.0),
                "root_pose": jp.array(0.0),
                "root_velocity": jp.array(0.0),
                "key_position": jp.array(0.0),
                "pose_error": jp.array(0.0),
                "velocity_error": jp.array(0.0),
                "root_xy_error": jp.array(0.0),
                "root_height_error": jp.array(0.0),
                "root_vel_error": jp.array(0.0),
                "root_angvel_error": jp.array(0.0),
                "key_pos_error": jp.array(0.0),
                "max_key_dist": jp.array(0.0),
            }
        initial_done = self._get_done(data, info).astype(jp.float32)
        initial_done_low_height, initial_done_tipped, initial_done_invalid = (
            self._get_done_reasons(data)
        )
        initial_done_motion_over = self._get_bvh_motion_over(info).astype(jp.float32)
        initial_done_pose_termination = self._get_pose_termination(data, info).astype(
            jp.float32
        )
        initial_reference_trace = self._get_reference_trace_metrics(data, info)
        metrics = {
            "reward": jp.array(0.0),
            "tracking_lin_vel": jp.array(0.0),
            "tracking_lin": jp.array(0.0),
            "tracking_yaw": jp.array(0.0),
            "forward_vel": jp.array(0.0),
            "command_norm": jp.array(0.0),
            "command_lin_norm": jp.array(0.0),
            "command_yaw_abs": jp.array(0.0),
            "command_progress": jp.array(0.0),
            "torso_up": jp.array(1.0),
            "head_up": jp.array(1.0),
            "height": qpos[2],
            "foot_slip": jp.array(0.0),
            "swing_drag": jp.array(0.0),
            "swing_clearance": jp.array(0.0),
            "swing_clearance_deficit": jp.array(0.0),
            "base_height": jp.array(1.0),
            "com_height": qpos[2],
            "root_vertical_velocity": jp.array(0.0),
            "pelvis_vertical_velocity": jp.array(0.0),
            "left_foot_height": jp.array(0.0),
            "right_foot_height": jp.array(0.0),
            "left_foot_contact": initial_reference_trace["left_foot_contact"],
            "right_foot_contact": initial_reference_trace["right_foot_contact"],
            "action_magnitude": jp.array(0.0),
            "variable_posture": jp.array(1.0),
            "gait_reward": jp.array(0.0),
            "reference_gait": initial_reference_gait,
            "reference_velocity": initial_reference_velocity,
            "deepmimic_pose": initial_deepmimic["pose"],
            "deepmimic_velocity": initial_deepmimic["velocity"],
            "deepmimic_total_no_root_velocity": initial_deepmimic[
                "total_no_root_velocity"
            ],
            "deepmimic_end_effector": initial_deepmimic["end_effector"],
            "deepmimic_root": initial_deepmimic["root"],
            "deepmimic_com": initial_deepmimic["com"],
            "deepmimic_root_pose": initial_deepmimic["root_pose"],
            "deepmimic_root_velocity": initial_deepmimic["root_velocity"],
            "deepmimic_key_position": initial_deepmimic["key_position"],
            "deepmimic_pose_error": initial_deepmimic["pose_error"],
            "deepmimic_velocity_error": initial_deepmimic["velocity_error"],
            "deepmimic_root_xy_error": initial_deepmimic["root_xy_error"],
            "deepmimic_root_height_error": initial_deepmimic["root_height_error"],
            "deepmimic_root_vel_error": initial_deepmimic["root_vel_error"],
            "deepmimic_root_angvel_error": initial_deepmimic["root_angvel_error"],
            "deepmimic_key_pos_error": initial_deepmimic["key_pos_error"],
            "deepmimic_max_key_dist": initial_deepmimic["max_key_dist"],
            "deepmimic_root_pose_raw": initial_deepmimic["root_pose"],
            "deepmimic_root_velocity_raw": initial_deepmimic["root_velocity"],
            "deepmimic_key_position_raw": initial_deepmimic["key_position"],
            "init_motion_count": sampled_valid.astype(jp.float32),
            "init_fallback_count": (~sampled_valid).astype(jp.float32),
            "init_rejected_count": rejected_count,
            "init_rejected_low_count": rejected_low_count,
            "init_rejected_tipped_count": rejected_tipped_count,
            "init_rejected_invalid_count": rejected_invalid_count,
            "init_exact_count": exact_motion_count,
            "init_exact_rejected_count": exact_rejected_count,
            "reference_fallback": (~sampled_valid).astype(jp.float32),
            "reference_clip_id": bvh_clip_id.astype(jp.float32),
            "reference_motion_time": bvh_time_offset.astype(jp.float32),
            "reference_root_height": initial_reference_trace["reference_root_height"],
            "reference_root_vertical_velocity": initial_reference_trace[
                "reference_root_vertical_velocity"
            ],
            "reference_left_foot_height": initial_reference_trace[
                "reference_left_foot_height"
            ],
            "reference_right_foot_height": initial_reference_trace[
                "reference_right_foot_height"
            ],
            "root_height_tracking_error": initial_reference_trace[
                "root_height_tracking_error"
            ],
            "root_vertical_velocity_tracking_error": initial_reference_trace[
                "root_vertical_velocity_tracking_error"
            ],
            "left_foot_height_tracking_error": initial_reference_trace[
                "left_foot_height_tracking_error"
            ],
            "right_foot_height_tracking_error": initial_reference_trace[
                "right_foot_height_tracking_error"
            ],
            "contact_force": jp.array(0.0),
            "done_low_height": initial_done_low_height.astype(jp.float32),
            "done_tipped": initial_done_tipped.astype(jp.float32),
            "done_invalid": initial_done_invalid.astype(jp.float32),
            "done_motion_over": initial_done_motion_over,
            "done_pose_termination": initial_done_pose_termination,
            "done": initial_done,
            "terminated": jp.array(0.0),
            "truncated": jp.array(0.0),
        }
        return mjx_env.State(
            data,
            obs,
            jp.array(0.0),
            jp.array(0.0),
            metrics,
            info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Izvrsi jedan locomotion korak za zadatu akciju politike."""
        info = dict(state.info)
        info["rng"], rfi_key, command_key = jax.random.split(info["rng"], 3)
        pure_imitation = self._is_pure_reference_imitation()

        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        policy_action = jp.clip(action, -1.0, 1.0)
        previous_action = info["last_action"]
        smoothed_action = (
            self._config.action_smoothing * policy_action
            + (1.0 - self._config.action_smoothing) * previous_action
        )
        motor_targets = self._policy_action_to_motor_targets(
            smoothed_action,
            info,
        )
        motor_targets = self._clip_motor_targets(motor_targets)
        data = self.step_with_joint_torque_injection(
            state.data,
            motor_targets,
            rfi_key,
            info["episode_torque_offset"],
            info["use_rfi"],
        )

        should_resample = (
            ~pure_imitation
        ) & (
            info["command_step"] > self._config.command_resample_steps
        )
        info["command"] = jp.where(
            should_resample,
            self.sample_command(command_key),
            info["command"],
        )
        info["last_action"] = smoothed_action
        info["motion_step"] = info["motion_step"] + jp.array(
            1,
            dtype=info["motion_step"].dtype,
        )
        info["command_step"] = jp.where(
            should_resample,
            jp.array(0, dtype=info["command_step"].dtype),
            info["command_step"] + jp.array(1, dtype=info["command_step"].dtype),
        )
        info["gait_step"] = jp.mod(
            info["gait_step"] + jp.array(1, dtype=info["gait_step"].dtype),
            self.GAIT_PERIOD_STEPS,
        )

        obs = self._get_obs(data, info)

        reward = self._get_reward(data, smoothed_action, previous_action, info)
        done = self._get_done(data, info)
        done_low_height, done_tipped, done_invalid = self._get_done_reasons(data)
        physical_fail = done_low_height | done_tipped | done_invalid
        done_motion_over = self._get_bvh_motion_over(info).astype(reward.dtype)
        done_pose_termination = self._get_pose_termination(data, info).astype(
            reward.dtype
        )
        info["truncation"] = (
            self._get_bvh_motion_over(info) & (~physical_fail)
        ).astype(reward.dtype)
        foot_slip = self._get_foot_slip_cost(data, info)
        swing_drag = self._get_swing_foot_drag_cost(data, info)
        swing_clearance = self._get_swing_clearance(data, info)
        swing_clearance_deficit = self._get_swing_clearance_deficit_cost(data, info)
        base_height = self._get_base_height_reward(data)
        foot_heights = self._foot_heights(data)
        com_height = self._center_of_mass(data)[2]
        root_vertical_velocity = self._reference_anchor_linvel(data)[2]
        pelvis_vertical_velocity = data.cvel[self._reference_anchor_body_id, 5]
        action_magnitude = jp.linalg.norm(action)
        reference_trace = self._get_reference_trace_metrics(data, info)
        variable_posture = self._get_variable_posture_reward(data, info)
        gait_reward = self._get_gait_reward(data, info)
        reference_gait = self._get_reference_gait_reward(data, info)
        reference_velocity = self._get_reference_velocity_reward(data, info)
        if self._config.get("reference_gait", "none") in ("bvh", "smpl"):
            deepmimic = self._get_bvh_deepmimic_reward(data, info)
        else:
            deepmimic = {
                "pose": jp.array(0.0),
                "velocity": jp.array(0.0),
                "end_effector": jp.array(0.0),
                "root": jp.array(0.0),
                "com": jp.array(0.0),
                "total_no_root_velocity": jp.array(0.0),
                "root_pose": jp.array(0.0),
                "root_velocity": jp.array(0.0),
                "key_position": jp.array(0.0),
                "pose_error": jp.array(0.0),
                "velocity_error": jp.array(0.0),
                "root_xy_error": jp.array(0.0),
                "root_height_error": jp.array(0.0),
                "root_vel_error": jp.array(0.0),
                "root_angvel_error": jp.array(0.0),
                "key_pos_error": jp.array(0.0),
                "max_key_dist": jp.array(0.0),
            }
        contact_force = self._get_contact_force_cost(data)
        metrics = dict(state.metrics)
        metrics["reward"] = reward
        metrics["tracking_lin_vel"] = self._get_tracking_reward(data, info)
        metrics["tracking_lin"] = self._get_tracking_lin_reward(data, info)
        metrics["tracking_yaw"] = self._get_tracking_yaw_reward(data, info)
        metrics["forward_vel"] = self._measured_command(data)[0]
        metrics["command_norm"] = jp.linalg.norm(info["command"])
        metrics["command_lin_norm"] = jp.linalg.norm(info["command"][:2])
        metrics["command_yaw_abs"] = jp.abs(info["command"][2])
        metrics["command_progress"] = self._get_command_progress(data, info)
        metrics["torso_up"] = self._torso_up(data)
        metrics["head_up"] = self._head_up(data)
        metrics["height"] = data.qpos[2]
        metrics["foot_slip"] = foot_slip
        metrics["swing_drag"] = swing_drag
        metrics["swing_clearance"] = swing_clearance
        metrics["swing_clearance_deficit"] = swing_clearance_deficit
        metrics["base_height"] = base_height
        metrics["com_height"] = com_height
        metrics["root_vertical_velocity"] = root_vertical_velocity
        metrics["pelvis_vertical_velocity"] = pelvis_vertical_velocity
        metrics["left_foot_height"] = foot_heights[0]
        metrics["right_foot_height"] = foot_heights[1]
        metrics["left_foot_contact"] = reference_trace["left_foot_contact"]
        metrics["right_foot_contact"] = reference_trace["right_foot_contact"]
        metrics["action_magnitude"] = action_magnitude
        metrics["variable_posture"] = variable_posture
        metrics["gait_reward"] = gait_reward
        metrics["reference_gait"] = reference_gait
        metrics["reference_velocity"] = reference_velocity
        metrics["deepmimic_pose"] = deepmimic["pose"]
        metrics["deepmimic_velocity"] = deepmimic["velocity"]
        metrics["deepmimic_total_no_root_velocity"] = deepmimic[
            "total_no_root_velocity"
        ]
        metrics["deepmimic_end_effector"] = deepmimic["end_effector"]
        metrics["deepmimic_root"] = deepmimic["root"]
        metrics["deepmimic_com"] = deepmimic["com"]
        metrics["deepmimic_root_pose"] = deepmimic["root_pose"]
        metrics["deepmimic_root_velocity"] = deepmimic["root_velocity"]
        metrics["deepmimic_key_position"] = deepmimic["key_position"]
        metrics["deepmimic_pose_error"] = deepmimic["pose_error"]
        metrics["deepmimic_velocity_error"] = deepmimic["velocity_error"]
        metrics["deepmimic_root_xy_error"] = deepmimic["root_xy_error"]
        metrics["deepmimic_root_height_error"] = deepmimic["root_height_error"]
        metrics["deepmimic_root_vel_error"] = deepmimic["root_vel_error"]
        metrics["deepmimic_root_angvel_error"] = deepmimic["root_angvel_error"]
        metrics["deepmimic_key_pos_error"] = deepmimic["key_pos_error"]
        metrics["deepmimic_max_key_dist"] = deepmimic["max_key_dist"]
        metrics["deepmimic_root_pose_raw"] = deepmimic["root_pose"]
        metrics["deepmimic_root_velocity_raw"] = deepmimic["root_velocity"]
        metrics["deepmimic_key_position_raw"] = deepmimic["key_position"]
        metrics["reference_fallback"] = self._reference_fallback_active(info).astype(
            reward.dtype
        )
        metrics["reference_clip_id"] = info["bvh_reference_clip_id"].astype(reward.dtype)
        metrics["reference_motion_time"] = self._get_bvh_reference_motion_time(
            info,
            0,
        ).astype(reward.dtype)
        metrics["reference_root_height"] = reference_trace["reference_root_height"]
        metrics["reference_root_vertical_velocity"] = reference_trace[
            "reference_root_vertical_velocity"
        ]
        metrics["reference_left_foot_height"] = reference_trace[
            "reference_left_foot_height"
        ]
        metrics["reference_right_foot_height"] = reference_trace[
            "reference_right_foot_height"
        ]
        metrics["root_height_tracking_error"] = reference_trace[
            "root_height_tracking_error"
        ]
        metrics["root_vertical_velocity_tracking_error"] = reference_trace[
            "root_vertical_velocity_tracking_error"
        ]
        metrics["left_foot_height_tracking_error"] = reference_trace[
            "left_foot_height_tracking_error"
        ]
        metrics["right_foot_height_tracking_error"] = reference_trace[
            "right_foot_height_tracking_error"
        ]
        metrics["contact_force"] = contact_force
        metrics["done_low_height"] = done_low_height.astype(reward.dtype)
        metrics["done_tipped"] = done_tipped.astype(reward.dtype)
        metrics["done_invalid"] = done_invalid.astype(reward.dtype)
        metrics["done_motion_over"] = done_motion_over
        metrics["done_pose_termination"] = done_pose_termination
        metrics["done"] = done.astype(reward.dtype)
        metrics["terminated"] = done.astype(reward.dtype) * (
            1.0 - info["truncation"]
        )
        metrics["truncated"] = done.astype(reward.dtype) * info["truncation"]
        info["last_foot_xy"] = self._foot_xy(data)

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(reward.dtype),
            metrics=metrics,
            info=info,
        )

    def _is_pure_reference_imitation(self) -> jax.Array:
        """Whether joystick commands should be suppressed for pure BVH/SMPL imitation."""
        return jp.array(
            self._config.get("reference_gait", "none") in ("bvh", "smpl")
            and self._config.get("deepmimic_reward_mode", "pure") == "pure"
        )

    def _policy_action_to_motor_targets(
        self,
        smoothed_action: jax.Array,
        info: dict[str, jax.Array],
    ) -> jax.Array:
        """Map normalized policy actions to PD position targets."""
        reference_gait = self._config.get("reference_gait", "none")
        reference_action_mode = self._config.get("reference_action_mode", "mimickit")
        if reference_gait in ("bvh", "smpl") and reference_action_mode == "mimickit":
            return (
                self._mimickit_action_center
                + smoothed_action * self._mimickit_action_half_range
            )

        if reference_gait in ("bvh", "smpl"):
            target_step = int(self._config.get("reference_replay_target_step", 1))
            reference_ctrl = self._query_bvh_reference(info, target_step)["qpos"]
            residual_scale = jp.array(
                self._config.get("reference_residual_scale", 0.25),
                dtype=smoothed_action.dtype,
            )
        else:
            reference_ctrl = self._default_ctrl
            residual_scale = jp.array(1.0, dtype=smoothed_action.dtype)
        return reference_ctrl + (smoothed_action * self._action_scale * residual_scale)

    def _clip_motor_targets(self, motor_targets: jax.Array) -> jax.Array:
        """Clip PD targets to the active action convention."""
        reference_gait = self._config.get("reference_gait", "none")
        reference_action_mode = self._config.get("reference_action_mode", "mimickit")
        if reference_gait in ("bvh", "smpl") and reference_action_mode == "mimickit":
            return jp.clip(
                motor_targets,
                jp.maximum(
                    self._mimickit_action_lower_limits,
                    self._actuator_ctrl_lower_limits,
                ),
                jp.minimum(
                    self._mimickit_action_upper_limits,
                    self._actuator_ctrl_upper_limits,
                ),
            )
        return jp.clip(
            motor_targets,
            self._actuator_ctrl_lower_limits,
            self._actuator_ctrl_upper_limits,
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

    def step_with_joint_torque_injection(
        self,
        data: mjx.Data,
        motor_targets: jax.Array,
        rng: jax.Array,
        episode_offset: jax.Array,
        use_rfi: jax.Array,
    ) -> mjx.Data:
        """Korak fizike uz RFI uzorkovanje na svakom impedance substep-u."""
        substep_keys = jax.random.split(rng, self._n_substeps)

        def single_step(current_data, substep_key):
            erfi_torque = self.sample_erfi_torque(
                substep_key,
                episode_offset,
                use_rfi,
            )
            current_data = self.apply_joint_torque_injection(
                current_data,
                erfi_torque,
            )
            current_data = current_data.replace(ctrl=motor_targets)
            current_data = mjx.step(self._mjx_model, current_data)
            return current_data, None

        return jax.lax.scan(single_step, data, substep_keys)[0]

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict,
    ) -> dict[str, jax.Array] | jax.Array:
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
            joint_pos,
            joint_vel,
            info["last_action"],
        ]
        optional_parts = []
        if self._include_gait_phase_observation:
            optional_parts.append(self._get_gait_phase_obs(info))
        if self._include_reference_target_observation:
            optional_parts.append(self._get_reference_target_observation(data, info))
        state_parts[4:4] = optional_parts
        state_obs = jp.concatenate(state_parts)
        state_obs = jp.nan_to_num(state_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        if not self._config.get("policy_observation_dict", True):
            return state_obs

        privileged_obs = jp.concatenate([
            state_obs,
            data.qpos[:3],
            data.qvel[:6],
            self._gymnasium_privileged_obs(data),
            data.qfrc_actuator[self._actuator_dof_indices],
            self._foot_positions(data).reshape(-1),
            self._foot_contact(data, info),
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

    def _get_reference_target_observation(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Policy sees DeepMimic-style current/future target frames."""
        if self._reference_target_observation_mode == "legacy":
            return self._get_reference_target_delta(info)
        if self._config.get("reference_gait", "none") == "none":
            return jp.zeros(self._reference_target_observation_size())
        return jp.concatenate([
            self._get_reference_target_frame_observation(data, info, step)
            for step in self._bvh_target_observation_steps
        ])

    def _get_reference_target_delta(self, info: dict) -> jax.Array:
        """Legacy actuator target delta for old checkpoint layouts."""
        if self._config.get("reference_gait", "none") == "none":
            return jp.zeros(self.action_size)
        target_qpos = self._get_reference_gait_target(info)
        return jp.where(
            self._reference_gait_mask,
            target_qpos - self._default_ctrl,
            0.0,
        )

    def _get_reference_target_frame_observation(
        self,
        data: mjx.Data,
        info: dict,
        future_step: int,
    ) -> jax.Array:
        """One target frame: actuator, root, and key-body references."""
        reference = self._query_bvh_reference(info, future_step)
        target_qpos = reference["qpos"]
        target_qvel = reference["qvel"]
        target_root_pos = reference["root_pos"]
        target_root_quat = reference["root_quat"]
        target_key_pos = reference["key_pos"]
        sim_root_pos = self._reference_anchor_pos(data)
        sim_root_quat = self._reference_anchor_quat(data)
        sim_heading = self._heading_world_to_local_from_quat(sim_root_quat)
        ref_heading = self._heading_world_to_local_from_quat(target_root_quat)
        root_delta = (target_root_pos - sim_root_pos) @ sim_heading.T
        # Local-root mode: XY progress is still useful to the policy, but height
        # is the only root position term used by the DeepMimic reward.
        root_quat_delta = self._quat_mul(
            self._quat_conjugate(self._heading_localize_quat(sim_root_quat)),
            self._heading_localize_quat(target_root_quat),
        )
        key_rel = (target_key_pos - target_root_pos) @ ref_heading.T
        current_qpos = data.qpos[self._actuator_qpos_indices]
        current_qvel = data.qvel[self._actuator_dof_indices]
        return jp.concatenate([
            jp.where(self._reference_gait_mask, target_qpos - current_qpos, 0.0),
            jp.where(
                self._reference_gait_mask,
                0.1 * (target_qvel - current_qvel),
                0.0,
            ),
            root_delta,
            root_quat_delta,
            key_rel.reshape(-1),
        ])

    def _reference_target_observation_size(self) -> int:
        """Size of the new MimicKit-style target observation block."""
        per_frame = (
            2 * self.action_size
            + 3
            + 4
            + 3 * self._deepmimic_key_count
        )
        return per_frame * len(self._bvh_target_observation_steps)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> jax.Array:
        """Nagrada za joystick hod: prati command vektor, stabilnost je uslov."""
        # REF: PROJECT-COMMAND-TRACKING-SPLIT
        # TYPE: ENGINEERING_DEFAULT
        local_angvel = self._local_root_angvel(data)
        measured_command = self._measured_command(data)
        command_lin_norm = jp.linalg.norm(info["command"][:2])
        measured_lin_norm = jp.linalg.norm(measured_command[:2])
        command_yaw_abs = jp.abs(info["command"][2])
        measured_yaw_abs = jp.abs(measured_command[2])
        command_active = (
            command_lin_norm > self.STUCK_COMMAND_THRESHOLD
        ) | (command_yaw_abs > self.STUCK_COMMAND_THRESHOLD)
        tracking = self._get_tracking_reward(data, info)
        tracking_lin = self._get_tracking_lin_reward(data, info)
        tracking_yaw = self._get_tracking_yaw_reward(data, info)
        command_progress = self._get_command_progress(data, info)
        overspeed_denominator = jp.maximum(command_lin_norm, 0.05)
        overspeed = jp.maximum(measured_lin_norm - 1.25 * command_lin_norm, 0.0)
        yaw_overspeed_denominator = jp.maximum(command_yaw_abs, 0.05)
        yaw_overspeed = jp.maximum(measured_yaw_abs - 1.25 * command_yaw_abs, 0.0)
        overspeed_cost = self.OVERSPEED_COST_SCALE * jp.square(
            overspeed / overspeed_denominator
        ) + 0.25 * self.OVERSPEED_COST_SCALE * jp.square(
            yaw_overspeed / yaw_overspeed_denominator
        )
        idle_motion_cost = jp.where(
            command_active,
            0.0,
            0.25 * (jp.square(measured_lin_norm) + 0.25 * jp.square(measured_yaw_abs)),
        )
        commanded_axis = info["command"][:2] / jp.maximum(command_lin_norm, 1e-6)
        velocity_along_command = jp.dot(measured_command[:2], commanded_axis)
        stuck_penalty = jp.where(
            (command_lin_norm > self.STUCK_COMMAND_THRESHOLD)
            & (velocity_along_command < self.STUCK_VELOCITY_THRESHOLD),
            self.STUCK_PENALTY,
            0.0,
        )
        upright = jp.clip(self._torso_up(data), 0.0, 1.0)
        head_up = jp.clip(self._head_up(data), 0.0, 1.0)
        low_height = jp.maximum(
            0.0,
            self.HEIGHT_PENALTY_START_RATIO * self._standing_height()
            - data.qpos[2],
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
        variable_posture_reward = (
            self.VARIABLE_POSTURE_REWARD_SCALE
            * jp.exp(-0.5 * variable_posture_error)
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
        base_height_reward = self.BASE_HEIGHT_REWARD_SCALE * (
            self._get_base_height_reward(data)
        )
        bvh_mode = self._config.get("reference_gait", "none") in ("bvh", "smpl")
        if bvh_mode:
            gait_reward = jp.array(0.0)
            swing_clearance_deficit_cost = jp.array(0.0)
        else:
            gait_reward = self._get_gait_reward(data, info)
            swing_clearance_deficit_cost = (
                self.SWING_CLEARANCE_DEFICIT_COST_SCALE
                * self._get_swing_clearance_deficit_cost(data, info)
            )
        reference_gait_reward = (
            self.REFERENCE_GAIT_REWARD_SCALE
            * self._get_reference_gait_reward(data, info)
        )
        if (
            bvh_mode
            and self._config.get("deepmimic_reward_mode", "pure") == "pure"
        ):
            # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
            # TYPE: REFERENCE_CODE_DERIVED
            # Pure DeepMimic pretraining should not fight joystick progress,
            # overspeed, or old posture priors. Physical falls still keep the
            # project-level terminal penalty so PPO does not treat early falls
            # as neutral episode endings.
            reward = self.REWARD_MAX * self._get_bvh_deepmimic_reward(
                data,
                info,
            )["total"]
            reward = jp.clip(reward, 0.0, self.REWARD_MAX)
            reward = jp.nan_to_num(
                reward,
                nan=0.0,
                posinf=self.REWARD_MAX,
                neginf=0.0,
            )
            physical_fail = self._get_physical_fail_done(data)
            terminal_reward = jp.where(
                physical_fail,
                jp.array(self.FALL_REWARD, dtype=reward.dtype),
                jp.array(0.0, dtype=reward.dtype),
            )
            return jp.where(self._get_done(data, info), terminal_reward, reward)
        reference_velocity_reward = (
            self.REFERENCE_VELOCITY_REWARD_SCALE
            * self._get_reference_velocity_reward(data, info)
        )
        contact_force_cost = self._get_contact_force_cost(data)
        if bvh_mode:
            foot_slip_cost = jp.array(0.0)
            swing_drag_cost = jp.array(0.0)
        else:
            foot_slip_cost = self.FOOT_SLIP_COST_SCALE * self._get_foot_slip_cost(
                data,
                info,
            )
            swing_drag_cost = (
                self.SWING_FOOT_DRAG_COST_SCALE
                * self._get_swing_foot_drag_cost(data, info)
            )
        velocity_tracking_scale = self.VELOCITY_TRACKING_REWARD_SCALE
        forward_progress_scale = self.FORWARD_PROGRESS_REWARD_SCALE
        if self._config.command_profile == "forward_slow":
            velocity_tracking_scale = 0.8
            forward_progress_scale = 0.25
        height_cost = self.LOW_HEIGHT_COST_SCALE * jp.square(low_height)
        if bvh_mode:
            deepmimic = self._get_bvh_deepmimic_reward(data, info)
            foot_contact = self._foot_contact(data, info)
            support_reward = jp.maximum(foot_contact[0], foot_contact[1])
            # Clean mixed objective:
            # - DeepMimic handles "look like the reference"
            # - task reward handles "stay alive and follow the command"
            # Avoid duplicating reference/posture/gait priors on top of
            # imitation, because they pull the policy toward overlapping but
            # slightly different notions of "good walking".
            task_reward = (
                self.ALIVE_REWARD_SCALE
                + velocity_tracking_scale * tracking
                + forward_progress_scale * command_progress
                + self.UPRIGHT_REWARD_SCALE * upright
                + self.HEAD_UP_REWARD_SCALE * head_up
                + base_height_reward
                + 0.25 * support_reward
                - stuck_penalty
                - action_cost
                - action_rate_cost
                - contact_force_cost
                - height_cost
                - overspeed_cost
                - idle_motion_cost
                - vertical_velocity_cost
                - angular_velocity_cost
                - 0.25 * self.FOOT_SLIP_COST_SCALE * self._get_foot_slip_cost(data, info)
                - 0.35
                * self.SWING_FOOT_DRAG_COST_SCALE
                * self._get_swing_foot_drag_cost(data, info)
                - 0.25
                * self.SWING_CLEARANCE_DEFICIT_COST_SCALE
                * self._get_swing_clearance_deficit_cost(data, info)
            )
            reward = (
                0.85 * deepmimic["total"]
                + 0.65 * task_reward
            )
        else:
            reward = (
                self.ALIVE_REWARD_SCALE
                + velocity_tracking_scale * tracking
                + forward_progress_scale * command_progress
                + self.UPRIGHT_REWARD_SCALE * upright
                + self.HEAD_UP_REWARD_SCALE * head_up
                + base_height_reward
                + posture_reward
                + variable_posture_reward
                + gait_reward
                + reference_gait_reward
                + reference_velocity_reward
                - stuck_penalty
                - action_cost
                - action_rate_cost
                - trunk_posture_cost
                - variable_posture_cost
                - contact_force_cost
                - foot_slip_cost
                - swing_drag_cost
                - swing_clearance_deficit_cost
                - height_cost
                - overspeed_cost
                - idle_motion_cost
                - vertical_velocity_cost
                - angular_velocity_cost
            )
        reward = jp.clip(reward, self.REWARD_MIN, self.REWARD_MAX)
        reward = jp.nan_to_num(
            reward,
            nan=0.0,
            posinf=self.REWARD_MAX,
            neginf=self.REWARD_MIN,
        )
        physical_fail = self._get_physical_fail_done(data)
        terminal_reward = jp.where(physical_fail, self.FALL_REWARD, jp.array(0.0))
        return jp.where(self._get_done(data, info), terminal_reward, reward)

    def _get_done(self, data: mjx.Data, info: dict | None = None) -> jax.Array:
        """Zavrsi epizodu ako human padne, numerika ode u NaN, ili CLAMP motion istekne."""
        too_low, tipped_over, invalid = self._get_done_reasons(data)
        done = too_low | tipped_over | invalid
        if info is not None:
            done = done | self._get_bvh_motion_over(info)
            done = done | self._get_pose_termination(data, info)
        return done

    def _get_physical_fail_done(self, data: mjx.Data) -> jax.Array:
        too_low, tipped_over, invalid = self._get_done_reasons(data)
        return too_low | tipped_over | invalid

    def _get_done_reasons(
        self,
        data: mjx.Data,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Vrati pojedinacne termination razloge za debug logove."""
        # REF: BIOHUMANOID-FALL-HEIGHT
        # TYPE: MODEL_CALIBRATED
        too_low = (
            data.qpos[2] < self.MIN_STANDING_HEIGHT_RATIO * self._standing_height()
        )
        tipped_over = self._torso_up(data) < 0.25
        invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return too_low, tipped_over, invalid

    def _get_pose_termination(self, data: mjx.Data, info: dict) -> jax.Array:
        """Terminate when key bodies drift too far from the reference pose."""
        # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
        # TYPE: REFERENCE_CODE_DERIVED
        # Copied from MimicKit compute_done pose_termination on key/body positions.
        if self._config.get("reference_gait", "none") not in ("bvh", "smpl"):
            return jp.array(False)
        if not bool(self._config.get("pose_termination", True)):
            return jp.array(False)
        fallback_active = self._reference_fallback_active(info)

        reference = self._query_bvh_reference(info, 0)
        root_pos = self._reference_anchor_pos(data)
        ref_root_pos = reference["root_pos"]
        key_pos = self._key_marker_positions(data)
        ref_key_pos = reference["key_pos"]
        key_rel = key_pos - root_pos
        ref_key_rel = ref_key_pos - ref_root_pos
        body_pos_dist = jp.sum(jp.square(key_rel - ref_key_rel), axis=-1)
        threshold = jp.square(
            jp.array(
                float(self._config.get("pose_termination_dist", self.POSE_TERMINATION_DIST)),
                dtype=jp.float32,
            )
        )
        pose_termination = jp.max(body_pos_dist) > threshold
        return jp.where(fallback_active, jp.array(False), pose_termination)

    def _get_bvh_motion_over(self, info: dict) -> jax.Array:
        """Terminate CLAMP clips when motion time reaches the last frame."""
        # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
        # TYPE: REFERENCE_CODE_DERIVED
        if self._config.get("reference_gait", "none") not in ("bvh", "smpl"):
            return jp.array(False)
        fallback_active = self._reference_fallback_active(info)
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        loop_mode = self._bvh_reference_loop_modes[clip_id]
        motion_time = self._get_bvh_reference_motion_time(info, 0)
        motion_length = self._bvh_reference_motion_lengths[clip_id]
        motion_over = (loop_mode == int(LoopMode.CLAMP)) & (
            motion_time >= motion_length
        )
        return jp.where(fallback_active, jp.array(False), motion_over)

    def _get_gait_phase_angle(self, info: dict) -> jax.Array:
        """Periodican signal koji govori politici koja noga treba da bude swing."""
        if self._config.get("reference_gait", "none") in ("bvh", "smpl"):
            return self._get_bvh_reference_phase_angle(info)
        phase = jp.mod(
            info["gait_step"].astype(jp.float32),
            self.GAIT_PERIOD_STEPS,
        ) / self.GAIT_PERIOD_STEPS
        return 2.0 * jp.pi * phase

    def _get_bvh_reference_phase_angle(self, info: dict) -> jax.Array:
        """Phase signal izveden iz aktivnog BVH clip-a."""
        fallback_active = self._reference_fallback_active(info)
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        motion_time = self._get_bvh_reference_motion_time(info, 0)
        motion_length = self._bvh_reference_motion_lengths[clip_id]
        loop_mode = self._bvh_reference_loop_modes[clip_id]
        phase = motion_time / jp.maximum(motion_length, 1e-6)
        phase = jp.where(
            loop_mode == int(LoopMode.WRAP),
            phase - jp.floor(phase),
            jp.clip(phase, 0.0, 1.0),
        )
        return jp.where(fallback_active, jp.array(0.0), 2.0 * jp.pi * phase)

    def _get_gait_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Nagradi swing clearance i stance contact u fazi koraka."""
        left_swing = self._is_left_swing(info)
        foot_heights = self._foot_heights(data)
        foot_contact = self._foot_contact(data, info)
        stance_contact = jp.where(left_swing, foot_contact[1], foot_contact[0])
        relative_clearance = self._get_swing_clearance(data, info)
        clearance_reward = jp.clip(
            relative_clearance / self.FOOT_CLEARANCE_TARGET,
            0.0,
            1.0,
        )
        command_active = self._command_active(info)
        return jp.where(
            command_active,
            self.FOOT_CLEARANCE_REWARD_SCALE * clearance_reward
            + self.STANCE_CONTACT_REWARD_SCALE * stance_contact,
            0.0,
        )

    def _get_swing_foot_drag_cost(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kazni kada swing noga ostane zalepljena za pod umesto da se podigne."""
        left_swing = self._is_left_swing(info)
        foot_contact = self._foot_contact(data, info)
        swing_contact = jp.where(left_swing, foot_contact[0], foot_contact[1])
        command_active = self._command_active(info)
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
        command_active = self._command_active(info)
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
        if self._config.get("reference_gait", "none") in ("bvh", "smpl"):
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
        if reference_gait not in ("sine", "bvh", "smpl"):
            raise ValueError("reference_gait mora biti 'none', 'sine', 'bvh' ili 'smpl'.")
        if reference_gait in ("bvh", "smpl"):
            return self._get_bvh_deepmimic_reward(data, info)["total"]

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
        command_active = self._command_active(info)
        return jp.where(command_active, pose_reward, 0.0)

    def _get_reference_velocity_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Nagradi BVH-like joint brzine, ne samo staticku pozu."""
        if self._config.get("reference_gait", "none") not in ("bvh", "smpl"):
            return jp.array(0.0)
        deepmimic = self._get_bvh_deepmimic_reward(data, info)
        return deepmimic["velocity"]

    def _get_bvh_deepmimic_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> dict[str, jax.Array]:
        """DeepMimic/MimicKit imitation reward on the clocked reference frame."""
        # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
        # TYPE: REFERENCE_CODE_DERIVED
        # Score only the wall-clock motion frame. Temporal best-match was removed
        # because it rewarded phase free-riding / stalling.
        return self._get_bvh_deepmimic_frame_reward(
            data,
            info,
            jp.array(0, dtype=jp.int32),
            jp.array(0, dtype=jp.int32),
        )

    def _get_bvh_deepmimic_frame_reward(
        self,
        data: mjx.Data,
        info: dict,
        clip_id: jax.Array,
        frame_index: jax.Array,
    ) -> dict[str, jax.Array]:
        """Score one simulated state against one BVH reference frame."""
        # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
        # TYPE: REFERENCE_CODE_DERIVED
        # Pose/velocity use actuator DOFs (MimicKit joint_rot/dof_vel). Root XY is
        # ignored like MimicKit local-root mode; key bodies are heading-local.
        del clip_id, frame_index
        reference = self._query_bvh_reference(info, 0)
        ref_qpos = reference["qpos"]
        ref_qvel = reference["qvel"]
        ref_key_pos = reference["key_pos"]
        ref_root_pos = reference["root_pos"]
        ref_root_quat = reference["root_quat"]
        ref_root_vel = reference["root_vel"]
        ref_root_angvel = reference["root_angvel"]
        qpos = data.qpos[self._actuator_qpos_indices]
        qvel = data.qvel[self._actuator_dof_indices]
        key_pos = self._key_marker_positions(data)
        root_pos = self._reference_anchor_pos(data)
        root_quat = self._reference_anchor_quat(data)
        root_vel = self._reference_anchor_linvel(data)
        root_angvel = self._reference_anchor_angvel(data)

        pose_error = jp.mean(jp.square(qpos - ref_qpos))
        velocity_error = jp.mean(jp.square(qvel - ref_qvel))

        sim_heading = self._heading_world_to_local_from_quat(root_quat)
        ref_heading = self._heading_world_to_local_from_quat(ref_root_quat)
        key_rel = (key_pos - root_pos) @ sim_heading.T
        ref_key_rel = (ref_key_pos - ref_root_pos) @ ref_heading.T
        key_delta = key_rel - ref_key_rel
        key_position_error = jp.sum(jp.square(key_delta))
        key_distances = jp.linalg.norm(key_delta, axis=-1)
        key_pos_error = jp.mean(key_distances)
        max_key_dist = jp.max(key_distances)

        # Local-root mode: ignore absolute XY. Height still matters.
        root_pos_diff = root_pos - ref_root_pos
        root_xy_error = jp.linalg.norm(root_pos_diff[:2])
        root_height_error = jp.abs(root_pos_diff[2])
        root_pos_diff = root_pos_diff.at[:2].set(0.0)
        root_pos_error = jp.sum(jp.square(root_pos_diff))

        local_root_quat = self._heading_localize_quat(root_quat)
        local_ref_root_quat = self._heading_localize_quat(ref_root_quat)
        root_rot_error = self._quat_distance(local_root_quat, local_ref_root_quat)

        local_root_vel = sim_heading @ root_vel
        local_ref_root_vel = ref_heading @ ref_root_vel
        local_root_angvel = sim_heading @ root_angvel
        local_ref_root_angvel = ref_heading @ ref_root_angvel
        root_vel_delta = local_root_vel - local_ref_root_vel
        root_angvel_delta = local_root_angvel - local_ref_root_angvel
        root_vel_error = jp.sum(jp.square(root_vel_delta))
        root_angvel_error = jp.sum(jp.square(root_angvel_delta))
        root_vel_error_norm = jp.linalg.norm(root_vel_delta)
        root_angvel_error_norm = jp.linalg.norm(root_angvel_delta)
        root_pose_error = root_pos_error + 0.1 * root_rot_error
        root_velocity_error = root_vel_error + 0.1 * root_angvel_error

        # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
        # TYPE: REFERENCE_CODE_DERIVED
        # Matches MimicKit's weighted pose/vel/root_pose/root_vel/key_pos split.
        pose_reward = jp.exp(-self.DEEPMIMIC_POSE_SCALE * pose_error)
        velocity_reward = jp.exp(-self.DEEPMIMIC_VELOCITY_SCALE * velocity_error)
        root_pose_reward = jp.exp(-self.DEEPMIMIC_ROOT_POSE_SCALE * root_pose_error)
        root_velocity_reward = jp.exp(
            -self.DEEPMIMIC_ROOT_VELOCITY_SCALE * root_velocity_error
        )
        key_position_reward = jp.exp(
            -self.DEEPMIMIC_KEY_POSITION_SCALE * key_position_error
        )
        root_velocity_weight = (
            self.DEEPMIMIC_ROOT_VELOCITY_WEIGHT
            * float(self._config.get("deepmimic_root_velocity_weight_scale", 1.0))
        )
        total_weight = (
            self.DEEPMIMIC_POSE_WEIGHT
            + self.DEEPMIMIC_VELOCITY_WEIGHT
            + self.DEEPMIMIC_ROOT_POSE_WEIGHT
            + root_velocity_weight
            + self.DEEPMIMIC_KEY_POSITION_WEIGHT
        )
        total_reward = (
            self.DEEPMIMIC_POSE_WEIGHT * pose_reward
            + self.DEEPMIMIC_VELOCITY_WEIGHT * velocity_reward
            + self.DEEPMIMIC_ROOT_POSE_WEIGHT * root_pose_reward
            + root_velocity_weight * root_velocity_reward
            + self.DEEPMIMIC_KEY_POSITION_WEIGHT * key_position_reward
        ) / max(total_weight, 1e-6)
        no_root_velocity_weight = (
            self.DEEPMIMIC_POSE_WEIGHT
            + self.DEEPMIMIC_VELOCITY_WEIGHT
            + self.DEEPMIMIC_ROOT_POSE_WEIGHT
            + self.DEEPMIMIC_KEY_POSITION_WEIGHT
        )
        total_no_root_velocity = (
            self.DEEPMIMIC_POSE_WEIGHT * pose_reward
            + self.DEEPMIMIC_VELOCITY_WEIGHT * velocity_reward
            + self.DEEPMIMIC_ROOT_POSE_WEIGHT * root_pose_reward
            + self.DEEPMIMIC_KEY_POSITION_WEIGHT * key_position_reward
        ) / max(no_root_velocity_weight, 1e-6)
        return {
            "total": total_reward,
            "total_no_root_velocity": total_no_root_velocity,
            "pose": pose_reward,
            "velocity": velocity_reward,
            "end_effector": key_position_reward,
            "root": root_pose_reward,
            "com": root_velocity_reward,
            "root_pose": root_pose_reward,
            "root_velocity": root_velocity_reward,
            "key_position": key_position_reward,
            "pose_error": pose_error,
            "velocity_error": velocity_error,
            "root_xy_error": root_xy_error,
            "root_height_error": root_height_error,
            "root_vel_error": root_vel_error_norm,
            "root_angvel_error": root_angvel_error_norm,
            "key_pos_error": key_pos_error,
            "max_key_dist": max_key_dist,
        }

    def _reference_anchor_pos(self, data: mjx.Data) -> jax.Array:
        """World position of the locomotion anchor body used for imitation."""
        return data.xpos[self._reference_anchor_body_id]

    def _reference_anchor_linvel(self, data: mjx.Data) -> jax.Array:
        """World linear velocity of the locomotion anchor body."""
        return data.cvel[self._reference_anchor_body_id, 3:6]

    def _reference_anchor_angvel(self, data: mjx.Data) -> jax.Array:
        """World angular velocity of the locomotion anchor body."""
        return data.cvel[self._reference_anchor_body_id, :3]

    def _reference_anchor_quat(self, data: mjx.Data) -> jax.Array:
        """World quaternion of the locomotion anchor body from its xmat."""
        return self._xmat_to_quat(data.xmat[self._reference_anchor_body_id])

    def _reference_key_foot_heights(
        self,
        reference: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        """Approximate reference left/right foot heights from key markers.

        The default imitation setup uses right then left metatarsal markers as
        DeepMimic key points, so these heights are primarily a forensic signal
        for playback audits rather than a new training objective.
        """
        key_pos = reference["key_pos"]
        zero = jp.array(0.0, dtype=jp.float32)
        right_height = key_pos[0, 2] if self._deepmimic_key_count >= 1 else zero
        left_height = key_pos[1, 2] if self._deepmimic_key_count >= 2 else right_height
        return left_height.astype(jp.float32), right_height.astype(jp.float32)

    def _get_reference_trace_metrics(
        self,
        data: mjx.Data,
        info: dict,
    ) -> dict[str, jax.Array]:
        """Collect reference-vs-sim root/foot diagnostics for playback traces."""
        foot_contact = self._foot_contact(data, info)
        if self._config.get("reference_gait", "none") not in ("bvh", "smpl"):
            zero = jp.array(0.0, dtype=jp.float32)
            return {
                "reference_root_height": zero,
                "reference_root_vertical_velocity": zero,
                "reference_left_foot_height": zero,
                "reference_right_foot_height": zero,
                "root_height_tracking_error": zero,
                "root_vertical_velocity_tracking_error": zero,
                "left_foot_height_tracking_error": zero,
                "right_foot_height_tracking_error": zero,
                "left_foot_contact": foot_contact[0].astype(jp.float32),
                "right_foot_contact": foot_contact[1].astype(jp.float32),
            }

        reference = self._query_bvh_reference(info, 0)
        foot_heights = self._foot_heights(data)
        root_height = self._reference_anchor_pos(data)[2]
        root_vertical_velocity = self._reference_anchor_linvel(data)[2]
        reference_root_height = reference["root_pos"][2].astype(jp.float32)
        reference_root_vertical_velocity = reference["root_vel"][2].astype(jp.float32)
        reference_left_foot_height, reference_right_foot_height = (
            self._reference_key_foot_heights(reference)
        )
        return {
            "reference_root_height": reference_root_height,
            "reference_root_vertical_velocity": reference_root_vertical_velocity,
            "reference_left_foot_height": reference_left_foot_height,
            "reference_right_foot_height": reference_right_foot_height,
            "root_height_tracking_error": jp.abs(
                root_height - reference_root_height
            ).astype(jp.float32),
            "root_vertical_velocity_tracking_error": jp.abs(
                root_vertical_velocity - reference_root_vertical_velocity
            ).astype(jp.float32),
            "left_foot_height_tracking_error": jp.abs(
                foot_heights[0] - reference_left_foot_height
            ).astype(jp.float32),
            "right_foot_height_tracking_error": jp.abs(
                foot_heights[1] - reference_right_foot_height
            ).astype(jp.float32),
            "left_foot_contact": foot_contact[0].astype(jp.float32),
            "right_foot_contact": foot_contact[1].astype(jp.float32),
        }

    def _quat_distance(self, quat_a: jax.Array, quat_b: jax.Array) -> jax.Array:
        """Quaternion orientation distance, invariant to q and -q."""
        dot = jp.abs(jp.sum(quat_a * quat_b, axis=-1))
        angle = 2.0 * jp.arccos(jp.clip(dot, -1.0 + 1e-6, 1.0 - 1e-6))
        return jp.square(angle)

    def _quat_conjugate(self, quat: jax.Array) -> jax.Array:
        """Quaternion conjugate in MuJoCo wxyz order."""
        return jp.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def _quat_mul(self, left: jax.Array, right: jax.Array) -> jax.Array:
        """Quaternion multiply in MuJoCo wxyz order."""
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        quat = jp.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
        return quat / jp.maximum(jp.linalg.norm(quat), 1e-6)

    def _quat_slerp(
        self,
        quat0: jax.Array,
        quat1: jax.Array,
        alpha: jax.Array,
    ) -> jax.Array:
        """Shortest-path quaternion interpolation in MuJoCo wxyz order."""
        quat0 = quat0 / jp.maximum(jp.linalg.norm(quat0), 1e-6)
        quat1 = quat1 / jp.maximum(jp.linalg.norm(quat1), 1e-6)
        dot = jp.sum(quat0 * quat1)
        quat1 = jp.where(dot < 0.0, -quat1, quat1)
        dot = jp.abs(jp.sum(quat0 * quat1))
        linear = quat0 + alpha * (quat1 - quat0)
        linear = linear / jp.maximum(jp.linalg.norm(linear), 1e-6)

        theta0 = jp.arccos(jp.clip(dot, -1.0 + 1e-6, 1.0 - 1e-6))
        sin_theta0 = jp.sin(theta0)
        theta = theta0 * alpha
        scale0 = jp.sin(theta0 - theta) / jp.maximum(sin_theta0, 1e-6)
        scale1 = jp.sin(theta) / jp.maximum(sin_theta0, 1e-6)
        spherical = scale0 * quat0 + scale1 * quat1
        spherical = spherical / jp.maximum(jp.linalg.norm(spherical), 1e-6)
        return jp.where(dot > 0.9995, linear, spherical)

    def _xmat_to_quat(self, xmat: jax.Array) -> jax.Array:
        """Convert MuJoCo 3x3/9-body matrix into a normalized wxyz quaternion."""
        matrix = xmat.reshape((3, 3))
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        r = jp.sqrt(jp.maximum(1.0 + trace, 1e-6))
        w = 0.5 * r
        denom = 0.5 / jp.maximum(r, 1e-6)
        x = (matrix[2, 1] - matrix[1, 2]) * denom
        y = (matrix[0, 2] - matrix[2, 0]) * denom
        z = (matrix[1, 0] - matrix[0, 1]) * denom
        quat = jp.array([w, x, y, z], dtype=matrix.dtype)
        quat = quat / jp.maximum(jp.linalg.norm(quat), 1e-6)
        return jp.where(quat[0] < 0.0, -quat, quat)

    def _resolve_deepmimic_key_marker_ids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve key markers as body or site ids, preferring locomotion sites."""
        # REF: MIMICKIT-DEEPMIMIC-HUMANOID-CONFIG
        # TYPE: REFERENCE_CODE_DERIVED
        configured = self._config.get(
            "deepmimic_key_bodies",
            ("metatarsal_midpoint_right", "metatarsal_midpoint_left"),
        )
        if isinstance(configured, str):
            key_body_names = tuple(
                body.strip()
                for body in configured.split(",")
                if body.strip()
            )
        else:
            key_body_names = tuple(str(body) for body in configured)
        body_ids: list[int] = []
        site_ids: list[int] = []
        is_site: list[bool] = []
        marker_aliases = {
            "right_foot": "metatarsal_midpoint_right",
            "left_foot": "metatarsal_midpoint_left",
        }
        for body_name in key_body_names:
            preferred_name = marker_aliases.get(body_name, body_name)
            if preferred_name != body_name:
                try:
                    site_ids.append(self._mj_model.site(preferred_name).id)
                    body_ids.append(0)
                    is_site.append(True)
                    continue
                except KeyError:
                    pass
            try:
                site_ids.append(self._mj_model.site(body_name).id)
                body_ids.append(0)
                is_site.append(True)
                continue
            except KeyError:
                pass
            try:
                body_ids.append(self._mj_model.body(body_name).id)
                site_ids.append(0)
                is_site.append(False)
            except KeyError:
                continue
        if not body_ids and not site_ids:
            for body_name in (
                "metatarsal_midpoint_right",
                "metatarsal_midpoint_left",
                "right_foot",
                "left_foot",
            ):
                try:
                    site_ids.append(self._mj_model.site(body_name).id)
                    body_ids.append(0)
                    is_site.append(True)
                    continue
                except KeyError:
                    pass
                try:
                    body_ids.append(self._mj_model.body(body_name).id)
                    site_ids.append(0)
                    is_site.append(False)
                except KeyError:
                    continue
        if not body_ids and not site_ids:
            body_ids = [self._head_body_id]
            site_ids = [0]
            is_site = [False]
        return (
            np.asarray(body_ids, dtype=np.int32),
            np.asarray(site_ids, dtype=np.int32),
            np.asarray(is_site, dtype=bool),
        )

    def _key_marker_positions_host(self, data: mujoco.MjData) -> np.ndarray:
        """Return configured key marker positions from host MuJoCo data."""
        body_positions = np.asarray(data.xpos[self._deepmimic_key_body_ids_np], dtype=np.float32)
        site_positions = np.asarray(
            data.site_xpos[self._deepmimic_key_site_ids_np],
            dtype=np.float32,
        )
        return np.where(
            self._deepmimic_key_is_site_np[:, None],
            site_positions,
            body_positions,
        )

    def _key_marker_positions(self, data: mjx.Data) -> jax.Array:
        """Return configured key marker positions from MJX data."""
        body_positions = data.xpos[self._deepmimic_key_body_ids]
        site_positions = data.site_xpos[self._deepmimic_key_site_ids]
        return jp.where(
            self._deepmimic_key_is_site[:, None],
            site_positions,
            body_positions,
        )

    def _center_of_mass(self, data: mjx.Data) -> jax.Array:
        """Mass-weighted body center in world coordinates."""
        return jp.sum(
            data.xpos[self._deepmimic_body_ids] * self._deepmimic_body_weights[:, None],
            axis=0,
        )

    def _center_of_mass_velocity(self, data: mjx.Data) -> jax.Array:
        """Mass-weighted linear body velocity from MJX body spatial velocities."""
        return jp.sum(
            data.cvel[self._deepmimic_body_ids, 3:6]
            * self._deepmimic_body_weights[:, None],
            axis=0,
        )

    def _get_reference_gait_target(self, info: dict) -> jax.Array:
        """Vrati target pozu za aktivni reference gait izvor."""
        if self._config.get("reference_gait", "none") in ("bvh", "smpl"):
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
        """Vrati interpoliranu BVH pozu za trenutno motion vreme."""
        return self._query_bvh_reference(info, 0)["qpos"]

    def _get_bvh_reference_velocity_target(self, info: dict) -> jax.Array:
        """Vrati interpoliranu BVH joint brzinu za trenutno motion vreme."""
        return self._query_bvh_reference(info, 0)["qvel"]

    def _sample_weighted_bvh_clip_id(self, rng: jax.Array) -> jax.Array:
        """Sample clip ids using MimicKit-style motion weights."""
        if self._bvh_reference_clip_count == 1:
            return jp.array(0, dtype=jp.int32)
        return jax.random.choice(
            rng,
            a=jp.arange(self._bvh_reference_clip_count, dtype=jp.int32),
            p=self._bvh_reference_weights,
            shape=(),
        ).astype(jp.int32)

    def _sample_bvh_motion_time(
        self,
        rng: jax.Array,
        clip_id: jax.Array,
    ) -> jax.Array:
        """Sample motion time, keeping finite clips away from the terminal edge."""
        motion_length = self._bvh_reference_motion_lengths[clip_id]
        loop_mode = self._bvh_reference_loop_modes[clip_id]
        min_remaining_steps = int(
            self._config.get("reference_reset_min_steps_remaining", 1)
        )
        clamp_margin = jp.array(
            max(1, min_remaining_steps) * self.dt,
            dtype=jp.float32,
        )
        usable_length = jp.where(
            loop_mode == int(LoopMode.CLAMP),
            jp.maximum(motion_length - clamp_margin, 0.0),
            motion_length,
        )
        phase = jax.random.uniform(rng, shape=(), minval=0.0, maxval=1.0)
        return phase * jp.maximum(usable_length, 0.0)

    def _reference_fallback_active(self, info: dict) -> jax.Array:
        return info.get("reference_fallback_standing", jp.array(False))

    def _get_bvh_reference_motion_time(
        self,
        info: dict,
        future_step: int,
    ) -> jax.Array:
        """Continuous motion time offset used by every BVH query."""
        return (
            info.get("bvh_reference_time_offset", jp.array(0.0, dtype=jp.float32))
            + (
                info["motion_step"].astype(jp.float32)
                + jp.array(future_step, dtype=jp.float32)
            )
            * jp.array(self.dt, dtype=jp.float32)
        )

    def _query_bvh_reference_clip(
        self,
        clip_id: jax.Array,
        motion_time: jax.Array,
    ) -> dict[str, jax.Array]:
        """Interpolate one BVH clip at continuous motion time."""
        motion_length = self._bvh_reference_motion_lengths[clip_id]
        frame_count = self._bvh_reference_frame_counts[clip_id]
        loop_mode = self._bvh_reference_loop_modes[clip_id]
        loop_count = jp.where(
            loop_mode == int(LoopMode.WRAP),
            jp.floor(motion_time / jp.maximum(motion_length, 1e-6)),
            0.0,
        )
        phase = motion_time / jp.maximum(motion_length, 1e-6)
        phase = jp.where(
            loop_mode == int(LoopMode.WRAP),
            phase - jp.floor(phase),
            jp.clip(phase, 0.0, 1.0),
        )
        frame_float = phase * jp.maximum(frame_count.astype(jp.float32) - 1.0, 0.0)
        frame_index0 = jp.floor(frame_float).astype(jp.int32)
        frame_index1 = jp.minimum(frame_index0 + 1, frame_count - 1)
        alpha = frame_float - frame_index0.astype(jp.float32)
        root_offset = self._bvh_reference_wrap_deltas[clip_id] * loop_count

        qpos0 = self._bvh_reference_qpos_targets[clip_id, frame_index0]
        qpos1 = self._bvh_reference_qpos_targets[clip_id, frame_index1]
        qvel0 = self._bvh_reference_qvel_targets[clip_id, frame_index0]
        qvel1 = self._bvh_reference_qvel_targets[clip_id, frame_index1]
        root_pos0 = self._bvh_reference_root_pos_targets[clip_id, frame_index0]
        root_pos1 = self._bvh_reference_root_pos_targets[clip_id, frame_index1]
        root_quat0 = self._bvh_reference_root_quat_targets[clip_id, frame_index0]
        root_quat1 = self._bvh_reference_root_quat_targets[clip_id, frame_index1]
        root_vel0 = self._bvh_reference_root_vel_targets[clip_id, frame_index0]
        root_vel1 = self._bvh_reference_root_vel_targets[clip_id, frame_index1]
        root_angvel0 = self._bvh_reference_root_angvel_targets[
            clip_id,
            frame_index0,
        ]
        root_angvel1 = self._bvh_reference_root_angvel_targets[
            clip_id,
            frame_index1,
        ]
        sim_root_pos0 = self._bvh_reference_sim_root_pos_targets[clip_id, frame_index0]
        sim_root_pos1 = self._bvh_reference_sim_root_pos_targets[clip_id, frame_index1]
        sim_root_quat0 = self._bvh_reference_sim_root_quat_targets[clip_id, frame_index0]
        sim_root_quat1 = self._bvh_reference_sim_root_quat_targets[clip_id, frame_index1]
        sim_root_vel0 = self._bvh_reference_sim_root_vel_targets[clip_id, frame_index0]
        sim_root_vel1 = self._bvh_reference_sim_root_vel_targets[clip_id, frame_index1]
        sim_root_angvel0 = self._bvh_reference_sim_root_angvel_targets[clip_id, frame_index0]
        sim_root_angvel1 = self._bvh_reference_sim_root_angvel_targets[clip_id, frame_index1]
        reset_root_pos0 = self._bvh_reference_reset_root_pos_targets[
            clip_id,
            frame_index0,
        ]
        reset_root_pos1 = self._bvh_reference_reset_root_pos_targets[
            clip_id,
            frame_index1,
        ]
        reset_root_quat0 = self._bvh_reference_reset_root_quat_targets[
            clip_id,
            frame_index0,
        ]
        reset_root_quat1 = self._bvh_reference_reset_root_quat_targets[
            clip_id,
            frame_index1,
        ]
        reset_root_vel0 = self._bvh_reference_reset_root_vel_targets[
            clip_id,
            frame_index0,
        ]
        reset_root_vel1 = self._bvh_reference_reset_root_vel_targets[
            clip_id,
            frame_index1,
        ]
        reset_root_angvel0 = self._bvh_reference_reset_root_angvel_targets[
            clip_id,
            frame_index0,
        ]
        reset_root_angvel1 = self._bvh_reference_reset_root_angvel_targets[
            clip_id,
            frame_index1,
        ]
        key_rel0 = self._bvh_reference_key_rel_local_targets[clip_id, frame_index0]
        key_rel1 = self._bvh_reference_key_rel_local_targets[clip_id, frame_index1]

        qpos = qpos0 + alpha * (qpos1 - qpos0)
        qvel = qvel0 + alpha * (qvel1 - qvel0)
        root_pos = root_pos0 + alpha * (root_pos1 - root_pos0) + root_offset
        root_quat = self._quat_slerp(root_quat0, root_quat1, alpha)
        root_vel = root_vel0 + alpha * (root_vel1 - root_vel0)
        sim_root_pos = sim_root_pos0 + alpha * (sim_root_pos1 - sim_root_pos0) + root_offset
        sim_root_quat = self._quat_slerp(sim_root_quat0, sim_root_quat1, alpha)
        sim_root_vel = sim_root_vel0 + alpha * (sim_root_vel1 - sim_root_vel0)
        root_angvel = root_angvel0 + alpha * (root_angvel1 - root_angvel0)
        sim_root_angvel = sim_root_angvel0 + alpha * (sim_root_angvel1 - sim_root_angvel0)
        reset_root_pos = (
            reset_root_pos0 + alpha * (reset_root_pos1 - reset_root_pos0) + root_offset
        )
        reset_root_quat = self._quat_slerp(
            reset_root_quat0,
            reset_root_quat1,
            alpha,
        )
        reset_root_vel = reset_root_vel0 + alpha * (
            reset_root_vel1 - reset_root_vel0
        )
        reset_root_angvel = reset_root_angvel0 + alpha * (
            reset_root_angvel1 - reset_root_angvel0
        )
        ref_heading = self._heading_world_to_local_from_quat(root_quat)
        key_rel = key_rel0 + alpha * (key_rel1 - key_rel0)
        key_pos = root_pos + key_rel @ ref_heading
        return {
            "clip_id": clip_id,
            "frame_index0": frame_index0,
            "frame_index1": frame_index1,
            "alpha": alpha,
            "loop_count": loop_count,
            "qpos": qpos,
            "qvel": qvel,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "root_vel": root_vel,
            "root_angvel": root_angvel,
            "sim_root_pos": sim_root_pos,
            "sim_root_quat": sim_root_quat,
            "sim_root_vel": sim_root_vel,
            "sim_root_angvel": sim_root_angvel,
            "reset_root_pos": reset_root_pos,
            "reset_root_quat": reset_root_quat,
            "reset_root_vel": reset_root_vel,
            "reset_root_angvel": reset_root_angvel,
            "key_pos": key_pos,
        }

    def _query_bvh_reference(
        self,
        info: dict,
        future_step: int,
    ) -> dict[str, jax.Array]:
        """Query the active BVH reference at continuous time, with standing fallback."""
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        motion_time = self._get_bvh_reference_motion_time(info, future_step)
        queried = self._query_bvh_reference_clip(clip_id, motion_time)
        use_standing = self._reference_fallback_active(info)
        standing = {
            "clip_id": jp.array(0, dtype=jp.int32),
            "frame_index0": jp.array(0, dtype=jp.int32),
            "frame_index1": jp.array(0, dtype=jp.int32),
            "alpha": jp.array(0.0, dtype=jp.float32),
            "loop_count": jp.array(0.0, dtype=jp.float32),
            "qpos": self._standing_reference_qpos_targets[0, 0],
            "qvel": self._standing_reference_qvel_targets[0, 0],
            "root_pos": self._standing_reference_root_pos_targets[0, 0],
            "root_quat": self._standing_reference_root_quat_targets[0, 0],
            "root_vel": self._standing_reference_root_vel_targets[0, 0],
            "root_angvel": self._standing_reference_root_angvel_targets[0, 0],
            "sim_root_pos": self._standing_reference_sim_root_pos_targets[0, 0],
            "sim_root_quat": self._standing_reference_sim_root_quat_targets[0, 0],
            "sim_root_vel": self._standing_reference_sim_root_vel_targets[0, 0],
            "sim_root_angvel": self._standing_reference_sim_root_angvel_targets[0, 0],
            "reset_root_pos": self._standing_reference_reset_root_pos_targets[0, 0],
            "reset_root_quat": self._standing_reference_reset_root_quat_targets[0, 0],
            "reset_root_vel": self._standing_reference_reset_root_vel_targets[0, 0],
            "reset_root_angvel": self._standing_reference_reset_root_angvel_targets[
                0,
                0,
            ],
            "key_pos": self._standing_reference_key_pos_targets[0, 0],
        }
        return {
            key: jp.where(use_standing, standing[key], queried[key])
            if queried[key].ndim == 0
            else jp.where(use_standing, standing[key], queried[key])
            for key in queried
        }

    def _quat_interval_angular_velocity(
        self,
        quat0: jax.Array,
        quat1: jax.Array,
        dt: jax.Array,
    ) -> jax.Array:
        """Angular velocity implied by one quaternion interval."""
        delta = self._quat_mul(self._quat_conjugate(quat0), quat1)
        return self._quat_to_rotvec(delta) / jp.maximum(dt, 1e-6)

    def _quat_to_rotvec(self, quat: jax.Array) -> jax.Array:
        """Quaternion logarithm map in MuJoCo wxyz order."""
        quat = quat / jp.maximum(jp.linalg.norm(quat), 1e-6)
        quat = jp.where(quat[0] < 0.0, -quat, quat)
        xyz = quat[1:]
        xyz_norm = jp.linalg.norm(xyz)
        angle = 2.0 * jp.arctan2(xyz_norm, jp.maximum(quat[0], 1e-6))
        axis = xyz / jp.maximum(xyz_norm, 1e-6)
        return jp.where(
            xyz_norm > 1e-6,
            axis * angle,
            jp.zeros(3, dtype=quat.dtype),
        )

    def _sample_initial_state_from_reference(
        self,
        clip_id: jax.Array,
        motion_time: jax.Array,
        projection_level: float = 1.0,
        exact: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """DeepMimic-style reset into one coherent reference frame."""
        if self._config.get("reference_gait", "none") not in ("bvh", "smpl"):
            qpos = self._init_q.at[2].set(self._standing_height())
            qvel = jp.zeros(self._mjx_model.nv)
            return qpos, qvel, self._default_ctrl, jp.zeros(self.action_size)

        reference = self._query_bvh_reference_clip(clip_id, motion_time)
        target_qpos = reference["qpos"]
        target_qvel = reference["qvel"]
        if exact:
            qpos = self._init_q
            qpos = qpos.at[:3].set(reference["reset_root_pos"])
            qpos = qpos.at[3:7].set(reference["reset_root_quat"])
            qpos = qpos.at[self._actuator_qpos_indices].set(target_qpos)
            qvel = jp.zeros(self._mjx_model.nv)
            qvel = qvel.at[:3].set(reference["reset_root_vel"])
            qvel = qvel.at[3:6].set(reference["reset_root_angvel"])
            qvel = qvel.at[self._actuator_dof_indices].set(target_qvel)
            qpos = self._align_sampled_reference_reset_height(qpos, qvel, target_qpos)
            return qpos, qvel, target_qpos, jp.zeros(self.action_size)

        projected_ctrl = self._project_reference_reset_ctrl(
            target_qpos,
            projection_level,
        )
        qpos = self._init_q
        qpos = qpos.at[:3].set(
            self._project_reference_reset_root_pos(
                reference["reset_root_pos"],
                projection_level,
            )
        )
        qpos = qpos.at[3:7].set(
            self._project_reference_reset_root_quat(
                reference["reset_root_quat"],
                projection_level,
            )
        )
        qpos = qpos.at[self._actuator_qpos_indices].set(projected_ctrl)
        qvel = jp.zeros(self._mjx_model.nv)
        qvel = qvel.at[:3].set(
            self._project_reference_reset_root_vel(
                reference["reset_root_vel"],
                projection_level,
            )
        )
        qvel = qvel.at[3:6].set(
            self._project_reference_reset_root_angvel(
                reference["reset_root_angvel"],
                projection_level,
            )
        )
        qvel = qvel.at[self._actuator_dof_indices].set(
            self._project_reference_reset_qvel(
                target_qvel,
                projection_level,
            )
        )
        qpos = self._align_sampled_reference_reset_height(qpos, qvel, projected_ctrl)
        return qpos, qvel, projected_ctrl, jp.zeros(self.action_size)

    def _project_reference_reset_root_pos(
        self,
        root_pos: jax.Array,
        projection_level: float,
    ) -> jax.Array:
        """Keep sampled root translation so reset matches the sampled motion phase."""
        del projection_level
        return root_pos

    def _project_reference_reset_ctrl(
        self,
        target_qpos: jax.Array,
        projection_level: float,
    ) -> jax.Array:
        """Blend retargeted actuator pose toward stable neutral locomotion pose."""
        level = jp.array(projection_level, dtype=target_qpos.dtype)
        weights = jp.clip(self._reset_pose_projection_weights * level, 0.0, 1.0)
        projected = self._default_ctrl + weights * (target_qpos - self._default_ctrl)
        return jp.clip(
            projected,
            self._actuator_qpos_lower_limits,
            self._actuator_qpos_upper_limits,
        )

    def _project_reference_reset_qvel(
        self,
        target_qvel: jax.Array,
        projection_level: float,
    ) -> jax.Array:
        """Attenuate retargeted joint velocities for a physically valid reset."""
        level = jp.array(projection_level, dtype=target_qvel.dtype)
        return self._reset_velocity_projection_weights * level * target_qvel

    def _project_reference_reset_root_vel(
        self,
        root_vel: jax.Array,
        projection_level: float,
    ) -> jax.Array:
        """Preserve sampled root linear velocity, softened only by projection level."""
        level = jp.array(projection_level, dtype=root_vel.dtype)
        return level * root_vel

    def _project_reference_reset_root_quat(
        self,
        root_quat: jax.Array,
        projection_level: float,
    ) -> jax.Array:
        """Keep sampled root heading so reward/obs/reset all describe one frame."""
        del projection_level
        return root_quat

    def _project_reference_reset_root_angvel(
        self,
        root_angvel: jax.Array,
        projection_level: float,
    ) -> jax.Array:
        """Preserve sampled root angular velocity, softened only by projection level."""
        level = jp.array(projection_level, dtype=root_angvel.dtype)
        return level * root_angvel

    def _reference_reset_noise_scale(self, projection_level: float) -> jax.Array:
        """Reference resets should stay close to the sampled frame, unlike fallback init."""
        if self._config.get("reference_gait", "none") not in ("bvh", "smpl"):
            return self._init_actuator_noise
        del projection_level
        return jp.zeros_like(self._init_actuator_noise)

    def _align_sampled_reference_reset_height(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        ctrl: jax.Array,
    ) -> jax.Array:
        """Shift sampled reference resets so the lowest foot starts near the floor.

        The BVH/root retarget can still be spatially coherent while carrying a
        global vertical offset that does not match this MuJoCo humanoid. For
        reset validation we want the sampled frame to start in contact-ready
        stance instead of being rejected purely for root-height mismatch.
        """
        data = self._fresh_reset_data().replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
        )
        data = mjx.forward(self._mjx_model, data)
        desired_lowest_height = jp.array(
            -self.FOOT_CONTACT_PRELOAD,
            dtype=qpos.dtype,
        )
        lowest_foot_height = jp.min(self._foot_lowest_heights(data))
        z_offset = desired_lowest_height - lowest_foot_height
        qpos = qpos.at[2].add(z_offset)
        minimum_reset_height = (
            self.MIN_STANDING_HEIGHT_RATIO * self._standing_height()
            + jp.array(0.05, dtype=qpos.dtype)
        )
        return qpos.at[2].set(jp.maximum(qpos[2], minimum_reset_height))

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
        height_error = (data.qpos[2] - self._standing_height()) / 0.15
        return jp.exp(-jp.square(height_error))

    def _standing_height(self) -> jax.Array:
        """Root height for the active MJX model, including size randomization."""
        return self._mjx_model.qpos0[2]

    def _foot_positions(self, data: mjx.Data) -> jax.Array:
        """World pozicije oba djona, redosled: levo, desno."""
        return data.geom_xpos[self._foot_geom_ids]

    def _foot_xy(self, data: mjx.Data) -> jax.Array:
        """World XY pozicije oba djona za slip procenu."""
        return self._foot_positions(data)[:, :2]

    def _foot_heights(self, data: mjx.Data) -> jax.Array:
        """World Z visine centara oba foot-sole geom-a."""
        return self._foot_positions(data)[:, 2]

    def _foot_lowest_heights(self, data: mjx.Data) -> jax.Array:
        """Approximate lowest world-Z point of each foot sole geom."""
        foot_pos = self._foot_positions(data)
        foot_xmat = data.geom_xmat[self._foot_geom_ids].reshape((-1, 3, 3))
        center_z = foot_pos[:, 2]
        size = self._foot_geom_sizes
        geom_type = self._foot_geom_types

        box_extent_z = jp.sum(jp.abs(foot_xmat[:, 2, :]) * size, axis=1)
        box_low = center_z - box_extent_z
        sphere_low = center_z - size[:, 0]
        capsule_low = center_z - jp.abs(foot_xmat[:, 2, 2]) * size[:, 1] - size[:, 0]
        fallback_low = center_z - jp.max(size, axis=1)

        return jp.where(
            geom_type == int(mujoco.mjtGeom.mjGEOM_BOX),
            box_low,
            jp.where(
                geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE),
                sphere_low,
                jp.where(
                    geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                    capsule_low,
                    fallback_low,
                ),
            ),
        )

    def _foot_contact(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kontakt stopala sa podom iz MuJoCo contact parova, ne samo iz visine."""
        if self._physics_backend == "mjx_warp":
            # REF: BULLET-WARP-CONTACT-BUFFERS
            # TYPE: REFERENCE_CODE_DERIVED
            warp_data = data._impl
            contact_geom = warp_data.contact__geom
            contact_distance = warp_data.contact__dist
            contact_world = warp_data.contact__worldid
            world_id = info.get("warp_world_id", jp.array(0, dtype=jp.int32))
            valid_world = contact_world == world_id
        else:
            contact_geom = data.contact.geom
            contact_distance = data.contact.dist
            valid_world = jp.ones(contact_distance.shape, dtype=bool)
        geom_a = contact_geom[:, 0]
        geom_b = contact_geom[:, 1]
        valid_contact = (
            (contact_distance <= self.FOOT_CONTACT_DISTANCE) & valid_world
        )

        def touches_floor(foot_geom_id: int) -> jax.Array:
            foot_floor = (
                ((geom_a == foot_geom_id) & (geom_b == self._floor_geom_id))
                | ((geom_b == foot_geom_id) & (geom_a == self._floor_geom_id))
            )
            return jp.any(foot_floor & valid_contact).astype(jp.float32)

        return jp.array([
            touches_floor(self._left_foot_sole_geom_id),
            touches_floor(self._right_foot_sole_geom_id),
        ])

    def _get_foot_slip_cost(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kazni horizontalno klizanje stopala dok je stopalo u kontaktu."""
        foot_velocity_xy = (self._foot_xy(data) - info["last_foot_xy"]) / self.dt
        slip_speed = jp.linalg.norm(foot_velocity_xy, axis=1)
        slip_speed = jp.maximum(slip_speed - self.FOOT_SLIP_FREE_SPEED, 0.0)
        return jp.sum(jp.square(slip_speed) * self._foot_contact(data, info))

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
        """Track linear and yaw command as separate physical quantities."""
        # REF: PROJECT-COMMAND-TRACKING-SPLIT
        # TYPE: ENGINEERING_DEFAULT
        return 0.75 * self._get_tracking_lin_reward(
            data,
            info,
        ) + 0.25 * self._get_tracking_yaw_reward(data, info)

    def _get_tracking_lin_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        """Track commanded horizontal velocity in m/s."""
        error = jp.sum(jp.square(info["command"][:2] - self._measured_command(data)[:2]))
        return jp.exp(-error / self._config.tracking_sigma)

    def _get_tracking_yaw_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        """Track commanded yaw velocity in rad/s."""
        error = jp.square(info["command"][2] - self._measured_command(data)[2])
        return jp.exp(-error / self._config.tracking_yaw_sigma)

    def _measured_command(self, data: mjx.Data) -> jax.Array:
        """Measure command in a yaw-only heading frame, not tilted torso frame."""
        # REF: BIOHUMANOID-HEADING-FRAME-COMMAND
        # TYPE: MODEL_CALIBRATED
        world_to_heading = self._heading_world_to_local(data)
        heading_linvel = world_to_heading @ data.qvel[:3]
        return jp.array([
            heading_linvel[0],
            heading_linvel[1],
            data.qvel[5],
        ])

    def _heading_world_to_local(self, data: mjx.Data) -> jax.Array:
        """Yaw-only world-to-heading frame from projected torso forward axis."""
        torso_xmat = self._body_xmat(data)
        forward = jp.array([torso_xmat[0, 0], torso_xmat[1, 0], 0.0])
        forward = forward / jp.maximum(jp.linalg.norm(forward), 1e-6)
        lateral = jp.array([-forward[1], forward[0], 0.0])
        return jp.stack([
            forward,
            lateral,
            jp.array([0.0, 0.0, 1.0]),
        ])

    def _heading_world_to_local_from_quat(self, quat: jax.Array) -> jax.Array:
        """Yaw-only world-to-heading frame from a root/body quaternion."""
        # MuJoCo wxyz: rotate local +X into world, then project to XY.
        x, y, z, w = quat[1], quat[2], quat[3], quat[0]
        forward = jp.array([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            0.0,
        ])
        forward = forward / jp.maximum(jp.linalg.norm(forward), 1e-6)
        lateral = jp.array([-forward[1], forward[0], 0.0])
        return jp.stack([
            forward,
            lateral,
            jp.array([0.0, 0.0, 1.0]),
        ])

    def _heading_localize_quat(self, quat: jax.Array) -> jax.Array:
        """Remove yaw from a quaternion, matching MimicKit local-root mode."""
        heading = self._heading_world_to_local_from_quat(quat)
        # heading rows are world axes expressed in heading frame; rebuild yaw quat.
        yaw_cos = heading[0, 0]
        yaw_sin = heading[0, 1]
        half_angle_cos = jp.sqrt(jp.maximum(0.5 * (1.0 + yaw_cos), 0.0))
        half_angle_sin = jp.where(
            half_angle_cos > 1e-6,
            0.5 * yaw_sin / jp.maximum(half_angle_cos, 1e-6),
            0.0,
        )
        heading_quat = jp.array([half_angle_cos, 0.0, 0.0, half_angle_sin])
        heading_quat = heading_quat / jp.maximum(jp.linalg.norm(heading_quat), 1e-6)
        return self._quat_mul(self._quat_conjugate(heading_quat), quat)

    def _command_active(self, info: dict) -> jax.Array:
        """Return true when either linear or yaw joystick command is meaningful."""
        command = info["command"]
        return (
            jp.linalg.norm(command[:2]) > self.STUCK_COMMAND_THRESHOLD
        ) | (jp.abs(command[2]) > self.STUCK_COMMAND_THRESHOLD)

    def _get_command_progress(self, data: mjx.Data, info: dict) -> jax.Array:
        """Reward progress without mixing m/s and rad/s in one norm."""
        command = info["command"]
        measured = self._measured_command(data)
        lin_norm_sq = jp.sum(jp.square(command[:2]))
        lin_active = lin_norm_sq > self.STUCK_COMMAND_THRESHOLD**2
        lin_alignment = jp.dot(measured[:2], command[:2]) / jp.maximum(
            lin_norm_sq,
            1e-6,
        )
        yaw_active = jp.abs(command[2]) > self.STUCK_COMMAND_THRESHOLD
        yaw_alignment = measured[2] * command[2] / jp.maximum(
            jp.square(command[2]),
            1e-6,
        )
        progress = 0.75 * jp.where(lin_active, jp.clip(lin_alignment, 0.0, 1.0), 0.0)
        progress += 0.25 * jp.where(yaw_active, jp.clip(yaw_alignment, 0.0, 1.0), 0.0)
        return jp.where(
            lin_active | yaw_active,
            progress * self._get_tracking_reward(data, info),
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


# Pre-jit vmap za domain randomization.
_randomize_vmap = jax.jit(jax.vmap(randomize_single_model, in_axes=(None, 0)))


def domain_randomize(model, rng):
    """Randomizuje velicinu i masu direktno u MJX model arrays.

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
