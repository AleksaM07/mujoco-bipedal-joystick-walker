import re
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from bvh_reference import load_bvh_references
from biomechanics_model import (
    HumanSpec,
    LEG_ACTUATED_JOINTS,
    TRUNK_ACTUATED_JOINTS,
    build_trainable_scene_xml,
)
from config import default_biomechanics_env_config, resolve_project_path


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

    FOOT_SOLE_GEOMS = ("left_foot_sole", "right_foot_sole")
    FOOT_CONTACT_PRELOAD = 0.005
    FOOT_CONTACT_HEIGHT = 0.095
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
    REFERENCE_GAIT_REWARD_SCALE = 0.35
    REFERENCE_GAIT_ERROR_SCALE = 8.0
    REFERENCE_VELOCITY_REWARD_SCALE = 0.15
    REFERENCE_VELOCITY_ERROR_SCALE = 0.25
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
            self._xml_path = build_trainable_scene_xml(env_version, human_spec)
        self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._mj_model.opt.timestep = self._sim_dt
        init_q = np.array(self._mj_model.keyframe("a-pose").qpos, copy=True)
        init_q = self._build_initial_qpos(init_q)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._init_q = jp.array(init_q)
        self._default_qpos = self._init_q[7:]
        self._actuator_qpos_indices = jp.array([
            self._mj_model.jnt_qposadr[joint_id]
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ])
        self._actuator_dof_indices = jp.array([
            self._mj_model.jnt_dofadr[joint_id]
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ])
        actuator_joint_names = tuple(
            mujoco.mj_id2name(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        )
        supported_joint_orders = (
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
        self._default_ctrl = self._init_q[self._actuator_qpos_indices]
        self._bvh_reference_qpos_targets = jp.expand_dims(
            jp.expand_dims(self._default_ctrl, axis=0),
            axis=0,
        )
        self._bvh_reference_qvel_targets = jp.zeros_like(
            self._bvh_reference_qpos_targets
        )
        self._bvh_reference_frame_times = jp.array([self.dt], dtype=jp.float32)
        self._bvh_reference_frame_counts = jp.array([1], dtype=jp.int32)
        self._bvh_reference_clip_count = 1
        self._configure_bvh_reference()
        self._n_substeps = int(round(self._ctrl_dt / self._sim_dt))
        self._torso_body_id = self._mj_model.body("thorax").id
        self._head_body_id = self._mj_model.body("head").id
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._left_foot_sole_geom_id = self._mj_model.geom("left_foot_sole").id
        self._right_foot_sole_geom_id = self._mj_model.geom("right_foot_sole").id
        self._foot_geom_ids = jp.array([
            self._left_foot_sole_geom_id,
            self._right_foot_sole_geom_id,
        ])
        self._configure_policy_observation_layout()

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
            0: (False, False),
            2: (True, False),
            self.action_size: (False, True),
            self.action_size + 2: (True, True),
        }
        if optional_size not in layouts:
            raise ValueError(
                "Checkpoint observation layout ne odgovara izabranom XML-u: "
                f"checkpoint={expected_size}, osnovni_env={base_size}, "
                f"action_size={self.action_size}."
            )

        (
            self._include_gait_phase_observation,
            self._include_reference_target_observation,
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
        self._bvh_reference_clip_count = len(references.source_paths)

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
        rng, command_key, pose_key, erfi_key, bias_key, gait_key, bvh_key = (
            jax.random.split(rng, 7)
        )
        qpos = self._init_q.at[2].set(self._standing_height())
        pose_noise = (
            jax.random.uniform(
                pose_key,
                shape=(self.action_size,),
                minval=-1.0,
                maxval=1.0,
            )
            * self._init_actuator_noise
        )
        qpos = qpos.at[self._actuator_qpos_indices].add(pose_noise)
        qvel = jp.zeros(self._mjx_model.nv)
        ctrl = self._default_ctrl
        data = mjx_env.make_data(
            self._mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
        )
        data = mjx.forward(self._mjx_model, data)
        command = self.sample_command(command_key)
        info = {
            "rng": rng,
            "command": command,
            "last_action": jp.zeros(self.action_size),
            "episode_torque_offset": self.sample_episode_torque_offset(bias_key),
            "use_rfi": jax.random.bernoulli(erfi_key, p=0.5),
            "step": jp.array(0),
            "gait_step": jax.random.randint(
                gait_key,
                shape=(),
                minval=0,
                maxval=int(self.GAIT_PERIOD_STEPS),
            ),
            "bvh_reference_clip_id": jax.random.randint(
                bvh_key,
                shape=(),
                minval=0,
                maxval=int(self._bvh_reference_clip_count),
            ),
            "last_foot_xy": self._foot_xy(data),
        }
        obs = self._get_obs(data, info)
        metrics = {
            "reward": jp.array(0.0),
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
            "base_height": jp.array(1.0),
            "variable_posture": jp.array(1.0),
            "gait_reward": jp.array(0.0),
            "reference_gait": jp.array(0.0),
            "reference_velocity": jp.array(0.0),
            "contact_force": jp.array(0.0),
            "done_low_height": jp.array(0.0),
            "done_tipped": jp.array(0.0),
            "done_invalid": jp.array(0.0),
            "done": jp.array(0.0),
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

        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        policy_action = jp.clip(action, -1.0, 1.0)
        previous_action = info["last_action"]
        smoothed_action = (
            self._config.action_smoothing * policy_action
            + (1.0 - self._config.action_smoothing) * previous_action
        )
        motor_targets = self._default_ctrl + (smoothed_action * self._action_scale)
        data = self.step_with_joint_torque_injection(
            state.data,
            motor_targets,
            rfi_key,
            info["episode_torque_offset"],
            info["use_rfi"],
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

        obs = self._get_obs(data, info)

        reward = self._get_reward(data, smoothed_action, previous_action, info)
        done = self._get_done(data)
        done_low_height, done_tipped, done_invalid = self._get_done_reasons(data)
        foot_slip = self._get_foot_slip_cost(data, info)
        swing_drag = self._get_swing_foot_drag_cost(data, info)
        swing_clearance = self._get_swing_clearance(data, info)
        swing_clearance_deficit = self._get_swing_clearance_deficit_cost(data, info)
        base_height = self._get_base_height_reward(data)
        variable_posture = self._get_variable_posture_reward(data, info)
        gait_reward = self._get_gait_reward(data, info)
        reference_gait = self._get_reference_gait_reward(data, info)
        reference_velocity = self._get_reference_velocity_reward(data, info)
        contact_force = self._get_contact_force_cost(data)
        metrics = dict(state.metrics)
        metrics["reward"] = reward
        metrics["tracking_lin_vel"] = self._get_tracking_reward(data, info)
        metrics["forward_vel"] = self._local_root_linvel(data)[0]
        metrics["command_norm"] = jp.linalg.norm(info["command"])
        metrics["command_progress"] = self._get_command_progress(data, info)
        metrics["torso_up"] = self._torso_up(data)
        metrics["head_up"] = self._head_up(data)
        metrics["height"] = data.qpos[2]
        metrics["foot_slip"] = foot_slip
        metrics["swing_drag"] = swing_drag
        metrics["swing_clearance"] = swing_clearance
        metrics["swing_clearance_deficit"] = swing_clearance_deficit
        metrics["base_height"] = base_height
        metrics["variable_posture"] = variable_posture
        metrics["gait_reward"] = gait_reward
        metrics["reference_gait"] = reference_gait
        metrics["reference_velocity"] = reference_velocity
        metrics["contact_force"] = contact_force
        metrics["done_low_height"] = done_low_height.astype(reward.dtype)
        metrics["done_tipped"] = done_tipped.astype(reward.dtype)
        metrics["done_invalid"] = done_invalid.astype(reward.dtype)
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
            optional_parts.append(self._get_reference_target_delta(info))
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
        local_linvel = self._local_root_linvel(data)
        local_angvel = self._local_root_angvel(data)
        measured_command = self._measured_command(data)
        command_norm = jp.linalg.norm(info["command"])
        measured_norm = jp.linalg.norm(measured_command)
        command_active = command_norm > self.STUCK_COMMAND_THRESHOLD
        tracking = self._get_tracking_reward(data, info)
        command_progress = self._get_command_progress(data, info)
        overspeed_denominator = jp.maximum(command_norm, 0.05)
        overspeed = jp.maximum(measured_norm - 1.25 * command_norm, 0.0)
        overspeed_cost = self.OVERSPEED_COST_SCALE * jp.square(
            overspeed / overspeed_denominator
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
        gait_reward = self._get_gait_reward(data, info)
        swing_clearance_deficit_cost = (
            self.SWING_CLEARANCE_DEFICIT_COST_SCALE
            * self._get_swing_clearance_deficit_cost(data, info)
        )
        reference_gait_reward = (
            self.REFERENCE_GAIT_REWARD_SCALE
            * self._get_reference_gait_reward(data, info)
        )
        reference_velocity_reward = (
            self.REFERENCE_VELOCITY_REWARD_SCALE
            * self._get_reference_velocity_reward(data, info)
        )
        contact_force_cost = self._get_contact_force_cost(data)
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
        return jp.where(self._get_done(data), self.FALL_REWARD, reward)

    def _get_done(self, data: mjx.Data) -> jax.Array:
        """Zavrsi epizodu ako human padne ili numerika ode u NaN."""
        too_low, tipped_over, invalid = self._get_done_reasons(data)
        return too_low | tipped_over | invalid

    def _get_done_reasons(
        self,
        data: mjx.Data,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Vrati pojedinacne termination razloge za debug logove."""
        too_low = (
            data.qpos[2] < self.MIN_STANDING_HEIGHT_RATIO * self._standing_height()
        )
        tipped_over = self._torso_up(data) < 0.25
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
            + self.STANCE_CONTACT_REWARD_SCALE * stance_contact,
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
        command_active = jp.linalg.norm(info["command"]) > self.STUCK_COMMAND_THRESHOLD
        return jp.where(command_active, velocity_reward, 0.0)

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

    def _get_bvh_reference_frame(self, info: dict) -> tuple[jax.Array, jax.Array]:
        """Vrati aktivni BVH clip i frame indeks."""
        clip_id = info["bvh_reference_clip_id"].astype(jp.int32)
        frame_count = self._bvh_reference_frame_counts[clip_id]
        frame_time = self._bvh_reference_frame_times[clip_id]
        frame_float = (
            info["step"].astype(jp.float32)
            * jp.array(self.dt, dtype=jp.float32)
            / frame_time
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

    def _foot_contact(self, data: mjx.Data) -> jax.Array:
        """Kontakt stopala sa podom iz MuJoCo contact parova, ne samo iz visine."""
        contact_geom = data.contact.geom
        geom_a = contact_geom[:, 0]
        geom_b = contact_geom[:, 1]
        valid_contact = data.contact.dist <= self.FOOT_CONTACT_DISTANCE

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
