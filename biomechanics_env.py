from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from biomechanics_model import (
    HumanSpec,
    LEG_ACTUATED_JOINTS,
    TRUNK_ACTUATED_JOINTS,
    build_trainable_scene_xml,
)
from mujoco_playground._src import mjx_env


@dataclass(frozen=True)
class BiomechanicsEnvConfig:
    env_version: str = "standard"
    impl: str = "jax"
    ctrl_dt: float = 0.02
    sim_dt: float = 0.005
    episode_length: int = 1000
    action_scale: float = 0.5
    action_smoothing: float = 0.5
    command_profile: str = "forward"
    command_resample_steps: int = 500
    tracking_sigma: float = 0.25
    action_noise_std: float = 0.03
    episode_bias_std: float = 0.02
    rfi_torque_limit: float = 2.0
    rao_torque_limit: float = 2.0
    enable_erfi: bool = True


def default_config() -> config_dict.ConfigDict:
    """Vraca config kompatibilan sa MuJoCo Playground MjxEnv bazom."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        action_scale=0.5,
        action_smoothing=0.5,
        command_profile="forward",
        command_resample_steps=500,
        tracking_sigma=0.25,
        action_noise_std=0.03,
        episode_bias_std=0.02,
        rfi_torque_limit=2.0,
        rao_torque_limit=2.0,
        enable_erfi=True,
        impl="jax",
    )


class BiomechanicsJoystickEnv(mjx_env.MjxEnv):
    """Joystick locomotion env za humanoida iz `mujoco-biomechanics`."""

    WORLD_GRAVITY = jp.array([0.0, 0.0, -1.0])
    FOOT_SOLE_GEOMS = ("left_foot_sole", "right_foot_sole")
    FOOT_CONTACT_PRELOAD = 0.005
    FOOT_CONTACT_HEIGHT = 0.095
    FORWARD_COMMAND_RANGE = (0.15, 0.35)
    ZERO_COMMAND_PROBABILITY = 0.0
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
    INIT_TRUNK_NOISE = 0.005
    INIT_LEG_NOISE = 0.02
    HEIGHT_PENALTY_START_RATIO = 0.8
    MIN_STANDING_HEIGHT_RATIO = 0.6
    ALIVE_REWARD_SCALE = 0.05
    ACTION_COST_SCALE = 0.01
    ACTION_RATE_COST_SCALE = 0.005
    BASE_HEIGHT_REWARD_SCALE = 0.25
    POSTURE_REWARD_SCALE = 0.02
    TRUNK_POSTURE_COST_SCALE = 0.15
    VELOCITY_TRACKING_REWARD_SCALE = 1.5
    FORWARD_PROGRESS_REWARD_SCALE = 1.0
    UPRIGHT_REWARD_SCALE = 0.2
    OVERSPEED_COST_SCALE = 0.75
    VERTICAL_VELOCITY_COST_SCALE = 0.05
    ANGULAR_VELOCITY_COST_SCALE = 0.02
    GAIT_PERIOD_STEPS = 50
    FOOT_CLEARANCE_TARGET = 0.08
    FOOT_CLEARANCE_REWARD_SCALE = 0.4
    STANCE_CONTACT_REWARD_SCALE = 0.2
    FOOT_SLIP_COST_SCALE = 0.06
    STUCK_COMMAND_THRESHOLD = 0.10
    STUCK_VELOCITY_THRESHOLD = 0.05
    STUCK_PENALTY = 1.0
    FALL_REWARD = -10.0

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
        self._xml_path = build_trainable_scene_xml(env_version, human_spec)
        self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._mj_model.opt.timestep = self._sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        init_q = np.array(self._mj_model.keyframe("a-pose").qpos, copy=True)
        init_q = self._apply_locomotion_neutral_pose(init_q)
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
            self.TRUNK_ACTION_SCALE.get(joint_name, self._config.action_scale)
            for joint_name in actuator_joint_names
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
        self._default_ctrl = self._init_q[self._actuator_qpos_indices]
        self._n_substeps = int(round(self._ctrl_dt / self._sim_dt))
        self._torso_body_id = self._mj_model.body("thorax").id
        self._left_foot_sole_geom_id = self._mj_model.geom("left_foot_sole").id
        self._right_foot_sole_geom_id = self._mj_model.geom("right_foot_sole").id
        self._foot_geom_ids = jp.array([
            self._left_foot_sole_geom_id,
            self._right_foot_sole_geom_id,
        ])

    def _apply_locomotion_neutral_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Centira akcije oko stabilnije stojece poze, ne oko krute A-poze."""
        for joint_name, joint_value in self.NEUTRAL_JOINT_POSE.items():
            joint_id = mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            qpos[self._mj_model.jnt_qposadr[joint_id]] = joint_value

        data = mujoco.MjData(self._mj_model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self._mj_model, data)
        qpos[2] -= self._minimum_geom_z(data) + self.FOOT_CONTACT_PRELOAD
        return qpos

    def _minimum_geom_z(self, data: mujoco.MjData) -> float:
        """Vraca najnizu world-Z tacku djonova u trenutnoj pozi."""
        min_z = np.inf
        for geom_name in self.FOOT_SOLE_GEOMS:
            geom_id = mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            geom_pos = data.geom_xpos[geom_id]
            geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
            geom_size = self._mj_model.geom_size[geom_id]
            for signs in product((-1.0, 1.0), repeat=3):
                corner = geom_pos + geom_xmat @ (geom_size * np.array(signs))
                min_z = min(min_z, corner[2])
        return float(min_z)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resetuje human u pocetnu pozu i uzorkuje joystick komandu."""
        rng, command_key, pose_key, erfi_key, bias_key, gait_key = (
            jax.random.split(rng, 6)
        )
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
        data = mjx_env.make_data(self._mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
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
            "last_foot_xy": self._foot_xy(data),
        }
        obs = self._get_obs(data, info)
        metrics = {
            "reward": jp.array(0.0),
            "tracking_lin_vel": jp.array(0.0),
            "forward_vel": jp.array(0.0),
            "torso_up": jp.array(1.0),
            "height": qpos[2],
            "foot_slip": jp.array(0.0),
            "base_height": jp.array(1.0),
            "gait_reward": jp.array(0.0),
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

        obs = self._get_obs(data, info)
        reward = self._get_reward(data, smoothed_action, previous_action, info)
        done = self._get_done(data)
        foot_slip = self._get_foot_slip_cost(data, info)
        base_height = self._get_base_height_reward(data)
        gait_reward = self._get_gait_reward(data, info)
        metrics = dict(state.metrics)
        metrics["reward"] = reward
        metrics["tracking_lin_vel"] = self._get_tracking_reward(data, info)
        metrics["forward_vel"] = self._local_root_linvel(data)[0]
        metrics["torso_up"] = self._torso_up(data)
        metrics["height"] = data.qpos[2]
        metrics["foot_slip"] = foot_slip
        metrics["base_height"] = base_height
        metrics["gait_reward"] = gait_reward
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
        if self._config.command_profile in ("forward", "walk"):
            x_key, zero_key = jax.random.split(rng, 2)
            command = jp.array([
                jax.random.uniform(
                    x_key,
                    minval=self.FORWARD_COMMAND_RANGE[0],
                    maxval=self.FORWARD_COMMAND_RANGE[1],
                ),
                0.0,
                0.0,
            ])
            return jp.where(
                jax.random.bernoulli(zero_key, p=self.ZERO_COMMAND_PROBABILITY),
                jp.zeros(3),
                command,
            )

        if self._config.command_profile != "standard":
            raise ValueError(
                "command_profile mora biti 'forward', 'walk' ili 'standard'."
            )

        x_key, y_key, yaw_key, zero_key = jax.random.split(rng, 4)
        command = jp.array([
            jax.random.uniform(x_key, minval=-1.0, maxval=1.0),
            jax.random.uniform(y_key, minval=-0.8, maxval=0.8),
            jax.random.uniform(yaw_key, minval=-1.0, maxval=1.0),
        ])
        return jp.where(jax.random.bernoulli(zero_key, p=0.1), jp.zeros(3), command)

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
        state_obs = jp.concatenate([
            local_linvel,
            local_angvel,
            projected_gravity,
            info["command"],
            self._get_gait_phase_obs(info),
            joint_pos,
            joint_vel,
            info["last_action"],
        ])
        state_obs = jp.nan_to_num(state_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        privileged_obs = jp.concatenate([
            state_obs,
            data.qpos[:3],
            data.qvel[:6],
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

    def _get_gait_phase_obs(self, info: dict) -> jax.Array:
        """Dodaje clock signal politici za lakse uskladjivanje nogu."""
        phase_angle = self._get_gait_phase_angle(info)
        return jp.array([jp.sin(phase_angle), jp.cos(phase_angle)])

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> jax.Array:
        """Nagrada za hod: brzina napred je cilj, stabilnost je uslov."""
        local_linvel = self._local_root_linvel(data)
        local_angvel = self._local_root_angvel(data)
        target_forward_vel = jp.maximum(info["command"][0], 0.0)
        forward_vel = local_linvel[0]
        tracking = self._get_tracking_reward(data, info)
        target_denominator = jp.maximum(target_forward_vel, 0.05)
        raw_forward_progress = jp.clip(
            forward_vel / target_denominator,
            0.0,
            1.0,
        )
        forward_progress = jp.where(
            target_forward_vel > 0.05,
            raw_forward_progress * tracking,
            0.0,
        )
        overspeed = jp.maximum(forward_vel - target_forward_vel, 0.0)
        overspeed_cost = self.OVERSPEED_COST_SCALE * jp.square(
            overspeed / target_denominator
        )
        stuck_penalty = jp.where(
            (target_forward_vel > self.STUCK_COMMAND_THRESHOLD)
            & (forward_vel < self.STUCK_VELOCITY_THRESHOLD),
            self.STUCK_PENALTY,
            0.0,
        )
        upright = jp.clip(self._torso_up(data), 0.0, 1.0)
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
        trunk_error = jp.sum(
            jp.square(jp.where(self._trunk_actuator_mask, actuator_position_error, 0.0))
        )
        trunk_posture_cost = self.TRUNK_POSTURE_COST_SCALE * trunk_error
        base_height_reward = self.BASE_HEIGHT_REWARD_SCALE * (
            self._get_base_height_reward(data)
        )
        gait_reward = self._get_gait_reward(data, info)
        foot_slip_cost = self.FOOT_SLIP_COST_SCALE * self._get_foot_slip_cost(
            data,
            info,
        )
        height_cost = 8.0 * jp.square(low_height)
        reward = (
            self.ALIVE_REWARD_SCALE
            + self.VELOCITY_TRACKING_REWARD_SCALE * tracking
            + self.FORWARD_PROGRESS_REWARD_SCALE * forward_progress
            + self.UPRIGHT_REWARD_SCALE * upright
            + base_height_reward
            + posture_reward
            + gait_reward
            - stuck_penalty
            - action_cost
            - action_rate_cost
            - trunk_posture_cost
            - foot_slip_cost
            - height_cost
            - overspeed_cost
            - vertical_velocity_cost
            - angular_velocity_cost
        )
        reward = jp.clip(reward, 0.0, 3.0)
        reward = jp.nan_to_num(
            reward,
            nan=0.0,
            posinf=3.0,
            neginf=0.0,
        )
        return jp.where(self._get_done(data), self.FALL_REWARD, reward)

    def _get_done(self, data: mjx.Data) -> jax.Array:
        """Zavrsi epizodu ako human padne ili numerika ode u NaN."""
        too_low = data.qpos[2] < self.MIN_STANDING_HEIGHT_RATIO * self._init_q[2]
        tipped_over = self._torso_up(data) < 0.25
        invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return too_low | tipped_over | invalid

    def _get_gait_phase_angle(self, info: dict) -> jax.Array:
        """Periodican signal koji govori politici koja noga treba da bude swing."""
        phase = jp.mod(
            info["gait_step"].astype(jp.float32),
            self.GAIT_PERIOD_STEPS,
        ) / self.GAIT_PERIOD_STEPS
        return 2.0 * jp.pi * phase

    def _get_gait_reward(
        self,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        """Nagradi swing clearance i stance contact u fazi koraka."""
        phase_angle = self._get_gait_phase_angle(info)
        left_swing = jp.sin(phase_angle) > 0.0
        foot_heights = self._foot_heights(data)
        foot_contact = self._foot_contact(data)
        swing_foot_z = jp.where(left_swing, foot_heights[0], foot_heights[1])
        stance_foot_z = jp.where(left_swing, foot_heights[1], foot_heights[0])
        stance_contact = jp.where(left_swing, foot_contact[1], foot_contact[0])
        relative_clearance = swing_foot_z - stance_foot_z
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

    def _get_base_height_reward(self, data: mjx.Data) -> jax.Array:
        """Mala gusta nagrada za drzanje root visine blizu pocetne stojece poze."""
        height_error = (data.qpos[2] - self._init_q[2]) / 0.15
        return jp.exp(-jp.square(height_error))

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
        """Pseudo contact iz visine stopala; dovoljno stabilno za reward/critic."""
        return (self._foot_heights(data) < self.FOOT_CONTACT_HEIGHT).astype(jp.float32)

    def _get_foot_slip_cost(self, data: mjx.Data, info: dict) -> jax.Array:
        """Kazni horizontalno klizanje stopala dok je stopalo u kontaktu."""
        foot_velocity_xy = (self._foot_xy(data) - info["last_foot_xy"]) / self.dt
        slip_speed = jp.linalg.norm(foot_velocity_xy, axis=1)
        return jp.sum(slip_speed * self._foot_contact(data))

    def _body_xmat(self, data: mjx.Data) -> jax.Array:
        """Vraca 3x3 rotaciju toraksa iz body-local u world frame."""
        return data.xmat[self._torso_body_id].reshape((3, 3))

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

    def _get_tracking_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        """Prati joystick u anatomskim osama: X napred, Z lateralno, Y yaw."""
        local_linvel = self._local_root_linvel(data)
        local_angvel = self._local_root_angvel(data)
        measured_command = jp.array([
            local_linvel[0],
            local_linvel[2],
            local_angvel[1],
        ])
        error = jp.sum(jp.square(info["command"] - measured_command))
        return jp.exp(-error / self._config.tracking_sigma)

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
