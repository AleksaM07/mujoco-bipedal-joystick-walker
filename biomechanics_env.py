from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from biomechanics_model import HumanSpec, build_trainable_scene_xml
from mujoco_playground._src import mjx_env


@dataclass(frozen=True)
class BiomechanicsEnvConfig:
    env_version: str = "standard"
    impl: str = "jax"
    ctrl_dt: float = 0.02
    sim_dt: float = 0.01
    episode_length: int = 1000
    action_scale: float = 0.5
    command_resample_steps: int = 500
    tracking_sigma: float = 0.5
    rfi_torque_limit: float = 2.0
    rao_torque_limit: float = 2.0
    enable_erfi: bool = True


def default_config() -> config_dict.ConfigDict:
    """Vraca config kompatibilan sa MuJoCo Playground MjxEnv bazom."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.01,
        episode_length=1000,
        action_scale=0.5,
        command_resample_steps=500,
        tracking_sigma=0.5,
        rfi_torque_limit=2.0,
        rao_torque_limit=2.0,
        enable_erfi=True,
        impl="jax",
    )


class BiomechanicsJoystickEnv(mjx_env.MjxEnv):
    """Joystick locomotion env za humanoida iz `mujoco-biomechanics`."""

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
        self._init_q = jp.array(self._mj_model.keyframe("a-pose").qpos)
        self._default_qpos = self._init_q[7:]
        self._actuator_qpos_indices = jp.array([
            self._mj_model.jnt_qposadr[joint_id]
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ])
        self._actuator_dof_indices = jp.array([
            self._mj_model.jnt_dofadr[joint_id]
            for joint_id in self._mj_model.actuator_trnid[:, 0]
        ])
        self._default_ctrl = self._init_q[self._actuator_qpos_indices]
        self._n_substeps = int(round(self._ctrl_dt / self._sim_dt))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resetuje human u pocetnu pozu i uzorkuje joystick komandu."""
        rng, command_key, pose_key, erfi_key, bias_key = jax.random.split(rng, 5)
        qpos = self._init_q
        qpos = qpos.at[7:].add(
            jax.random.uniform(
                pose_key,
                shape=qpos[7:].shape,
                minval=-0.03,
                maxval=0.03,
            )
        )
        qvel = jp.zeros(self._mjx_model.nv)
        ctrl = self._default_ctrl
        data = mjx_env.make_data(self._mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        command = self.sample_command(command_key)
        info = {
            "rng": rng,
            "command": command,
            "last_action": jp.zeros(self.action_size),
            "episode_torque_offset": self.sample_episode_torque_offset(bias_key),
            "use_rfi": jax.random.bernoulli(erfi_key, p=0.5),
            "step": jp.array(0),
        }
        obs = self._get_obs(data, info)
        metrics = {
            "reward": jp.array(0.0),
            "tracking_lin_vel": jp.array(0.0),
        }
        return mjx_env.State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Izvrsi jedan locomotion korak za zadatu akciju politike."""
        info = dict(state.info)
        info["rng"], rfi_key, command_key = jax.random.split(info["rng"], 3)

        action = jp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        policy_action = jp.clip(action, -1.0, 1.0)
        erfi_torque = self.sample_erfi_torque(
            rfi_key,
            info["episode_torque_offset"],
            info["use_rfi"],
        )
        data_with_erfi = self.apply_joint_torque_injection(
            state.data,
            erfi_torque,
        )
        motor_targets = self._default_ctrl + policy_action * self._config.action_scale
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
        info["last_action"] = policy_action
        info["step"] = jp.where(should_resample, 0, info["step"] + 1)

        obs = self._get_obs(data, info)
        reward = self._get_reward(data, policy_action, info)
        done = self._get_done(data)
        metrics = dict(state.metrics)
        metrics["reward"] = reward
        metrics["tracking_lin_vel"] = reward
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
        return jp.where(use_rfi, rfi_torque, episode_offset)

    def apply_joint_torque_injection(
        self,
        data: mjx.Data,
        joint_torque: jax.Array,
    ) -> mjx.Data:
        """Upise RFI/RAO torques u qfrc_applied na kontrolisane DoF-ove."""
        qfrc_applied = jp.zeros_like(data.qfrc_applied)
        qfrc_applied = qfrc_applied.at[self._actuator_dof_indices].set(joint_torque)
        return data.replace(qfrc_applied=qfrc_applied)

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        """Sastavi observation koji policy dobija."""
        joint_pos = data.qpos[7:] - self._default_qpos
        joint_vel = data.qvel[6:]
        obs = jp.concatenate([
            data.qvel[:3],
            data.qvel[3:6],
            info["command"],
            joint_pos,
            joint_vel,
            info["last_action"],
        ])
        return jp.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict) -> jax.Array:
        """Nagrada za pracenje joystick brzine uz malu kaznu za energiju."""
        lin_vel_error = jp.sum(jp.square(info["command"][:2] - data.qvel[:2]))
        yaw_error = jp.square(info["command"][2] - data.qvel[5])
        tracking = jp.exp(-(lin_vel_error + 0.5 * yaw_error) / self._config.tracking_sigma)
        action_cost = 0.001 * jp.sum(jp.square(action))
        height_cost = 0.5 * jp.square(data.qpos[2] - self._init_q[2])
        reward = jp.clip(tracking - action_cost - height_cost, -1.0, 2.0)
        reward = jp.nan_to_num(reward, nan=-1.0, posinf=2.0, neginf=-1.0)
        return jp.where(self._get_done(data), -1.0, reward)

    def _get_done(self, data: mjx.Data) -> jax.Array:
        """Zavrsi epizodu ako human padne ili numerika ode u NaN."""
        fallen = data.qpos[2] < 0.35 * self._init_q[2]
        invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return fallen | invalid

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
        qpos0,
        body_mass,
        body_inertia,
        geom_friction,
    )


_randomize_vmap = jax.jit(jax.vmap(randomize_single_model, in_axes=(None, 0)))


def domain_randomize(model, rng):
    """Randomizuje human velicinu i masu direktno u MJX model arrays.

    Ovo je deo koji sprecava generisanje novog XML-a po epizodi: topologija
    ostaje ista, a Brax/MJX dobija razlicite numericke modele po env-u.
    """
    randomized = _randomize_vmap(model, rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({
        "body_pos": 0,
        "geom_pos": 0,
        "geom_size": 0,
        "qpos0": 0,
        "body_mass": 0,
        "body_inertia": 0,
        "geom_friction": 0,
    })

    randomized_model = model.tree_replace({
        "body_pos": randomized[0],
        "geom_pos": randomized[1],
        "geom_size": randomized[2],
        "qpos0": randomized[3],
        "body_mass": randomized[4],
        "body_inertia": randomized[5],
        "geom_friction": randomized[6],
    })
    return randomized_model, in_axes
