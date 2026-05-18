from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import LOCOMOTION_JOINTS, EnvConfig
from .model_factory import HumanModelFactory, HumanSpec


class HumanWalkEnv(gym.Env):
    """MuJoCo/Gymnasium okruzenje za robustno humanoidno hodanje.

    Akcija je normalizovana u [-1, 1]. Okruzenje je prevodi u motorne momente.
    Observacija sadrzi pozu bez globalne x/y pozicije, brzine, komandu i parametre tela.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.factory = HumanModelFactory(self.config)

        self.model, self.data, self.spec = None, None, None
        self.command = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(len(LOCOMOTION_JOINTS), dtype=np.float32)
        self.episode_bias = np.zeros(len(LOCOMOTION_JOINTS), dtype=np.float32)
        self.steps = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(LOCOMOTION_JOINTS),), dtype=np.float32)

        # Pravimo jedan model u konstruktoru samo da znamo dimenziju observacije.
        self._new_episode_model()
        obs = self._observation()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._new_episode_model()
        self.command = self._sample_command(options)
        self.prev_action[:] = 0.0
        self.episode_bias = self.rng.normal(
            0.0,
            self.config.episode_torque_bias_std,
            size=self.action_space.shape,
        ).astype(np.float32)
        self.steps = 0

        self._set_initial_pose()
        return self._observation(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        x_before = float(self.data.qpos[0])
        y_before = float(self.data.qpos[1])

        torque = action * self.config.torque_limit
        torque += self.episode_bias
        torque += self.rng.normal(0.0, self.config.torque_noise_std, size=action.shape)
        self.data.ctrl[:] = np.clip(torque, -self.config.torque_limit, self.config.torque_limit)

        self._maybe_apply_push()
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        dt = self.model.opt.timestep * self.config.frame_skip
        velocity = np.array(
            [
                (self.data.qpos[0] - x_before) / dt,
                (self.data.qpos[1] - y_before) / dt,
                self.data.qvel[5],
            ],
            dtype=np.float32,
        )

        reward = self._reward(action, velocity)
        terminated = not self._is_healthy()
        self.steps += 1
        truncated = self.steps >= int(self.config.episode_seconds / dt)

        self.prev_action = action.copy()
        return self._observation(), reward, terminated, truncated, self._info(velocity)

    def close(self):
        self.factory.close()

    def _new_episode_model(self) -> None:
        self.spec = self.factory.sample_spec(self.rng)
        self.model, self.data, _ = self.factory.build_model(self.spec)

    def _set_initial_pose(self) -> None:
        # Koristimo a-pose iz generatora: stabilnija je od T-pose za pocetak hoda.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "a-pose")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Mala perturbacija pocetne poze sprecava politiku da pamti samo jedan start.
        hinge_qpos_start = 7
        noise = self.rng.normal(0.0, 0.02, size=self.model.nq - hinge_qpos_start)
        self.data.qpos[hinge_qpos_start:] += noise
        mujoco.mj_forward(self.model, self.data)

    def _sample_command(self, options: dict | None) -> np.ndarray:
        if options and "command" in options:
            return np.asarray(options["command"], dtype=np.float32)
        vx = self.rng.uniform(*self.config.command_x_range)
        vy = self.rng.uniform(*self.config.command_y_range)
        wz = self.rng.uniform(*self.config.command_yaw_range)
        return np.array([vx, vy, wz], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        qpos_without_global_xy = self.data.qpos[2:].copy()
        qvel = self.data.qvel.copy()
        body_params = np.array(
            [
                self.spec.mass / 100.0,
                self.spec.height / 2.0,
                1.0 if self.spec.sex == "male" else -1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                qpos_without_global_xy,
                qvel,
                self.command,
                body_params,
                self.prev_action,
            ]
        ).astype(np.float32)

    def _reward(self, action: np.ndarray, velocity: np.ndarray) -> float:
        velocity_error = np.linalg.norm(velocity - self.command)
        tracking = np.exp(-(velocity_error / self.config.velocity_sigma) ** 2)
        effort = self.config.control_cost * float(np.square(action).sum())
        smooth = self.config.action_rate_cost * float(np.square(action - self.prev_action).sum())
        side_drift = self.config.lateral_height_cost * abs(float(self.data.qpos[1]))
        return float(tracking + self.config.healthy_reward - effort - smooth - side_drift)

    def _is_healthy(self) -> bool:
        height = float(self.data.qpos[2])
        return (
            self.config.healthy_min_height_fraction * self.spec.height
            <= height
            <= self.config.healthy_max_height_fraction * self.spec.height
            and np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        )

    def _maybe_apply_push(self) -> None:
        if self.rng.random() >= self.config.push_probability:
            return
        thorax_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        if thorax_id < 0:
            return
        force_xy = self.rng.normal(0.0, self.config.push_force_std, size=2)
        self.data.xfrc_applied[thorax_id, 0] = force_xy[0]
        self.data.xfrc_applied[thorax_id, 1] = force_xy[1]

    def _info(self, velocity: np.ndarray | None = None) -> dict:
        return {
            "command": self.command.copy(),
            "mass": self.spec.mass,
            "height": self.spec.height,
            "sex": self.spec.sex,
            "velocity": None if velocity is None else velocity.copy(),
        }
