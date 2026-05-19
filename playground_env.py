from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import mjx

try:
    from .config import EnvConfig
except ImportError:
    from config import EnvConfig


@dataclass(frozen=True)
class PlaygroundEnvSpec:
    env_name: str
    env_version: str


def playground_env_name(config: EnvConfig) -> str:
    if config.env_version == "standard":
        return config.playground_flat_env
    if config.env_version == "hardcore":
        return config.playground_hardcore_env
    raise ValueError("env_version mora biti 'standard' ili 'hardcore'.")


class PlaygroundJoystickEnv(gym.Env):
    """Gymnasium adapter za MuJoCo Playground humanoid joystick env."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.env_name = playground_env_name(self.config)

        from mujoco_playground import locomotion

        self.playground_env = locomotion.load(
            self.env_name,
            config_overrides={"impl": self.config.playground_impl},
        )
        self.spec = PlaygroundEnvSpec(self.env_name, self.config.env_version)
        self.rng = jax.random.PRNGKey(0 if seed is None else seed)
        self.state = None

        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.playground_env.action_size,),
            dtype=np.float32,
        )
        obs_size = self.playground_env.observation_size
        obs_shape = tuple(obs_size["state"]) if isinstance(obs_size, dict) else (int(obs_size),)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(jax.device_get(self.state.info["command"]), dtype=np.float32)

    def set_command(self, command: np.ndarray) -> None:
        if self.state is not None:
            self.state.info["command"] = jnp.asarray(command, dtype=jnp.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, reset_rng = jax.random.split(self.rng)
        self.state = self.playground_env.reset(reset_rng)
        if options and "command" in options:
            self.set_command(np.asarray(options["command"], dtype=np.float32))
        return self._observation(), self._info()

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Pozovi reset() pre step().")
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.state = self.playground_env.step(self.state, jnp.asarray(action))
        reward = float(jax.device_get(self.state.reward))
        terminated = bool(jax.device_get(self.state.done))
        truncated = False
        return self._observation(), reward, terminated, truncated, self._info()

    def close(self):
        pass

    def render_data(self) -> mujoco.MjData:
        if self.state is None:
            raise RuntimeError("Pozovi reset() pre render_data().")
        return mjx.get_data(self.playground_env.mj_model, self.state.data)

    def _observation(self) -> np.ndarray:
        obs = self.state.obs
        if isinstance(obs, dict):
            obs = obs["state"]
        return np.asarray(jax.device_get(obs), dtype=np.float32)

    def _info(self) -> dict:
        return {
            "backend": "playground",
            "env_name": self.env_name,
            "env_version": self.config.env_version,
            "command": self.command.copy(),
        }
