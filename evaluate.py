from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np

try:
    from .config import EnvConfig
    from .deep_rl import TrainConfig, make_agent
    from .env_factory import make_env
except ImportError:
    from config import EnvConfig
    from deep_rl import TrainConfig, make_agent
    from env_factory import make_env


KEY_SPACE = 32
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_DOWN = 264
KEY_UP = 265


def choose_device(name: str):
    if name == "cpu":
        return jax.devices("cpu")[0]
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        raise RuntimeError("JAX ne vidi GPU. Za CPU probu dodaj --device cpu.")
    return gpus[0]


def load_agent(path: Path, device):
    with path.open("rb") as f:
        data = pickle.load(f)
    jax.config.update("jax_default_device", device)
    cfg = TrainConfig(algo=data["algo"], hidden=data["hidden"], device=str(device))
    agent = make_agent(data["algo"], data["obs_dim"], data["action_dim"], cfg, jax.random.PRNGKey(0))
    agent.load_dict(data["agent"])
    return agent, data.get("env_config", {})


def clip_command(command: np.ndarray) -> None:
    command[0] = np.clip(command[0], -1.2, 1.4)
    command[1] = np.clip(command[1], -1.0, 1.0)
    command[2] = np.clip(command[2], -1.5, 1.5)


def main():
    parser = argparse.ArgumentParser(description="Gledanje JAX politike u MuJoCo viewer-u.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--command-x", type=float, default=0.8)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--env-backend", choices=["playground", "biomech"], default=None)
    parser.add_argument("--env-version", choices=["standard", "hardcore"], default=None)
    parser.add_argument("--playground-impl", choices=["jax", "warp"], default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    device = choose_device(args.device)
    agent, checkpoint_env_config = load_agent(args.model, device)
    env_config = EnvConfig(**checkpoint_env_config) if checkpoint_env_config else EnvConfig()
    if args.env_backend is not None:
        env_config.env_backend = args.env_backend
    if args.env_version is not None:
        env_config.env_version = args.env_version
    if args.playground_impl is not None:
        env_config.playground_impl = args.playground_impl
    env = make_env(config=env_config, seed=args.seed)
    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    clip_command(command)
    obs, _ = env.reset(options={"command": command})
    paused = False

    def key_callback(keycode):
        nonlocal paused
        current = env.command if hasattr(env, "command") else command
        if keycode == KEY_SPACE:
            paused = not paused
            return
        elif keycode in (KEY_UP, 87, 119):
            current[0] += env.config.command_change_rate
        elif keycode in (KEY_DOWN, 83, 115):
            current[0] -= env.config.command_change_rate
        elif keycode in (KEY_LEFT, 65, 97):
            current[1] += env.config.command_change_rate
        elif keycode in (KEY_RIGHT, 68, 100):
            current[1] -= env.config.command_change_rate
        elif keycode in (81, 113):
            current[2] += env.config.command_change_rate
        elif keycode in (69, 101):
            current[2] -= env.config.command_change_rate
        clip_command(current)
        if hasattr(env, "set_command"):
            env.set_command(current)
        else:
            env.command[:] = current

    model = env.playground_env.mj_model if hasattr(env, "playground_env") else env.model
    data = env.render_data() if hasattr(env, "render_data") else env.data

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20

        while viewer.is_running():
            if not paused:
                action = agent.act(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset(options={"command": env.command})

                if hasattr(env, "render_data"):
                    latest_data = env.render_data()
                    data.qpos[:] = latest_data.qpos
                    data.qvel[:] = latest_data.qvel
                    mujoco.mj_forward(model, data)

            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(model.opt.timestep)

    env.close()


if __name__ == "__main__":
    main()
