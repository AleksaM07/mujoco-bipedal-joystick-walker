from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np

from .deep_rl import TrainConfig, make_agent
from .env import HumanWalkEnv


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
    return agent


def main():
    parser = argparse.ArgumentParser(description="Gledanje JAX politike u MuJoCo viewer-u.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--command-x", type=float, default=0.8)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    device = choose_device(args.device)
    agent = load_agent(args.model, device)
    env = HumanWalkEnv(seed=args.seed)
    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    obs, _ = env.reset(options={"command": command})
    paused = False

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:
            paused = not paused
        elif keycode in (87, 119):
            env.command[0] += 0.1
        elif keycode in (83, 115):
            env.command[0] -= 0.1
        elif keycode in (65, 97):
            env.command[1] += 0.1
        elif keycode in (68, 100):
            env.command[1] -= 0.1
        elif keycode in (81, 113):
            env.command[2] += 0.1
        elif keycode in (69, 101):
            env.command[2] -= 0.1

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20

        while viewer.is_running():
            if not paused:
                action = agent.act(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset(options={"command": env.command})

            viewer.cam.lookat[:] = env.data.qpos[:3]
            viewer.sync()
            time.sleep(env.model.opt.timestep * env.config.frame_skip)

    env.close()


if __name__ == "__main__":
    main()
