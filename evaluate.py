import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground import locomotion

from config import EnvConfig, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE, KEY_UP


class JoystickController:
    """Cita tastaturu iz MuJoCo viewer-a i menja command vektor politike."""

    def __init__(self, command: np.ndarray, step: float):
        self.command = command
        self.step = step
        self.paused = False

    def key_callback(self, keycode: int) -> None:
        if keycode == KEY_SPACE:
            self.paused = not self.paused
            return
        if keycode in (KEY_UP, 87, 119):
            self.command[0] += self.step
        elif keycode in (KEY_DOWN, 83, 115):
            self.command[0] -= self.step
        elif keycode in (KEY_LEFT, 65, 97):
            self.command[1] += self.step
        elif keycode in (KEY_RIGHT, 68, 100):
            self.command[1] -= self.step
        elif keycode in (81, 113):
            self.command[2] += self.step
        elif keycode in (69, 101):
            self.command[2] -= self.step
        clip_command(self.command)


def choose_device(name: str):
    """Bira JAX uredjaj za inference."""
    if name == "cpu":
        return jax.devices("cpu")[0]
    gpus = jax.devices("gpu")
    if not gpus:
        raise RuntimeError("JAX ne vidi GPU. Za CPU probu dodaj --device cpu.")
    return gpus[0]


def clip_command(command: np.ndarray) -> None:
    """Drzi joystick komandu u razumnom opsegu za hod."""
    command[0] = np.clip(command[0], -1.2, 1.4)
    command[1] = np.clip(command[1], -1.0, 1.0)
    command[2] = np.clip(command[2], -1.5, 1.5)


def load_ppo_policy(checkpoint_path: Path, deterministic: bool):
    """Ucita Brax PPO checkpoint i rekonstruise inference policy."""
    params = checkpoint.load(str(checkpoint_path))
    network_config = checkpoint.load_config(checkpoint_path / "config.json")
    networks = checkpoint.get_network(
        network_config,
        ppo_networks.make_ppo_networks,
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    return make_policy(params, deterministic=deterministic)


def set_command(state, command: np.ndarray):
    """Upise joystick komandu u Playground state info."""
    state.info["command"] = jnp.asarray(command, dtype=jnp.float32)
    return state


def update_viewer_data(model, data, state) -> None:
    """Kopira MJX state u MuJoCo viewer data."""
    latest_data = mjx.get_data(model, state.data)
    data.qpos[:] = latest_data.qpos
    data.qvel[:] = latest_data.qvel
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(
        description="Gledanje Brax PPO politike u MuJoCo viewer-u."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--env-version",
        choices=["standard", "hardcore"],
        default="standard",
    )
    parser.add_argument(
        "--playground-impl",
        choices=["jax", "warp"],
        default="jax",
    )
    parser.add_argument("--command-x", type=float, default=0.8)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-step", type=float, default=0.1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    device = choose_device(args.device)
    jax.config.update("jax_default_device", device)

    env_config = EnvConfig(
        env_version=args.env_version,
        playground_impl=args.playground_impl,
    )
    env_name = env_config.playground_env_name()
    env = locomotion.load(
        env_name,
        config_overrides={"impl": env_config.playground_impl},
    )
    policy = load_ppo_policy(args.checkpoint, deterministic=not args.stochastic)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = np.array(
        [args.command_x, args.command_y, args.command_yaw],
        dtype=np.float32,
    )
    clip_command(command)
    state = set_command(state, command)

    model = env.mj_model
    data = mjx.get_data(model, state.data)
    controller = JoystickController(command, args.command_step)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=controller.key_callback,
    ) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20

        while viewer.is_running():
            if not controller.paused:
                rng, action_key = jax.random.split(rng)
                state = set_command(state, controller.command)
                action, _ = policy(state.obs, action_key)
                state = env.step(state, action)
                update_viewer_data(model, data, state)

            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(env.dt)


if __name__ == "__main__":
    main()
