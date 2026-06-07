import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training import checkpoint
from brax.training import networks as brax_networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground import locomotion

from config import (
    EnvConfig,
    KEY_A,
    KEY_D,
    KEY_DOWN,
    KEY_E,
    KEY_LEFT,
    KEY_NUMPAD_2,
    KEY_NUMPAD_4,
    KEY_NUMPAD_6,
    KEY_NUMPAD_7,
    KEY_NUMPAD_8,
    KEY_NUMPAD_9,
    KEY_Q,
    KEY_RIGHT,
    KEY_S,
    KEY_SPACE,
    KEY_UP,
    KEY_W,
)


OBS_COMMAND_START = 9
OBS_COMMAND_END = 12
DEBUG_PRINT_INTERVAL = 120


class JoystickController:
    """Cita tastaturu iz MuJoCo viewer-a i menja command vektor politike."""

    def __init__(self, command: np.ndarray, step: float):
        self.command = command
        self.step = step
        self.paused = False

    def key_callback(self, keycode: int) -> None:
        changed = False
        if keycode == KEY_SPACE:
            self.paused = not self.paused
            print(f"paused={self.paused}", flush=True)
            return
        if keycode in (KEY_UP, KEY_W, KEY_NUMPAD_8):
            self.command[0] += self.step
            changed = True
        elif keycode in (KEY_DOWN, KEY_S, KEY_NUMPAD_2):
            self.command[0] -= self.step
            changed = True
        elif keycode in (KEY_LEFT, KEY_A, KEY_NUMPAD_4):
            self.command[1] += self.step
            changed = True
        elif keycode in (KEY_RIGHT, KEY_D, KEY_NUMPAD_6):
            self.command[1] -= self.step
            changed = True
        elif keycode in (KEY_Q, KEY_NUMPAD_7):
            self.command[2] += self.step
            changed = True
        elif keycode in (KEY_E, KEY_NUMPAD_9):
            self.command[2] -= self.step
            changed = True

        if not changed:
            return

        clip_command(self.command)
        print(
            "command "
            f"x={self.command[0]:.2f} "
            f"y={self.command[1]:.2f} "
            f"yaw={self.command[2]:.2f}",
            flush=True,
        )


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
    checkpoint_path = checkpoint_path.resolve()
    params = checkpoint.load(str(checkpoint_path))
    networks = load_ppo_networks(checkpoint_path)
    make_policy = ppo_networks.make_inference_fn(networks)
    return make_policy(params, deterministic=deterministic)


def identity_observation_preprocessor(normalizer_params, observations):
    """Vraca observation bez normalizacije kada checkpoint to ne trazi."""
    return observations


def load_ppo_networks(checkpoint_path: Path):
    """Rekonstruise PPO mreze iz Brax checkpoint config-a.

    Brax `checkpoint.load_config` u ovoj verziji puca kada checkpoint JSON ima
    `null` za opcione kernel init parametre. Zato ovde citamo JSON direktno i
    konvertujemo samo vrednosti koje zaista imaju ime funkcije.
    """
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        config_path = checkpoint_path / "ppo_network_config.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    kwargs = config["network_factory_kwargs"]
    kwargs["activation"] = brax_networks.ACTIVATION[kwargs["activation"]]

    for key in (
        "policy_network_kernel_init_fn",
        "value_network_kernel_init_fn",
        "mean_kernel_init_fn",
    ):
        if kwargs.get(key) is not None:
            kwargs[key] = brax_networks.KERNEL_INITIALIZER[kwargs[key]]

    observation_size = {
        name: tuple(value["shape"])
        for name, value in config["observation_size"].items()
    }
    preprocess_observations_fn = (
        running_statistics.normalize
        if config["normalize_observations"]
        else identity_observation_preprocessor
    )
    return ppo_networks.make_ppo_networks(
        observation_size,
        config["action_size"],
        preprocess_observations_fn=preprocess_observations_fn,
        **kwargs,
    )


def set_command(state, command: np.ndarray):
    """Upise joystick komandu u info i observation koje politika cita."""
    command_array = jnp.asarray(command, dtype=jnp.float32)

    info = dict(state.info)
    info["command"] = command_array
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[OBS_COMMAND_START:OBS_COMMAND_END].set(
        command_array
    )
    obs["privileged_state"] = obs["privileged_state"].at[
        OBS_COMMAND_START:OBS_COMMAND_END
    ].set(command_array)
    return state.replace(info=info, obs=obs)


def update_viewer_data(model, data, state) -> None:
    """Kopira MJX state u MuJoCo viewer data."""
    latest_data = mjx.get_data(model, state.data)
    data.qpos[:] = latest_data.qpos
    data.qvel[:] = latest_data.qvel
    mujoco.mj_forward(model, data)


def reset_state(env, rng, command: np.ndarray):
    """Resetuje epizodu i zadrzava trenutnu joystick komandu."""
    state = env.reset(rng)
    return set_command(state, command)


def print_debug(step: int, state, action) -> None:
    """Ispise signal da proverimo da li politika vidi komandu."""
    if step % DEBUG_PRINT_INTERVAL != 0:
        return

    command = np.asarray(state.obs["state"][OBS_COMMAND_START:OBS_COMMAND_END])
    qpos = np.asarray(state.data.qpos[:3])
    qvel = np.asarray(state.data.qvel[:6])
    action_norm = float(jnp.linalg.norm(action))
    print(
        "debug "
        f"step={step} "
        f"obs_command={command.round(3)} "
        f"qpos={qpos.round(3)} "
        f"qvel={qvel.round(3)} "
        f"action_norm={action_norm:.3f}",
        flush=True,
    )


@functools.partial(jax.jit, static_argnames=("env", "policy"))
def simulation_step(env, policy, state, rng, command):
    """Izvrsi jedan JIT-ovan policy+environment korak."""
    state = set_command(state, command)
    action, _ = policy(state.obs, rng)
    next_state = env.step(state, action)
    return next_state, action


def main():
    parser = argparse.ArgumentParser(
        description="Gledanje Brax PPO politike u MuJoCo viewer-u."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--env-source",
        choices=["prototip"],
        default="prototip",
    )
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    jax.config.update("jax_default_device", device)

    env_config = EnvConfig(
        env_source=args.env_source,
        env_version=args.env_version,
        playground_impl=args.playground_impl,
    )
    env_name = env_config.prototype_env_name()
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

    print("compiling first JAX simulation step...", flush=True)
    rng, action_key = jax.random.split(rng)
    state, action = simulation_step(env, policy, state, action_key, command)
    jax.block_until_ready(action)
    print("compile done, opening MuJoCo viewer", flush=True)

    model = env.mj_model
    data = mjx.get_data(model, state.data)
    controller = JoystickController(command, args.command_step)
    step = 0

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
                rng, action_key, reset_key = jax.random.split(rng, 3)
                state, action = simulation_step(
                    env,
                    policy,
                    state,
                    action_key,
                    controller.command,
                )
                if bool(np.asarray(state.done)):
                    print("episode done, resetting", flush=True)
                    state = reset_state(env, reset_key, controller.command)
                if args.debug:
                    print_debug(step, state, action)
                update_viewer_data(model, data, state)
                step += 1

            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(env.dt)


if __name__ == "__main__":
    main()
