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

from biomechanics_env import BiomechanicsJoystickEnv
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
    expand_reference_gait_files,
)


OBS_COMMAND_START = 9
OBS_COMMAND_END = 12
BIOMECH_COMMAND_START = 9
BIOMECH_COMMAND_END = 12
DEBUG_PRINT_INTERVAL = 120
DEFAULT_WALK_COMMAND_X = 0.25


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


def infer_command_profile(checkpoint_path: Path) -> str:
    """Procitaj command_profile iz run configa, uz fallback na obs size."""
    run_config = find_run_config(checkpoint_path)
    if run_config is not None:
        profile = run_config.get("env", {}).get("command_profile")
        if profile in {"forward", "walk", "steer", "standard_easy", "standard"}:
            return profile

    obs_size = read_checkpoint_observation_size(checkpoint_path)
    if obs_size == 92:
        return "walk"
    return "forward"


def infer_init_qpos_file(checkpoint_path: Path) -> str | None:
    """Procitaj init_qpos_file iz run configa ako je trening koristio tu opciju."""
    run_config = find_run_config(checkpoint_path)
    if run_config is None:
        return None
    return run_config.get("env", {}).get("init_qpos_file")


def infer_reference_gait(checkpoint_path: Path) -> str:
    """Procitaj reference_gait iz run configa."""
    run_config = find_run_config(checkpoint_path)
    if run_config is None:
        return "none"
    reference_gait = run_config.get("env", {}).get("reference_gait", "none")
    if reference_gait in {"none", "sine", "bvh"}:
        return reference_gait
    return "none"


def infer_reference_gait_file(checkpoint_path: Path) -> str | list[str] | None:
    """Procitaj reference_gait_file iz run configa."""
    run_config = find_run_config(checkpoint_path)
    if run_config is None:
        return None
    return run_config.get("env", {}).get("reference_gait_file")


def find_run_config(checkpoint_path: Path) -> dict | None:
    """Nadji config.json u roditeljskom run direktorijumu checkpointa."""
    for path in (checkpoint_path, *checkpoint_path.parents):
        config_path = path / "config.json"
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    return None


def read_checkpoint_observation_size(checkpoint_path: Path) -> int | None:
    """Procitaj observation_size iz Brax checkpoint metadata fajla."""
    network_config_path = checkpoint_path / "ppo_network_config.json"
    if not network_config_path.exists():
        return None
    with network_config_path.open("r", encoding="utf-8") as config_file:
        network_config = json.load(config_file)

    observation_size = network_config.get("observation_size", {})
    if "state" in observation_size:
        shape = observation_size["state"].get("shape")
    else:
        shape = observation_size.get("shape")
    if not shape:
        return None
    return int(shape[0])


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

    observation_size = parse_observation_size(config["observation_size"])
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


def parse_observation_size(raw_observation_size):
    """Procita Brax observation_size za dict i array observation formate."""
    if isinstance(raw_observation_size, list):
        shape = tuple(raw_observation_size)
        return shape[0] if len(shape) == 1 else shape
    if "shape" in raw_observation_size:
        shape = tuple(raw_observation_size["shape"])
        return shape[0] if len(shape) == 1 else shape
    return {
        name: tuple(value["shape"])
        for name, value in raw_observation_size.items()
    }


def set_command(state, command: np.ndarray):
    """Upise joystick komandu u info i observation koje politika cita."""
    command_array = jnp.asarray(command, dtype=jnp.float32)

    info = dict(state.info)
    info["command"] = command_array
    if isinstance(state.obs, dict):
        obs = dict(state.obs)
        obs["state"] = obs["state"].at[OBS_COMMAND_START:OBS_COMMAND_END].set(
            command_array
        )
        obs["privileged_state"] = obs["privileged_state"].at[
            OBS_COMMAND_START:OBS_COMMAND_END
        ].set(command_array)
    else:
        obs = state.obs.at[BIOMECH_COMMAND_START:BIOMECH_COMMAND_END].set(
            command_array
        )
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

    if isinstance(state.obs, dict):
        command = np.asarray(state.obs["state"][OBS_COMMAND_START:OBS_COMMAND_END])
    else:
        command = np.asarray(state.obs[BIOMECH_COMMAND_START:BIOMECH_COMMAND_END])
    qpos = np.asarray(state.data.qpos[:3])
    qvel = np.asarray(state.data.qvel[:6])
    torso_up = get_torso_up(state)
    action_norm = float(jnp.linalg.norm(action))
    print(
        "debug "
        f"step={step} "
        f"obs_command={command.round(3)} "
        f"qpos={qpos.round(3)} "
        f"qvel={qvel.round(3)} "
        f"torso_up={torso_up:.3f} "
        f"done={float(np.asarray(state.done)):.1f} "
        f"action_norm={action_norm:.3f}",
        flush=True,
    )


def get_torso_up(state) -> float:
    """Procita anatomski upright signal za biomechanics env."""
    torso_xmat = np.asarray(state.data.xmat[1]).reshape(3, 3)
    return float(torso_xmat[2, 1])


def print_done_reason(state) -> None:
    """Objasni zasto je biomechanics epizoda resetovana."""
    qpos_z = float(np.asarray(state.data.qpos[2]))
    torso_up = get_torso_up(state)
    print(
        "episode done, resetting | "
        f"z={qpos_z:.3f} "
        f"torso_up={torso_up:.3f}",
        flush=True,
    )


@functools.partial(jax.jit, static_argnames=("env", "policy"))
def simulation_step(env, policy, state, rng, command):
    """Izvrsi jedan JIT-ovan policy+environment korak."""
    state = set_command(state, command)
    action, _ = policy(state.obs, rng)
    next_state = env.step(state, action)
    return next_state, action


def inspect_policy(env, policy, rng, command: np.ndarray, steps: int) -> None:
    """Pokreni headless rollout i ispisi objektivne survival metrike."""
    state = reset_state(env, rng, command)
    total_reward = 0.0
    reset_count = 0
    episode_lengths = []
    current_episode_length = 0
    z_values = []
    torso_up_values = []
    action_norm_values = []

    print("compiling first JAX inspect step...", flush=True)
    rng, action_key = jax.random.split(rng)
    state, action = simulation_step(env, policy, state, action_key, command)
    jax.block_until_ready(action)
    print("compile done, running inspect rollout", flush=True)

    for step in range(steps):
        rng, action_key, reset_key = jax.random.split(rng, 3)
        state, action = simulation_step(env, policy, state, action_key, command)
        reward = float(np.asarray(state.reward))
        done = bool(np.asarray(state.done))
        qpos_z = float(np.asarray(state.data.qpos[2]))
        torso_up = get_torso_up(state)
        action_norm = float(np.asarray(jnp.linalg.norm(action)))

        total_reward += reward
        current_episode_length += 1
        z_values.append(qpos_z)
        torso_up_values.append(torso_up)
        action_norm_values.append(action_norm)

        if step % DEBUG_PRINT_INTERVAL == 0 or done:
            print(
                "inspect "
                f"step={step} "
                f"reward={reward:.3f} "
                f"z={qpos_z:.3f} "
                f"torso_up={torso_up:.3f} "
                f"done={int(done)} "
                f"action_norm={action_norm:.3f}",
                flush=True,
            )

        if done:
            reset_count += 1
            episode_lengths.append(current_episode_length)
            state = reset_state(env, reset_key, command)
            current_episode_length = 0

    if current_episode_length:
        episode_lengths.append(current_episode_length)

    print(
        "inspect summary | "
        f"steps={steps} "
        f"resets={reset_count} "
        f"mean_episode_length={np.mean(episode_lengths):.1f} "
        f"total_reward={total_reward:.3f} "
        f"mean_z={np.mean(z_values):.3f} "
        f"min_z={np.min(z_values):.3f} "
        f"mean_torso_up={np.mean(torso_up_values):.3f} "
        f"min_torso_up={np.min(torso_up_values):.3f} "
        f"mean_action_norm={np.mean(action_norm_values):.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gledanje Brax PPO politike u MuJoCo viewer-u."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--env-source",
        choices=["biomechanics", "prototip"],
        default="biomechanics",
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
    parser.add_argument(
        "--command-profile",
        choices=["auto", "forward", "walk", "steer", "standard_easy", "standard"],
        default="auto",
    )
    parser.add_argument(
        "--reference-gait",
        choices=["auto", "none", "sine", "bvh"],
        default="auto",
    )
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--reference-gait-list",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--action-smoothing", type=float, default=0.5)
    parser.add_argument("--init-qpos-file", type=Path, default=None)
    parser.add_argument("--command-x", type=float, default=None)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-step", type=float, default=0.05)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-steps", type=int, default=2000)
    parser.add_argument(
        "--accurate-physics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fast-physics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    jax.config.update("jax_default_device", device)

    command_profile = (
        infer_command_profile(args.checkpoint)
        if args.command_profile == "auto"
        else args.command_profile
    )
    command_x = args.command_x
    if command_x is None:
        command_x = DEFAULT_WALK_COMMAND_X if command_profile == "walk" else 0.0
    init_qpos_file = (
        str(args.init_qpos_file)
        if args.init_qpos_file is not None
        else infer_init_qpos_file(args.checkpoint)
    )
    reference_gait = (
        infer_reference_gait(args.checkpoint)
        if args.reference_gait == "auto"
        else args.reference_gait
    )
    if args.reference_gait_file is not None or args.reference_gait_list is not None:
        reference_gait_file = expand_reference_gait_files(
            args.reference_gait_file,
            args.reference_gait_list,
        )
    else:
        reference_gait_file = infer_reference_gait_file(args.checkpoint)
    print(
        "eval config | "
        f"command_profile={command_profile} | "
        f"command_x={command_x} | "
        f"init_qpos_file={init_qpos_file} | "
        f"reference_gait={reference_gait} | "
        f"reference_gait_file={reference_gait_file}",
        flush=True,
    )

    env_config = EnvConfig(
        env_source=args.env_source,
        env_version=args.env_version,
        playground_impl=args.playground_impl,
        command_profile=command_profile,
        reference_gait=reference_gait,
        reference_gait_file=reference_gait_file,
        action_smoothing=args.action_smoothing,
        init_qpos_file=init_qpos_file,
        accurate_physics=not args.fast_physics,
    )
    env = make_environment(env_config)
    policy = load_ppo_policy(args.checkpoint, deterministic=not args.stochastic)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = np.array(
        [command_x, args.command_y, args.command_yaw],
        dtype=np.float32,
    )
    clip_command(command)
    state = set_command(state, command)

    if args.inspect:
        inspect_policy(env, policy, rng, command, args.inspect_steps)
        return

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
                    print_done_reason(state)
                    state = reset_state(env, reset_key, controller.command)
                if args.debug:
                    print_debug(step, state, action)
                update_viewer_data(model, data, state)
                step += 1

            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(env.dt)


def make_environment(env_config: EnvConfig):
    """Napravi env za viewer."""
    if env_config.env_source == "prototip":
        return locomotion.load(
            env_config.prototype_env_name(),
            config_overrides={"impl": env_config.playground_impl},
        )
    config_overrides = {
        "impl": env_config.playground_impl,
        "enable_erfi": False,
        "command_profile": env_config.command_profile,
        "reference_gait": env_config.reference_gait,
        "action_smoothing": env_config.action_smoothing,
    }
    if env_config.reference_gait_file is not None:
        config_overrides["reference_gait_file"] = env_config.reference_gait_file
    if env_config.init_qpos_file is not None:
        config_overrides["init_qpos_file"] = env_config.init_qpos_file
    if env_config.accurate_physics:
        config_overrides["sim_dt"] = 0.005
    return BiomechanicsJoystickEnv(
        env_version=env_config.env_version,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()
