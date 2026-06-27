"""Legacy Berkeley humanoid joystick training from commit 0168f50.

This is the old MuJoCo Playground Berkeley PPO path isolated from the active
biomechanics training code. Keep this runner intentionally boring: it should
match the working Berkeley setup from the old commit as closely as possible.
"""

import argparse
import functools
import json
import time
from dataclasses import dataclass
from datetime import datetime
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
from brax.training.agents.ppo import train as ppo
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loguru import logger
from mujoco import mjx
from mujoco_playground import locomotion
from mujoco_playground._src import wrapper
from mujoco_playground.config import locomotion_params

from config import (
    COMMAND_OBS_END,
    COMMAND_OBS_START,
    DEBUG_PRINT_INTERVAL,
    DEFAULT_COMMAND_STEP,
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
    RUNS_DIR,
)


class JoystickController:
    """Cita tastaturu iz MuJoCo viewer-a i menja Berkeley command vektor."""

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


@dataclass
class EnvConfig:
    """Old Berkeley env selection config from the working commit."""

    env_version: str = "standard"
    playground_impl: str = "jax"
    playground_flat_env: str = "BerkeleyHumanoidJoystickFlatTerrain"
    playground_hardcore_env: str = "BerkeleyHumanoidJoystickRoughTerrain"
    command_change_rate: float = 0.1

    def playground_env_name(self) -> str:
        """Map standard/hardcore to the old MuJoCo Playground env name."""
        if self.env_version == "standard":
            return self.playground_flat_env
        if self.env_version == "hardcore":
            return self.playground_hardcore_env
        raise ValueError("env_version mora biti 'standard' ili 'hardcore'.")


@dataclass
class TrainConfig:
    """Old PPO train config from the working Berkeley commit."""

    seed: int = 7
    num_timesteps: int | None = None
    num_evals: int | None = None
    num_envs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    algorithm: str = "ppo"


class TrainingProgressLogger:
    """Prima PPO evaluacione metrike i upisuje ih u loguru logger."""

    def __call__(self, step: int, metrics: dict) -> None:
        reward = metrics.get("eval/episode_reward")
        episode_length = metrics.get("eval/episode_length")
        logger.info(
            "eval | step={} | reward={} | episode_length={}",
            step,
            None if reward is None else float(reward),
            None if episode_length is None else float(episode_length),
        )


def identity_observation_preprocessor(normalizer_params, observations):
    """Vrati observation bez normalizacije kada checkpoint to ne koristi."""
    del normalizer_params
    return observations


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


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve checkpoint dir, checkpoints dir, or run dir to one checkpoint."""
    candidate = Path(path).expanduser().resolve(strict=False)
    if is_checkpoint_dir(candidate):
        return candidate

    checkpoint_root = candidate / "checkpoints"
    if checkpoint_root.is_dir():
        return latest_checkpoint_dir(checkpoint_root)
    if candidate.is_dir():
        return latest_checkpoint_dir(candidate)
    return candidate


def is_checkpoint_dir(path: Path) -> bool:
    """Brax checkpoint dirs contain PPO network metadata."""
    return path.is_dir() and (path / "ppo_network_config.json").exists()


def latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    """Find numerically latest checkpoint under a checkpoints directory."""
    checkpoint_dirs = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"Nema checkpoint foldera u {checkpoint_root}")
    return max(checkpoint_dirs, key=lambda path: int(path.name))


def find_run_config(checkpoint_path: Path) -> dict | None:
    """Nadji config.json u roditeljskom run direktorijumu checkpointa."""
    for path in (checkpoint_path, *checkpoint_path.parents):
        config_path = path / "config.json"
        if not config_path.exists():
            continue
        return json.loads(config_path.read_text(encoding="utf-8"))
    return None


def load_ppo_policy(checkpoint_path: Path, deterministic: bool):
    """Ucita Brax PPO checkpoint i rekonstruise inference policy."""
    params = checkpoint.load(str(checkpoint_path))
    networks = load_ppo_networks(checkpoint_path)
    make_policy = ppo_networks.make_inference_fn(networks)
    return make_policy(params, deterministic=deterministic)


def load_ppo_networks(checkpoint_path: Path):
    """Rekonstruise PPO mreze iz checkpoint metadata fajla."""
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


def read_checkpoint_observation_size(checkpoint_path: Path) -> int | None:
    """Procitaj policy observation size iz checkpoint metadata fajla."""
    network_config_path = checkpoint_path / "ppo_network_config.json"
    if not network_config_path.exists():
        return None
    network_config = json.loads(network_config_path.read_text(encoding="utf-8"))

    observation_size = network_config.get("observation_size", {})
    if "state" in observation_size:
        shape = observation_size["state"].get("shape")
    else:
        shape = observation_size.get("shape")
    if not shape:
        return None
    return int(shape[0])


def policy_observation_size(obs) -> int:
    """Vrati velicinu observation-a koji policy cita."""
    if isinstance(obs, dict):
        return int(obs["state"].shape[0])
    return int(obs.shape[0])


def validate_observation_compatibility(checkpoint_path: Path, obs) -> None:
    """Uhvatiti checkpoint/env mismatch pre nejasnog JAX broadcast error-a."""
    expected_size = read_checkpoint_observation_size(checkpoint_path)
    if expected_size is None:
        return

    actual_size = policy_observation_size(obs)
    if actual_size == expected_size:
        return

    raise ValueError(
        "Checkpoint i env nisu kompatibilni: "
        f"checkpoint ocekuje policy observation {expected_size}, "
        f"a trenutni env daje {actual_size}. Proveri --env-version."
    )


def clip_command(command: np.ndarray) -> None:
    """Drzi Berkeley command u opsegu na kom je env treniran."""
    command[0] = np.clip(command[0], -1.0, 1.0)
    command[1] = np.clip(command[1], -1.0, 1.0)
    command[2] = np.clip(command[2], -1.0, 1.0)


def set_legacy_command(state, command: np.ndarray):
    """Upise joystick komandu u Berkeley info i policy observation."""
    command_array = jnp.asarray(command, dtype=jnp.float32)
    info = dict(state.info)
    info["command"] = command_array

    if isinstance(state.obs, dict):
        obs = dict(state.obs)
        obs["state"] = obs["state"].at[
            COMMAND_OBS_START:COMMAND_OBS_END
        ].set(command_array)
        if "privileged_state" in obs:
            obs["privileged_state"] = obs["privileged_state"].at[
                COMMAND_OBS_START:COMMAND_OBS_END
            ].set(command_array)
    else:
        obs = state.obs.at[COMMAND_OBS_START:COMMAND_OBS_END].set(
            command_array
        )
    return state.replace(info=info, obs=obs)


def reset_legacy_state(env, rng, command: np.ndarray):
    """Resetuje Berkeley epizodu i zadrzava trenutnu joystick komandu."""
    return set_legacy_command(env.reset(rng), command)


def patch_jax_for_brax_compatibility() -> None:
    """Vraca API koji Brax jos koristi, a JAX 0.10 ga je uklonio.

    Brax 0.14 jos poziva `jax.device_put_replicated`. U novom JAX-u je taj
    helper uklonjen, a zvanicna zamena koristi `jax.device_put` sa sharding-om.
    Ovaj patch je lokalni most dok se Brax/Playground ne usklade sa JAX 0.10.
    """
    if hasattr(jax, "device_put_replicated"):
        return

    def device_put_replicated(x, devices):
        mesh = Mesh(np.array(devices), ("x",))
        sharding = NamedSharding(mesh, PartitionSpec("x"))
        return jax.tree.map(
            lambda leaf: jax.device_put(
                jnp.stack([leaf] * len(devices)),
                sharding,
            ),
            x,
        )

    jax.device_put_replicated = device_put_replicated


def choose_device(name: str, allow_cpu: bool):
    """Bira JAX uredjaj i eksplicitno odbija tihi fallback sa GPU-a na CPU."""
    if name == "cpu":
        return jax.devices("cpu")[0]

    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if gpus:
        return gpus[0]
    if allow_cpu:
        return jax.devices("cpu")[0]
    raise RuntimeError(
        "JAX ne vidi GPU. Dodaj --allow-cpu za mali CPU run."
    )


def format_steps(steps: int) -> str:
    """Formatira broj stepova za ime run foldera."""
    if steps >= 1_000_000:
        whole = steps // 1_000_000
        rest = (steps % 1_000_000) // 100_000
        return f"{whole}m{rest}" if rest else f"{whole}m"
    if steps >= 1_000 and steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def make_run_dir(base_dir: Path, env_name: str, timesteps: int, seed: int) -> Path:
    """Pravi deterministicki citljivo ime run foldera."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return base_dir / f"ppo_{env_name}_{stamp}_{format_steps(timesteps)}_seed{seed}"


def make_ppo_config(
    env_name: str,
    env_config: EnvConfig,
    train_config: TrainConfig,
):
    """Uzima tuned Playground PPO config i primenjuje samo eksplicitne override-e."""
    rl_config = locomotion_params.brax_ppo_config(
        env_name,
        impl=env_config.playground_impl,
    )
    overrides = {
        "num_timesteps": train_config.num_timesteps,
        "num_evals": train_config.num_evals,
        "num_envs": train_config.num_envs,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
    }
    for key, value in overrides.items():
        if value is not None:
            rl_config[key] = value

    return rl_config


def make_network_factory(rl_config):
    """Pretvara Playground network config u Brax PPO network factory callable."""
    return functools.partial(
        ppo_networks.make_ppo_networks,
        **dict(rl_config.network_factory),
    )


def save_run_config(
    run_dir: Path,
    env_config: EnvConfig,
    train_config: TrainConfig,
    rl_config,
) -> None:
    """Snima nas config i finalni Brax PPO config koji stvarno ide u trening."""
    serializable_rl_config = rl_config.to_dict()
    serializable_rl_config["network_factory_fn"] = (
        "brax.training.agents.ppo.networks.make_ppo_networks"
    )
    data = {
        "env": env_config.__dict__,
        "train": train_config.__dict__,
        "brax_ppo": serializable_rl_config,
    }
    (run_dir / "config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_training(
    env_config: EnvConfig,
    train_config: TrainConfig,
    out_dir: Path,
) -> Path:
    """Pokrece MuJoCo Playground env kroz Brax PPO/MJX training pipeline."""
    patch_jax_for_brax_compatibility()

    env_name = env_config.playground_env_name()
    rl_config = make_ppo_config(env_name, env_config, train_config)
    run_dir = make_run_dir(
        out_dir,
        env_name,
        rl_config.num_timesteps,
        train_config.seed,
    )
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(run_dir / "train.log", level="INFO", encoding="utf-8", mode="w")
    save_run_config(run_dir, env_config, train_config, rl_config)

    env = locomotion.load(
        env_name,
        config_overrides={"impl": env_config.playground_impl},
    )
    logger.info(
        "trening start | env={} | impl={} | run_dir={}",
        env_name,
        env_config.playground_impl,
        run_dir,
    )
    logger.info(
        "ppo | timesteps={} | num_envs={} | batch_size={} | lr={}",
        rl_config.num_timesteps,
        rl_config.num_envs,
        rl_config.batch_size,
        rl_config.learning_rate,
    )

    train_kwargs = rl_config.to_dict()
    train_kwargs["network_factory"] = make_network_factory(rl_config)
    ppo.train(
        environment=env,
        seed=train_config.seed,
        progress_fn=TrainingProgressLogger(),
        save_checkpoint_path=str(checkpoint_dir),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **train_kwargs,
    )
    logger.info("trening gotov | checkpoints={}", checkpoint_dir)
    return run_dir


@functools.partial(jax.jit, static_argnames=("env", "policy"))
def legacy_simulation_step(env, policy, state, rng, command):
    """Izvrsi jedan JIT-ovan legacy policy+environment korak."""
    state = set_legacy_command(state, command)
    action, _ = policy(state.obs, rng)
    next_state = env.step(state, action)
    next_state = set_legacy_command(next_state, command)
    return next_state, action


def update_viewer_data(model, data, state) -> None:
    """Kopira MJX state u MuJoCo viewer data."""
    latest_data = mjx.get_data(model, state.data)
    data.qpos[:] = latest_data.qpos
    data.qvel[:] = latest_data.qvel
    mujoco.mj_forward(model, data)


def print_legacy_debug(step: int, state, action) -> None:
    """Ispise osnovni signal tokom legacy viewer rollout-a."""
    if step % DEBUG_PRINT_INTERVAL != 0:
        return

    obs_command = np.asarray(
        state.obs["state"][COMMAND_OBS_START:COMMAND_OBS_END]
    )
    qpos = np.asarray(state.data.qpos[:3])
    action_norm = float(np.asarray(jnp.linalg.norm(action)))
    print(
        "debug "
        f"step={step} "
        f"command={obs_command.round(3)} "
        f"reward={float(np.asarray(state.reward)):.3f} "
        f"qpos={qpos.round(3)} "
        f"done={float(np.asarray(state.done)):.1f} "
        f"action_norm={action_norm:.3f}",
        flush=True,
    )


def inspect_legacy_policy(env, policy, rng, command: np.ndarray, steps: int) -> None:
    """Pokreni headless legacy rollout i ispisi kratke metrike."""
    state = reset_legacy_state(env, rng, command)
    total_reward = 0.0
    reset_count = 0
    episode_lengths = []
    current_episode_length = 0
    z_values = []
    action_norm_values = []

    print("compiling first JAX inspect step...", flush=True)
    rng, action_key = jax.random.split(rng)
    state, action = legacy_simulation_step(
        env,
        policy,
        state,
        action_key,
        command,
    )
    jax.block_until_ready(action)
    print("compile done, running inspect rollout", flush=True)

    for step in range(steps):
        rng, action_key, reset_key = jax.random.split(rng, 3)
        state, action = legacy_simulation_step(
            env,
            policy,
            state,
            action_key,
            command,
        )
        reward = float(np.asarray(state.reward))
        done = bool(np.asarray(state.done))
        qpos_z = float(np.asarray(state.data.qpos[2]))
        action_norm = float(np.asarray(jnp.linalg.norm(action)))

        total_reward += reward
        current_episode_length += 1
        z_values.append(qpos_z)
        action_norm_values.append(action_norm)

        if step % DEBUG_PRINT_INTERVAL == 0 or done:
            obs_command = np.asarray(
                state.obs["state"][COMMAND_OBS_START:COMMAND_OBS_END]
            )
            print(
                "inspect "
                f"step={step} "
                f"command={obs_command.round(3)} "
                f"reward={reward:.3f} "
                f"z={qpos_z:.3f} "
                f"done={int(done)} "
                f"action_norm={action_norm:.3f}",
                flush=True,
            )

        if done:
            reset_count += 1
            episode_lengths.append(current_episode_length)
            state = reset_legacy_state(env, reset_key, command)
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
        f"mean_action_norm={np.mean(action_norm_values):.3f}",
        flush=True,
    )


def run_evaluation(
    env_config: EnvConfig,
    checkpoint_path: Path,
    seed: int,
    command: np.ndarray,
    command_step: float,
    stochastic: bool,
    inspect: bool,
    inspect_steps: int,
    debug: bool,
) -> None:
    """Pusti legacy Berkeley checkpoint kroz odgovarajuci Playground env."""
    patch_jax_for_brax_compatibility()

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    run_config = find_run_config(checkpoint_path)
    if run_config is not None:
        env_version = run_config.get("env", {}).get("env_version")
        playground_impl = run_config.get("env", {}).get("playground_impl")
        if env_version in {"standard", "hardcore"}:
            env_config.env_version = env_version
        if playground_impl in {"jax", "warp"}:
            env_config.playground_impl = playground_impl

    env_name = env_config.playground_env_name()
    print(
        "legacy eval config | "
        f"env={env_name} | "
        f"impl={env_config.playground_impl} | "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    env = locomotion.load(
        env_name,
        config_overrides={"impl": env_config.playground_impl},
    )
    policy = load_ppo_policy(checkpoint_path, deterministic=not stochastic)

    rng = jax.random.PRNGKey(seed)
    clip_command(command)
    state = reset_legacy_state(env, rng, command)
    validate_observation_compatibility(checkpoint_path, state.obs)
    print(
        "initial command | "
        f"x={command[0]:.2f} | y={command[1]:.2f} | yaw={command[2]:.2f}",
        flush=True,
    )

    if inspect:
        inspect_legacy_policy(env, policy, rng, command, inspect_steps)
        return

    print("compiling first JAX simulation step...", flush=True)
    rng, action_key = jax.random.split(rng)
    state, action = legacy_simulation_step(
        env,
        policy,
        state,
        action_key,
        command,
    )
    jax.block_until_ready(action)
    print("compile done, opening MuJoCo viewer", flush=True)

    model = env.mj_model
    data = mjx.get_data(model, state.data)
    controller = JoystickController(command, command_step)
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
                state, action = legacy_simulation_step(
                    env,
                    policy,
                    state,
                    action_key,
                    controller.command,
                )
                if bool(np.asarray(state.done)):
                    print("episode done, resetting", flush=True)
                    state = reset_legacy_state(
                        env,
                        reset_key,
                        controller.command,
                    )
                if debug:
                    print_legacy_debug(step, state, action)
                update_viewer_data(model, data, state)
                step += 1

            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(env.dt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy MuJoCo Playground/Brax Berkeley PPO trening."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--allow-cpu", action="store_true")
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
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-evals", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--eval-checkpoint",
        "--checkpoint",
        dest="eval_checkpoint",
        type=Path,
        default=None,
        help="Legacy checkpoint, checkpoints folder, or run folder to view.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-steps", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--command-x", type=float, default=0.0)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-step", type=float, default=DEFAULT_COMMAND_STEP)
    args = parser.parse_args()

    device = choose_device(args.device, args.allow_cpu)
    jax.config.update("jax_default_device", device)

    env_config = EnvConfig(
        env_version=args.env_version,
        playground_impl=args.playground_impl,
    )
    if args.eval_checkpoint is not None:
        command = np.array(
            [args.command_x, args.command_y, args.command_yaw],
            dtype=np.float32,
        )
        run_evaluation(
            env_config=env_config,
            checkpoint_path=args.eval_checkpoint,
            seed=args.seed,
            command=command,
            command_step=args.command_step,
            stochastic=args.stochastic,
            inspect=args.inspect,
            inspect_steps=args.inspect_steps,
            debug=args.debug,
        )
        return

    train_config = TrainConfig(
        seed=args.seed,
        num_timesteps=args.timesteps,
        num_evals=args.num_evals,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    run_training(env_config, train_config, args.out)


if __name__ == "__main__":
    main()
