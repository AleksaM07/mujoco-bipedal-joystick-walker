"""Legacy Berkeley humanoid joystick training from commit 0168f50.

This is the old MuJoCo Playground Berkeley PPO path isolated from the active
biomechanics training code. Keep this runner intentionally boring: it should
match the working Berkeley setup from the old commit as closely as possible.
"""

import argparse
import functools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loguru import logger
from mujoco_playground import locomotion
from mujoco_playground._src import wrapper
from mujoco_playground.config import locomotion_params

from config import RUNS_DIR


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
        "JAX ne vidi GPU. Dodaj --allow-cpu samo za mali CPU run."
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
    args = parser.parse_args()

    device = choose_device(args.device, args.allow_cpu)
    jax.config.update("jax_default_device", device)

    env_config = EnvConfig(
        env_version=args.env_version,
        playground_impl=args.playground_impl,
    )
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
