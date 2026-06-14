import argparse
import functools
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loguru import logger
from ml_collections import config_dict
from mujoco_playground import locomotion
from mujoco_playground._src import wrapper
from mujoco_playground.config import locomotion_params

from biomechanics_env import BiomechanicsJoystickEnv, domain_randomize
from config import RUNS_DIR, EnvConfig, TrainConfig


@contextmanager
def logged_stage(name: str):
    """Loguje pocetak, kraj i trajanje jedne faze."""
    start_time = time.perf_counter()
    logger.info("stage start | {}", name)
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info("stage done | {} | {:.2f}s", name, duration)


class TrainingProgressLogger:
    """Prima PPO metrike i upisuje samo korisne linije."""

    def __init__(self):
        self.final_reward = None
        self.best_reward = None
        self.best_step = None

    def __call__(self, step: int, metrics: dict) -> None:
        reward = metrics.get("eval/episode_reward")
        episode_length = metrics.get("eval/episode_length")
        if reward is None and episode_length is None:
            logger.info("train progress callback | step={}", step)
            return

        logger.info(
            "eval | step={} | reward={} | episode_length={}",
            step,
            None if reward is None else float(reward),
            None if episode_length is None else float(episode_length),
        )
        if reward is not None:
            self.final_reward = float(reward)
            if self.best_reward is None or self.final_reward > self.best_reward:
                self.best_reward = self.final_reward
                self.best_step = step


class PpoPercentLogger:
    """Loguje grubi PPO progres bez oslanjanja na evaluaciju."""

    def __init__(self, total_timesteps: int, percent_step: int = 5):
        self.total_timesteps = total_timesteps
        self.percent_step = percent_step
        self.next_percent = 0
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time

    def __call__(self, step: int, make_policy, params) -> None:
        del make_policy, params
        percent = min(100, int(step * 100 / self.total_timesteps))
        if percent < self.next_percent:
            return

        now = time.perf_counter()
        elapsed = now - self.start_time
        since_last = now - self.last_log_time
        eta = estimate_eta(elapsed, percent)
        logger.info(
            "ppo progress | step={} / {} | {}% | elapsed={} | "
            "last={} | eta={}",
            step,
            self.total_timesteps,
            percent,
            format_duration(elapsed),
            format_duration(since_last),
            format_duration(eta),
        )
        self.last_log_time = now
        while self.next_percent <= percent:
            self.next_percent += self.percent_step


def estimate_eta(elapsed: float, percent: int) -> float:
    """Proceni preostalo vreme iz procenta i proteklog vremena."""
    if percent <= 0:
        return 0.0
    return elapsed * (100 - percent) / percent


def format_duration(seconds: float) -> str:
    """Formatira trajanje kao 1h23m, 12m04s ili 8s."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


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


def configure_stdout_encoding() -> None:
    """Omoguci da Loguru traceback ne pukne na Windows legacy encoding-u."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


def make_run_dir(
    base_dir: Path,
    env_source: str,
    env_name: str,
    timesteps: int,
    seed: int,
) -> Path:
    """Pravi deterministicki citljivo ime run foldera."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return (
        base_dir
        / (
            f"{env_source}_ppo_{env_name}_{stamp}_"
            f"{format_steps(timesteps)}_seed{seed}_running"
        )
    )


def mark_run_status(
    run_dir: Path,
    status: str,
    final_reward: float | None = None,
    best_reward: float | None = None,
) -> Path:
    """Preimenuje run folder na `_s` ili `_f` suffix."""
    if status not in {"s", "f"}:
        raise ValueError("status mora biti 's' ili 'f'.")

    name = run_dir.name
    reward_suffix = ""
    if status == "s" and final_reward is not None:
        reward_suffix = f"_rew_{format_reward_for_path(final_reward)}"
    if status == "s" and best_reward is not None:
        reward_suffix += f"_best_{format_reward_for_path(best_reward)}"

    if name.endswith("_running"):
        new_name = name.removesuffix("_running") + f"{reward_suffix}_{status}"
    elif name.endswith("_s") or name.endswith("_f"):
        new_name = name[:-2] + f"{reward_suffix}_{status}"
    else:
        new_name = f"{name}{reward_suffix}_{status}"

    target = run_dir.with_name(new_name)
    if target.exists():
        target = run_dir.with_name(f"{new_name}_{datetime.now().strftime('%H%M%S')}")
    run_dir.rename(target)
    return target


def format_reward_for_path(reward: float) -> str:
    """Formatira reward za ime foldera bez tacke i suvisne duzine."""
    return f"{reward:.4f}".replace("-", "m").replace(".", "p")


def close_file_logger() -> None:
    """Zatvori loguru sinkove da Windows/WSL dozvoli rename run foldera."""
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")


def make_ppo_config(
    env_name: str,
    env_config: EnvConfig,
    train_config: TrainConfig,
):
    """Pravi PPO config za prototip ili biomehanicki human env."""
    if env_config.env_source == "prototip":
        rl_config = locomotion_params.brax_ppo_config(
            env_name,
            impl=env_config.playground_impl,
        )
    else:
        rl_config = biomechanics_ppo_config()

    overrides = {
        "num_timesteps": train_config.num_timesteps,
        "num_evals": train_config.num_evals,
        "num_envs": train_config.num_envs,
        "episode_length": train_config.episode_length,
        "unroll_length": train_config.unroll_length,
        "batch_size": train_config.batch_size,
        "num_minibatches": train_config.num_minibatches,
        "num_updates_per_batch": train_config.num_updates_per_batch,
        "learning_rate": train_config.learning_rate,
    }
    for key, value in overrides.items():
        if value is not None:
            rl_config[key] = value

    return rl_config


def biomechanics_ppo_config():
    """PPO polazne vrednosti za nas custom human joystick env."""
    return config_dict.create(
        num_timesteps=50_000_000,
        num_evals=10,
        num_envs=1024,
        num_eval_envs=32,
        episode_length=500,
        action_repeat=1,
        learning_rate=3e-4,
        entropy_cost=3e-3,
        discounting=0.97,
        unroll_length=20,
        batch_size=512,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        normalize_observations_std_eps=1e-3,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            activation=jax.nn.silu,
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )


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
    serializable_rl_config = make_json_safe(rl_config.to_dict())
    serializable_rl_config["network_factory_fn"] = (
        "brax.training.agents.ppo.networks.make_ppo_networks"
    )
    data = {
        "env": env_config.__dict__,
        "train": train_config.__dict__,
        "brax_ppo": serializable_rl_config,
    }
    (run_dir / "config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_json_safe(value):
    """Pretvori config vrednosti u JSON-safe oblik za run/config.json."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__name__}"
    return value


def run_training(
    env_config: EnvConfig,
    train_config: TrainConfig,
    out_dir: Path,
) -> Path:
    """Pokrece MuJoCo Playground env kroz Brax PPO/MJX training pipeline."""
    patch_jax_for_brax_compatibility()

    env_name = env_display_name(env_config)
    rl_config = make_ppo_config(env_name, env_config, train_config)
    run_dir = make_run_dir(
        out_dir,
        run_source_name(env_config, train_config),
        env_name,
        rl_config.num_timesteps,
        train_config.seed,
    )
    checkpoint_dir = (
        Path(train_config.checkpoint_out).expanduser()
        if train_config.checkpoint_out
        else run_dir / "checkpoints"
    )
    restore_checkpoint_path = (
        str(Path(train_config.resume_from).expanduser())
        if train_config.resume_from
        else None
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if train_config.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(run_dir / "train.log", level="INFO", encoding="utf-8", mode="w")
    save_run_config(run_dir, env_config, train_config, rl_config)

    enable_erfi = not train_config.bare
    with logged_stage("make_environment"):
        env = make_environment(env_config, enable_erfi=enable_erfi)
        eval_env = make_environment(env_config, enable_erfi=False)
    log_environment_summary(env, label="train env")
    log_eval_environment_summary(eval_env)
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
    if train_config.save_checkpoints:
        logger.info("checkpoints enabled | path={}", checkpoint_dir)
    else:
        logger.info("checkpoints disabled for this run")
    if restore_checkpoint_path is not None:
        logger.info(
            "resume enabled | restore_checkpoint_path={} | "
            "requires compatible env/action/obs/network config",
            restore_checkpoint_path,
        )
    logger.info(
        "ppo detail | episode_length={} | unroll_length={} | "
        "num_minibatches={} | updates_per_batch={} | num_evals={} | "
        "num_eval_envs={} | env_steps_per_training_block={}",
        rl_config.episode_length,
        rl_config.unroll_length,
        rl_config.num_minibatches,
        rl_config.num_updates_per_batch,
        rl_config.num_evals,
        rl_config.get("num_eval_envs", None),
        env_steps_per_training_block(rl_config),
    )
    if rl_config.num_timesteps < env_steps_per_training_block(rl_config):
        logger.warning(
            "requested timesteps is smaller than one PPO block; "
            "run will overshoot to at least {} env steps",
            env_steps_per_training_block(rl_config),
        )
    if env_config.env_source == "biomechanics" and rl_config.num_envs < 512:
        logger.warning(
            "biomechanics run uses only {} envs; GPU throughput is usually "
            "better with --num-envs 512 or 1024",
            rl_config.num_envs,
        )

    train_kwargs = rl_config.to_dict()
    train_kwargs["network_factory"] = make_network_factory(rl_config)
    if rl_config.num_evals == 0:
        train_kwargs["run_evals"] = False
        train_kwargs["num_evals"] = 11
    train_kwargs_extra = {}
    if (
        env_config.env_source == "biomechanics"
        and not train_config.bare
        and not train_config.no_domain_randomization
    ):
        train_kwargs_extra["randomization_fn"] = domain_randomize
    if train_config.bare:
        logger.info("bare mode | ERFI disabled | domain randomization disabled")
    if train_config.no_domain_randomization:
        logger.info("domain randomization disabled for this run")

    if train_config.debug_run:
        with logged_stage("debug_preflight"):
            debug_preflight(env, train_config.seed)

    logger.info("calling ppo.train")
    progress_logger = TrainingProgressLogger()
    save_checkpoint_path = (
        str(checkpoint_dir) if train_config.save_checkpoints else None
    )
    try:
        with logged_stage("ppo.train"):
            ppo.train(
                environment=env,
                eval_env=eval_env,
                seed=train_config.seed,
                progress_fn=progress_logger,
                policy_params_fn=PpoPercentLogger(rl_config.num_timesteps),
                save_checkpoint_path=save_checkpoint_path,
                restore_checkpoint_path=restore_checkpoint_path,
                wrap_env_fn=wrapper.wrap_for_brax_training,
                **train_kwargs_extra,
                **train_kwargs,
            )
    except Exception:
        logger.exception("trening pukao pre status rename-a")
        close_file_logger()
        failed_dir = mark_run_status(run_dir, "f")
        logger.error("trening pukao | run_dir={}", failed_dir)
        raise

    close_file_logger()
    success_dir = mark_run_status(
        run_dir,
        "s",
        progress_logger.final_reward,
        progress_logger.best_reward,
    )
    logger.info(
        "trening gotov | run_dir={} | checkpoints={} | best_reward={} | best_step={}",
        success_dir,
        checkpoint_dir if train_config.save_checkpoints else None,
        progress_logger.best_reward,
        progress_logger.best_step,
    )
    return success_dir


def log_environment_summary(env, label: str = "env") -> None:
    """Ispise dimenzije modela pre treninga."""
    model = env.mj_model
    logger.info(
        "{} summary | nq={} | nv={} | nu={} | nbody={} | ngeom={} | "
        "nsite={} | action_size={} | substeps={} | erfi_enabled={} | "
        "command_profile={} | action_smoothing={} | rfi_limit={} | "
        "rao_limit={} | xml={}",
        label,
        model.nq,
        model.nv,
        model.nu,
        model.nbody,
        model.ngeom,
        model.nsite,
        env.action_size,
        getattr(env, "n_substeps", None),
        getattr(env._config, "enable_erfi", None),
        getattr(env._config, "command_profile", None),
        getattr(env._config, "action_smoothing", None),
        getattr(env._config, "rfi_torque_limit", None),
        getattr(env._config, "rao_torque_limit", None),
        getattr(env, "xml_path", None),
    )


def log_eval_environment_summary(env) -> None:
    """Ispise najbitniju razliku eval env-a."""
    logger.info(
        "eval env | erfi_enabled={} | rfi_limit={} | rao_limit={}",
        getattr(env._config, "enable_erfi", None),
        getattr(env._config, "rfi_torque_limit", None),
        getattr(env._config, "rao_torque_limit", None),
    )


def env_steps_per_training_block(rl_config) -> int:
    """Vraca koliko env koraka Brax PPO napravi u jednom training bloku."""
    return int(
        rl_config.batch_size
        * rl_config.unroll_length
        * rl_config.num_minibatches
        * rl_config.action_repeat
    )


def debug_preflight(env, seed: int) -> None:
    """Proveri reset/step/JIT pre ulaska u Brax PPO."""
    rng = jax.random.PRNGKey(seed)
    action = None

    with logged_stage("preflight reset"):
        state = env.reset(rng)
        obs_shape = jax.tree_util.tree_map(
            lambda value: getattr(value, "shape", None),
            state.obs,
        )
        logger.info(
            "preflight reset ok | obs_shape={} | reward_dtype={} | done_dtype={}",
            obs_shape,
            state.reward.dtype,
            state.done.dtype,
        )

    with logged_stage("preflight eager step"):
        action = jnp.zeros(env.action_size)
        next_state = env.step(state, action)
        logger.info(
            "preflight eager step ok | reward={} | done={}",
            float(next_state.reward),
            float(next_state.done),
        )

    with logged_stage("preflight jit step compile"):
        jit_step = jax.jit(env.step)
        compiled_state = jit_step(state, action)
        jax.block_until_ready(compiled_state.reward)
        logger.info(
            "preflight jit step ok | reward={} | done={}",
            float(compiled_state.reward),
            float(compiled_state.done),
        )


def env_display_name(env_config: EnvConfig) -> str:
    """Vraca ime env-a za logove i run foldere."""
    if env_config.env_source == "prototip":
        return env_config.prototype_env_name()
    return f"BiomechanicsHumanJoystick{env_config.env_version.title()}"


def run_source_name(env_config: EnvConfig, train_config: TrainConfig) -> str:
    """Dodaje mode u ime run foldera kada nije standardni trening."""
    env_source = env_config.env_source
    if train_config.bare:
        env_source = f"{env_source}_bare"
    if env_config.command_profile != "standard":
        env_source = f"{env_source}_{env_config.command_profile}"
    if env_config.accurate_physics:
        env_source = f"{env_source}_accurate"
    return env_source


def make_environment(env_config: EnvConfig, enable_erfi: bool = True):
    """Napravi prototip ili pravi biomehanicki joystick env."""
    if env_config.env_source == "prototip":
        return locomotion.load(
            env_config.prototype_env_name(),
            config_overrides={"impl": env_config.playground_impl},
        )
    if env_config.env_source == "biomechanics":
        config_overrides = {
            "impl": env_config.playground_impl,
            "enable_erfi": enable_erfi,
            "command_profile": env_config.command_profile,
            "action_smoothing": env_config.action_smoothing,
        }
        if env_config.accurate_physics:
            config_overrides["sim_dt"] = 0.005
        return BiomechanicsJoystickEnv(
            env_version=env_config.env_version,
            config_overrides=config_overrides,
        )
    raise ValueError("env_source mora biti 'biomechanics' ili 'prototip'.")


def main() -> None:
    configure_stdout_encoding()
    parser = argparse.ArgumentParser(
        description="MuJoCo Playground/Brax PPO trening."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--allow-cpu", action="store_true")
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
        choices=["forward", "walk", "standard"],
        default="forward",
        help=(
            "Curriculum komande: forward za kompatibilan prvi hod, "
            "walk za gait-clock trening, standard za pun joystick."
        ),
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.5,
        help="Filtriranje policy akcije pre servo targeta; 0.5 prati walking repo.",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-evals", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--debug-run",
        action="store_true",
        help="Mali izolacioni run bez domain randomization.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Baseline: bez ERFI i bez domain randomization.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Ne snimaj Orbax checkpointove tokom ovog treninga.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help=(
            "Alternativni folder za checkpointove; korisno u WSL-u da se pise "
            "na Linux filesystem umesto na /mnt/c."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Putanja do kompatibilnog Brax/Orbax checkpointa za nastavak "
            "treninga. Mora biti isti env/action/obs/network setup."
        ),
    )
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
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    device = choose_device(args.device, args.allow_cpu)
    jax.config.update("jax_default_device", device)

    env_config = EnvConfig(
        env_source=args.env_source,
        env_version=args.env_version,
        playground_impl=args.playground_impl,
        command_profile=args.command_profile,
        action_smoothing=args.action_smoothing,
        accurate_physics=not args.fast_physics,
    )
    debug_defaults = debug_run_defaults(args.debug_run)
    train_config = TrainConfig(
        seed=args.seed,
        num_timesteps=args.timesteps or debug_defaults.get("num_timesteps"),
        num_evals=args.num_evals
        if args.num_evals is not None
        else debug_defaults.get("num_evals"),
        num_envs=args.num_envs or debug_defaults.get("num_envs"),
        episode_length=args.episode_length or debug_defaults.get("episode_length"),
        unroll_length=debug_defaults.get("unroll_length"),
        batch_size=args.batch_size or debug_defaults.get("batch_size"),
        num_minibatches=debug_defaults.get("num_minibatches"),
        num_updates_per_batch=debug_defaults.get("num_updates_per_batch"),
        learning_rate=args.learning_rate,
        no_domain_randomization=debug_defaults.get(
            "no_domain_randomization",
            False,
        ),
        save_checkpoints=not args.no_checkpoints,
        checkpoint_out=(
            str(args.checkpoint_out) if args.checkpoint_out is not None else None
        ),
        resume_from=str(args.resume_from) if args.resume_from is not None else None,
        debug_run=args.debug_run,
        bare=args.bare,
    )
    run_training(env_config, train_config, args.out)


def debug_run_defaults(enabled: bool) -> dict:
    """Vraca mali debug preset umesto gomile CLI opcija."""
    if not enabled:
        return {}
    return {
        "num_timesteps": 1000,
        "num_envs": 4,
        "num_evals": 0,
        "episode_length": 20,
        "unroll_length": 5,
        "batch_size": 4,
        "num_minibatches": 1,
        "num_updates_per_batch": 1,
        "no_domain_randomization": True,
    }


if __name__ == "__main__":
    main()
