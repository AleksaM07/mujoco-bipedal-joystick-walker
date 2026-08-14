import argparse
import contextlib
import functools
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# REF: PROJECT-XLA-PREALLOCATE-DEFAULT
# TYPE: ENGINEERING_DEFAULT
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_training
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loguru import logger
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as playground_wrapper

from biomechanics_env import BiomechanicsJoystickEnv, domain_randomize
from config import (
    DEFAULT_BVH_REFERENCE_LIST,
    DEFAULT_SMPL_REFERENCE_FILES,
    DOMAIN_RANDOMIZATION_ID,
    PROJECT_ROOT,
    RUNS_DIR,
    EnvConfig,
    TRAIN_DIAGNOSTIC_METRICS,
    TrainConfig,
    default_biomechanics_ppo_config,
    expand_reference_gait_files,
)
from phase1_backends import resolve_warp_capacities


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
        episode_length = metrics.get(
            "eval/avg_episode_length",
            metrics.get("eval/episode_length"),
        )
        if reward is None and episode_length is None:
            logger.info("train progress callback | step={}", step)
            return

        diagnostics = []
        for key, label in TRAIN_DIAGNOSTIC_METRICS:
            value = metrics.get(key)
            if value is not None:
                diagnostics.append(f"{label}={float(value):.3f}")

        if episode_length is not None and float(episode_length) > 1e-6:
            length = float(episode_length)
            per_step_metrics = (
                ("eval/episode_reward", "reward_step"),
                ("eval/episode_deepmimic_pose", "dm_pose_step"),
                ("eval/episode_deepmimic_velocity", "dm_vel_step"),
                ("eval/episode_deepmimic_root_pose", "dm_root_step"),
                ("eval/episode_deepmimic_root_velocity", "dm_root_vel_step"),
                ("eval/episode_deepmimic_key_position", "dm_key_step"),
                ("eval/episode_action_magnitude", "act_mag_step"),
                ("eval/episode_tracking_lin", "track_lin_step"),
                ("eval/episode_command_progress", "progress_step"),
                ("eval/episode_height", "height_step"),
                ("eval/episode_torso_up", "torso_step"),
            )
            for key, label in per_step_metrics:
                value = metrics.get(key)
                if value is not None:
                    diagnostics.append(f"{label}={float(value) / length:.3f}")

        logger.info(
            "eval | step={} | reward={} | episode_length={}{}",
            step,
            None if reward is None else float(reward),
            None if episode_length is None else float(episode_length),
            "" if not diagnostics else " | " + " | ".join(diagnostics),
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
    run_label: str,
    timesteps: int,
) -> Path:
    """Create a compact, readable run folder name."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # REF: PROJECT-RUN-NAMING-V1
    # TYPE: ENGINEERING_DEFAULT
    return (
        base_dir
        / (
            f"{stamp}_{run_label}_steps-{format_steps(timesteps)}"
            "_reward-pending_running"
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
    reward_value = best_reward if best_reward is not None else final_reward
    reward_label = (
        format_reward_for_path(reward_value)
        if reward_value is not None
        else "unknown"
    )

    if name.endswith("_running"):
        new_name = name.removesuffix("_running")
        new_name = new_name.replace("reward-pending", f"reward-{reward_label}")
        new_name = f"{new_name}_{status}"
    elif name.endswith("_s") or name.endswith("_f"):
        new_name = name[:-2]
        new_name = new_name.replace("reward-pending", f"reward-{reward_label}")
        new_name = f"{new_name}_{status}"
    else:
        new_name = f"{name}_reward-{reward_label}_{status}"

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


def resolve_resume_checkpoint_path(path: str | Path | None) -> Path | None:
    """Resolve checkpoint, checkpoints dir, or run dir to a concrete checkpoint."""
    if path is None:
        return None

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve(strict=False)

    if is_checkpoint_dir(candidate):
        return candidate

    checkpoint_root = candidate / "checkpoints"
    if checkpoint_root.is_dir():
        return latest_checkpoint_dir(checkpoint_root)

    if candidate.is_dir():
        return latest_checkpoint_dir(candidate)

    return candidate


def is_checkpoint_dir(path: Path) -> bool:
    """Brax/Orbax checkpoint dirs contain a network config metadata file."""
    return path.is_dir() and (path / "ppo_network_config.json").exists()


def latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    """Find the numerically latest checkpoint under a checkpoints directory."""
    checkpoint_dirs = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"Nema checkpoint foldera u {checkpoint_root}")
    return max(checkpoint_dirs, key=lambda path: int(path.name))


def infer_xml_path_from_resume(checkpoint_path: Path | None) -> str | None:
    """Infer the original XML path from a resumed run config or train log."""
    if checkpoint_path is None:
        return None

    run_dir = find_run_dir_for_checkpoint(checkpoint_path)
    if run_dir is None:
        return None

    config_path = run_dir / "config.json"
    if config_path.exists():
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        xml_path = run_config.get("env", {}).get("xml_path")
        if xml_path:
            return str(xml_path)
    manifest_path = run_dir / "xml_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        xml_path = manifest.get("xml_path")
        if xml_path:
            return str(xml_path)

    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None

    marker = " | xml="
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        return normalize_logged_xml_path(line.split(marker, 1)[1].strip())
    return None


def find_run_dir_for_checkpoint(checkpoint_path: Path) -> Path | None:
    """Find the parent run directory that owns a checkpoint."""
    for path in (checkpoint_path, *checkpoint_path.parents):
        if (path / "config.json").exists():
            return path
    return None


def normalize_logged_xml_path(raw_path: str) -> str:
    """Prefer repo-relative generated_models paths from old WSL logs."""
    normalized = raw_path.replace("\\", "/")
    marker = "generated_models/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        return normalized[marker_index:]
    return raw_path


def make_ppo_config(
    train_config: TrainConfig,
):
    """Pravi PPO config za biomehanicki human env."""
    rl_config = default_biomechanics_ppo_config()

    overrides = {
        "num_timesteps": train_config.num_timesteps,
        "num_evals": train_config.num_evals,
        "num_envs": train_config.num_envs,
        "num_eval_envs": train_config.num_eval_envs,
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
    if train_config.distribution_type is not None:
        rl_config.network_factory.distribution_type = train_config.distribution_type

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


def file_sha256(path: str | Path | None) -> str | None:
    """Return a SHA-256 hash for compatibility-critical files."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_xml_manifest(run_dir: Path, env: BiomechanicsJoystickEnv) -> None:
    """Persist the exact XML identity used by a run."""
    # REF: PROJECT-XML-HASH-GUARD
    # TYPE: ENGINEERING_DEFAULT
    manifest = {
        "xml_path": env.xml_path,
        "xml_sha256": file_sha256(env.xml_path),
        "action_size": env.action_size,
    }
    (run_dir / "xml_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def validate_resume_xml_guard(
    checkpoint_path: Path | None,
    env: BiomechanicsJoystickEnv,
) -> None:
    """Stop resume when checkpoint XML and runtime XML differ."""
    # REF: PROJECT-XML-HASH-GUARD
    # TYPE: ENGINEERING_DEFAULT
    if checkpoint_path is None:
        return
    run_dir = find_run_dir_for_checkpoint(checkpoint_path)
    if run_dir is None:
        return
    manifest_path = run_dir / "xml_manifest.json"
    if not manifest_path.exists():
        logger.warning(
            "resume XML guard missing | checkpoint run has no xml_manifest.json"
        )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_hash = manifest.get("xml_sha256")
    current_hash = file_sha256(env.xml_path)
    if saved_hash and current_hash and saved_hash != current_hash:
        raise ValueError(
            "Checkpoint/XML mismatch: saved checkpoint XML hash "
            f"{saved_hash[:12]} != runtime XML hash {current_hash[:12]}. "
            "Pass the original --xml-path or retrain for this XML."
        )


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
    rl_config = make_ppo_config(train_config)
    user_warp_naconmax = env_config.warp_naconmax
    user_warp_njmax = env_config.warp_njmax
    if env_config.physics_backend == "mjx_warp":
        env_config.playground_impl = "warp"
        env_config.warp_num_worlds = int(rl_config.num_envs)
        capacity_plan = resolve_warp_capacities(
            env_config.warp_num_worlds,
            user_warp_naconmax,
            user_warp_njmax,
        )
        env_config.warp_naconmax = capacity_plan.naconmax
        env_config.warp_njmax = capacity_plan.njmax
    eval_env_config = replace(env_config)
    if eval_env_config.physics_backend == "mjx_warp":
        eval_env_config.playground_impl = "warp"
        eval_worlds = int(rl_config.get("num_eval_envs", 1) or 1)
        eval_env_config.warp_num_worlds = eval_worlds
        eval_capacity_plan = resolve_warp_capacities(
            eval_env_config.warp_num_worlds,
            user_warp_naconmax,
            user_warp_njmax,
        )
        eval_env_config.warp_naconmax = eval_capacity_plan.naconmax
        eval_env_config.warp_njmax = eval_capacity_plan.njmax
    run_dir = make_run_dir(
        out_dir,
        run_source_name(env_config, train_config),
        rl_config.num_timesteps,
    )
    checkpoint_dir = (
        Path(train_config.checkpoint_out).expanduser()
        if train_config.checkpoint_out
        else run_dir / "checkpoints"
    )
    restore_checkpoint = resolve_resume_checkpoint_path(train_config.resume_from)
    restore_checkpoint_path = str(restore_checkpoint) if restore_checkpoint else None
    if restore_checkpoint is not None and env_config.xml_path is None:
        inferred_xml_path = infer_xml_path_from_resume(restore_checkpoint)
        if inferred_xml_path is not None:
            env_config.xml_path = inferred_xml_path

    run_dir.mkdir(parents=True, exist_ok=True)
    if train_config.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(run_dir / "train.log", level="INFO", encoding="utf-8", mode="w")
    save_run_config(run_dir, env_config, train_config, rl_config)
    reference_files = env_config.reference_gait_file
    reference_count = len(reference_files) if isinstance(reference_files, list) else 0
    logger.info(
        "reference files | gait={} | count={} | first={}",
        env_config.reference_gait,
        reference_count,
        reference_files[0] if reference_count else reference_files,
    )

    enable_erfi = (
        train_config.enable_erfi
        and not train_config.bare
        and not train_config.no_erfi
    )
    with logged_stage("make_environment"):
        env = make_environment(env_config, enable_erfi=enable_erfi)
        eval_env = make_environment(eval_env_config, enable_erfi=False)
    validate_resume_xml_guard(restore_checkpoint, env)
    write_xml_manifest(run_dir, env)
    log_environment_summary(env, label="train env")
    log_eval_environment_summary(eval_env)
    if getattr(env._config, "reference_gait", "none") in ("bvh", "smpl"):
        with logged_stage("reference reset diagnostics"):
            log_reference_reset_diagnostics(env.reset(jax.random.PRNGKey(train_config.seed)))
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
        if env_config.xml_path is not None:
            logger.info("resume xml locked | xml_path={}", env_config.xml_path)
        else:
            logger.warning(
                "resume xml was not inferred; pass --xml-path if this checkpoint "
                "was trained on an older generated XML"
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
    if rl_config.num_envs < 512:
        logger.warning(
            "biomechanics run uses only {} envs; GPU throughput is usually "
            "better with --num-envs 4096 on the MJX-Warp path",
            rl_config.num_envs,
        )

    train_kwargs = rl_config.to_dict()
    train_kwargs["network_factory"] = make_network_factory(rl_config)
    if rl_config.num_evals == 0:
        train_kwargs["run_evals"] = False
        train_kwargs["num_evals"] = 11
    train_kwargs_extra = {}
    enable_domain_randomization = (
        train_config.enable_domain_randomization
        and not train_config.bare
        and not train_config.no_domain_randomization
    )
    if enable_domain_randomization:
        randomization_rng = jax.random.split(
            jax.random.PRNGKey(train_config.seed + 10_000),
            rl_config.num_envs,
        )
        train_kwargs_extra["randomization_fn"] = functools.partial(
            domain_randomize,
            rng=randomization_rng,
        )
        logger.info(
            "domain randomization enabled | train resamples numeric MJX "
            "model on full episode reset | eval uses nominal model | bank_size={}",
            rl_config.num_envs,
        )
    if train_config.bare:
        logger.info("bare mode | ERFI disabled | domain randomization disabled")
    if not enable_erfi:
        logger.info("ERFI disabled for this run")
    if not enable_domain_randomization:
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
                wrap_env_fn=make_wrap_env_fn(eval_env),
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


def make_wrap_env_fn(eval_env):
    """Return Brax wrapper with nominal eval and reset-time DR for training."""

    def wrap_env(
        wrapped_env,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn=None,
        full_reset: bool = False,
    ):
        del full_reset
        if wrapped_env is eval_env:
            randomization_fn = None
        return wrap_biomechanics_training(
            wrapped_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
        )

    return wrap_env


class BiomechanicsVmapWrapper(playground_wrapper.Wrapper):
    """Vectorizes the env and keeps ERFI-50 split exact per parallel batch."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = jax.vmap(self.env.reset)(rng)
        if "warp_world_id" in state.info:
            state.info["warp_world_id"] = jnp.arange(rng.shape[0], dtype=jnp.int32)
        return _with_erfi50_split(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return jax.vmap(self.env.step)(state, action)


class PerEpisodeDomainRandomizationVmapWrapper(playground_wrapper.Wrapper):
    """Vectorizes envs with a prebuilt randomized MJX model bank.

    The expensive XML/MjModel construction stays outside the hot path. Each env
    reset samples one model id and subsequent steps gather that numeric MJX
    model from the bank.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]],
    ) -> None:
        super().__init__(env)
        self._mjx_model_bank, self._in_axes = randomization_fn(self.mjx_model)
        self._bank_size = self._infer_bank_size()

    def _infer_bank_size(self) -> int:
        sizes: list[int] = []

        def collect_size(value, axis) -> None:
            if axis == 0:
                sizes.append(int(value.shape[0]))

        jax.tree_util.tree_map(collect_size, self._mjx_model_bank, self._in_axes)
        if not sizes:
            raise ValueError("domain randomization did not create a model bank.")

        unique_sizes = set(sizes)
        if len(unique_sizes) != 1:
            raise ValueError(
                "domain randomization model bank has inconsistent leaf sizes: "
                f"{sorted(unique_sizes)}"
            )
        return unique_sizes.pop()

    def _select_model(self, model_id: jax.Array) -> mjx.Model:
        return jax.tree_util.tree_map(
            lambda value, axis: value[model_id] if axis == 0 else value,
            self._mjx_model_bank,
            self._in_axes,
        )

    @contextlib.contextmanager
    def _using_model(self, mjx_model: mjx.Model) -> Iterator[mjx_env.MjxEnv]:
        env = self.env.unwrapped
        old_mjx_model = env._mjx_model
        try:
            env._mjx_model = mjx_model
            yield env
        finally:
            env._mjx_model = old_mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        def reset_one(reset_rng):
            model_key, env_key = jax.random.split(reset_rng)
            model_id = jax.random.randint(
                model_key,
                shape=(),
                minval=0,
                maxval=self._bank_size,
            )
            mjx_model = self._select_model(model_id)
            with self._using_model(mjx_model) as env:
                state = env.reset(env_key)
            state.info[DOMAIN_RANDOMIZATION_ID] = model_id
            return state

        state = jax.vmap(reset_one)(rng)
        if "warp_world_id" in state.info:
            state.info["warp_world_id"] = jnp.arange(rng.shape[0], dtype=jnp.int32)
        return _with_erfi50_split(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def step_one(model_id, env_state, env_action):
            mjx_model = self._select_model(model_id.astype(jnp.int32))
            with self._using_model(mjx_model) as env:
                return env.step(env_state, env_action)

        return jax.vmap(step_one)(
            state.info[DOMAIN_RANDOMIZATION_ID],
            state,
            action,
        )


class ConditionalAutoResetWrapper(playground_wrapper.Wrapper):
    """Full-reset auto wrapper that only builds reset states when needed."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self._info_key = "AutoResetWrapper"

    def _key(self, name: str) -> str:
        return f"{self._info_key}_{name}"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)
        state.info[self._key("first_data")] = state.data
        state.info[self._key("first_obs")] = state.obs
        state.info[self._key("rng")] = rng
        state.info[self._key("done_count")] = jnp.zeros(
            key.shape[:-1],
            dtype=int,
        )
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(state.info[self._key("rng")])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]

        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jnp.zeros_like(state.done))
        stepped_state = self.env.step(state, action)

        reset_state = jax.lax.cond(
            jnp.any(stepped_state.done),
            lambda _: self.reset(reset_key),
            lambda _: state,
            operand=None,
        )

        def where_done(reset_value, step_value):
            done = stepped_state.done
            if done.shape and done.shape[0] != reset_value.shape[0]:
                return step_value
            if done.shape:
                done = jnp.reshape(
                    done,
                    [reset_value.shape[0]] + [1] * (len(reset_value.shape) - 1),
                )
            return jnp.where(done, reset_value, step_value)

        data = jax.tree.map(where_done, reset_state.data, stepped_state.data)
        obs = jax.tree.map(where_done, reset_state.obs, stepped_state.obs)
        next_info = jax.tree.map(
            where_done,
            reset_state.info,
            stepped_state.info,
        )

        done_count_key = self._key("done_count")
        next_info[done_count_key] = stepped_state.info[done_count_key]
        if "steps" in next_info:
            next_info["steps"] = stepped_state.info["steps"]

        preserve_info_key = self._key("preserve_info")
        if preserve_info_key in next_info:
            next_info[preserve_info_key] = stepped_state.info[preserve_info_key]

        next_info[done_count_key] += stepped_state.done.astype(int)
        next_info[self._key("rng")] = reset_rng
        truncation = stepped_state.info.get(
            "truncation",
            jnp.zeros_like(stepped_state.done),
        )
        # REF: PROJECT-TERMINATED-TRUNCATED-SPLIT
        # TYPE: ENGINEERING_DEFAULT
        metrics = dict(stepped_state.metrics)
        metrics["terminated"] = stepped_state.done * (1.0 - truncation)
        metrics["truncated"] = stepped_state.done * truncation

        return stepped_state.replace(
            data=data,
            obs=obs,
            info=next_info,
            metrics=metrics,
        )


def wrap_biomechanics_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]]
    | None = None,
) -> playground_wrapper.Wrapper:
    """Wrap biomechanics envs for PPO with true reset-time randomization."""
    if randomization_fn is None:
        env = BiomechanicsVmapWrapper(env)
    else:
        env = PerEpisodeDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return ConditionalAutoResetWrapper(env)


def _with_erfi50_split(state: mjx_env.State) -> mjx_env.State:
    """Force exactly half of batched envs into RFI and half into RAO."""
    if "use_rfi" not in state.info:
        return state
    batch_size = state.info["use_rfi"].shape[0]
    split = batch_size // 2
    state.info["use_rfi"] = jnp.arange(batch_size) < split
    return state


def log_environment_summary(env, label: str = "env") -> None:
    """Ispise dimenzije modela pre treninga."""
    model = env.mj_model
    logger.info(
        "{} summary | nq={} | nv={} | nu={} | nbody={} | ngeom={} | "
        "nsite={} | action_size={} | substeps={} | erfi_enabled={} | "
        "physics_backend={} | warp_naconmax={} | warp_njmax={} | "
        "command_profile={} | action_smoothing={} | rfi_limit={} | "
        "rao_limit={} | reference_target_observation={} | "
        "reference_action_mode={} | reference_action_center={} | "
        "reference_action_range={} | reference_action_range_scale={} | "
        "reference_replay_target_step={} | dm_root_vel_weight_scale={} | "
        "legacy_action_prior={} | "
        "init_qpos_file={} | xml={}",
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
        getattr(env._config, "physics_backend", None),
        getattr(env._config, "warp_naconmax", None),
        getattr(env._config, "warp_njmax", None),
        getattr(env._config, "command_profile", None),
        getattr(env._config, "action_smoothing", None),
        getattr(env._config, "rfi_torque_limit", None),
        getattr(env._config, "rao_torque_limit", None),
        getattr(env._config, "reference_target_observation", None),
        getattr(env._config, "reference_action_mode", None),
        getattr(env._config, "reference_action_center", None),
        getattr(env._config, "reference_action_range", None),
        getattr(env._config, "reference_action_range_scale", None),
        getattr(env._config, "reference_replay_target_step", None),
        getattr(env._config, "deepmimic_root_velocity_weight_scale", None),
        getattr(env._config, "legacy_action_prior", None),
        getattr(env._config, "init_qpos_file", None),
        getattr(env, "xml_path", None),
    )


def log_eval_environment_summary(env) -> None:
    """Ispise najbitniju razliku eval env-a."""
    logger.info(
        "eval env | erfi_enabled={} | physics_backend={} | warp_worlds={} | "
        "warp_naconmax={} | warp_njmax={} | rfi_limit={} | rao_limit={}",
        getattr(env._config, "enable_erfi", None),
        getattr(env._config, "physics_backend", None),
        getattr(env._config, "warp_num_worlds", None),
        getattr(env._config, "warp_naconmax", None),
        getattr(env._config, "warp_njmax", None),
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
        log_reference_reset_diagnostics(state)

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


def log_reference_reset_diagnostics(state: mjx_env.State) -> None:
    """Log reset-time imitation diagnostics so bad starts are obvious in train.log."""
    metrics = jax.device_get(state.metrics)
    info = jax.device_get(state.info)
    logger.info(
        "reference reset | clip_id={} | motion_time={:.3f} | fallback={} | "
        "init_motion={} | init_fallback={} | init_rejected={} | "
        "init_exact={} | init_exact_rejected={} | "
        "done_low={} | done_tipped={} | done_invalid={} | "
        "done_motion_over={} | done_pose={} | "
        "dm_pose={:.3f} | dm_vel={:.3f} | dm_root_pose={:.3f} | "
        "dm_root_vel={:.3f} | dm_key={:.3f} | "
        "pose_err={:.3f} | vel_err={:.3f} | root_xy_err={:.3f} | "
        "root_h_err={:.3f} | root_vel_err={:.3f} | root_angvel_err={:.3f} | "
        "key_err={:.3f} | max_key={:.3f}",
        int(info.get("bvh_reference_clip_id", 0)),
        float(info.get("bvh_reference_time_offset", 0.0)),
        bool(info.get("reference_fallback_standing", False)),
        float(metrics.get("init_motion_count", 0.0)),
        float(metrics.get("init_fallback_count", 0.0)),
        float(metrics.get("init_rejected_count", 0.0)),
        float(metrics.get("init_exact_count", 0.0)),
        float(metrics.get("init_exact_rejected_count", 0.0)),
        bool(metrics.get("done_low_height", 0.0)),
        bool(metrics.get("done_tipped", 0.0)),
        bool(metrics.get("done_invalid", 0.0)),
        bool(metrics.get("done_motion_over", 0.0)),
        bool(metrics.get("done_pose_termination", 0.0)),
        float(metrics.get("deepmimic_pose", 0.0)),
        float(metrics.get("deepmimic_velocity", 0.0)),
        float(metrics.get("deepmimic_root_pose", 0.0)),
        float(metrics.get("deepmimic_root_velocity", 0.0)),
        float(metrics.get("deepmimic_key_position", 0.0)),
        float(metrics.get("deepmimic_pose_error", 0.0)),
        float(metrics.get("deepmimic_velocity_error", 0.0)),
        float(metrics.get("deepmimic_root_xy_error", 0.0)),
        float(metrics.get("deepmimic_root_height_error", 0.0)),
        float(metrics.get("deepmimic_root_vel_error", 0.0)),
        float(metrics.get("deepmimic_root_angvel_error", 0.0)),
        float(metrics.get("deepmimic_key_pos_error", 0.0)),
        float(metrics.get("deepmimic_max_key_dist", 0.0)),
    )


def env_display_name(env_config: EnvConfig) -> str:
    """Vraca ime env-a za logove i run foldere."""
    return f"BiomechanicsHumanJoystick{env_config.env_version.title()}"


def run_source_name(env_config: EnvConfig, train_config: TrainConfig) -> str:
    """Build a compact run mode label for the folder name."""
    # REF: PROJECT-RUN-NAMING-V1
    # TYPE: ENGINEERING_DEFAULT
    if train_config.run_tag:
        return f"{sanitize_run_tag(train_config.run_tag)}_V1"

    return f"{sanitize_run_tag(env_config.command_profile)}_V1"


def sanitize_run_tag(run_tag: str) -> str:
    """Pretvori opisni tag u kratak filesystem-safe suffix."""
    normalized = []
    previous_was_separator = False
    for character in run_tag.strip().lower():
        if character.isalnum():
            normalized.append(character)
            previous_was_separator = False
        elif character in {" ", "-", "_", "."} and not previous_was_separator:
            normalized.append("_")
            previous_was_separator = True

    cleaned = "".join(normalized).strip("_")
    if not cleaned:
        raise ValueError("--run-tag mora imati bar jedan alfanumericki znak.")
    return cleaned[:40]


def make_environment(env_config: EnvConfig, enable_erfi: bool = False):
    """Napravi biomehanicki joystick env."""
    config_overrides = {
        "impl": env_config.playground_impl,
        "physics_backend": env_config.physics_backend,
        "warp_num_worlds": env_config.warp_num_worlds,
        "warp_naconmax": env_config.warp_naconmax,
        "warp_njmax": env_config.warp_njmax,
        "warp_graph_mode": env_config.warp_graph_mode,
        "enable_erfi": enable_erfi,
        "command_profile": env_config.command_profile,
        "reference_gait": env_config.reference_gait,
        "reference_target_observation": env_config.reference_target_observation,
        "reference_action_mode": env_config.reference_action_mode,
        "reference_action_center": env_config.reference_action_center,
        "reference_action_range": env_config.reference_action_range,
        "reference_action_range_scale": env_config.reference_action_range_scale,
        "bvh_target_observation_steps": env_config.bvh_target_observation_steps,
        "reference_replay_target_step": env_config.reference_replay_target_step,
        "deepmimic_root_velocity_weight_scale": (
            env_config.deepmimic_root_velocity_weight_scale
        ),
        "deepmimic_reward_mode": env_config.deepmimic_reward_mode,
        "deepmimic_key_bodies": env_config.deepmimic_key_bodies,
        "pose_termination": env_config.pose_termination,
        "pose_termination_dist": env_config.pose_termination_dist,
        "reset_sample_attempts": env_config.reset_sample_attempts,
        "reset_projection_levels": env_config.reset_projection_levels,
        "action_smoothing": env_config.action_smoothing,
        "legacy_action_prior": env_config.legacy_action_prior,
    }
    if env_config.xml_path is not None:
        config_overrides["xml_path"] = env_config.xml_path
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


def run_reference_playback_audit(
    env_config: EnvConfig,
    resets: int,
    steps: int,
    seed: int,
    physics_backend: str = "mjx_jax",
    clip_mode: str = "all",
    trace_resets: int = 0,
    trace_steps: int = 0,
    out_dir: Path | None = None,
) -> Path:
    """Play zero residual actions against the BVH reference without PPO.

    Optimized path: JIT reset/step and one packed device sync per episode via
    ``lax.while_loop``. Writes a detailed log under ``runs/``.
    """
    patch_jax_for_brax_compatibility()
    env_config.physics_backend = physics_backend
    env_config.playground_impl = "warp" if physics_backend == "mjx_warp" else "jax"
    env_config.reference_target_observation = False
    env_config.reference_action_mode = "residual"
    env_config.reference_action_center = "default"
    env_config.reference_action_range = "action_scale"
    env_config.reference_action_range_scale = 1.0
    env_config.bvh_target_observation_steps = (0,)
    env_config.reset_sample_attempts = min(int(env_config.reset_sample_attempts), 2)
    env_config.reset_projection_levels = tuple(
        level for level in env_config.reset_projection_levels if level >= 0.45
    ) or (1.0,)
    if env_config.physics_backend == "mjx_warp":
        env_config.warp_num_worlds = 1
        capacity_plan = resolve_warp_capacities(
            env_config.warp_num_worlds,
            env_config.warp_naconmax,
            env_config.warp_njmax,
        )
        env_config.warp_naconmax = capacity_plan.naconmax
        env_config.warp_njmax = capacity_plan.njmax

    base_dir = Path(out_dir) if out_dir is not None else RUNS_DIR
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{stamp}_reference_playback_{physics_backend}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "playback.log"
    summary_path = run_dir / "playback_summary.json"
    trace_path = run_dir / "playback_trace.jsonl"

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(log_path, level="DEBUG", encoding="utf-8", mode="w")
    logger.info("reference playback audit start | run_dir={}", run_dir)
    logger.info(
        "playback config | resets={} | steps={} | seed={} | backend={} | "
        "requested_clip_mode={} | reference_gait={} | reference_gait_file={} | "
        "audit_target_obs={} | audit_action_mode={} | audit_reset_attempts={} | "
        "audit_projection_levels={} | trace_resets={} | trace_steps={}",
        resets,
        steps,
        seed,
        physics_backend,
        clip_mode,
        env_config.reference_gait,
        env_config.reference_gait_file,
        env_config.reference_target_observation,
        env_config.reference_action_mode,
        env_config.reset_sample_attempts,
        env_config.reset_projection_levels,
        trace_resets,
        trace_steps,
    )

    with logged_stage("reference_playback/make_environment"):
        env = make_environment(env_config, enable_erfi=False)
    actual_clip_mode = _apply_reference_playback_clip_mode(env, clip_mode)
    log_environment_summary(env, label="playback env")
    _log_reference_library_summary(env)

    total_resets = max(int(resets), 1)
    max_steps = int(max(int(steps), 1))
    metric_keys = (
        "reward",
        "reference_gait",
        "deepmimic_pose",
        "deepmimic_velocity",
        "deepmimic_total_no_root_velocity",
        "deepmimic_root_pose",
        "deepmimic_root_velocity",
        "deepmimic_key_position",
        "deepmimic_pose_error",
        "deepmimic_velocity_error",
        "deepmimic_root_xy_error",
        "deepmimic_root_height_error",
        "deepmimic_root_vel_error",
        "deepmimic_root_angvel_error",
        "deepmimic_key_pos_error",
        "deepmimic_max_key_dist",
        "height",
        "com_height",
        "root_vertical_velocity",
        "pelvis_vertical_velocity",
        "left_foot_height",
        "right_foot_height",
        "foot_slip",
        "action_magnitude",
        "torso_up",
    )
    done_keys = (
        "done_low_height",
        "done_tipped",
        "done_invalid",
        "done_motion_over",
        "done_pose_termination",
    )

    zero_action = jnp.zeros(env.action_size, dtype=jnp.float32)
    max_steps_jax = jnp.asarray(max_steps, dtype=jnp.int32)

    trace_metric_keys = (
        "reward",
        "height",
        "com_height",
        "torso_up",
        "head_up",
        "root_vertical_velocity",
        "pelvis_vertical_velocity",
        "left_foot_height",
        "right_foot_height",
        "left_foot_contact",
        "right_foot_contact",
        "foot_slip",
        "action_magnitude",
        "reference_gait",
        "deepmimic_pose",
        "deepmimic_velocity",
        "deepmimic_total_no_root_velocity",
        "deepmimic_root_pose",
        "deepmimic_root_velocity",
        "deepmimic_key_position",
        "deepmimic_root_pose_raw",
        "deepmimic_root_velocity_raw",
        "deepmimic_key_position_raw",
        "deepmimic_pose_error",
        "deepmimic_velocity_error",
        "deepmimic_root_xy_error",
        "deepmimic_root_height_error",
        "deepmimic_root_vel_error",
        "deepmimic_root_angvel_error",
        "deepmimic_key_pos_error",
        "deepmimic_max_key_dist",
        "reference_motion_time",
        "reference_root_height",
        "reference_root_vertical_velocity",
        "reference_left_foot_height",
        "reference_right_foot_height",
        "root_height_tracking_error",
        "root_vertical_velocity_tracking_error",
        "left_foot_height_tracking_error",
        "right_foot_height_tracking_error",
        "done_low_height",
        "done_tipped",
        "done_invalid",
        "done_motion_over",
        "done_pose_termination",
        "done",
    )

    trace_limit = min(max(int(trace_steps), 0), max_steps)

    def _trace_episode_device(rng: jax.Array) -> dict[str, jax.Array]:
        state = env.reset(rng)

        def body(carry, step_index):
            current_state, stopped = carry
            metric_values = jnp.stack(
                [current_state.metrics[key] for key in trace_metric_keys]
            ).astype(jnp.float32)
            clip_id = current_state.info["bvh_reference_clip_id"].astype(jnp.int32)
            fallback = current_state.info["reference_fallback_standing"]
            done = current_state.done > 0.5
            valid = ~stopped
            should_step = valid & (~done) & (step_index < trace_limit)
            next_state = jax.lax.cond(
                should_step,
                lambda _: env.step(current_state, zero_action),
                lambda _: current_state,
                operand=None,
            )
            return (
                next_state,
                stopped | done | (step_index >= trace_limit),
            ), {
                "valid": valid,
                "clip_id": clip_id,
                "fallback": fallback,
                "metrics": metric_values,
            }

        _, rows = jax.lax.scan(
            body,
            (state, jnp.array(False)),
            jnp.arange(trace_limit + 1, dtype=jnp.int32),
        )
        return rows

    trace_episode_fn = (
        jax.jit(_trace_episode_device) if trace_limit > 0 and trace_resets > 0 else None
    )

    def _trace_episode(reset_index: int) -> None:
        if trace_episode_fn is None:
            return
        rng = jax.random.PRNGKey(seed + reset_index)
        host = jax.tree_util.tree_map(
            lambda value: jax.device_get(value),
            trace_episode_fn(rng),
        )
        valid_mask = np.asarray(host["valid"], dtype=bool)
        metric_values = np.asarray(host["metrics"], dtype=np.float64)
        clip_ids = np.asarray(host["clip_id"], dtype=np.int32)
        fallbacks = np.asarray(host["fallback"], dtype=bool)
        rows = []
        for step_index in np.where(valid_mask)[0]:
            clip_id = int(clip_ids[step_index])
            metrics = {
                key: float(metric_values[step_index, metric_index])
                for metric_index, key in enumerate(trace_metric_keys)
            }
            rows.append(
                {
                    "reset_index": reset_index,
                    "step": int(step_index),
                    "clip_id": clip_id,
                    "loop_mode": _loop_mode_name(env, clip_id),
                    "support_foot": _clip_support_foot(env, clip_id),
                    "fallback": bool(fallbacks[step_index]),
                    **metrics,
                }
            )

        with trace_path.open("a", encoding="utf-8") as trace_file:
            for row in rows:
                trace_file.write(json.dumps(row, sort_keys=True) + "\n")
        last = rows[-1]
        logger.info(
            "trace episode | reset={} | rows={} | final_step={} | clip={} | "
            "mode={} | done={} | low={} | tipped={} | motion_over={} | "
            "height={:.3f} | reward={:.3f} | trace={}",
            reset_index,
            len(rows),
            last["step"],
            last["clip_id"],
            last["loop_mode"],
            bool(last.get("done", 0.0) > 0.5),
            bool(last.get("done_low_height", 0.0) > 0.5),
            bool(last.get("done_tipped", 0.0) > 0.5),
            bool(last.get("done_motion_over", 0.0) > 0.5),
            last.get("height", 0.0),
            last.get("reward", 0.0),
            trace_path,
        )

    def _episode(rng: jax.Array):
        state = env.reset(rng)
        init_stats = jnp.stack(
            [
                state.metrics["init_motion_count"],
                state.metrics["init_fallback_count"],
                state.metrics["init_rejected_count"],
                state.metrics["init_rejected_low_count"],
                state.metrics["init_rejected_tipped_count"],
                state.metrics["init_rejected_invalid_count"],
            ]
        ).astype(jnp.float32)
        metric_sum = jnp.zeros(len(metric_keys), dtype=jnp.float32)

        def cond(carry):
            current_state, step_index, _, _ = carry
            return (step_index < max_steps_jax) & (current_state.done < 0.5)

        def body(carry):
            current_state, step_index, current_sum, _ = carry
            next_state = env.step(current_state, zero_action)
            step_vals = jnp.stack(
                [next_state.metrics[key] for key in metric_keys]
            ).astype(jnp.float32)
            return (
                next_state,
                step_index + jnp.asarray(1, dtype=jnp.int32),
                current_sum + step_vals,
                next_state.done,
            )

        state, steps_taken, metric_sum, _ = jax.lax.while_loop(
            cond,
            body,
            (state, jnp.asarray(0, dtype=jnp.int32), metric_sum, state.done),
        )
        done_flags = jnp.stack(
            [state.metrics[key] for key in done_keys]
        ).astype(jnp.float32)
        return {
            "steps_taken": steps_taken,
            "done": state.done.astype(jnp.float32),
            "metric_sum": metric_sum,
            "done_flags": done_flags,
            "init_stats": init_stats,
            "clip_id": state.info["bvh_reference_clip_id"].astype(jnp.int32),
            "height": state.metrics["height"].astype(jnp.float32),
            "reward": state.metrics["reward"].astype(jnp.float32),
            "deepmimic_pose": state.metrics["deepmimic_pose"].astype(jnp.float32),
            "deepmimic_total_no_root_velocity": state.metrics[
                "deepmimic_total_no_root_velocity"
            ].astype(jnp.float32),
            "deepmimic_key_position": state.metrics[
                "deepmimic_key_position"
            ].astype(jnp.float32),
            "deepmimic_pose_error": state.metrics[
                "deepmimic_pose_error"
            ].astype(jnp.float32),
            "deepmimic_velocity_error": state.metrics[
                "deepmimic_velocity_error"
            ].astype(jnp.float32),
            "deepmimic_root_xy_error": state.metrics[
                "deepmimic_root_xy_error"
            ].astype(jnp.float32),
            "deepmimic_root_height_error": state.metrics[
                "deepmimic_root_height_error"
            ].astype(jnp.float32),
            "deepmimic_root_vel_error": state.metrics[
                "deepmimic_root_vel_error"
            ].astype(jnp.float32),
            "deepmimic_root_angvel_error": state.metrics[
                "deepmimic_root_angvel_error"
            ].astype(jnp.float32),
            "deepmimic_key_pos_error": state.metrics[
                "deepmimic_key_pos_error"
            ].astype(jnp.float32),
            "deepmimic_max_key_dist": state.metrics[
                "deepmimic_max_key_dist"
            ].astype(jnp.float32),
        }

    with logged_stage("reference_playback/jit_compile"):
        episode_fn = jax.jit(_episode)
        warm_rng = jax.random.PRNGKey(seed)
        warm = episode_fn(warm_rng)
        jax.block_until_ready(warm["steps_taken"])
        logger.info(
            "jit warmup done | steps_taken={} | done={} | clip_id={}",
            int(jax.device_get(warm["steps_taken"])),
            float(jax.device_get(warm["done"])),
            int(jax.device_get(warm["clip_id"])),
        )

    valid = 0
    failed = 0
    low = 0
    tipped = 0
    invalid = 0
    motion_over = 0
    natural_end = 0
    physical_fail = 0
    unknown_fail = 0
    init_motion = 0.0
    init_fallback = 0.0
    init_rejected = 0.0
    init_rejected_low = 0.0
    init_rejected_tipped = 0.0
    init_rejected_invalid = 0.0
    first_failure_steps: list[int] = []
    metric_sums = {key: 0.0 for key in metric_keys}
    metric_steps = 0
    episode_rows: list[dict] = []
    mode_counts = {"wrap": 0, "clamp": 0}
    clip_stats: dict[int, dict] = {}
    audit_start = time.perf_counter()

    for reset_index in range(total_resets):
        episode_start = time.perf_counter()
        rng = jax.random.PRNGKey(seed + reset_index)
        # Skip re-running seed episode already used for warmup compile.
        result = episode_fn(rng)
        jax.block_until_ready(result["steps_taken"])
        host = jax.tree_util.tree_map(lambda value: jax.device_get(value), result)
        steps_taken = int(host["steps_taken"])
        done = float(host["done"]) > 0.5
        init_stats = np.asarray(host["init_stats"], dtype=np.float64)
        init_motion += float(init_stats[0])
        init_fallback += float(init_stats[1])
        init_rejected += float(init_stats[2])
        init_rejected_low += float(init_stats[3])
        init_rejected_tipped += float(init_stats[4])
        init_rejected_invalid += float(init_stats[5])
        metric_sum = np.asarray(host["metric_sum"], dtype=np.float64)
        for key, value in zip(metric_keys, metric_sum, strict=True):
            metric_sums[key] += float(value)
        metric_steps += max(steps_taken, 0)
        done_flags = np.asarray(host["done_flags"], dtype=np.float64)
        episode_seconds = time.perf_counter() - episode_start
        clip_id = int(host["clip_id"])
        loop_mode = _loop_mode_name(env, clip_id)
        mode_counts[loop_mode] = mode_counts.get(loop_mode, 0) + 1
        if reset_index < max(int(trace_resets), 0) and int(trace_steps) > 0:
            _trace_episode(reset_index)
        is_motion_over = bool(done_flags[3] > 0.5)
        is_pose_termination = bool(done_flags[4] > 0.5)
        is_physical_fail = bool(
            (done_flags[0] > 0.5)
            or (done_flags[1] > 0.5)
            or (done_flags[2] > 0.5)
            or is_pose_termination
        )
        is_natural_end = bool(done and is_motion_over and not is_physical_fail)
        is_unknown_fail = bool(done and not is_motion_over and not is_physical_fail)

        row = {
            "reset_index": reset_index,
            "clip_id": clip_id,
            "loop_mode": loop_mode,
            "steps_taken": steps_taken,
            "done": done,
            "natural_end": is_natural_end,
            "physical_fail": is_physical_fail,
            "unknown_fail": is_unknown_fail,
            "init_motion": float(init_stats[0]),
            "init_fallback": float(init_stats[1]),
            "init_rejected": float(init_stats[2]),
            "init_rejected_low": float(init_stats[3]),
            "init_rejected_tipped": float(init_stats[4]),
            "init_rejected_invalid": float(init_stats[5]),
            "done_low_height": float(done_flags[0]),
            "done_tipped": float(done_flags[1]),
            "done_invalid": float(done_flags[2]),
            "done_motion_over": float(done_flags[3]),
            "done_pose_termination": float(done_flags[4]),
            "final_height": float(host["height"]),
            "final_reward": float(host["reward"]),
            "final_pose": float(host["deepmimic_pose"]),
            "final_key": float(host["deepmimic_key_position"]),
            "final_pose_error": float(host["deepmimic_pose_error"]),
            "final_velocity_error": float(host["deepmimic_velocity_error"]),
            "final_root_xy_error": float(host["deepmimic_root_xy_error"]),
            "final_root_height_error": float(host["deepmimic_root_height_error"]),
            "final_root_vel_error": float(host["deepmimic_root_vel_error"]),
            "final_root_angvel_error": float(host["deepmimic_root_angvel_error"]),
            "final_key_pos_error": float(host["deepmimic_key_pos_error"]),
            "final_max_key_dist": float(host["deepmimic_max_key_dist"]),
            "seconds": episode_seconds,
        }
        episode_rows.append(row)
        clip_stat = clip_stats.setdefault(
            clip_id,
            {
                "clip_id": clip_id,
                "loop_mode": loop_mode,
                "source_path": _clip_source_path(env, clip_id),
                "source_start_frame": _clip_source_start_frame(env, clip_id),
                "source_end_frame": _clip_source_end_frame(env, clip_id),
                "support_foot": _clip_support_foot(env, clip_id),
                "frame_count": _clip_frame_count(env, clip_id),
                "motion_length_seconds": _clip_motion_length(env, clip_id),
                "episodes": 0,
                "valid": 0,
                "natural_end": 0,
                "physical_fail": 0,
                "unknown_fail": 0,
                "low": 0,
                "tipped": 0,
                "invalid": 0,
                "motion_over": 0,
                "pose_termination": 0,
                "steps_total": 0,
                "reward_total": 0.0,
                "pose_total": 0.0,
                "key_total": 0.0,
                "pose_error_total": 0.0,
                "velocity_error_total": 0.0,
                "root_xy_error_total": 0.0,
                "root_height_error_total": 0.0,
                "root_vel_error_total": 0.0,
                "root_angvel_error_total": 0.0,
                "key_pos_error_total": 0.0,
                "max_key_dist_total": 0.0,
            },
        )
        clip_stat["episodes"] += 1
        clip_stat["steps_total"] += steps_taken
        clip_stat["reward_total"] += row["final_reward"]
        clip_stat["pose_total"] += row["final_pose"]
        clip_stat["key_total"] += row["final_key"]
        clip_stat["pose_error_total"] += row["final_pose_error"]
        clip_stat["velocity_error_total"] += row["final_velocity_error"]
        clip_stat["root_xy_error_total"] += row["final_root_xy_error"]
        clip_stat["root_height_error_total"] += row["final_root_height_error"]
        clip_stat["root_vel_error_total"] += row["final_root_vel_error"]
        clip_stat["root_angvel_error_total"] += row["final_root_angvel_error"]
        clip_stat["key_pos_error_total"] += row["final_key_pos_error"]
        clip_stat["max_key_dist_total"] += row["final_max_key_dist"]

        if done:
            failed += 1
            first_failure_steps.append(steps_taken)
            low += int(done_flags[0] > 0.5)
            tipped += int(done_flags[1] > 0.5)
            invalid += int(done_flags[2] > 0.5)
            motion_over += int(done_flags[3] > 0.5)
            natural_end += int(is_natural_end)
            physical_fail += int(is_physical_fail)
            unknown_fail += int(is_unknown_fail)
            clip_stat["natural_end"] += int(is_natural_end)
            clip_stat["physical_fail"] += int(is_physical_fail)
            clip_stat["unknown_fail"] += int(is_unknown_fail)
            clip_stat["low"] += int(done_flags[0] > 0.5)
            clip_stat["tipped"] += int(done_flags[1] > 0.5)
            clip_stat["invalid"] += int(done_flags[2] > 0.5)
            clip_stat["motion_over"] += int(done_flags[3] > 0.5)
            clip_stat["pose_termination"] += int(done_flags[4] > 0.5)
            logger.info(
                "episode fail | reset={} | clip={} | mode={} | steps={} | "
                "natural_end={} physical_fail={} unknown_fail={} | "
                "low={} tipped={} invalid={} motion_over={} pose_term={} | "
                "height={:.3f} pose={:.3f} key={:.3f} | "
                "root_xy_err={:.3f} root_h_err={:.3f} root_vel_err={:.3f} "
                "key_err={:.3f} max_key={:.3f} | {:.2f}s",
                reset_index,
                row["clip_id"],
                loop_mode,
                steps_taken,
                is_natural_end,
                is_physical_fail,
                is_unknown_fail,
                row["done_low_height"] > 0.5,
                row["done_tipped"] > 0.5,
                row["done_invalid"] > 0.5,
                row["done_motion_over"] > 0.5,
                row["done_pose_termination"] > 0.5,
                row["final_height"],
                row["final_pose"],
                row["final_key"],
                row["final_root_xy_error"],
                row["final_root_height_error"],
                row["final_root_vel_error"],
                row["final_key_pos_error"],
                row["final_max_key_dist"],
                episode_seconds,
            )
        else:
            valid += 1
            clip_stat["valid"] += 1
            logger.info(
                "episode ok | reset={} | clip={} | mode={} | steps={} | "
                "height={:.3f} pose={:.3f} key={:.3f} | "
                "root_xy_err={:.3f} root_h_err={:.3f} root_vel_err={:.3f} "
                "key_err={:.3f} max_key={:.3f} | {:.2f}s",
                reset_index,
                row["clip_id"],
                loop_mode,
                steps_taken,
                row["final_height"],
                row["final_pose"],
                row["final_key"],
                row["final_root_xy_error"],
                row["final_root_height_error"],
                row["final_root_vel_error"],
                row["final_key_pos_error"],
                row["final_max_key_dist"],
                episode_seconds,
            )

    avg_failure_step = (
        sum(first_failure_steps) / len(first_failure_steps)
        if first_failure_steps
        else None
    )
    metric_means = {
        name: (value / metric_steps if metric_steps else None)
        for name, value in metric_sums.items()
    }
    clip_summaries = []
    for clip_id in sorted(clip_stats):
        clip_stat = clip_stats[clip_id]
        episodes = max(int(clip_stat["episodes"]), 1)
        clip_summaries.append(
            {
                **clip_stat,
                "avg_steps": clip_stat["steps_total"] / episodes,
                "avg_reward": clip_stat["reward_total"] / episodes,
                "avg_pose": clip_stat["pose_total"] / episodes,
                "avg_key": clip_stat["key_total"] / episodes,
                "avg_pose_error": clip_stat["pose_error_total"] / episodes,
                "avg_velocity_error": clip_stat["velocity_error_total"] / episodes,
                "avg_root_xy_error": clip_stat["root_xy_error_total"] / episodes,
                "avg_root_height_error": clip_stat["root_height_error_total"]
                / episodes,
                "avg_root_vel_error": clip_stat["root_vel_error_total"] / episodes,
                "avg_root_angvel_error": clip_stat["root_angvel_error_total"]
                / episodes,
                "avg_key_pos_error": clip_stat["key_pos_error_total"] / episodes,
                "avg_max_key_dist": clip_stat["max_key_dist_total"] / episodes,
            }
        )
    total_seconds = time.perf_counter() - audit_start
    summary = {
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "physics_backend": env_config.physics_backend,
        "requested_clip_mode": clip_mode,
        "clip_mode": actual_clip_mode,
        "reference_gait": env_config.reference_gait,
        "reference_gait_file": env_config.reference_gait_file,
        "resets": total_resets,
        "max_steps": max_steps,
        "trace_path": str(trace_path) if trace_resets > 0 and trace_steps > 0 else None,
        "trace_resets": int(trace_resets),
        "trace_steps": int(trace_steps),
        "seed": seed,
        "valid": valid,
        "failed": failed,
        "low": low,
        "tipped": tipped,
        "invalid": invalid,
        "motion_over": motion_over,
        "natural_end": natural_end,
        "physical_fail": physical_fail,
        "unknown_fail": unknown_fail,
        "mode_counts": mode_counts,
        "init_motion": init_motion,
        "init_fallback": init_fallback,
        "avg_init_rejected": init_rejected / total_resets,
        "avg_init_rejected_low": init_rejected_low / total_resets,
        "avg_init_rejected_tipped": init_rejected_tipped / total_resets,
        "avg_init_rejected_invalid": init_rejected_invalid / total_resets,
        "avg_failure_step": avg_failure_step,
        "metric_means": metric_means,
        "total_seconds": total_seconds,
        "clip_summaries": clip_summaries,
        "episodes": episode_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_line = (
        "reference_playback_audit | "
        f"physics_backend={env_config.physics_backend} | "
        f"clip_mode={actual_clip_mode} | "
        f"valid={valid} | failed={failed} | resets={total_resets} | "
        f"steps={max_steps} | low={low} | tipped={tipped} | invalid={invalid} | "
        f"motion_over={motion_over} | natural_end={natural_end} | "
        f"physical_fail={physical_fail} | unknown_fail={unknown_fail} | "
        f"wrap_sampled={mode_counts.get('wrap', 0)} | "
        f"clamp_sampled={mode_counts.get('clamp', 0)} | "
        f"init_motion={init_motion:.0f} | "
        f"init_fallback={init_fallback:.0f} | "
        f"avg_init_rejected={init_rejected / total_resets:.2f} | "
        f"avg_init_rej_low={init_rejected_low / total_resets:.2f} | "
        f"avg_init_rej_tipped={init_rejected_tipped / total_resets:.2f} | "
        f"avg_init_rej_invalid={init_rejected_invalid / total_resets:.2f} | "
        f"avg_reward={metric_means['reward']} | "
        f"avg_reference_gait={metric_means['reference_gait']} | "
        f"avg_pose={metric_means['deepmimic_pose']} | "
        f"avg_vel={metric_means['deepmimic_velocity']} | "
        f"avg_no_root_vel={metric_means['deepmimic_total_no_root_velocity']} | "
        f"avg_root_pose={metric_means['deepmimic_root_pose']} | "
        f"avg_root_vel={metric_means['deepmimic_root_velocity']} | "
        f"avg_key_pos={metric_means['deepmimic_key_position']} | "
        f"avg_pose_err={metric_means['deepmimic_pose_error']} | "
        f"avg_vel_err={metric_means['deepmimic_velocity_error']} | "
        f"avg_root_xy_err={metric_means['deepmimic_root_xy_error']} | "
        f"avg_root_h_err={metric_means['deepmimic_root_height_error']} | "
        f"avg_root_vel_err={metric_means['deepmimic_root_vel_error']} | "
        f"avg_root_angvel_err={metric_means['deepmimic_root_angvel_error']} | "
        f"avg_key_err={metric_means['deepmimic_key_pos_error']} | "
        f"avg_max_key={metric_means['deepmimic_max_key_dist']} | "
        f"avg_failure_step={avg_failure_step} | "
        f"total_seconds={total_seconds:.2f} | "
        f"log={log_path}"
    )
    logger.info(summary_line)
    logger.info("playback summary json | {}", summary_path)
    print(summary_line)
    close_file_logger()
    return run_dir


def _apply_reference_playback_clip_mode(env, clip_mode: str) -> str:
    """Restrict reset sampling to all clips, wrap-only clips, or clamp-only clips."""
    weights = np.asarray(getattr(env, "_bvh_reference_weights", np.array([1.0])))
    loop_modes = np.asarray(getattr(env, "_bvh_reference_loop_modes", np.zeros_like(weights)))
    if clip_mode == "all":
        return "all"
    if clip_mode == "wrap":
        mask = loop_modes == 1
    elif clip_mode == "clamp":
        mask = loop_modes == 0
    else:
        raise ValueError(f"Unsupported reference playback clip mode: {clip_mode}")
    filtered = np.where(mask, weights, 0.0).astype(np.float32)
    total = float(filtered.sum())
    if total <= 0.0:
        wrap_count = int(np.sum(loop_modes == 1))
        clamp_count = int(np.sum(loop_modes == 0))
        logger.warning(
            "playback clip filter fallback | requested_clip_mode={} | wrap_clips={} "
            "| clamp_clips={} | using=all",
            clip_mode,
            wrap_count,
            clamp_count,
        )
        return "all"
    env._bvh_reference_weights = jnp.asarray(filtered / total, dtype=jnp.float32)
    logger.info(
        "playback clip filter | clip_mode={} | selected_clips={} | total_clips={}",
        clip_mode,
        int(mask.sum()),
        int(mask.shape[0]),
    )
    return clip_mode


def _loop_mode_name(env, clip_id: int) -> str:
    """Return a human-readable loop mode name for one reference clip."""
    loop_modes = np.asarray(getattr(env, "_bvh_reference_loop_modes", np.array([0])))
    return "wrap" if int(loop_modes[clip_id]) == 1 else "clamp"


def _clip_source_path(env, clip_id: int) -> str | None:
    paths = getattr(env, "_bvh_reference_source_paths", ())
    return paths[clip_id] if clip_id < len(paths) else None


def _clip_source_start_frame(env, clip_id: int) -> int | None:
    starts = np.asarray(getattr(env, "_bvh_reference_source_start_frames", np.array([])))
    return int(starts[clip_id]) if clip_id < starts.shape[0] else None


def _clip_source_end_frame(env, clip_id: int) -> int | None:
    ends = np.asarray(getattr(env, "_bvh_reference_source_end_frames", np.array([])))
    return int(ends[clip_id]) if clip_id < ends.shape[0] else None


def _clip_support_foot(env, clip_id: int) -> str | None:
    support_feet = getattr(env, "_bvh_reference_support_feet", ())
    return support_feet[clip_id] if clip_id < len(support_feet) else None


def _clip_frame_count(env, clip_id: int) -> int | None:
    frame_counts = np.asarray(getattr(env, "_bvh_reference_frame_counts", np.array([])))
    return int(frame_counts[clip_id]) if clip_id < frame_counts.shape[0] else None


def _clip_motion_length(env, clip_id: int) -> float | None:
    motion_lengths = np.asarray(
        getattr(env, "_bvh_reference_motion_lengths", np.array([])),
        dtype=np.float32,
    )
    return float(motion_lengths[clip_id]) if clip_id < motion_lengths.shape[0] else None


def _log_reference_library_summary(env) -> None:
    """Log BVH clip library stats that dominate playback startup cost."""
    clip_count = int(getattr(env, "_bvh_reference_clip_count", 0))
    frame_counts = getattr(env, "_bvh_reference_frame_counts", None)
    motion_lengths = getattr(env, "_bvh_reference_motion_lengths", None)
    loop_modes = getattr(env, "_bvh_reference_loop_modes", None)
    support_feet = tuple(getattr(env, "_bvh_reference_support_feet", ()))
    diagnostic_summaries = tuple(
        summary
        for summary in getattr(env, "_bvh_reference_diagnostic_summaries", ())
        if summary
    )
    if frame_counts is None:
        logger.info("reference library | clip_count={}", clip_count)
        return
    counts = np.asarray(frame_counts)
    lengths = (
        np.asarray(motion_lengths)
        if motion_lengths is not None
        else np.zeros_like(counts, dtype=np.float32)
    )
    modes = (
        np.asarray(loop_modes)
        if loop_modes is not None
        else np.zeros_like(counts, dtype=np.int32)
    )
    logger.info(
        "reference library | clips={} | frames_total={} | "
        "frames_min={} | frames_max={} | frames_mean={:.1f} | "
        "duration_sum_s={:.2f} | wrap_clips={} | clamp_clips={}",
        clip_count,
        int(counts.sum()),
        int(counts.min()) if counts.size else 0,
        int(counts.max()) if counts.size else 0,
        float(counts.mean()) if counts.size else 0.0,
        float(lengths.sum()) if lengths.size else 0.0,
        int(np.sum(modes == 1)),
        int(np.sum(modes == 0)),
    )
    if support_feet:
        left_support = sum(foot == "left_foot" for foot in support_feet)
        right_support = sum(foot == "right_foot" for foot in support_feet)
        unknown_support = sum(not foot for foot in support_feet)
        logger.info(
            "reference support feet | left={} | right={} | unknown={}",
            left_support,
            right_support,
            unknown_support,
        )
    if diagnostic_summaries:
        logger.info(
            "reference retarget diagnostics | clips_with_clipping={} | sample={}",
            len(diagnostic_summaries),
            diagnostic_summaries[:3],
        )


def main() -> None:
    configure_stdout_encoding()
    parser = argparse.ArgumentParser(
        description="MuJoCo Playground/Brax PPO trening."
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
        default="warp",
    )
    parser.add_argument(
        "--command-profile",
        choices=[
            "forward_slow",
            "forward",
            "walk",
            "steer",
            "standard_easy",
            "standard",
        ],
        default="standard",
        help=(
            "standard je pun joystick; standard_easy je meksi svi-pravci "
            "curriculum; steer je forward + skretanje; forward_slow/forward "
            "su bootstrap curriculum; walk je forward sa gait-clock setupom."
        ),
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=1.0,
        help="Policy-to-PD blend; 1.0 is MimicKit-style unsmoothed PD targets.",
    )
    parser.add_argument(
        "--reference-gait",
        choices=["none", "sine", "bvh", "smpl"],
        default="bvh",
        help=(
            "BVH/MimicKit-style imitation je default; smpl cita CMU/AMASS "
            "SMPL+H-G npz reference; none/sine su compatibility/debug modovi."
        ),
    )
    parser.add_argument(
        "--reference-action-mode",
        choices=["mimickit", "residual"],
        default=EnvConfig.reference_action_mode,
        help=(
            "mimickit: policy action is an absolute PD target in DOF space; "
            "residual: old reference_ctrl + action_scale * action playback mode."
        ),
    )
    parser.add_argument(
        "--reference-action-center",
        choices=["default", "joint_midpoint"],
        default=EnvConfig.reference_action_center,
        help=(
            "Action-space zero point for MimicKit mode. default keeps zero "
            "action at the XML standing pose; joint_midpoint reproduces raw "
            "MimicKit bounds centering."
        ),
    )
    parser.add_argument(
        "--reference-action-range",
        choices=["action_scale", "joint_limits"],
        default=EnvConfig.reference_action_range,
        help=(
            "PD target half-range for MimicKit mode. action_scale uses the "
            "locally tuned actuator scales; joint_limits uses raw 1.4x "
            "joint-limit bounds."
        ),
    )
    parser.add_argument(
        "--reference-action-range-scale",
        type=float,
        default=EnvConfig.reference_action_range_scale,
        help="Multiplier for --reference-action-range.",
    )
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        action="append",
        default=None,
        help=(
            "Reference fajl za --reference-gait bvh ili smpl. Moze se navesti "
            "vise puta; env bira jedan reference clip po epizodi."
        ),
    )
    parser.add_argument(
        "--reference-gait-list",
        type=Path,
        action="append",
        default=None,
        help=(
            "Text fajl sa jednim reference path-om po liniji (BVH ili SMPL npz). "
            "Moze se navesti vise puta."
        ),
    )
    parser.add_argument(
        "--deepmimic-reward-mode",
        choices=["pure", "mixed"],
        default="pure",
        help=(
            "pure koristi MimicKit/DeepMimic imitation-only reward; mixed "
            "vraca joystick task shaping preko imitacije."
        ),
    )
    parser.add_argument(
        "--deepmimic-key-bodies",
        default="metatarsal_midpoint_right,metatarsal_midpoint_left",
        help=(
            "Comma-separated key bodies for key-position imitation. Default "
            "je feet-only dok BVH retarget ne kontrolise ruke/glavu."
        ),
    )
    parser.add_argument(
        "--pose-termination",
        dest="pose_termination",
        action="store_true",
        help=(
            "Enable MimicKit-style pose termination on configured key bodies."
        ),
    )
    parser.add_argument(
        "--no-pose-termination",
        dest="pose_termination",
        action="store_false",
        help="Disable pose termination explicitly.",
    )
    parser.set_defaults(pose_termination=None)
    parser.add_argument(
        "--pose-termination-dist",
        type=float,
        default=None,
        help="Distance threshold for pose termination when enabled.",
    )
    parser.add_argument(
        "--init-qpos-file",
        type=Path,
        default=None,
        help=(
            "Opcioni MJDATA/QPOS fajl za pocetnu pozu, npr. "
            "assets/poses/MJDATA_neutral_poze.TXT."
        ),
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=None,
        help=(
            "Opcioni konkretan generated XML za trening/resume. Korisno kada "
            "nastavljas checkpoint treniran na starijem XML version-u."
        ),
    )
    parser.add_argument(
        "--legacy-action-prior",
        action="store_true",
        help=(
            "Compatibility mode za V10/slow checkpoint-eve: koristi stari "
            "leg action scale i gasi novi variable posture prior."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=None)
    parser.add_argument("--num-evals", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--unroll-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--updates-per-batch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--distribution-type",
        choices=["normal", "tanh_normal"],
        default=None,
        help=(
            "Override Brax policy action distribution. Default comes from "
            "config.py and is tanh_normal for bounded MimicKit-style actions."
        ),
    )
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
        "--erfi",
        action="store_true",
        help="Enable ERFI/RFI torque perturbations. Disabled by default.",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Enable model size/mass/friction randomization. Disabled by default.",
    )
    parser.add_argument(
        "--no-erfi",
        action="store_true",
        help="Deprecated compatibility flag; ERFI is already disabled by default.",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help=(
            "Deprecated compatibility flag; domain randomization is already "
            "disabled by default."
        ),
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
            "Putanja do kompatibilnog Brax/Orbax checkpointa, checkpoints "
            "foldera ili run foldera. Ako je folder, koristi najnoviji "
            "numericki checkpoint. Mora biti isti env/action/obs/network setup."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Kratak suffix za runs folder, npr. stylev1, trajectory, "
            "animation_bvh ili mocap_cmu."
        ),
    )
    parser.add_argument(
        "--fast-physics",
        action="store_true",
        help="Use sim_dt=0.01 instead of the default accurate 0.005 setup.",
    )
    parser.add_argument(
        "--reference-playback-audit",
        action="store_true",
        help=(
            "Do not train; JIT zero-action BVH reference playback audit. "
            "Writes playback.log + playback_summary.json under --out."
        ),
    )
    parser.add_argument(
        "--reference-playback-resets",
        type=int,
        default=8,
        help="Number of resets for --reference-playback-audit (smoke: 2).",
    )
    parser.add_argument(
        "--reference-playback-steps",
        type=int,
        default=500,
        help="Max steps per reset for --reference-playback-audit (smoke: 50).",
    )
    parser.add_argument(
        "--reference-playback-backend",
        choices=["mjx_jax", "mjx_warp"],
        default="mjx_jax",
        help=(
            "Physics backend for --reference-playback-audit. Default is "
            "mjx_jax so the diagnostic avoids Warp allocator/OOM noise; "
            "training still defaults to MJX-Warp."
        ),
    )
    parser.add_argument(
        "--reference-playback-clip-mode",
        choices=["all", "wrap", "clamp"],
        default="all",
        help=(
            "Audit all clips, only loopable wrap clips, or only short clamp "
            "clips. Useful for separating sustainable playback from clip "
            "library coverage issues."
        ),
    )
    parser.add_argument(
        "--reference-playback-trace-resets",
        type=int,
        default=0,
        help=(
            "For playback audit, also write per-step JSONL traces for this "
            "many reset seeds."
        ),
    )
    parser.add_argument(
        "--reference-playback-trace-steps",
        type=int,
        default=0,
        help=(
            "Max per-step rows per traced playback episode. Use with "
            "--reference-playback-trace-resets."
        ),
    )
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    device = choose_device(args.device, args.allow_cpu)
    jax.config.update("jax_default_device", device)

    reference_gait_file = expand_reference_gait_files(
        args.reference_gait_file,
        args.reference_gait_list,
    )
    if args.reference_gait == "bvh" and reference_gait_file is None:
        reference_gait_file = expand_reference_gait_files(
            reference_gait_lists=[DEFAULT_BVH_REFERENCE_LIST],
        )
    if args.reference_gait == "smpl" and reference_gait_file is None:
        reference_gait_file = expand_reference_gait_files(
            reference_gait_files=[
                path.relative_to(PROJECT_ROOT)
                for path in DEFAULT_SMPL_REFERENCE_FILES
            ],
        )

    env_config = EnvConfig(
        env_version=args.env_version,
        playground_impl=args.playground_impl,
        command_profile=args.command_profile,
        reference_gait=args.reference_gait,
        reference_gait_file=reference_gait_file,
        reference_action_mode=args.reference_action_mode,
        reference_action_center=args.reference_action_center,
        reference_action_range=args.reference_action_range,
        reference_action_range_scale=args.reference_action_range_scale,
        reference_target_observation=(
            args.reference_gait in ("bvh", "smpl")
            and EnvConfig.reference_target_observation
        ),
        deepmimic_reward_mode=args.deepmimic_reward_mode,
        deepmimic_key_bodies=tuple(
            body.strip()
            for body in args.deepmimic_key_bodies.split(",")
            if body.strip()
        ),
        pose_termination=(
            args.pose_termination
            if args.pose_termination is not None
            else EnvConfig.pose_termination
        ),
        pose_termination_dist=(
            args.pose_termination_dist
            if args.pose_termination_dist is not None
            else EnvConfig.pose_termination_dist
        ),
        xml_path=str(args.xml_path) if args.xml_path is not None else None,
        legacy_action_prior=args.legacy_action_prior,
        action_smoothing=args.action_smoothing,
        init_qpos_file=(
            str(args.init_qpos_file) if args.init_qpos_file is not None else None
        ),
        accurate_physics=not args.fast_physics,
    )
    if args.reference_playback_audit:
        run_dir = run_reference_playback_audit(
            env_config,
            resets=args.reference_playback_resets,
            steps=args.reference_playback_steps,
            seed=args.seed,
            physics_backend=args.reference_playback_backend,
            clip_mode=args.reference_playback_clip_mode,
            trace_resets=args.reference_playback_trace_resets,
            trace_steps=args.reference_playback_trace_steps,
            out_dir=args.out,
        )
        print(f"reference playback logs: {run_dir}")
        return

    debug_defaults = debug_run_defaults(args.debug_run)
    train_config = TrainConfig(
        seed=args.seed,
        num_timesteps=args.timesteps or debug_defaults.get("num_timesteps"),
        num_evals=args.num_evals
        if args.num_evals is not None
        else debug_defaults.get("num_evals"),
        num_envs=args.num_envs or debug_defaults.get("num_envs"),
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length or debug_defaults.get("episode_length"),
        unroll_length=(
            args.unroll_length
            if args.unroll_length is not None
            else debug_defaults.get("unroll_length")
        ),
        batch_size=args.batch_size or debug_defaults.get("batch_size"),
        num_minibatches=(
            args.num_minibatches
            if args.num_minibatches is not None
            else debug_defaults.get("num_minibatches")
        ),
        num_updates_per_batch=(
            args.updates_per_batch
            if args.updates_per_batch is not None
            else debug_defaults.get("num_updates_per_batch")
        ),
        learning_rate=args.learning_rate,
        distribution_type=args.distribution_type,
        enable_erfi=args.erfi,
        enable_domain_randomization=(
            args.domain_randomization
            and not debug_defaults.get("no_domain_randomization", False)
        ),
        no_erfi=args.no_erfi,
        no_domain_randomization=(
            args.no_domain_randomization
            or debug_defaults.get("no_domain_randomization", False)
        ),
        save_checkpoints=not args.no_checkpoints,
        checkpoint_out=(
            str(args.checkpoint_out) if args.checkpoint_out is not None else None
        ),
        resume_from=str(args.resume_from) if args.resume_from is not None else None,
        run_tag=args.run_tag,
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
