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
from config import (
    PROJECT_ROOT,
    RUNS_DIR,
    EnvConfig,
    TrainConfig,
    expand_reference_gait_files,
)


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

    PER_STEP_DIAGNOSTICS = (
        ("eval/episode_reward_raw", "raw_step"),
        ("eval/episode_reward_nonterminal", "rew_floor"),
        ("eval/episode_reward_clip_low", "clip_lo"),
        ("eval/episode_forward_vel", "fwd_avg"),
        ("eval/episode_locomotion_quality", "loco_avg"),
        ("eval/episode_swing_contact", "swing_r"),
        ("eval/episode_stance_contact", "stance_r"),
        ("eval/episode_double_contact", "double_r"),
        ("eval/episode_bvh_locomotion_gate", "gate_avg"),
        ("eval/episode_bvh_bootstrap_reward", "boot_avg"),
        ("eval/episode_bvh_regularization_raw", "reg_raw"),
    )

    DIAGNOSTIC_METRICS = (
        ("eval/episode_reward_raw", "raw"),
        ("eval/episode_reward_clip_low", "clip_lo"),
        ("eval/episode_reward_clip_high", "clip_hi"),
        ("eval/episode_reward_done_override", "fall_rew"),
        ("eval/episode_reward_nonterminal", "rew_nonterm"),
        ("eval/episode_tracking_lin_vel", "tracking"),
        ("eval/episode_forward_vel", "fwd"),
        ("eval/episode_command_progress", "progress"),
        ("eval/episode_command_norm", "cmd_norm"),
        ("eval/episode_torso_up", "torso_up"),
        ("eval/episode_head_up", "head_up"),
        ("eval/episode_height", "height"),
        ("eval/episode_foot_slip", "foot_slip"),
        ("eval/episode_swing_drag", "swing_drag"),
        ("eval/episode_swing_clearance", "swing_clear"),
        ("eval/episode_swing_clearance_deficit", "clear_deficit"),
        ("eval/episode_locomotion_quality", "loco_q"),
        ("eval/episode_gated_tracking", "gated_track"),
        ("eval/episode_gated_progress", "gated_prog"),
        ("eval/episode_swing_contact", "swing_ct"),
        ("eval/episode_stance_contact", "stance_ct"),
        ("eval/episode_double_contact", "double_ct"),
        ("eval/episode_double_support_drag", "dbl_drag"),
        ("eval/episode_task_reward", "task_rew"),
        ("eval/episode_zombie_sine_reward", "zombie_rew"),
        ("eval/episode_bvh_mimic_reward", "mimic_rew"),
        ("eval/episode_bvh_mimic_core", "bvh_core"),
        ("eval/episode_bvh_stability_reward", "bvh_stab"),
        ("eval/episode_bvh_joystick_reward", "bvh_task"),
        ("eval/episode_bvh_regularization_cost", "bvh_reg"),
        ("eval/episode_bvh_regularization_raw", "bvh_reg_raw"),
        ("eval/episode_bvh_locomotion_gate", "bvh_gate"),
        ("eval/episode_bvh_bootstrap_reward", "bvh_boot"),
        ("eval/episode_gait_cost_scale", "gait_scale"),
        ("eval/episode_variable_posture", "var_pose"),
        ("eval/episode_gait_reward", "gait"),
        ("eval/episode_reference_gait", "ref_gait"),
        ("eval/episode_reference_velocity", "ref_vel"),
        ("eval/episode_reference_foot", "ref_foot"),
        ("eval/episode_reference_root", "ref_root"),
        ("eval/episode_contact_force", "contact_force"),
        ("eval/episode_joint_limit", "joint_limit"),
        ("eval/episode_illegal_contact", "illegal_ct"),
        ("eval/episode_foot_slip_cost", "foot_slip_c"),
        ("eval/episode_swing_drag_cost", "swing_drag_c"),
        ("eval/episode_swing_contact_cost", "swing_ct_c"),
        ("eval/episode_clearance_deficit_cost", "clear_def_c"),
        ("eval/episode_double_contact_cost", "double_ct_c"),
        ("eval/episode_double_support_drag_cost", "dbl_drag_c"),
        ("eval/episode_overspeed_cost", "overspeed_c"),
        ("eval/episode_height_cost", "height_c"),
        ("eval/episode_done_low_height", "done_low"),
        ("eval/episode_done_tipped", "done_tip"),
        ("eval/episode_done_illegal_contact", "done_illegal"),
        ("eval/episode_done_invalid", "done_nan"),
        ("eval/episode_done", "done"),
    )

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
        for key, label in self.DIAGNOSTIC_METRICS:
            value = metrics.get(key)
            if value is not None:
                diagnostics.append(f"{label}={float(value):.3f}")
        if episode_length is not None and float(episode_length) > 1e-6:
            for key, label in self.PER_STEP_DIAGNOSTICS:
                value = metrics.get(key)
                if value is not None:
                    diagnostics.append(
                        f"{label}={float(value) / float(episode_length):.3f}"
                    )

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

    return rl_config


def validate_ppo_batch_config(rl_config) -> None:
    """Fail fast for Brax PPO batch/env divisibility constraints."""
    batch_size = int(rl_config.batch_size)
    num_minibatches = int(rl_config.num_minibatches)
    num_envs = int(rl_config.num_envs)
    batch_env_slots = batch_size * num_minibatches
    if batch_env_slots % num_envs == 0:
        return

    raise ValueError(
        "Invalid PPO config: batch_size * num_minibatches must be divisible "
        f"by num_envs. Got {batch_size} * {num_minibatches} = "
        f"{batch_env_slots}, which is not divisible by {num_envs}. "
        "Try --num-envs 768 --batch-size 384, or use the smaller safe "
        "fallback --num-envs 512 --batch-size 256."
    )


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
    validate_ppo_batch_config(rl_config)
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

    enable_erfi = not train_config.bare and not train_config.no_erfi
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
    if train_config.no_erfi:
        logger.info("ERFI disabled for this run")
    if train_config.no_domain_randomization:
        logger.info("domain randomization disabled for this run")

    if train_config.debug_run or train_config.diagnostic_rollout:
        with logged_stage("debug_preflight"):
            debug_preflight(
                env,
                train_config.seed,
                rollout_steps=train_config.diagnostic_rollout_steps,
                include_eager_step=train_config.debug_run,
                include_rollouts=train_config.diagnostic_rollout,
            )

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
        "rao_limit={} | reference_target_observation={} | "
        "reference_phase_randomization={} | reference_state_init={} | "
        "legacy_action_prior={} | init_qpos_file={} | xml={}",
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
        getattr(env._config, "reference_target_observation", None),
        getattr(env._config, "reference_phase_randomization", None),
        getattr(env._config, "reference_state_init", None),
        getattr(env._config, "legacy_action_prior", None),
        getattr(env._config, "init_qpos_file", None),
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


def debug_preflight(
    env,
    seed: int,
    rollout_steps: int = 20,
    include_eager_step: bool = False,
    include_rollouts: bool = False,
) -> None:
    """Proveri reset/step/JIT pre ulaska u Brax PPO."""
    rng = jax.random.PRNGKey(seed)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    with logged_stage("preflight reset"):
        state = jit_reset(rng)
        jax.block_until_ready(state.reward)
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
        log_state_snapshot("preflight reset metrics", state)

    action = jnp.zeros(env.action_size)
    if include_eager_step:
        with logged_stage("preflight eager step"):
            next_state = env.step(state, action)
            logger.info(
                "preflight eager step ok | reward={} | done={}",
                float(next_state.reward),
                float(next_state.done),
            )
            log_state_snapshot("preflight eager step metrics", next_state)

    with logged_stage("preflight jit step compile"):
        compiled_state = jit_step(state, action)
        jax.block_until_ready(compiled_state.reward)
        logger.info(
            "preflight jit step ok | reward={} | done={}",
            float(compiled_state.reward),
            float(compiled_state.done),
        )
        log_state_snapshot("preflight jit step metrics", compiled_state)

    if not include_rollouts:
        return

    with logged_stage("preflight zero-action rollout"):
        rollout_state = jit_reset(jax.random.PRNGKey(seed + 101))
        jax.block_until_ready(rollout_state.reward)
        zero_action = jnp.zeros(env.action_size)
        summarize_debug_rollout(
            "zero",
            rollout_state,
            lambda step_index, key: zero_action,
            jit_step,
            rollout_steps,
            jax.random.PRNGKey(seed + 102),
        )

    with logged_stage("preflight small-random rollout"):
        rollout_state = jit_reset(jax.random.PRNGKey(seed + 201))
        jax.block_until_ready(rollout_state.reward)
        summarize_debug_rollout(
            "small_random",
            rollout_state,
            lambda step_index, key: jax.random.uniform(
                key,
                (env.action_size,),
                minval=-0.2,
                maxval=0.2,
            ),
            jit_step,
            rollout_steps,
            jax.random.PRNGKey(seed + 202),
        )


DEBUG_ROLLOUT_METRICS = (
    "reward",
    "reward_raw",
    "reward_clip_low",
    "reward_clip_high",
    "reward_done_override",
    "reward_nonterminal",
    "bvh_mimic_reward",
    "bvh_mimic_core",
    "bvh_stability_reward",
    "bvh_joystick_reward",
    "bvh_regularization_cost",
    "bvh_regularization_raw",
    "bvh_locomotion_gate",
    "bvh_bootstrap_reward",
    "reference_gait",
    "reference_velocity",
    "reference_foot",
    "reference_root",
    "locomotion_quality",
    "forward_vel",
    "command_norm",
    "command_progress",
    "stance_contact",
    "swing_contact",
    "double_contact",
    "gait_cost_scale",
    "illegal_contact",
    "joint_limit",
    "foot_slip_cost",
    "swing_drag_cost",
    "swing_contact_cost",
    "clearance_deficit_cost",
    "double_contact_cost",
    "double_support_drag_cost",
    "overspeed_cost",
    "height_cost",
    "height",
    "torso_up",
    "done_low_height",
    "done_tipped",
    "done_illegal_contact",
    "done_invalid",
    "done",
)


def summarize_debug_rollout(
    label: str,
    state,
    action_fn,
    step_fn,
    rollout_steps: int,
    rng,
) -> None:
    """Run a tiny local rollout and log mean/final diagnostics before PPO."""
    values = {key: [] for key in DEBUG_ROLLOUT_METRICS}
    first_done_step = None
    final_state = state
    for step_index in range(rollout_steps):
        rng, action_key = jax.random.split(rng)
        action = action_fn(step_index, action_key)
        final_state = step_fn(final_state, action)
        jax.block_until_ready(final_state.reward)
        for key in DEBUG_ROLLOUT_METRICS:
            values[key].append(metric_float(final_state, key))
        if values["done"][-1] >= 0.5 and first_done_step is None:
            first_done_step = step_index + 1
            break

    summary = []
    for key in DEBUG_ROLLOUT_METRICS:
        metric_values = values[key]
        if not metric_values:
            continue
        mean_value = float(np.mean(metric_values))
        final_value = metric_values[-1]
        summary.append(f"{key}_mean={mean_value:.3f}")
        summary.append(f"{key}_final={final_value:.3f}")

    logger.info(
        "diagnostic rollout | label={} | steps={} | first_done_step={} | {}",
        label,
        len(values["reward"]),
        first_done_step,
        " | ".join(summary),
    )
    log_state_snapshot(f"diagnostic rollout {label} final metrics", final_state)


def log_state_snapshot(label: str, state) -> None:
    """Log one state snapshot with the same labels as rollout summaries."""
    metrics = [
        f"{key}={metric_float(state, key):.3f}"
        for key in DEBUG_ROLLOUT_METRICS
        if key in state.metrics
    ]
    info = []
    for key in (
        "command",
        "gait_step",
        "bvh_reference_clip_id",
        "bvh_reference_frame_offset",
    ):
        if key in state.info:
            info.append(f"{key}={metric_value_to_python(state.info[key])}")
    logger.info(
        "{} | {}{}",
        label,
        " | ".join(metrics),
        "" if not info else " | " + " | ".join(info),
    )


def metric_float(state, key: str) -> float:
    """Convert a scalar JAX metric to a Python float for loguru."""
    if key not in state.metrics:
        return 0.0
    value = metric_value_to_python(state.metrics[key])
    if isinstance(value, list):
        return float(np.mean(value)) if value else 0.0
    return float(value)


def metric_value_to_python(value):
    """Convert JAX/NumPy scalars and small arrays into readable Python values."""
    array = np.asarray(jax.device_get(value))
    if array.shape == ():
        return float(array)
    return array.tolist()


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
    elif train_config.no_erfi:
        env_source = f"{env_source}_noerfi"
    if train_config.no_domain_randomization and not train_config.bare:
        env_source = f"{env_source}_nodr"
    if env_config.command_profile != "standard":
        env_source = f"{env_source}_{env_config.command_profile}"
    if env_config.reference_gait != "none":
        env_source = f"{env_source}_ref_{sanitize_run_tag(env_config.reference_gait)}"
    if env_config.init_qpos_file:
        env_source = f"{env_source}_initqpos"
    if train_config.run_tag:
        env_source = f"{env_source}_{sanitize_run_tag(train_config.run_tag)}"
    if env_config.accurate_physics:
        env_source = f"{env_source}_accurate"
    return env_source


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
            "reference_gait": env_config.reference_gait,
            "reference_target_observation": env_config.reference_target_observation,
            "reference_phase_randomization": (
                env_config.reference_phase_randomization
            ),
            "reference_state_init": env_config.reference_state_init,
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
        else:
            config_overrides["sim_dt"] = 0.01
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
        default=0.5,
        help="Filtriranje policy akcije pre servo targeta; 0.5 prati walking repo.",
    )
    parser.add_argument(
        "--reference-gait",
        choices=["none", "sine", "bvh"],
        default="none",
        help=(
            "Opcioni pose-imitation prior: sine je rucna cyclic putanja, "
            "bvh koristi jednu ili vise BVH animacija."
        ),
    )
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        action="append",
        default=None,
        help=(
            "BVH fajl za --reference-gait bvh. Moze se navesti vise puta; "
            "env bira jedan reference clip po epizodi."
        ),
    )
    parser.add_argument(
        "--reference-gait-list",
        type=Path,
        action="append",
        default=None,
        help=(
            "Text fajl sa jednim BVH path-om po liniji. Moze se navesti "
            "vise puta za tier1+tier2 curriculum run."
        ),
    )
    parser.add_argument(
        "--init-qpos-file",
        type=Path,
        default=None,
        help=(
            "Opcioni MJDATA/QPOS fajl za pocetnu pozu, npr. "
            "MJDATA_neutral_poze.TXT."
        ),
    )
    parser.add_argument(
        "--reference-phase-randomization",
        action="store_true",
        help=(
            "Randomizuje BVH phase/frame offset po epizodi. Ovo prati "
            "DeepMimic/DRLoco RSI ideju, ali ne menja reset pozu samo po sebi."
        ),
    )
    parser.add_argument(
        "--reference-state-init",
        action="store_true",
        help=(
            "Resetuje humanoida u retargetovanu BVH pozu i qvel na random "
            "reference frame-u. Ovo je najblize DRLoco/LocoMuJoCo RSI setup-u."
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--debug-run",
        action="store_true",
        help="Mali izolacioni run bez domain randomization.",
    )
    parser.add_argument(
        "--diagnostic-rollout",
        action="store_true",
        help=(
            "Pre PPO treninga pusti kratke zero/small-random rollout-e i "
            "uloguj raw reward, clipping, done razloge i BVH reward breakdown."
        ),
    )
    parser.add_argument(
        "--diagnostic-rollout-steps",
        type=int,
        default=20,
        help="Broj koraka po kratkom diagnostic rollout-u.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Baseline: bez ERFI i bez domain randomization.",
    )
    parser.add_argument(
        "--no-erfi",
        action="store_true",
        help="Iskljuci random force injection, ali ostavi domain randomization.",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="Iskljuci model size/mass/friction randomization, ali ostavi ERFI.",
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
        reference_gait=args.reference_gait,
        reference_gait_file=expand_reference_gait_files(
            args.reference_gait_file,
            args.reference_gait_list,
        ),
        reference_target_observation=args.reference_gait == "bvh",
        reference_phase_randomization=args.reference_phase_randomization,
        reference_state_init=args.reference_state_init,
        xml_path=str(args.xml_path) if args.xml_path is not None else None,
        legacy_action_prior=args.legacy_action_prior,
        action_smoothing=args.action_smoothing,
        init_qpos_file=(
            str(args.init_qpos_file) if args.init_qpos_file is not None else None
        ),
        accurate_physics=args.accurate_physics or not args.fast_physics,
    )
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
        unroll_length=debug_defaults.get("unroll_length"),
        batch_size=args.batch_size or debug_defaults.get("batch_size"),
        num_minibatches=debug_defaults.get("num_minibatches"),
        num_updates_per_batch=debug_defaults.get("num_updates_per_batch"),
        learning_rate=args.learning_rate,
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
        diagnostic_rollout=args.diagnostic_rollout,
        diagnostic_rollout_steps=args.diagnostic_rollout_steps,
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
