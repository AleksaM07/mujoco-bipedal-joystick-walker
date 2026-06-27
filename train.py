import argparse
import contextlib
import functools
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

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
    DOMAIN_RANDOMIZATION_ID,
    PROJECT_ROOT,
    RUNS_DIR,
    EnvConfig,
    TRAIN_DIAGNOSTIC_METRICS,
    TrainConfig,
    default_biomechanics_ppo_config,
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
    return (
        base_dir
        / (
            f"{stamp}_{run_label}_{format_steps(timesteps)}_running"
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
    if rl_config.num_envs < 512:
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
    if not train_config.bare and not train_config.no_domain_randomization:
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
    if train_config.no_erfi:
        logger.info("ERFI disabled for this run")
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

        return stepped_state.replace(data=data, obs=obs, info=next_info)


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
        "command_profile={} | action_smoothing={} | rfi_limit={} | "
        "rao_limit={} | reference_target_observation={} | "
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
    return f"BiomechanicsHumanJoystick{env_config.env_version.title()}"


def run_source_name(env_config: EnvConfig, train_config: TrainConfig) -> str:
    """Build a compact run mode label for the folder name."""
    if train_config.run_tag:
        return f"bio_{sanitize_run_tag(train_config.run_tag)}"

    run_mode = "bio"
    if env_config.env_version != "standard":
        run_mode = f"{run_mode}_{sanitize_run_tag(env_config.env_version)}"
    if train_config.bare:
        run_mode = f"{run_mode}_bare"
    elif train_config.no_erfi:
        run_mode = f"{run_mode}_noerfi"
    if train_config.no_domain_randomization and not train_config.bare:
        run_mode = f"{run_mode}_nodr"
    if env_config.command_profile != "standard":
        run_mode = f"{run_mode}_{env_config.command_profile}"
    if env_config.reference_gait != "none":
        run_mode = f"{run_mode}_{sanitize_run_tag(env_config.reference_gait)}"
    if env_config.init_qpos_file:
        run_mode = f"{run_mode}_init"
    if not env_config.accurate_physics:
        run_mode = f"{run_mode}_fast"
    return run_mode


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
    """Napravi biomehanicki joystick env."""
    config_overrides = {
        "impl": env_config.playground_impl,
        "enable_erfi": enable_erfi,
        "command_profile": env_config.command_profile,
        "reference_gait": env_config.reference_gait,
        "reference_target_observation": env_config.reference_target_observation,
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
        "--fast-physics",
        action="store_true",
        help="Use sim_dt=0.01 instead of the default accurate 0.005 setup.",
    )
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    device = choose_device(args.device, args.allow_cpu)
    jax.config.update("jax_default_device", device)

    env_config = EnvConfig(
        env_version=args.env_version,
        playground_impl=args.playground_impl,
        command_profile=args.command_profile,
        reference_gait=args.reference_gait,
        reference_gait_file=expand_reference_gait_files(
            args.reference_gait_file,
            args.reference_gait_list,
        ),
        reference_target_observation=args.reference_gait == "bvh",
        xml_path=str(args.xml_path) if args.xml_path is not None else None,
        legacy_action_prior=args.legacy_action_prior,
        action_smoothing=args.action_smoothing,
        init_qpos_file=(
            str(args.init_qpos_file) if args.init_qpos_file is not None else None
        ),
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
