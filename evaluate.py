import argparse
import functools
import hashlib
import json
import os
import re
import time
from pathlib import Path

# REF: PROJECT-XLA-PREALLOCATE-DEFAULT
# TYPE: ENGINEERING_DEFAULT
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

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

from biomechanics_env import BiomechanicsJoystickEnv
from config import (
    COMMAND_OBS_END,
    COMMAND_OBS_START,
    DEBUG_PRINT_INTERVAL,
    DEFAULT_WALK_COMMAND_X,
    GENERATED_MODEL_DIR,
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
from phase1_backends import resolve_warp_capacities


PERFECT_WALK_STEPS = 500


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


def run_env_value(run_config: dict | None, name: str, default=None):
    """Read one value from a saved run config."""
    if run_config is None:
        return default
    return run_config.get("env", {}).get(name, default)


def infer_command_profile(checkpoint_path: Path, run_config: dict | None) -> str:
    """Procitaj command_profile iz run configa, uz fallback na obs size."""
    profile = run_env_value(run_config, "command_profile")
    if profile in {
        "forward_slow",
        "forward",
        "walk",
        "steer",
        "standard_easy",
        "standard",
    }:
        return profile

    obs_size = read_checkpoint_observation_size(checkpoint_path)
    if obs_size == 92:
        return "walk"
    return "forward"


def infer_reference_gait(run_config: dict | None) -> str:
    """Procitaj reference_gait iz run configa."""
    reference_gait = run_env_value(run_config, "reference_gait", "none")
    if reference_gait in {"none", "sine", "bvh"}:
        return reference_gait
    return "none"


def find_run_config(checkpoint_path: Path) -> dict | None:
    """Nadji config.json u roditeljskom run direktorijumu checkpointa."""
    for path in (checkpoint_path, *checkpoint_path.parents):
        config_path = path / "config.json"
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        if "env" in config:
            return config
    return None


def find_run_dir(checkpoint_path: Path) -> Path | None:
    """Find the run directory that owns a checkpoint."""
    for path in (checkpoint_path, *checkpoint_path.parents):
        if (path / "config.json").exists() or (path / "xml_manifest.json").exists():
            return path
    return None


def file_sha256(path: str | Path | None) -> str | None:
    """Return a SHA-256 hash for compatibility-critical files."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_xml_manifest(checkpoint_path: Path, env: BiomechanicsJoystickEnv) -> None:
    """Catch checkpoint/XML mismatch before policy compilation."""
    # REF: PROJECT-XML-HASH-GUARD
    # TYPE: ENGINEERING_DEFAULT
    run_dir = find_run_dir(checkpoint_path)
    if run_dir is None:
        return
    manifest_path = run_dir / "xml_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_hash = manifest.get("xml_sha256")
    current_hash = file_sha256(env.xml_path)
    if saved_hash and current_hash and saved_hash != current_hash:
        raise ValueError(
            "Checkpoint/XML mismatch: saved XML hash "
            f"{saved_hash[:12]} != runtime XML hash {current_hash[:12]}."
        )


def resolve_saved_xml_path(raw_path: str) -> str | None:
    """Resolve a saved Windows/WSL XML path against this checkout."""
    direct_path = Path(raw_path).expanduser()
    if direct_path.exists():
        return str(direct_path)

    local_path = GENERATED_MODEL_DIR / direct_path.name
    if local_path.exists():
        return str(local_path)
    return None


def infer_xml_path(checkpoint_path: Path, run_config: dict | None) -> str | None:
    """Infer the exact generated XML used by a run."""
    configured_path = run_env_value(run_config, "xml_path")
    if configured_path:
        resolved_path = resolve_saved_xml_path(str(configured_path))
        if resolved_path is not None:
            return resolved_path

    for path in (checkpoint_path, *checkpoint_path.parents):
        manifest_path = path / "xml_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            xml_path = manifest.get("xml_path")
            if xml_path:
                resolved_path = resolve_saved_xml_path(str(xml_path))
                if resolved_path is not None:
                    return resolved_path
        log_path = path / "train.log"
        if not log_path.exists():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        logged_paths = re.findall(
            r"\bxml=(.+?\.xml)(?=\s*(?:\||$))",
            log_text,
            flags=re.MULTILINE,
        )
        for logged_path in reversed(logged_paths):
            resolved_path = resolve_saved_xml_path(logged_path)
            if resolved_path is not None:
                return resolved_path
    return None


def read_checkpoint_observation_size(checkpoint_path: Path) -> int | None:
    """Procitaj observation_size iz Brax checkpoint metadata fajla."""
    network_config = read_checkpoint_network_config(checkpoint_path)
    if network_config is None:
        return None

    observation_size = network_config.get("observation_size", {})
    if isinstance(observation_size, list):
        shape = observation_size
    elif "state" in observation_size:
        shape = observation_size["state"].get("shape")
    else:
        shape = observation_size.get("shape")
    if not shape:
        return None
    return int(shape[0])


def read_checkpoint_network_config(checkpoint_path: Path) -> dict | None:
    """Read Brax network metadata across old and current file names."""
    for file_name in ("ppo_network_config.json", "config.json"):
        config_path = checkpoint_path / file_name
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        if "observation_size" in config and "action_size" in config:
            return config
    return None


def read_checkpoint_action_size(checkpoint_path: Path) -> int | None:
    """Read the policy action size from Brax checkpoint metadata."""
    network_config = read_checkpoint_network_config(checkpoint_path)
    if network_config is None:
        return None
    action_size = network_config.get("action_size")
    return int(action_size) if action_size is not None else None


def checkpoint_uses_dict_observation(checkpoint_path: Path) -> bool:
    """Return whether checkpoint normalizer/network expects keyed observations."""
    network_config = read_checkpoint_network_config(checkpoint_path)
    if network_config is None:
        return True
    observation_size = network_config.get("observation_size", {})
    return isinstance(observation_size, dict) and "shape" not in observation_size


def validate_action_compatibility(checkpoint_path: Path, action_size: int) -> None:
    """Catch a checkpoint/XML action mismatch before JAX compilation."""
    expected_size = read_checkpoint_action_size(checkpoint_path)
    if expected_size is None or expected_size == action_size:
        return
    raise ValueError(
        "Checkpoint i XML nisu kompatibilni: "
        f"checkpoint ocekuje {expected_size} akcija, a XML daje {action_size}."
    )


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
        f"a trenutni env daje {actual_size}. "
        "Proveri --env-version, --reference-gait, "
        "--reference-target-observation i --xml-path."
    )


def set_command(state, command: np.ndarray):
    """Upise joystick komandu u info i observation koje politika cita."""
    command_array = jnp.asarray(command, dtype=jnp.float32)

    info = dict(state.info)
    info["command"] = command_array
    if isinstance(state.obs, dict):
        obs = dict(state.obs)
        obs["state"] = obs["state"].at[COMMAND_OBS_START:COMMAND_OBS_END].set(
            command_array
        )
        obs["privileged_state"] = obs["privileged_state"].at[
            COMMAND_OBS_START:COMMAND_OBS_END
        ].set(command_array)
    else:
        obs = state.obs.at[COMMAND_OBS_START:COMMAND_OBS_END].set(
            command_array
        )
    return state.replace(info=info, obs=obs)


def update_viewer_data(model, data, state) -> None:
    """Kopira MJX state u MuJoCo viewer data."""
    latest_data = mjx.get_data(model, state.data)
    data.qpos[:] = latest_data.qpos
    data.qvel[:] = latest_data.qvel
    mujoco.mj_forward(model, data)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Write an RGB frame list to MP4 using an installed lightweight writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mediapy as media

        media.write_video(path, frames, fps=fps)
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frames), fps=fps)
        return
    except ImportError as exc:
        raise RuntimeError(
            "Install mediapy or imageio to write MP4 files: "
            "pip install mediapy imageio imageio-ffmpeg"
        ) from exc


def score_percent(reward: float, max_steps: int = PERFECT_WALK_STEPS) -> float:
    """Normalize total reward so 100% means perfect reward for max_steps."""
    max_reward = float(BiomechanicsJoystickEnv.REWARD_MAX) * float(max_steps)
    return 100.0 * reward / max(max_reward, 1e-6)


def survival_percent(length: int, max_steps: int = PERFECT_WALK_STEPS) -> float:
    """Normalize episode length so 100% means surviving max_steps."""
    return 100.0 * float(length) / max(float(max_steps), 1e-6)


def quality_percent(reward: float, length: int) -> float:
    """Normalize reward quality over the steps that actually happened."""
    max_reward = float(BiomechanicsJoystickEnv.REWARD_MAX) * max(float(length), 1.0)
    return 100.0 * reward / max(max_reward, 1e-6)


def reset_state(env, rng, command: np.ndarray):
    """Resetuje epizodu i zadrzava trenutnu joystick komandu."""
    state = env.reset(rng)
    return set_command(state, command)


def print_debug(step: int, state, action) -> None:
    """Ispise signal da proverimo da li politika vidi komandu."""
    if step % DEBUG_PRINT_INTERVAL != 0:
        return

    if isinstance(state.obs, dict):
        command = np.asarray(state.obs["state"][COMMAND_OBS_START:COMMAND_OBS_END])
    else:
        command = np.asarray(state.obs[COMMAND_OBS_START:COMMAND_OBS_END])
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
        f"score_pct={score_percent(total_reward, steps):.1f} "
        f"mean_survive_pct={survival_percent(int(np.mean(episode_lengths))):.1f} "
        f"mean_z={np.mean(z_values):.3f} "
        f"min_z={np.min(z_values):.3f} "
        f"mean_torso_up={np.mean(torso_up_values):.3f} "
        f"min_torso_up={np.min(torso_up_values):.3f} "
        f"mean_action_norm={np.mean(action_norm_values):.3f}",
        flush=True,
    )


def record_policy_video(
    env,
    policy,
    rng,
    command: np.ndarray,
    output_path: Path,
    steps: int,
    fps: int,
    width: int,
    height: int,
    continue_after_done: bool,
) -> None:
    """Render one policy rollout to MP4 and print normalized score diagnostics."""
    state = reset_state(env, rng, command)
    model = env.mj_model
    data = mjx.get_data(model, state.data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.distance = 4.0
    camera.azimuth = 160
    camera.elevation = -20

    frames: list[np.ndarray] = []
    total_reward = 0.0
    episode_reward = 0.0
    episode_length = 0
    completed_episodes = 0
    done_reason = "none"

    print("compiling first JAX record step...", flush=True)
    rng, action_key = jax.random.split(rng)
    state, action = simulation_step(env, policy, state, action_key, command)
    jax.block_until_ready(action)
    print("compile done, recording video", flush=True)

    for step in range(steps):
        rng, action_key, reset_key = jax.random.split(rng, 3)
        state, _action = simulation_step(env, policy, state, action_key, command)
        reward = float(np.asarray(state.reward))
        done = bool(np.asarray(state.done))
        total_reward += reward
        episode_reward += reward
        episode_length += 1

        update_viewer_data(model, data, state)
        camera.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        if done:
            qpos_z = float(np.asarray(state.data.qpos[2]))
            torso_up = get_torso_up(state)
            done_reason = f"done z={qpos_z:.3f} torso_up={torso_up:.3f}"
            completed_episodes += 1
            print(
                "record episode done | "
                f"step={step} "
                f"episode_length={episode_length} "
                f"episode_reward={episode_reward:.3f} "
                f"score_pct={score_percent(episode_reward):.1f} "
                f"survive_pct={survival_percent(episode_length):.1f} "
                f"quality_pct={quality_percent(episode_reward, episode_length):.1f} "
                f"{done_reason}",
                flush=True,
            )
            if not continue_after_done:
                break
            state = reset_state(env, reset_key, command)
            episode_reward = 0.0
            episode_length = 0

    renderer.close()
    if not frames:
        raise RuntimeError("No frames were rendered.")
    write_video(output_path, frames, fps=fps)
    print(
        "record summary | "
        f"file={output_path} "
        f"frames={len(frames)} "
        f"fps={fps} "
        f"total_reward={total_reward:.3f} "
        f"score_pct={score_percent(total_reward, steps):.1f} "
        f"completed_episodes={completed_episodes} "
        f"last_episode_length={episode_length} "
        f"last_done={done_reason}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gledanje Brax PPO politike u MuJoCo viewer-u."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    # REF: PROJECT-EVAL-CPU-DEFAULT
    # TYPE: ENGINEERING_DEFAULT
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument(
        "--env-version",
        choices=["standard", "hardcore"],
        default=None,
    )
    parser.add_argument(
        "--playground-impl",
        choices=["jax", "warp"],
        default=None,
    )
    parser.add_argument(
        "--physics-backend",
        choices=["mjx_jax", "mjx_warp"],
        default=None,
    )
    parser.add_argument(
        "--command-profile",
        choices=[
            "auto",
            "forward_slow",
            "forward",
            "walk",
            "steer",
            "standard_easy",
            "standard",
        ],
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
    parser.add_argument("--reference-root-xy-scale", type=float, default=None)
    parser.add_argument("--action-smoothing", type=float, default=None)
    parser.add_argument("--init-qpos-file", type=Path, default=None)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument(
        "--legacy-action-prior",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
        "--record-video",
        type=Path,
        default=None,
        help="Headless MP4 output path for a policy rollout.",
    )
    parser.add_argument("--record-steps", type=int, default=500)
    parser.add_argument("--record-fps", type=int, default=0)
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    parser.add_argument(
        "--record-continue-after-done",
        action="store_true",
        help="Keep recording after terminal states by resetting the episode.",
    )
    parser.add_argument(
        "--fast-physics",
        action="store_true",
        help="Use sim_dt=0.01 instead of the default accurate 0.005 setup.",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    jax.config.update("jax_default_device", device)
    run_config = find_run_config(args.checkpoint)
    checkpoint_observation_size = read_checkpoint_observation_size(args.checkpoint)
    checkpoint_action_size = read_checkpoint_action_size(args.checkpoint)
    policy_observation_dict = checkpoint_uses_dict_observation(args.checkpoint)
    env_version = args.env_version or run_env_value(
        run_config,
        "env_version",
        "standard",
    )
    playground_impl = args.playground_impl or run_env_value(
        run_config,
        "playground_impl",
        "warp",
    )
    physics_backend = args.physics_backend or run_env_value(
        run_config,
        "physics_backend",
        "mjx_warp" if playground_impl == "warp" else "mjx_jax",
    )
    if args.device == "cpu" and physics_backend == "mjx_warp":
        physics_backend = "mjx_jax"
        playground_impl = "jax"
    action_smoothing = (
        args.action_smoothing
        if args.action_smoothing is not None
        else float(run_env_value(run_config, "action_smoothing", 0.5))
    )
    accurate_physics = bool(
        run_env_value(run_config, "accurate_physics", True)
    ) and not args.fast_physics

    command_profile = (
        infer_command_profile(args.checkpoint, run_config)
        if args.command_profile == "auto"
        else args.command_profile
    )
    command_x = args.command_x
    if command_x is None:
        command_x = DEFAULT_WALK_COMMAND_X if command_profile == "walk" else 0.0
    init_qpos_file = (
        str(args.init_qpos_file)
        if args.init_qpos_file is not None
        else run_env_value(run_config, "init_qpos_file")
    )
    xml_path = str(args.xml_path) if args.xml_path is not None else infer_xml_path(
        args.checkpoint,
        run_config,
    )
    reference_gait = (
        infer_reference_gait(run_config)
        if args.reference_gait == "auto"
        else args.reference_gait
    )
    arm_actuators = bool(run_env_value(run_config, "arm_actuators", False))
    if args.reference_gait_file is not None or args.reference_gait_list is not None:
        reference_gait_file = expand_reference_gait_files(
            args.reference_gait_file,
            args.reference_gait_list,
        )
        reference_target_observation = reference_gait == "bvh"
    else:
        reference_gait_file = run_env_value(run_config, "reference_gait_file")
        reference_target_observation = bool(
            run_env_value(run_config, "reference_target_observation", False)
        )
    reference_loop_mode = str(run_env_value(run_config, "reference_loop_mode", "auto"))
    reference_root_xy_scale = (
        args.reference_root_xy_scale
        if args.reference_root_xy_scale is not None
        else float(run_env_value(run_config, "reference_root_xy_scale", 1.0))
    )
    saved_legacy_action_prior = run_env_value(
        run_config,
        "legacy_action_prior",
        None,
    )
    deepmimic_reward_mode = str(
        run_env_value(run_config, "deepmimic_reward_mode", "pure")
    )
    deepmimic_key_bodies = run_env_value(
        run_config,
        "deepmimic_key_bodies",
        None,
    )
    if deepmimic_key_bodies is None:
        deepmimic_key_bodies = (
            ("head", "right_hand", "left_hand", "right_foot", "left_foot")
            if checkpoint_observation_size is not None
            else ("right_foot", "left_foot")
        )
    if isinstance(deepmimic_key_bodies, str):
        deepmimic_key_bodies = tuple(
            body.strip()
            for body in deepmimic_key_bodies.split(",")
            if body.strip()
        )
    else:
        deepmimic_key_bodies = tuple(str(body) for body in deepmimic_key_bodies)
    if args.legacy_action_prior is not None:
        legacy_action_prior = args.legacy_action_prior
    elif saved_legacy_action_prior is not None:
        legacy_action_prior = bool(saved_legacy_action_prior)
    else:
        # Configs without this field predate the Unitree-style leg scales.
        # Their trunk used per-joint scales, but every leg action used 0.5.
        legacy_action_prior = True
    print(
        "eval config | "
        f"env_version={env_version} | "
        f"playground_impl={playground_impl} | "
        f"physics_backend={physics_backend} | "
        f"command_profile={command_profile} | "
        f"command_x={command_x} | "
        f"init_qpos_file={init_qpos_file} | "
        f"xml_path={xml_path} | "
        f"legacy_action_prior={legacy_action_prior} | "
        f"reference_gait={reference_gait} | "
        f"arm_actuators={arm_actuators} | "
        f"reference_gait_file={reference_gait_file} | "
        f"reference_loop_mode={reference_loop_mode} | "
        f"reference_root_xy_scale={reference_root_xy_scale} | "
        f"reference_target_observation={reference_target_observation} | "
        f"deepmimic_reward_mode={deepmimic_reward_mode} | "
        f"deepmimic_key_bodies={deepmimic_key_bodies} | "
        f"checkpoint_obs={checkpoint_observation_size} | "
        f"checkpoint_actions={checkpoint_action_size} | "
        f"dict_observation={policy_observation_dict} | "
        f"action_smoothing={action_smoothing} | "
        f"accurate_physics={accurate_physics}",
        flush=True,
    )

    env_config = EnvConfig(
        env_version=env_version,
        playground_impl=playground_impl,
        physics_backend=physics_backend,
        warp_num_worlds=1,
        warp_naconmax=resolve_warp_capacities(1).naconmax,
        warp_njmax=resolve_warp_capacities(1).njmax,
        warp_graph_mode=str(run_env_value(run_config, "warp_graph_mode", "warp")),
        command_profile=command_profile,
        reference_gait=reference_gait,
        arm_actuators=arm_actuators,
        reference_gait_file=reference_gait_file,
        reference_loop_mode=reference_loop_mode,
        reference_root_xy_scale=reference_root_xy_scale,
        reference_target_observation=reference_target_observation,
        deepmimic_reward_mode=deepmimic_reward_mode,
        deepmimic_key_bodies=deepmimic_key_bodies,
        policy_observation_size=checkpoint_observation_size,
        policy_observation_dict=policy_observation_dict,
        xml_path=xml_path,
        legacy_action_prior=legacy_action_prior,
        action_smoothing=action_smoothing,
        init_qpos_file=init_qpos_file,
        accurate_physics=accurate_physics,
    )
    env = make_environment(env_config)
    validate_xml_manifest(args.checkpoint, env)
    validate_action_compatibility(args.checkpoint, env.action_size)
    policy = load_ppo_policy(args.checkpoint, deterministic=not args.stochastic)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    command = np.array(
        [command_x, args.command_y, args.command_yaw],
        dtype=np.float32,
    )
    clip_command(command)
    state = set_command(state, command)
    validate_observation_compatibility(args.checkpoint, state.obs)

    if args.inspect:
        inspect_policy(env, policy, rng, command, args.inspect_steps)
        return

    if args.record_video is not None:
        fps = args.record_fps if args.record_fps > 0 else int(round(1.0 / env.dt))
        record_policy_video(
            env,
            policy,
            rng,
            command,
            args.record_video,
            args.record_steps,
            fps,
            args.record_width,
            args.record_height,
            args.record_continue_after_done,
        )
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
    config_overrides = {
        "impl": env_config.playground_impl,
        "physics_backend": env_config.physics_backend,
        "warp_num_worlds": env_config.warp_num_worlds,
        "warp_naconmax": env_config.warp_naconmax,
        "warp_njmax": env_config.warp_njmax,
        "warp_graph_mode": env_config.warp_graph_mode,
        "enable_erfi": False,
        "command_profile": env_config.command_profile,
        "reference_gait": env_config.reference_gait,
        "arm_actuators": env_config.arm_actuators,
        "reference_loop_mode": env_config.reference_loop_mode,
        "reference_root_xy_scale": env_config.reference_root_xy_scale,
        "reference_target_observation": env_config.reference_target_observation,
        "deepmimic_reward_mode": env_config.deepmimic_reward_mode,
        "deepmimic_key_bodies": env_config.deepmimic_key_bodies,
        "policy_observation_size": env_config.policy_observation_size,
        "policy_observation_dict": env_config.policy_observation_dict,
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


if __name__ == "__main__":
    main()
