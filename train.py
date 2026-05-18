from __future__ import annotations

import argparse
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
from loguru import logger

from .config import RUNS_DIR
from .deep_rl import ReplayBuffer, TrainConfig, make_agent
from .env import HumanWalkEnv


def choose_device(name: str, allow_cpu: bool):
    """Bira JAX uredjaj. Ako trazis GPU, kod ne nastavlja tiho na CPU."""
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
    raise RuntimeError("JAX ne vidi GPU. Pokreni sa --allow-cpu samo za mali CPU test.")


def save_checkpoint(path: Path, algo: str, agent, obs_dim: int, action_dim: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "algo": algo,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden": cfg.hidden,
        "agent": agent.save_dict(),
    }
    with path.open("wb") as f:
        pickle.dump(data, f)


def format_steps(steps: int) -> str:
    """200000 -> 200k, 1500000 -> 1m5, 500 -> 500."""
    if steps >= 1_000_000:
        whole = steps // 1_000_000
        rest = (steps % 1_000_000) // 100_000
        return f"{whole}m{rest}" if rest else f"{whole}m"
    if steps >= 1_000 and steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def make_run_dir(base_dir: Path, algo: str, timesteps: int, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return base_dir / f"{algo}_{stamp}_{format_steps(timesteps)}_seed{seed}"


def train_off_policy(env: HumanWalkEnv, agent, cfg: TrainConfig, save_path: Path, algo: str) -> None:
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay = ReplayBuffer(obs_dim, action_dim, cfg.replay_size)

    obs, _ = env.reset(seed=cfg.seed)
    episode_reward = 0.0
    episode = 1

    logger.info("petlja start | total_steps={} | random_start_steps={}", cfg.timesteps, cfg.start_steps)
    for step in range(1, cfg.timesteps + 1):
        if step < cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, float(done))
        obs = next_obs
        episode_reward += float(reward)

        if done:
            logger.info("episode={} step={} reward={:.2f}", episode, step, episode_reward)
            obs, _ = env.reset()
            episode_reward = 0.0
            episode += 1

        if len(replay) >= cfg.batch_size and step >= cfg.start_steps:
            for _ in range(cfg.updates_per_step):
                agent.update(replay)

        if step == 1 or step % 1_000 == 0 or step == cfg.timesteps:
            logger.info(
                "progress | step={}/{} | replay_size={} | current_episode={} | current_episode_reward={:.2f}",
                step,
                cfg.timesteps,
                len(replay),
                episode,
                episode_reward,
            )

        if step % 10_000 == 0:
            save_checkpoint(save_path, algo, agent, obs_dim, action_dim, cfg)
            logger.info("checkpoint sacuvan | step={} | path={}", step, save_path)

    save_checkpoint(save_path, algo, agent, obs_dim, action_dim, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX deep RL trening bez Stable-Baselines3 i bez PyTorch-a.")
    parser.add_argument("--algo", choices=["sac", "td3"], default="sac")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "jax")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device, args.allow_cpu)
    jax.config.update("jax_default_device", device)
    print(f"jax_device={device}")

    cfg = TrainConfig(algo=args.algo, timesteps=args.timesteps, seed=args.seed, device=args.device)
    env = HumanWalkEnv(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    key = jax.random.PRNGKey(args.seed)
    agent = make_agent(args.algo, obs_dim, action_dim, cfg, key)

    run_dir = make_run_dir(args.out, args.algo, args.timesteps, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(run_dir / "train.log", level="INFO", encoding="utf-8", mode="w")

    save_path = run_dir / "policy.pkl"
    (run_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")

    logger.info("trening start")
    logger.info("algo={} seed={} timesteps={} run_dir={}", args.algo, args.seed, args.timesteps, run_dir)
    logger.info("jax_device={} available_devices={}", device, jax.devices())
    logger.info("obs_dim={} action_dim={}", obs_dim, action_dim)
    logger.info("batch_size={} replay_size={} start_steps={} lr={}", cfg.batch_size, cfg.replay_size, cfg.start_steps, cfg.lr)

    train_off_policy(env, agent, cfg, save_path, args.algo)

    env.close()
    logger.info("trening gotov | model={}", save_path)


if __name__ == "__main__":
    main()
