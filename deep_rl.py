from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    algo: str = "sac"
    timesteps: int = 200_000
    seed: int = 7
    device: str = "gpu"
    hidden: int = 256
    gamma: float = 0.98
    tau: float = 0.01
    batch_size: int = 256
    replay_size: int = 200_000
    start_steps: int = 2_000
    updates_per_step: int = 1
    lr: float = 3e-4
    alpha: float = 0.2


class ReplayBuffer:
    """Prost replay buffer za SAC i TD3."""

    def __init__(self, obs_dim: int, action_dim: int, size: int):
        self.size = int(size)
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self), size=batch_size)
        return {
            "obs": jnp.asarray(self.obs[idx]),
            "actions": jnp.asarray(self.actions[idx]),
            "rewards": jnp.asarray(self.rewards[idx]),
            "next_obs": jnp.asarray(self.next_obs[idx]),
            "dones": jnp.asarray(self.dones[idx]),
        }


def init_mlp(key, input_dim: int, output_dim: int, hidden: int):
    """Pravi obicnu MLP mrezu sa dva skrivena sloja."""
    sizes = [input_dim, hidden, hidden, output_dim]
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for layer_key, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        params.append(
            {
                "w": jax.random.uniform(layer_key, (fan_in, fan_out), minval=-limit, maxval=limit),
                "b": jnp.zeros((fan_out,), dtype=jnp.float32),
            }
        )
    return params


def mlp_apply(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def actor_stats(params, obs, action_dim: int):
    out = mlp_apply(params, obs)
    mean, log_std = jnp.split(out, 2, axis=-1)
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mean, log_std


def gaussian_sample(params, obs, key, action_dim: int):
    mean, log_std = actor_stats(params, obs, action_dim)
    noise = jax.random.normal(key, mean.shape)
    raw = mean + jnp.exp(log_std) * noise
    action = jnp.tanh(raw)
    logp = gaussian_log_prob_from_raw(raw, mean, log_std, action)
    return action, logp


def gaussian_log_prob_from_raw(raw, mean, log_std, action):
    normal_logp = -0.5 * (((raw - mean) / (jnp.exp(log_std) + 1e-6)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    squash_fix = jnp.log(1.0 - action**2 + 1e-6)
    return jnp.sum(normal_logp - squash_fix, axis=-1, keepdims=True)


def gaussian_log_prob(params, obs, action, action_dim: int):
    clipped = jnp.clip(action, -0.999, 0.999)
    raw = 0.5 * jnp.log((1.0 + clipped) / (1.0 - clipped))
    mean, log_std = actor_stats(params, obs, action_dim)
    return gaussian_log_prob_from_raw(raw, mean, log_std, clipped)


def q_apply(params, obs, action):
    return mlp_apply(params, jnp.concatenate([obs, action], axis=-1))


def soft_update(params, target_params, tau: float):
    return jax.tree_util.tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params)


def init_adam(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_step(params, grads, state, lr: float):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * (g * g), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - 0.9**t), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - 0.999**t), v)
    new_params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + 1e-8), params, m_hat, v_hat)
    return new_params, {"m": m, "v": v, "t": t}


class SacAgent:
    """Minimalan SAC agent u JAX-u."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: TrainConfig, key):
        self.cfg = cfg
        self.action_dim = action_dim
        self.key = key
        k1, k2, k3 = jax.random.split(key, 3)
        self.actor = init_mlp(k1, obs_dim, action_dim * 2, cfg.hidden)
        self.q1 = init_mlp(k2, obs_dim + action_dim, 1, cfg.hidden)
        self.q2 = init_mlp(k3, obs_dim + action_dim, 1, cfg.hidden)
        self.tq1 = self.q1
        self.tq2 = self.q2
        self.actor_opt = init_adam(self.actor)
        self.q1_opt = init_adam(self.q1)
        self.q2_opt = init_adam(self.q2)

    def act(self, obs, deterministic: bool = False):
        obs = jnp.asarray(obs[None, :], dtype=jnp.float32)
        if deterministic:
            mean, _ = actor_stats(self.actor, obs, self.action_dim)
            action = jnp.tanh(mean)
        else:
            self.key, subkey = jax.random.split(self.key)
            action, _ = gaussian_sample(self.actor, obs, subkey, self.action_dim)
        return np.asarray(action[0], dtype=np.float32)

    def update(self, replay: ReplayBuffer):
        batch = replay.sample(self.cfg.batch_size)
        self.key, next_key, actor_key = jax.random.split(self.key, 3)

        def q_loss(q1, q2):
            next_action, next_logp = gaussian_sample(self.actor, batch["next_obs"], next_key, self.action_dim)
            tq = jnp.minimum(q_apply(self.tq1, batch["next_obs"], next_action), q_apply(self.tq2, batch["next_obs"], next_action))
            target = batch["rewards"] + self.cfg.gamma * (1.0 - batch["dones"]) * (tq - self.cfg.alpha * next_logp)
            loss1 = jnp.mean((q_apply(q1, batch["obs"], batch["actions"]) - target) ** 2)
            loss2 = jnp.mean((q_apply(q2, batch["obs"], batch["actions"]) - target) ** 2)
            return loss1 + loss2

        (_, (gq1, gq2)) = jax.value_and_grad(q_loss, argnums=(0, 1))(self.q1, self.q2)
        self.q1, self.q1_opt = adam_step(self.q1, gq1, self.q1_opt, self.cfg.lr)
        self.q2, self.q2_opt = adam_step(self.q2, gq2, self.q2_opt, self.cfg.lr)

        def actor_loss(actor):
            action, logp = gaussian_sample(actor, batch["obs"], actor_key, self.action_dim)
            q = jnp.minimum(q_apply(self.q1, batch["obs"], action), q_apply(self.q2, batch["obs"], action))
            return jnp.mean(self.cfg.alpha * logp - q)

        _, gactor = jax.value_and_grad(actor_loss)(self.actor)
        self.actor, self.actor_opt = adam_step(self.actor, gactor, self.actor_opt, self.cfg.lr)
        self.tq1 = soft_update(self.q1, self.tq1, self.cfg.tau)
        self.tq2 = soft_update(self.q2, self.tq2, self.cfg.tau)

    def save_dict(self):
        return {
            "actor": jax.device_get(self.actor),
            "q1": jax.device_get(self.q1),
            "q2": jax.device_get(self.q2),
        }

    def load_dict(self, data):
        self.actor = data["actor"]


class Td3Agent:
    """Minimalan TD3 agent u JAX-u."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: TrainConfig, key):
        self.cfg = cfg
        self.action_dim = action_dim
        self.key = key
        self.update_count = 0
        k1, k2, k3 = jax.random.split(key, 3)
        self.actor = init_mlp(k1, obs_dim, action_dim, cfg.hidden)
        self.tactor = self.actor
        self.q1 = init_mlp(k2, obs_dim + action_dim, 1, cfg.hidden)
        self.q2 = init_mlp(k3, obs_dim + action_dim, 1, cfg.hidden)
        self.tq1 = self.q1
        self.tq2 = self.q2
        self.actor_opt = init_adam(self.actor)
        self.q1_opt = init_adam(self.q1)
        self.q2_opt = init_adam(self.q2)

    def act(self, obs, deterministic: bool = False):
        obs = jnp.asarray(obs[None, :], dtype=jnp.float32)
        action = jnp.tanh(mlp_apply(self.actor, obs))
        if not deterministic:
            self.key, subkey = jax.random.split(self.key)
            action = jnp.clip(action + 0.1 * jax.random.normal(subkey, action.shape), -1.0, 1.0)
        return np.asarray(action[0], dtype=np.float32)

    def update(self, replay: ReplayBuffer):
        self.update_count += 1
        batch = replay.sample(self.cfg.batch_size)
        self.key, noise_key = jax.random.split(self.key)

        def q_loss(q1, q2):
            noise = jnp.clip(0.2 * jax.random.normal(noise_key, batch["actions"].shape), -0.5, 0.5)
            next_action = jnp.clip(jnp.tanh(mlp_apply(self.tactor, batch["next_obs"])) + noise, -1.0, 1.0)
            tq = jnp.minimum(q_apply(self.tq1, batch["next_obs"], next_action), q_apply(self.tq2, batch["next_obs"], next_action))
            target = batch["rewards"] + self.cfg.gamma * (1.0 - batch["dones"]) * tq
            loss1 = jnp.mean((q_apply(q1, batch["obs"], batch["actions"]) - target) ** 2)
            loss2 = jnp.mean((q_apply(q2, batch["obs"], batch["actions"]) - target) ** 2)
            return loss1 + loss2

        (_, (gq1, gq2)) = jax.value_and_grad(q_loss, argnums=(0, 1))(self.q1, self.q2)
        self.q1, self.q1_opt = adam_step(self.q1, gq1, self.q1_opt, self.cfg.lr)
        self.q2, self.q2_opt = adam_step(self.q2, gq2, self.q2_opt, self.cfg.lr)

        if self.update_count % 2 == 0:
            def actor_loss(actor):
                action = jnp.tanh(mlp_apply(actor, batch["obs"]))
                return -jnp.mean(q_apply(self.q1, batch["obs"], action))

            _, gactor = jax.value_and_grad(actor_loss)(self.actor)
            self.actor, self.actor_opt = adam_step(self.actor, gactor, self.actor_opt, self.cfg.lr)
            self.tactor = soft_update(self.actor, self.tactor, self.cfg.tau)
            self.tq1 = soft_update(self.q1, self.tq1, self.cfg.tau)
            self.tq2 = soft_update(self.q2, self.tq2, self.cfg.tau)

    def save_dict(self):
        return {
            "actor": jax.device_get(self.actor),
            "q1": jax.device_get(self.q1),
            "q2": jax.device_get(self.q2),
        }

    def load_dict(self, data):
        self.actor = data["actor"]


def make_agent(algo: str, obs_dim: int, action_dim: int, cfg: TrainConfig, key):
    if algo == "sac":
        return SacAgent(obs_dim, action_dim, cfg, key)
    if algo == "td3":
        return Td3Agent(obs_dim, action_dim, cfg, key)
    raise ValueError(f"Nepoznat algoritam: {algo}")
