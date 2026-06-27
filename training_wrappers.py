import contextlib
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as playground_wrapper


DOMAIN_RANDOMIZATION_ID = "domain_randomization_id"


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
            mjx_model = self._select_model(model_id.astype(jp.int32))
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
        state.info[self._key("done_count")] = jp.zeros(
            key.shape[:-1],
            dtype=int,
        )
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(state.info[self._key("rng")])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]

        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        stepped_state = self.env.step(state, action)

        reset_state = jax.lax.cond(
            jp.any(stepped_state.done),
            lambda _: self.reset(reset_key),
            lambda _: state,
            operand=None,
        )

        def where_done(reset_value, step_value):
            done = stepped_state.done
            if done.shape and done.shape[0] != reset_value.shape[0]:
                return step_value
            if done.shape:
                done = jp.reshape(
                    done,
                    [reset_value.shape[0]] + [1] * (len(reset_value.shape) - 1),
                )
            return jp.where(done, reset_value, step_value)

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
    state.info["use_rfi"] = jp.arange(batch_size) < split
    return state
