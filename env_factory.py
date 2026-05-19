from __future__ import annotations

try:
    from .config import EnvConfig
    from .env import HumanWalkEnv
    from .playground_env import PlaygroundJoystickEnv
except ImportError:
    from config import EnvConfig
    from env import HumanWalkEnv
    from playground_env import PlaygroundJoystickEnv


def make_env(config: EnvConfig | None = None, seed: int | None = None):
    config = config or EnvConfig()
    if config.env_backend == "playground":
        return PlaygroundJoystickEnv(config=config, seed=seed)
    if config.env_backend == "biomech":
        return HumanWalkEnv(config=config, seed=seed)
    raise ValueError("env_backend mora biti 'playground' ili 'biomech'.")
