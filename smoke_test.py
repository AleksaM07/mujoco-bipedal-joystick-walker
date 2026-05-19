import argparse

try:
    from .config import EnvConfig
    from .env_factory import make_env
except ImportError:
    from config import EnvConfig
    from env_factory import make_env


def main():
    parser = argparse.ArgumentParser(description="Brza provera da se izabrano MuJoCo okruzenje startuje.")
    parser.add_argument("--env-backend", choices=["playground", "biomech"], default="playground")
    parser.add_argument("--env-version", choices=["standard", "hardcore"], default="standard")
    parser.add_argument("--playground-impl", choices=["jax", "warp"], default="jax")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    env_config = EnvConfig(
        env_backend=args.env_backend,
        env_version=args.env_version,
        playground_impl=args.playground_impl,
    )
    env = make_env(config=env_config, seed=123)
    obs, info = env.reset()
    print("Backend:", info.get("backend", args.env_backend))
    print("Env version:", info.get("env_version", args.env_version))
    if "env_name" in info:
        print("Env name:", info["env_name"])
    print("Observation shape:", obs.shape)
    print("Action shape:", env.action_space.shape)
    if "sex" in info:
        print("Random subject:", info["sex"], round(info["height"], 3), "m", round(info["mass"], 1), "kg")

    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    print("Last reward:", round(float(reward), 4))
    if "velocity" in info:
        print("Last velocity:", None if info["velocity"] is None else info["velocity"].round(3))
    print("Last command:", info["command"].round(3))
    env.close()


if __name__ == "__main__":
    main()
