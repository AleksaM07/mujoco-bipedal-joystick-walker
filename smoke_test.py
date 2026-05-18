from .env import HumanWalkEnv


def main():
    env = HumanWalkEnv(seed=123)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action shape:", env.action_space.shape)
    print("Random subject:", info["sex"], round(info["height"], 3), "m", round(info["mass"], 1), "kg")

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    print("Last reward:", round(float(reward), 4))
    print("Last velocity:", None if info["velocity"] is None else info["velocity"].round(3))
    env.close()


if __name__ == "__main__":
    main()

