import gymnasium as gym
import numpy as np

def experiment(seed, n_episodes=20):
    env = gym.make("CartPole-v1")

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_r = 0
        while True:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total_r += r
            if term or trunc:
                break

        rewards.append(total_r)
    env.close()

    return {
        "seed": seed,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
        "min_reward": round(float(np.min(rewards)), 2)
    }

for s in [10, 42, 100, 2024, 9999]:
    print(experiment(s))
