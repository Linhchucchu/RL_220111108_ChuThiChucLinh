import gymnasium as gym
import numpy as np

def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    env = gym.make(env_name)
    rewards, lengths = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        r_sum, l = 0, 0

        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            r_sum += r
            l += 1
            if term or trunc:
                break

        rewards.append(r_sum)
        lengths.append(l)

    env.close()

    return {
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "min_reward": round(float(np.min(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
        "mean_length": round(float(np.mean(lengths)), 2)
    }

print("Evaluation Result:", evaluate_policy("CartPole-v1", lambda obs: 1 if obs[2] > 0 else 0))
