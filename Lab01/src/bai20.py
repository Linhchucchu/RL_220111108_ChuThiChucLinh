import gymnasium as gym
import numpy as np

def test_seed(seed_val, n_episodes=20):
    env = gym.make("CartPole-v1")
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_val + ep)
        total_r = 0
        while True:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total_r += r
            if term or trunc:
                break
        rewards.append(total_r)

    env.close()
    return np.mean(rewards)

print(f"Mean Reward (Seed 42) : {test_seed(42):.2f}")
print(f"Mean Reward (Seed 100): {test_seed(100):.2f}")
