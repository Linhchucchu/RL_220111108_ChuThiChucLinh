import gymnasium as gym
import numpy as np

def improved_policy(obs):
    # PD heuristic: góc + 0.1 * vận tốc góc
    return 1 if (obs[2] + 0.1 * obs[3]) > 0 else 0

env = gym.make("CartPole-v1")

rewards = []
for _ in range(100):
    obs, _ = env.reset()
    total_r = 0
    while True:
        obs, r, term, trunc, _ = env.step(improved_policy(obs))
        total_r += r
        if term or trunc:
            break
    rewards.append(total_r)

env.close()

print(f"Improved Policy Mean Reward: {np.mean(rewards):.2f}")
