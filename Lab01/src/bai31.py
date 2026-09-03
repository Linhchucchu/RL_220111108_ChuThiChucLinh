import gymnasium as gym
import numpy as np

def angle_based_policy(obs):
    return 1 if obs[2] > 0 else 0

env = gym.make("CartPole-v1")

rewards = []
for _ in range(100):
    obs, _ = env.reset()
    total_r = 0
    while True:
        obs, r, term, trunc, _ = env.step(angle_based_policy(obs))
        total_r += r
        if term or trunc:
            break
    rewards.append(total_r)

env.close()

print(f"Angle-based Policy Mean Reward: {np.mean(rewards):.2f}")
