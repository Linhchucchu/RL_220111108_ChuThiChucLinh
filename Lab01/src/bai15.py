import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
rewards = []

for _ in range(100):
    obs, _ = env.reset()
    total_r = 0
    while True:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    rewards.append(total_r)

print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Min reward : {np.min(rewards):.2f}")
print(f"Max reward : {np.max(rewards):.2f}")
print(f"Std reward : {np.std(rewards):.2f}")

env.close()
