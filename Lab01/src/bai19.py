import gymnasium as gym
import numpy as np

observations = []
for _ in range(10):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)
    observations.append(obs)
    env.close()

is_identical = all(np.array_equal(observations[0], o) for o in observations)

print("Are all 10 initial observations identical?:", is_identical)
