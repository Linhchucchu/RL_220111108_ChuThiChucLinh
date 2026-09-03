import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)

print("Observation:", observation)
print("Type:", type(observation))
print("Shape:", observation.shape)
print("Info:", info)

# Observation components:
# 0: Cart Position (float32)
# 1: Cart Velocity (float32)
# 2: Pole Angle (float32)
# 3: Pole Angular Velocity (float32)

env.close()
