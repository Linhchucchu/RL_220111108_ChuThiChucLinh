import gymnasium as gym
env = gym.make("CartPole-v1")

obs_space = env.observation_space
print("Observation space:", obs_space)
print("Shape of observation:", obs_space.shape)
print("Data type:", obs_space.dtype)
print("Lower bound (low):", obs_space.low)
print("Upper bound (high):", obs_space.high)
env.close()
