import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)
obs, _ = env.reset()
print(env.render())

env.close()
