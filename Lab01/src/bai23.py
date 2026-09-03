import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print(f"Number of states : {env.observation_space.n}")
print(f"Number of actions: {env.action_space.n}")

env.close()
