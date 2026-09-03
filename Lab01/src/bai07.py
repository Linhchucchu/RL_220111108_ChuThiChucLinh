import gymnasium as gym
env = gym.make("CartPole-v1")

obs, info = env.reset(seed=42)
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

print("State before action:", obs)
print("Action:", action)
print("State after action:", next_obs)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

env.close()
