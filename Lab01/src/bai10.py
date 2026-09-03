import gymnasium as gym
env = gym.make("CartPole-v1")

obs, _ = env.reset(seed=42)
total_reward = 0.0
length = 0

for t in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    length += 1
    if terminated or truncated:
        break

print(f"Episode length: {length}")
print(f"Total reward: {total_reward}")

env.close()
