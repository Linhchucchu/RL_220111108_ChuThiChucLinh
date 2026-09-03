import gymnasium as gym

def random_agent(env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0.0
    episode_length = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_length += 1
        if terminated or truncated:
            break
    return total_reward, episode_length

env = gym.make("CartPole-v1")
r, l = random_agent(env)
print(f"Random Agent -> Reward: {r}, Length: {l}")

env.close()
