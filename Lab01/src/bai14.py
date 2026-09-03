import gymnasium as gym
env = gym.make("CartPole-v1")

episode_rewards = []
for ep in range(100):
    obs, _ = env.reset()
    total_r = 0
    while True:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    episode_rewards.append(total_r)

print(f"Collected rewards for {len(episode_rewards)} episodes.")

env.close()
