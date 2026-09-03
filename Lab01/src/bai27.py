import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
total_episodes = 100
success = 0

for _ in range(total_episodes):
    obs, _ = env.reset()
    while True:
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            if reward > 0:
                success += 1
            break

env.close()

print(f"Success: {success}, Failure: {total_episodes - success}")
print(f"Success rate: {success / total_episodes * 100:.2f}%")
