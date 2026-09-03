import gymnasium as gym
env = gym.make("CartPole-v1")

obs, _ = env.reset(seed=42)

for t in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"t={t:2d} | action={action} | reward={reward}")
    if terminated or truncated:
        print("Episode stopped early.")
        break

env.close()
