import gymnasium as gym
env = gym.make("CartPole-v1")

print(f"{'Episode':<10} | {'Reward':<10} | {'Length':<10}")
print("-" * 36)

for ep in range(1, 11):
    obs, _ = env.reset()
    total_r, length = 0, 0
    while True:
        action = env.action_space.sample()
        _, r, term, trunc, _ = env.step(action)
        total_r += r
        length += 1
        if term or trunc:
            break
    print(f"{ep:<10} | {total_r:<10.1f} | {length:<10}")

env.close()
