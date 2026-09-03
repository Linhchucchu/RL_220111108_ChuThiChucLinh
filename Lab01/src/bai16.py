import gymnasium as gym
env = gym.make("CartPole-v1")

history = []
for ep in range(100):
    obs, _ = env.reset()
    total_r, length = 0, 0
    while True:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total_r += r
        length += 1
        if term or trunc:
            break
    history.append((ep + 1, total_r, length))

best_ep = max(history, key=lambda x: x[1])
print(f"Best Episode: {best_ep[0]} | Reward: {best_ep[1]:.2f} | Length: {best_ep[2]}")

env.close()
