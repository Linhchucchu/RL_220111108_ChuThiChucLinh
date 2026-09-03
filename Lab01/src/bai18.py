import gymnasium as gym
import matplotlib.pyplot as plt
import os

def moving_average(values, window_size=10):
    ma = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        ma.append(sum(values[start:i+1]) / (i - start + 1))
    return ma

env = gym.make("CartPole-v1")

rewards = []
for _ in range(100):
    obs, _ = env.reset()
    total_r = 0
    while True:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    rewards.append(total_r)

env.close()

ma_rewards = moving_average(rewards, 10)
plt.figure(figsize=(8, 4))
plt.plot(rewards, alpha=0.4, label='Raw Reward')
plt.plot(ma_rewards, color='red', label='Moving Average (window=10)')
plt.title("CartPole-v1 Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
os.makedirs("Lab01/figures", exist_ok=True)
plt.savefig("Lab01/figures/moving_average.png")
print("Saved to Lab01/figures/moving_average.png")
