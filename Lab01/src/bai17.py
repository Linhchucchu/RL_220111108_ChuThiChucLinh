import gymnasium as gym
import matplotlib.pyplot as plt
import os

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

plt.figure(figsize=(8, 4))
plt.plot(rewards, color='blue', label='Episode Reward')
plt.title("CartPole-v1 Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
os.makedirs("Lab01/figures", exist_ok=True)
plt.savefig("Lab01/figures/reward_cartpole.png")
print("Saved to Lab01/figures/reward_cartpole.png")
