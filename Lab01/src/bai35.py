import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def eval_agent(policy, episodes=500, seed=42):
    env = gym.make("CartPole-v1")

    rewards, lengths = [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        r_sum, l = 0, 0

        while True:
            obs, r, term, trunc, _ = env.step(policy(obs, env))
            r_sum += r
            l += 1
            if term or trunc:
                break

        rewards.append(r_sum)
        lengths.append(l)

    env.close()

    return rewards, lengths

pol_random = lambda obs, env: env.action_space.sample()
pol_angle = lambda obs, env: 1 if obs[2] > 0 else 0
pol_improved = lambda obs, env: 1 if (obs[2] + 0.1 * obs[3]) > 0 else 0

r_rnd, l_rnd = eval_agent(pol_random)
r_ang, l_ang = eval_agent(pol_angle)
r_imp, l_imp = eval_agent(pol_improved)

print(f"{'Agent':<15} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'Mean Length':<10}")
print("-" * 65)
for name, r, l in [("Random", r_rnd, l_rnd), ("Angle-based", r_ang, l_ang), ("Improved", r_imp, l_imp)]:
    print(f"{name:<15} | {np.mean(r):<8.2f} | {np.std(r):<8.2f} | {np.min(r):<8.2f} | {np.max(r):<8.2f} | {np.mean(l):<10.2f}")

plt.figure(figsize=(8, 4))
plt.bar(["Random", "Angle-based", "Improved"], [np.mean(r_rnd), np.mean(r_ang), np.mean(r_imp)], color=['gray', 'orange', 'green'])
plt.title("Comparison of Agent Performance (500 Episodes)")
plt.ylabel("Mean Total Reward")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
os.makedirs("Lab01/figures", exist_ok=True)
plt.savefig("Lab01/figures/comparison_agents.png")
print("Saved comparison plot to Lab01/figures/comparison_agents.png")
