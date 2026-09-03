import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def create_environment(env_name="CartPole-v1"):
    return gym.make(env_name)

def policy(observation):
    pole_angle = observation[2]
    pole_vel = observation[3]
    return 1 if (pole_angle + 0.1 * pole_vel) > 0 else 0

def run_episode(env, seed=None):
    obs, _ = env.reset(seed=seed)
    total_r, length = 0, 0
    while True:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total_r += r
        length += 1
        if term or trunc:
            break
    return total_r, length

def evaluate_policy(n_episodes=500, seed=42):
    env = create_environment()
    rewards, lengths = [], []
    for ep in range(n_episodes):
        r, l = run_episode(env, seed=seed + ep)
        rewards.append(r)
        lengths.append(l)
    env.close()
    return rewards, lengths

def plot_results(rewards):
    ma = [sum(rewards[max(0, i-10+1):i+1]) / (i - max(0, i-10+1) + 1) for i in range(len(rewards))]
    plt.figure(figsize=(9, 4.5))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(ma, color='red', label='Moving Average (window=10)')
    plt.title("Mini-project: CartPole-v1 Improved Policy Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("Lab01/figures", exist_ok=True)
    plt.savefig("Lab01/figures/mini_project_reward.png")
    plt.close()

def main():
    rewards, lengths = evaluate_policy(n_episodes=500, seed=42)
    print("--- Thí nghiệm Mini Project (CartPole-v1) ---")
    print(f"Episodes      : {len(rewards)}")
    print(f"Mean Reward   : {np.mean(rewards):.2f}")
    print(f"Std Reward    : {np.std(rewards):.2f}")
    print(f"Best Episode  : {np.max(rewards):.2f}")
    print(f"Worst Episode : {np.min(rewards):.2f}")
    plot_results(rewards)
    print("Plot saved to Lab01/figures/mini_project_reward.png")

if __name__ == "__main__":
    main()
