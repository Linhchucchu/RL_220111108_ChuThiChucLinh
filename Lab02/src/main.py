import os, sys
sys.path.append(os.path.dirname(__file__))
from time import perf_counter
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import (
    evaluate_policy_by_simulation,
    policy_iteration,
    print_frozenlake_policy,
    value_iteration,
)

def main():
    os.makedirs("Lab02/figures", exist_ok=True)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    gamma = 0.99
    theta = 1e-8

    print("=== Chạy Value Iteration ===")
    t0 = perf_counter()
    vi_policy, vi_V, vi_iters, vi_deltas = value_iteration(env, gamma=gamma, theta=theta)
    vi_time = perf_counter() - t0
    print(f"VI hoàn thành sau {vi_iters} sweeps ({vi_time:.4f}s)")
    print_frozenlake_policy(env, vi_policy)

    plt.figure(figsize=(7, 4))
    plt.plot(vi_deltas, color="blue")
    plt.yscale("log")
    plt.title("Value Iteration Convergence Delta")
    plt.xlabel("Iteration")
    plt.ylabel("Delta (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Lab02/figures/value_iteration_convergence.png", dpi=300)
    plt.close()

    print("\n=== Chạy Policy Iteration ===")
    t0 = perf_counter()
    pi_policy, pi_V, pi_iters, pi_eval_iters, pi_deltas = policy_iteration(env, gamma=gamma, theta=theta)
    pi_time = perf_counter() - t0
    print(f"PI hoàn thành sau {pi_iters} iterations ({pi_time:.4f}s)")
    print_frozenlake_policy(env, pi_policy)

    plt.figure(figsize=(7, 4))
    plt.plot(pi_deltas, color="teal")
    plt.yscale("log")
    plt.title("Policy Iteration: Evaluation Convergence")
    plt.xlabel("Evaluation Sweeps")
    plt.ylabel("Delta (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Lab02/figures/policy_iteration_convergence.png", dpi=300)
    plt.close()

    print("\n=== Đánh giá mô phỏng 1000 Episodes ===")
    metrics_vi = evaluate_policy_by_simulation(env, vi_policy, 1000)
    metrics_pi = evaluate_policy_by_simulation(env, pi_policy, 1000)
    print(f"VI Success Rate: {metrics_vi['success_rate']*100:.2f}% | Mean Reward: {metrics_vi['mean_reward']:.3f}")
    print(f"PI Success Rate: {metrics_pi['success_rate']*100:.2f}% | Mean Reward: {metrics_pi['mean_reward']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["VI", "PI"], [vi_time, pi_time], color=["steelblue", "coral"])
    axes[0].set_title("Thời gian chạy (s)")
    axes[0].grid(axis="y", linestyle="--")

    axes[1].bar(["VI", "PI"], [metrics_vi["success_rate"]*100, metrics_pi["success_rate"]*100], color=["steelblue", "coral"])
    axes[1].set_title("Tỷ lệ thành công (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig("Lab02/figures/algorithm_comparison.png", dpi=300)
    plt.close()
    print("Đã lưu đầy đủ các biểu đồ so sánh!")
    env.close()

if __name__ == "__main__":
    main()
