import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mdp_utils import policy_evaluation

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    random_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    _, _, deltas = policy_evaluation(env, random_policy, gamma=0.99, theta=1e-8)
    plt.figure(figsize=(7, 4))
    plt.plot(deltas)
    plt.yscale("log")
    plt.title("Policy Evaluation Delta Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Delta (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Lab02/figures/policy_eval_deltas.png")
    print("Đã lưu Lab02/figures/policy_eval_deltas.png")
    env.close()

if __name__ == "__main__":
    main()
