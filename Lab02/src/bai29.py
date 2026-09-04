import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
from mdp_utils import policy_iteration, print_frozenlake_policy

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    pol, V, iters, _, _ = policy_iteration(env, gamma=0.99, theta=1e-8)
    print(f"Policy Iteration hội tụ sau {iters} bước.")
    print_frozenlake_policy(env, pol)
    env.close()

if __name__ == "__main__":
    main()
