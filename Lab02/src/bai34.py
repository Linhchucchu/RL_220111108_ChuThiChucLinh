import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import evaluate_policy_by_simulation, value_iteration, policy_iteration

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    rand_pol = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    vi_pol, _, _, _ = value_iteration(env)
    pi_pol, _, _, _, _ = policy_iteration(env)

    print("Random :", evaluate_policy_by_simulation(env, rand_pol, 1000))
    print("VI     :", evaluate_policy_by_simulation(env, vi_pol, 1000))
    print("PI     :", evaluate_policy_by_simulation(env, pi_pol, 1000))
    env.close()

if __name__ == "__main__":
    main()
