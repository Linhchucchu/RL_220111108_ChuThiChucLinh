import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import greedy_policy_from_value

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V = np.zeros(env.observation_space.n)
    pol = greedy_policy_from_value(env, V, gamma=0.99)
    print("Greedy policy trích xuất từ V=0:", pol)
    env.close()

if __name__ == "__main__":
    main()
