import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import print_frozenlake_policy

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    policy = np.zeros(env.observation_space.n, dtype=int)
    print("Biểu diễn policy dạng lưới:")
    print_frozenlake_policy(env, policy)
    env.close()

if __name__ == "__main__":
    main()
