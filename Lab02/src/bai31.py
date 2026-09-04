import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import action_values

def value_iteration_sweep(env, V, gamma):
    new_V = np.zeros_like(V)
    for s in range(env.observation_space.n):
        new_V[s] = np.max(action_values(env, V, s, gamma))
    return new_V

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V = np.zeros(env.observation_space.n)
    print("V sau 1 sweep của VI:\n", value_iteration_sweep(env, V, 0.99).reshape(4, 4))
    env.close()
