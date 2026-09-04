import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import q_from_v

def policy_evaluation_sweep(env, policy, V, gamma):
    new_V = np.zeros_like(V)
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            new_V[s] += policy[s][a] * q_from_v(env, V, s, a, gamma)
    return new_V

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    V = np.zeros(env.observation_space.n)
    print("V sau 1 sweep:\n", policy_evaluation_sweep(env, policy, V, 0.99).reshape(4, 4))
    env.close()
