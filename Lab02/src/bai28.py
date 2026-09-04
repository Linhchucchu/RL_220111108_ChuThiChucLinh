import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import policy_evaluation, greedy_policy_from_value

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    old_policy = np.zeros(env.observation_space.n, dtype=int)
    V, _, _ = policy_evaluation(env, old_policy, gamma=0.99)
    new_policy = greedy_policy_from_value(env, V, gamma=0.99)
    changed = np.sum(old_policy != new_policy)
    print(f"Số trạng thái đổi hành động: {changed}/{env.observation_space.n}")
    env.close()

if __name__ == "__main__":
    main()
