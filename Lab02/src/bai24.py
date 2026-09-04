import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import policy_evaluation

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    random_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    V, iters, _ = policy_evaluation(env, random_policy, gamma=0.99, theta=1e-8)
    print(f"Policy evaluation hội tụ sau {iters} iterations.")
    print("State values V:\n", V.reshape(4, 4))
    env.close()

if __name__ == "__main__":
    main()
