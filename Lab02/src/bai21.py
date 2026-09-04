import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import numpy as np
from mdp_utils import q_from_v

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V = np.zeros(env.observation_space.n)
    q = q_from_v(env, V, state=0, action=1, gamma=0.99)
    print(f"Q(state=0, action=1) khi V=0: {q}")
    env.close()

if __name__ == "__main__":
    main()
