import gymnasium as gym
import numpy as np

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    P = env.unwrapped.P
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            assert np.isclose(sum(p for p, _, _, _ in P[s][a]), 1.0)
    print("Kiểm tra hoàn tất: Mọi cặp (state, action) đều có tổng xác suất bằng 1.0!")
    env.close()

if __name__ == "__main__":
    main()
