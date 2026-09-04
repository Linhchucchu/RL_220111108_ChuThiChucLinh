import gymnasium as gym

def main():
    env_det = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env_stoch = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    s, a = 0, 2
    print("Non-slippery (State 0, RIGHT):", env_det.unwrapped.P[s][a])
    print("Slippery (State 0, RIGHT):    ", env_stoch.unwrapped.P[s][a])
    env_det.close()
    env_stoch.close()

if __name__ == "__main__":
    main()
