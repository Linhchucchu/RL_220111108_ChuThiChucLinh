import gymnasium as gym

def describe_state(env, state):
    print(f"\n--- State {state} Description ---")
    for a in range(env.action_space.n):
        print(f"Action {a}: {env.unwrapped.P[state][a]}")

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    for s in [0, 1, 14]:
        describe_state(env, s)
    env.close()
