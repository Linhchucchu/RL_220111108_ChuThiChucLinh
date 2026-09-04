import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    P = env.unwrapped.P
    state = 0
    print(f"Bảng transitions tại state = {state}:")
    for action, transitions in P[state].items():
        print(f"  Action {action}:")
        for prob, next_s, r, term in transitions:
            print(f"    Prob: {prob:.3f} | Next state: {next_s:2d} | Reward: {r} | Terminated: {term}")
    env.close()

if __name__ == "__main__":
    main()
