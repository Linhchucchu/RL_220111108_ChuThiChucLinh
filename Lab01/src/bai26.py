import gymnasium as gym

actions = [2, 2, 1, 1, 1, 2]
env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)
obs, _ = env.reset(seed=42)

for step_idx, act in enumerate(actions):
    obs, reward, term, trunc, _ = env.step(act)
    print(f"--- Step {step_idx+1}: Action {act} ---")
    print(env.render())
    if term or trunc:
        print(f"Reached Goal with Reward: {reward}")
        break

env.close()
