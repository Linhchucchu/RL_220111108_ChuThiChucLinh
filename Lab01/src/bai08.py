import gymnasium as gym

def run_one_step(env, action):
    return env.step(action)

env = gym.make("CartPole-v1")
env.reset(seed=42)

for i in range(5):
    action = env.action_space.sample()
    obs, r, term, trunc, info = run_one_step(env, action)
    print(f"Step {i+1}: Action={action}, Reward={r}, Term={term}, Trunc={trunc}")
    if term or trunc:
        env.reset()

env.close()
