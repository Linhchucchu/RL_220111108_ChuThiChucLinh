import gymnasium as gym

def run_agent_no_done(env, max_steps=500):
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_finished = terminated or truncated
        if episode_finished:
            if terminated:
                print("End Reason: Termination")
            if truncated:
                print("End Reason: Truncation")
            break

env = gym.make("CartPole-v1")
run_agent_no_done(env)

env.close()
