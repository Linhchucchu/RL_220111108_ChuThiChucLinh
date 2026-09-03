import gymnasium as gym

def run_episode(env, policy, seed=None, max_steps=1000):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated, truncated = False, False

    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated
    }

env = gym.make("CartPole-v1")

res = run_episode(env, lambda obs: 1 if obs[2] > 0 else 0, seed=42)
print("Run Episode Result:", res)

env.close()
