import gymnasium as gym

def policy(observation, env):
    return env.action_space.sample()

env = gym.make("CartPole-v1")

obs, _ = env.reset()
total_r = 0

while True:
    obs, r, term, trunc, _ = env.step(policy(obs, env))
    total_r += r
    if term or trunc:
        break

print(f"Random policy total reward: {total_r}")

env.close()
