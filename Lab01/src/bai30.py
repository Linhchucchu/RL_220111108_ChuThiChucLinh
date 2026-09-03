import gymnasium as gym
import numpy as np

def eval_policy(pol):
    env = gym.make("CartPole-v1")

    rewards = []
    for _ in range(100):
        obs, _ = env.reset()
        total_r = 0
        while True:
            obs, r, term, trunc, _ = env.step(pol(obs))
            total_r += r
            if term or trunc:
                break
        rewards.append(total_r)

    env.close()
    return np.mean(rewards)

print(f"Always Left Mean Reward : {eval_policy(lambda obs: 0):.2f}")
print(f"Always Right Mean Reward: {eval_policy(lambda obs: 1):.2f}")
