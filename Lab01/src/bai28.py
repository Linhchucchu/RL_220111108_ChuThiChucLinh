import gymnasium as gym
import numpy as np

def run_fl(is_slippery, episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    success, rewards, lengths = 0, [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        r_sum, l = 0, 0
        while True:
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            r_sum += reward
            l += 1
            if term or trunc:
                if reward > 0:
                    success += 1
                break

        rewards.append(r_sum)
        lengths.append(l)

    env.close()
    return success / episodes, np.mean(rewards), np.mean(lengths)

s_det, r_det, l_det = run_fl(is_slippery=False)
s_stoch, r_stoch, l_stoch = run_fl(is_slippery=True)

print(f"Deterministic (is_slippery=False): Success={s_det*100:.2f}%, Avg Reward={r_det:.4f}, Avg Length={l_det:.2f}")
print(f"Stochastic    (is_slippery=True) : Success={s_stoch*100:.2f}%, Avg Reward={r_stoch:.4f}, Avg Length={l_stoch:.2f}")
