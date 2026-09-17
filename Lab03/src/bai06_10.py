import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mc_utils import generate_episode, compute_returns

def main():
    env = gym.make("Blackjack-v1")
    random_policy = lambda s: env.action_space.sample()
    ep = generate_episode(env, random_policy, seed=42)
    rewards = [step[2] for step in ep]
    
    # Bài 7 & 8 & 9: Tính return
    returns = compute_returns(rewards, gamma=0.9)
    print("Rewards:", rewards)
    print("Discounted Returns (gamma=0.9):", returns)
    
    # Bài 10: So sánh gamma
    gammas = [0.5, 0.8, 0.9, 0.99, 1.0]
    mean_g0 = []
    for g in gammas:
        g0_list = []
        for _ in range(500):
            episode = generate_episode(env, random_policy)
            rew = [s[2] for s in episode]
            g0_list.append(compute_returns(rew, gamma=g)[0])
        mean_g0.append(np.mean(g0_list))
        
    print("Mean G0 ứng với các gamma:", dict(zip(gammas, mean_g0)))
    env.close()

if __name__ == "__main__":
    main()
