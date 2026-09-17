import gymnasium as gym
from mc_utils import generate_episode, ACTION_NAMES_BLACKJACK

def main():
    # Bài 1 & 2 & 3: Khởi tạo và khảo sát Blackjack-v1
    env = gym.make("Blackjack-v1")
    obs, info = env.reset(seed=42)
    print("Observation không gian:", env.observation_space)
    print("Action space:", env.action_space)
    print("Mẫu Observation:", obs)
    
    # Bài 4 & 5: Sinh episode ngẫu nhiên
    random_policy = lambda s: env.action_space.sample()
    ep = generate_episode(env, random_policy, seed=42)
    print(f"Độ dài episode ngẫu nhiên: {len(ep)}")
    for step in ep[:3]:
        print(f"  State: {step[0]} | Action: {ACTION_NAMES_BLACKJACK[step[1]]} | Reward: {step[2]}")
    env.close()

if __name__ == "__main__":
    main()
