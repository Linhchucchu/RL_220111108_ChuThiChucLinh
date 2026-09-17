import gymnasium as gym
import numpy as np
from mc_utils import generate_episode, compute_returns, first_visit_mc_prediction

def main():
    print("=== KIỂM THỬ MONTE CARLO TRÊN MÔI TRƯỜNG PHỤ THỨ HAI: Taxi-v4 ===")
    
    # Khởi tạo môi trường Taxi-v4 thay vì v3
    env = gym.make("Taxi-v4")
    
    # Policy ngẫu nhiên
    random_policy = lambda state: env.action_space.sample()

    # 1. Sinh một episode mẫu
    episode = generate_episode(env, random_policy, seed=42)
    rewards = [step[2] for step in episode]
    returns = compute_returns(rewards, gamma=0.95)
    
    print(f"Độ dài episode trên Taxi-v4: {len(episode)} steps")
    print(f"Tổng reward của episode: {sum(rewards)}")
    print(f"Discounted returns mẫu (5 bước đầu): {returns[:5]}")

    # 2. Chạy First-Visit MC Prediction trên Taxi-v4 với số lượng episode vừa phải
    print("\nĐang chạy First-Visit MC Prediction trên Taxi-v4 (2,000 episodes)...")
    V_taxi, counts = first_visit_mc_prediction(env, random_policy, n_episodes=2000, gamma=0.95)
    
    print(f"Số lượng state độc lập đã thu thập và ước lượng được: {len(V_taxi)}")
    
    # In một vài giá trị V(s) mẫu
    sample_states = list(V_taxi.keys())[:5]
    print("Mẫu giá trị V(s) của một số trạng thái:")
    for s in sample_states:
        print(f"  State {s}: V = {V_taxi[s]:.4f} (ghé thăm {counts[s]} lần)")

    env.close()

if __name__ == "__main__":
    main()
