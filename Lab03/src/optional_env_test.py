import gymnasium as gym
import numpy as np
from mc_utils import generate_episode, compute_returns, first_visit_mc_prediction

def main():
    print("=== KIỂM THỬ MONTE CARLO TRÊN MÔI TRƯỜNG PHỤ: FrozenLake-v1 ===")
    
    # Khởi tạo môi trường phụ FrozenLake-v1 (bật is_slippery=True để có tính ngẫu nhiên)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    
    # Định nghĩa một policy đơn giản: luôn đi ngẫu nhiên hoặc theo chính sách cố định
    def random_policy(state):
        return env.action_space.sample()

    # 1. Sinh một episode mẫu và tính return
    episode = generate_episode(env, random_policy, seed=42)
    rewards = [step[2] for step in episode]
    returns = compute_returns(rewards, gamma=0.99)
    
    print(f"Độ dài episode trên FrozenLake: {len(episode)} steps")
    print(f"Tổng reward nhận được: {sum(rewards)}")
    print(f"Discounted returns mẫu (5 bước đầu): {returns[:5]}")

    # 2. Chạy First-Visit MC Prediction trên FrozenLake
    print("\nĐang chạy First-Visit MC Prediction trên FrozenLake (3,000 episodes)...")
    V_frozen, counts = first_visit_mc_prediction(env, random_policy, n_episodes=3000, gamma=0.99)
    
    print(f"Số lượng trạng thái được ghé thăm và ước lượng: {len(V_frozen)}")
    print("Bảng giá trị V(s) ước lượng (định dạng lưới 4x4):")
    
    V_grid = np.zeros(16)
    for s, val in V_frozen.items():
        V_grid[s] = val
    print(V_grid.reshape(4, 4))

    env.close()

if __name__ == "__main__":
    main()
