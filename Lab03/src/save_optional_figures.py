import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from mc_utils import generate_episode, compute_returns, first_visit_mc_prediction

def generate_optional_figures():
    os.makedirs("Lab03/figures", exist_ok=True)
    
    # --- 1. Môi trường FrozenLake-v1 ---
    print("Đang tạo biểu đồ cho FrozenLake-v1...")
    env_fl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    random_policy_fl = lambda s: env_fl.action_space.sample()
    
    V_frozen, _ = first_visit_mc_prediction(env_fl, random_policy_fl, n_episodes=5000, gamma=0.99)
    V_grid = np.zeros(16)
    for s, val in V_frozen.items():
        V_grid[s] = val
        
    plt.figure(figsize=(6, 5))
    plt.imshow(V_grid.reshape(4, 4), cmap='coolwarm', interpolation='none')
    plt.colorbar(label="Estimated State Value V(s)")
    plt.title("FrozenLake-v1: State Value Heatmap (First-Visit MC)")
    plt.xticks(range(4))
    plt.yticks(range(4))
    plt.tight_layout()
    plt.savefig("Lab03/figures/frozenlake_values.png", dpi=300)
    plt.close()
    env_fl.close()

    # --- 2. Môi trường Taxi-v4 ---
    print("Đang tạo biểu đồ cho Taxi-v4...")
    env_taxi = gym.make("Taxi-v4")
    random_policy_taxi = lambda s: env_taxi.action_space.sample()
    
    V_taxi, _ = first_visit_mc_prediction(env_taxi, random_policy_taxi, n_episodes=3000, gamma=0.95)
    values_list = list(V_taxi.values())
    
    plt.figure(figsize=(7, 4))
    plt.hist(values_list, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.xlabel("Estimated V(s)")
    plt.ylabel("Frequency")
    plt.title("Taxi-v4: Distribution of Estimated State Values V(s)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Lab03/figures/taxi_value_distribution.png", dpi=300)
    plt.close()
    env_taxi.close()

    print("Đã tạo và lưu thành công các ảnh biểu đồ vào thư mục Lab03/figures/!")

if __name__ == "__main__":
    generate_optional_figures()
