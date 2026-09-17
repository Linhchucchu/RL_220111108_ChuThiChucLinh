import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from mc_utils import (
    generate_episode,
    compute_returns,
    first_visit_mc_prediction,
    on_policy_mc_control,
    evaluate_policy,
    ACTION_NAMES_BLACKJACK
)

def main():
    env = gym.make("Blackjack-v1")
    print("=== CHẠY CHƯƠNG TRÌNH TỔNG HỢP LAB 03 (MONTE CARLO METHODS) ===")
    
    # 1. Fixed Policy mẫu
    def stick_on_20_policy(state):
        player_sum, _, _ = state
        return 0 if player_sum >= 20 else 1

    # 2. Prediction
    print("\n[1] Đang chạy First-Visit MC Prediction (10,000 episodes)...")
    V_first, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=10000)
    print(f" -> Ước lượng thành công giá trị cho {len(V_first)} trạng thái.")

    # 3. Control (Training agent)
    print("\n[2] Đang huấn luyện On-policy MC Control (50,000 episodes)...")
    Q_optimal, episode_rewards = on_policy_mc_control(env, n_episodes=50000, epsilon=0.1)

    learned_policy_dict = {s: int(np.argmax(acts)) for s, acts in Q_optimal.items()}

    # 4. Evaluation & Comparison
    print("\n[3] Đang đánh giá hiệu năng các policy...")
    random_policy = lambda s: env.action_space.sample()
    
    eval_learned = evaluate_policy(env, learned_policy_dict, n_episodes=5000)
    eval_fixed = evaluate_policy(env, stick_on_20_policy, n_episodes=5000)
    eval_random = evaluate_policy(env, random_policy, n_episodes=5000)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM ---")
    print(f"Random Policy  -> Mean Reward: {eval_random['mean_reward']:+.4f} | Win Rate: {eval_random['win_rate']*100:.2f}%")
    print(f"Fixed Policy   -> Mean Reward: {eval_fixed['mean_reward']:+.4f} | Win Rate: {eval_fixed['win_rate']*100:.2f}%")
    print(f"Learned Policy -> Mean Reward: {eval_learned['mean_reward']:+.4f} | Win Rate: {eval_learned['win_rate']*100:.2f}%")

    # 5. Learning Curve Plotting
    print("\n[4] Đang vẽ và lưu Learning Curve...")
    window = 1000
    moving_avgs = [np.mean(episode_rewards[max(0, i-window):i+1]) for i in range(len(episode_rewards))]
    
    plt.figure(figsize=(8, 4))
    plt.plot(moving_avgs, color='crimson', label=f"Moving Average Reward (window={window})")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve - On-Policy MC Control (Blackjack-v1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Lab03/figures", exist_ok=True)
    plt.savefig("Lab03/figures/mc_convergence.png", dpi=300)
    plt.show()
    print("Đã lưu biểu đồ thành công vào Lab03/figures/mc_convergence.png")

    env.close()

if __name__ == "__main__":
    main()
