import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from mc_utils import first_visit_mc_prediction, every_visit_mc_prediction

def main():
    env = gym.make("Blackjack-v1")
    def stick_on_20_policy(state):
        player_sum, _, _ = state
        return 0 if player_sum >= 20 else 1

    # Bài 17 & 20: First-visit và Every-visit MC prediction
    n_ep = 5000
    V_fv, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_ep)
    V_ev, _ = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n_ep)
    
    print(f"First-Visit ước lượng được {len(V_fv)} states.")
    print(f"Every-Visit ước lượng được {len(V_ev)} states.")

    # Bài 23: Vẽ biểu đồ so sánh hội tụ giữa First-Visit và Every-Visit
    target_s = (18, 5, False)
    ep_list = [100, 500, 1000, 3000, 5000]
    fv_vals, ev_vals = [], []
    for n in ep_list:
        vf, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n)
        ve, _ = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n)
        fv_vals.append(vf.get(target_s, 0.0))
        ev_vals.append(ve.get(target_s, 0.0))

    plt.figure(figsize=(7, 4))
    plt.plot(ep_list, fv_vals, marker='o', label='First-Visit MC')
    plt.plot(ep_list, ev_vals, marker='x', label='Every-Visit MC')
    plt.xlabel("Episodes")
    plt.ylabel(f"Estimated V({target_s})")
    plt.title("First-Visit vs Every-Visit MC Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Lab03/figures", exist_ok=True)
    plt.savefig("Lab03/figures/first_vs_every_visit.png", dpi=300)
    print("Đã lưu Lab03/figures/first_vs_every_visit.png")
    env.close()

if __name__ == "__main__":
    main()
