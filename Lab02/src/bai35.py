import sys, os
sys.path.append(os.path.dirname(__file__))
from time import perf_counter
import gymnasium as gym
from mdp_utils import value_iteration, policy_iteration, evaluate_policy_by_simulation

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    t0 = perf_counter()
    vi_pol, _, vi_it, _ = value_iteration(env)
    vi_time = perf_counter() - t0
    vi_eval = evaluate_policy_by_simulation(env, vi_pol, 1000)

    t0 = perf_counter()
    pi_pol, _, pi_it, _, _ = policy_iteration(env)
    pi_time = perf_counter() - t0
    pi_eval = evaluate_policy_by_simulation(env, pi_pol, 1000)

    print(f"{'Thuật toán':<18} | {'Vòng lặp':<8} | {'Thời gian (s)':<14} | {'Success rate':<12}")
    print("-" * 60)
    print(f"{'Value Iteration':<18} | {vi_it:<8} | {vi_time:<14.5f} | {vi_eval['success_rate']*100:.2f}%")
    print(f"{'Policy Iteration':<18} | {pi_it:<8} | {pi_time:<14.5f} | {pi_eval['success_rate']*100:.2f}%")
    env.close()

if __name__ == "__main__":
    main()
