import numpy as np

def main():
    n_states, n_actions = 2, 2
    policy = np.ones((n_states, n_actions)) / n_actions
    for s in range(n_states):
        assert np.isclose(np.sum(policy[s]), 1.0)
    print("Stochastic Policy (Uniform):\n", policy)

if __name__ == "__main__":
    main()
