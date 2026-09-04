import numpy as np

def print_policy(policy):
    print("Deterministic policy:", policy)

if __name__ == "__main__":
    policy = np.array([0, 1])
    print_policy(policy)
