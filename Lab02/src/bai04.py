import numpy as np

def state_distribution(p0, P, n_steps):
    return p0 @ np.linalg.matrix_power(P, n_steps)

def main():
    P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    p0 = np.array([1.0, 0.0, 0.0])
    for step in [1, 2, 5, 10, 50]:
        print(f"t = {step:2d}: {state_distribution(p0, P, step)}")

if __name__ == "__main__":
    main()
