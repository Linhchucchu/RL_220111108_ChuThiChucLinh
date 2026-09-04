import numpy as np

def sample_next_state(current_state, P, rng):
    return rng.choice(len(P), p=P[current_state])

def main():
    P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    rng = np.random.default_rng(42)
    s = 0
    traj = [s]
    for _ in range(30):
        s = sample_next_state(s, P, rng)
        traj.append(s)
    print("Chuỗi 30 bước mô phỏng:", traj)

if __name__ == "__main__":
    main()
