import numpy as np
import matplotlib.pyplot as plt

def main():
    P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    rng = np.random.default_rng(42)
    n_steps = 100000
    counts = np.zeros(3)
    s = 0
    for _ in range(n_steps):
        s = rng.choice(3, p=P[s])
        counts[s] += 1

    print("Mô phỏng 100k bước :", counts / n_steps)
    print("Lý thuyết (t = 50) :", np.array([1.0, 0.0, 0.0]) @ np.linalg.matrix_power(P, 50))

    p = np.array([1.0, 0.0, 0.0])
    hist = [p]
    for _ in range(20):
        p = p @ P
        hist.append(p)
    hist = np.array(hist)

    plt.figure(figsize=(7, 4))
    plt.plot(hist[:, 0], label="Sunny", marker="o")
    plt.plot(hist[:, 1], label="Cloudy", marker="s")
    plt.plot(hist[:, 2], label="Rainy", marker="^")
    plt.title("Markov Chain: Phân phối trạng thái theo thời gian")
    plt.xlabel("Bước (t)")
    plt.ylabel("Xác suất")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Lab02/figures/markov_distribution.png", dpi=300)
    print("Đã lưu Lab02/figures/markov_distribution.png")

if __name__ == "__main__":
    main()
