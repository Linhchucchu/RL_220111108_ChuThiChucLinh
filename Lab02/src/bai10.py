import numpy as np
import matplotlib.pyplot as plt

def main():
    gammas = np.linspace(0, 1, 101)
    returns = [10.0 * (g ** 4) for g in gammas]
    plt.figure(figsize=(7, 4))
    plt.plot(gammas, returns, color="crimson", linewidth=2)
    plt.title("G0 phụ thuộc vào Gamma (R = [0, 0, 0, 0, 10])")
    plt.xlabel("Gamma")
    plt.ylabel("Return G0")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Lab02/figures/gamma_comparison.png", dpi=300)
    print("Đã lưu Lab02/figures/gamma_comparison.png")

if __name__ == "__main__":
    main()
