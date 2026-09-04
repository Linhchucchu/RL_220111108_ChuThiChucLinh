import numpy as np
from bai07 import compute_return

def main():
    seq_A = [5, 0, 0, 0, 0]
    seq_B = [0, 0, 0, 0, 10]
    gammas = np.linspace(0, 1, 1001)
    b_greater = [g for g in gammas if compute_return(seq_B, g) > compute_return(seq_A, g)]
    print(f"Khoảng gamma mà B > A: [{min(b_greater):.4f}, {max(b_greater):.4f}]")

if __name__ == "__main__":
    main()
