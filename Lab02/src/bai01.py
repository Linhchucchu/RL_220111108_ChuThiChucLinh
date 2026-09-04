import numpy as np

def main():
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ])
    print("Transition matrix P (3x3):")
    print(P)
    print("Tổng từng hàng:", P.sum(axis=1))

if __name__ == "__main__":
    main()
