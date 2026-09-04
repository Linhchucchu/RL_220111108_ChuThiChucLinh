import numpy as np

def main():
    P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    p0 = np.array([1.0, 0.0, 0.0])
    p1 = p0 @ P
    print("Phân phối sau 1 bước (p1):", p1)

if __name__ == "__main__":
    main()
