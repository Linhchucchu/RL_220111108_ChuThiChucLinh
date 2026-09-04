import numpy as np

def validate_transition_matrix(P, tol=1e-10):
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if np.any(P < 0.0) or np.any(P > 1.0):
        return False
    return np.allclose(np.sum(P, axis=1), 1.0, atol=tol)

if __name__ == "__main__":
    P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    print("Ma trận P hợp lệ:", validate_transition_matrix(P))
