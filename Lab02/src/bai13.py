import numpy as np
from bai12 import get_two_state_mdp

def validate_mdp(P, n_states, n_actions):
    for s in range(n_states):
        for a in range(n_actions):
            total_prob = sum(prob for prob, _, _, _ in P[s][a])
            if not np.isclose(total_prob, 1.0):
                print(f"Invalid transition at state={s}, action={a}")
                return False
    return True

if __name__ == "__main__":
    mdp = get_two_state_mdp()
    print("Mô hình MDP hợp lệ:", validate_mdp(mdp, 2, 2))
