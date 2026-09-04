def compute_return(rewards, gamma):
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

if __name__ == "__main__":
    rewards = [1, 1, 1, 1, 1]
    print("Undiscounted return (gamma=1.0):", compute_return(rewards, 1.0))
