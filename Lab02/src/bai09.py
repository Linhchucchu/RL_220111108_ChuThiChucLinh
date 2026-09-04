def discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))

if __name__ == "__main__":
    rewards = [0, 0, 0, 1]
    print("Returns [G0, G1, G2, G3] với gamma=0.9:", discounted_returns(rewards, 0.9))
