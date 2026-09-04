from bai07 import compute_return

def main():
    rewards = [1, 1, 1, 1, 1]
    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]
    print(f"{'Gamma':>6} | {'Return':>8}")
    print("-" * 18)
    for g in gammas:
        print(f"{g:6.2f} | {compute_return(rewards, g):8.4f}")

if __name__ == "__main__":
    main()
