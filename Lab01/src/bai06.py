import gymnasium as gym
from collections import Counter

env = gym.make("CartPole-v1")
actions = [env.action_space.sample() for _ in range(20)]

print("20 sampled actions:", actions)

freq = Counter(actions)

for action, count in freq.items():
    print(f"Action {action}: {count} times ({count/20*100:.1f}%)")

env.close()
