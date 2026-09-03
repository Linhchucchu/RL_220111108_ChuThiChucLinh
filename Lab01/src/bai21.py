import gymnasium as gym

def sample_actions(seed_val):
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed_val)
    actions = [env.action_space.sample() for _ in range(20)]
    env.close()
    return actions

run1 = sample_actions(42)
run2 = sample_actions(42)

print("Run 1:", run1)
print("Run 2:", run2)
print("Are both action lists identical?:", run1 == run2)
