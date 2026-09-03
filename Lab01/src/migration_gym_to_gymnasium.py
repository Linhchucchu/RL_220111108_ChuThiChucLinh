# 1. terminated: Agent đạt tới trạng thái kết thúc tự nhiên của bài toán (ngã gậy, về đích).
# 2. truncated: Episode dừng do chạm giới hạn thời gian/bước tối đa (Max timesteps).
# 3. Không dùng done: Vì done làm lẫn lộn giữa chết thật và hết giờ, khiến hàm học giá trị bị sai lệch.

import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

for t in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Finished at step {t+1} (Terminated: {terminated}, Truncated: {truncated})")
        break

env.close()
