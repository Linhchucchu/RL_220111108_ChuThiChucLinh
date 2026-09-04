import numpy as np

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def q_from_v(env, V, state, action, gamma=0.99):
    """Tính Q(s, a) = sum p(s', r | s, a) * [r + gamma * V(s') * (1 - terminated)]"""
    q_val = 0.0
    for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
        terminal_factor = 0.0 if terminated else 1.0
        q_val += prob * (reward + gamma * V[next_state] * terminal_factor)
    return q_val

def action_values(env, V, state, gamma=0.99):
    """Tính vector Q(s, a) cho mọi actions tại state s"""
    n_actions = env.action_space.n
    q_vals = np.zeros(n_actions)
    for a in range(n_actions):
        q_vals[a] = q_from_v(env, V, state, a, gamma)
    return q_vals

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Iterative Policy Evaluation cho cả policy đơn định và ngẫu nhiên"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    deltas = []
    is_deterministic = (policy.ndim == 1)

    for _ in range(max_iterations):
        delta = 0.0
        new_V = np.zeros(n_states)
        for s in range(n_states):
            if is_deterministic:
                a = policy[s]
                v_s = q_from_v(env, V, s, a, gamma)
            else:
                v_s = 0.0
                for a in range(n_actions):
                    if policy[s][a] > 0:
                        v_s += policy[s][a] * q_from_v(env, V, s, a, gamma)
            delta = max(delta, abs(v_s - V[s]))
            new_V[s] = v_s
        V = new_V
        deltas.append(delta)
        if delta < theta:
            break
    return V, len(deltas), deltas

def greedy_policy_from_value(env, V, gamma=0.99):
    """Trích xuất greedy policy từ hàm giá trị V"""
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = np.argmax(action_values(env, V, s, gamma))
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """Thuật toán Policy Iteration hoàn chỉnh"""
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)
    eval_iters_total = 0
    deltas_all = []

    for it in range(max_iterations):
        V, it_eval, deltas = policy_evaluation(env, policy, gamma, theta)
        eval_iters_total += it_eval
        deltas_all.extend(deltas)
        policy_stable = True
        for s in range(n_states):
            old_a = policy[s]
            new_a = np.argmax(action_values(env, V, s, gamma))
            if old_a != new_a:
                policy_stable = False
            policy[s] = new_a
        if policy_stable:
            return policy, V, it + 1, eval_iters_total, deltas_all
    return policy, V, max_iterations, eval_iters_total, deltas_all

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Thuật toán Value Iteration hoàn chỉnh"""
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    deltas = []
    for _ in range(max_iterations):
        delta = 0.0
        new_V = np.zeros(n_states)
        for s in range(n_states):
            max_q = np.max(action_values(env, V, s, gamma))
            delta = max(delta, abs(max_q - V[s]))
            new_V[s] = max_q
        V = new_V
        deltas.append(delta)
        if delta < theta:
            break
    policy = greedy_policy_from_value(env, V, gamma)
    return policy, V, len(deltas), deltas

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    """Đánh giá thực nghiệm policy qua n_episodes mô phỏng"""
    rng = np.random.default_rng(seed)
    success_count = 0
    total_rewards = []
    lengths = []
    for _ in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 1000000)))
        terminated, truncated = False, False
        ep_reward, ep_len = 0.0, 0
        while not (terminated or truncated):
            action = policy[state] if policy.ndim == 1 else rng.choice(env.action_space.n, p=policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len += 1
        total_rewards.append(ep_reward)
        lengths.append(ep_len)
        if ep_reward > 0:
            success_count += 1
    return {
        "success_rate": success_count / n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }

def print_frozenlake_policy(env, policy):
    """Hiển thị bảng ký tự mũi tên trên bản đồ 4x4"""
    desc = env.unwrapped.desc.astype(str)
    nrow, ncol = desc.shape
    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            tile = desc[r, c]
            char = tile if tile in ["H", "G"] else ACTION_SYMBOLS[policy[s]]
            print(f"{char:^4}", end="")
        print()
