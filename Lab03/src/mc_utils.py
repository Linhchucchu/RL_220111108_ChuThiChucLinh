import gymnasium as gym
import numpy as np
from collections import defaultdict

ACTION_NAMES_BLACKJACK = {
    0: "STICK (Dừng rút)",
    1: "HIT (Rút thêm)"
}

def generate_episode(env, policy, seed=None):
    """Bài 5: Sinh một episode từ môi trường dựa trên policy cho trước.
    Trả về trajectory dạng list các tuple: (state, action, reward)
    """
    if seed is not None:
        observation, info = env.reset(seed=seed)
    else:
        observation, info = env.reset()
        
    episode = []
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action = policy(observation) if callable(policy) else policy[observation]
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode.append((observation, action, reward))
        observation = next_observation
        
    return episode

def compute_returns(rewards, gamma=1.0):
    """Bài 7: Tính discounted return G_t cho từng timestep từ danh sách rewards của episode."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    """Bài 17: Ước lượng hàm giá trị trạng thái V(s) bằng First-Visit Monte Carlo."""
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        states = [step[0] for step in episode]
        rewards = [step[2] for step in episode]
        returns = compute_returns(rewards, gamma)
        
        visited_states = set()
        for t, state in enumerate(states):
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += returns[t]
                returns_count[state] += 1
                
    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count

def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    """Bài 20: Ước lượng hàm giá trị trạng thái V(s) bằng Every-Visit Monte Carlo."""
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        states = [step[0] for step in episode]
        rewards = [step[2] for step in episode]
        returns = compute_returns(rewards, gamma)
        
        for t, state in enumerate(states):
            returns_sum[state] += returns[t]
            returns_count[state] += 1
                
    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count

def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):
    """Bài 25: Ước lượng hàm giá trị hành động Q(s, a)."""
    returns_sum = defaultdict(lambda: defaultdict(float))
    returns_count = defaultdict(lambda: defaultdict(int))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        states = [step[0] for step in episode]
        actions = [step[1] for step in episode]
        rewards = [step[2] for step in episode]
        returns = compute_returns(rewards, gamma)
        
        visited_pairs = set()
        for t, (state, action) in enumerate(zip(states, actions)):
            pair = (state, action)
            if pair not in visited_pairs:
                visited_pairs.add(pair)
                returns_sum[state][action] += returns[t]
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                
    return Q

def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    """Bài 27: Chọn action theo chiến lược epsilon-greedy dựa trên Q-values."""
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    else:
        q_vals = Q[state]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return rng.choice(best_actions)

def on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):
    """Bài 32: Thuật toán On-policy First-Visit MC Control."""
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))
    episode_rewards = []
    
    for i in range(n_episodes):
        def current_policy(state):
            return epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
            
        episode = generate_episode(env, current_policy, seed=int(rng.integers(0, 1e9)))
        states = [step[0] for step in episode]
        actions = [step[1] for step in episode]
        rewards = [step[2] for step in episode]
        
        episode_rewards.append(sum(rewards))
        returns = compute_returns(rewards, gamma)
        
        visited_state_actions = set()
        for t, (state, action) in enumerate(zip(states, actions)):
            sa_pair = (state, action)
            if sa_pair not in visited_state_actions:
                visited_state_actions.add(sa_pair)
                returns_sum[state][action] += returns[t]
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                
    return Q, episode_rewards

def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    """Bài 34: Đánh giá policy đã học (tỷ lệ thắng, thua, hòa và reward trung bình)."""
    rng = np.random.default_rng(seed)
    wins = 0
    losses = 0
    draws = 0
    total_rewards = []
    
    for _ in range(n_episodes):
        if callable(policy):
            pol_func = policy
        else:
            pol_func = lambda s: int(np.argmax(policy[s])) if s in policy else env.action_space.sample()
            
        episode = generate_episode(env, pol_func, seed=int(rng.integers(0, 1e9)))
        tot_reward = sum([step[2] for step in episode])
        total_rewards.append(tot_reward)
        
        if tot_reward > 0: wins += 1
        elif tot_reward < 0: losses += 1
        else: draws += 1
            
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": float(np.mean(total_rewards))
    }
