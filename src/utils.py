"""
Training loops, experiment runners, statistical utilities, and all plotting
functions used across the three experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings

from .algorithms import QLearning, SARSA, ExpectedSARSA, DoubleQLearning, value_iteration
from .environments import make_env

sns.set_theme(style="whitegrid", font_scale=1.1)

ALGO_COLORS = {
    'Q-learning':        '#e74c3c',
    'SARSA':             '#2ecc71',
    'Expected SARSA':    '#3498db',
    'Double Q-learning': '#9b59b6',
}

ALGO_CLASSES = {
    'Q-learning':        QLearning,
    'SARSA':             SARSA,
    'Expected SARSA':    ExpectedSARSA,
    'Double Q-learning': DoubleQLearning,
}


def compute_confidence_interval(data, confidence=0.95):
    """
    Computes mean and 95% confidence interval across seeds (axis 0).
    Uses Student t distribution since n=30 seeds.
    Returns (mean, ci) where the band is mean +/- ci.
    """
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    se = stats.sem(data, axis=0)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, ci


def train_agent(env, agent, n_episodes, max_steps=200, record_q_every=None):
    """
    Runs one full training session for a single agent on one environment.

    SARSA needs special handling because its update uses the next action
    that was actually selected, not the greedy max. So we pre-select the
    first action before the loop and carry each action forward.

    All other algorithms follow the standard select, step, update pattern.

    Returns a numpy array of total rewards per episode, shape (n_episodes,).
    """
    episode_rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        if isinstance(agent, SARSA):
            action = agent.select_action(state)

        for step in range(max_steps):
            if isinstance(agent, SARSA):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_state) if not done else 0
                agent.update(state, action, reward, next_state, done, next_action)
                state = next_state
                action = next_action
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state

            total_reward += reward
            if done:
                break

        episode_rewards[ep] = total_reward
        agent.decay_epsilon()

        if record_q_every and (ep + 1) % record_q_every == 0:
            agent.q_history.append(agent.get_q_table())

    return episode_rewards


def run_experiment(env_name, algo_names, n_episodes=1000, n_seeds=30,
                   max_steps=200, alpha=0.1, gamma=0.99,
                   epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    """
    Runs all specified algorithms on one environment across multiple seeds.

    For each algorithm and each seed a fresh agent is created and trained.
    Also runs value iteration to get Q* for comparison if the environment
    supports it. Returns a results dict, a q_tables dict, and Q* (or None).

    results[algo_name] has shape (n_seeds, n_episodes).
    q_tables[algo_name] is a list of final Q tables, one per seed.
    """
    env, config = make_env(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_star = None
    try:
        q_star, v_star, opt_policy = value_iteration(env, gamma=gamma)
    except (ValueError, AttributeError):
        pass

    results = {}
    q_tables = {}

    for algo_name in algo_names:
        AgentClass = ALGO_CLASSES[algo_name]
        seed_rewards = np.zeros((n_seeds, n_episodes))
        final_q_tables = []

        for seed in range(n_seeds):
            np.random.seed(seed)
            env, _ = make_env(env_name)
            env.reset(seed=seed)

            agent = AgentClass(
                n_states=n_states, n_actions=n_actions,
                alpha=alpha, gamma=gamma,
                epsilon=epsilon, epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
            )
            rewards = train_agent(env, agent, n_episodes, max_steps)
            seed_rewards[seed] = rewards
            final_q_tables.append(agent.get_q_table())

        results[algo_name] = seed_rewards
        q_tables[algo_name] = final_q_tables

    return results, q_tables, q_star


def run_sensitivity_sweep(env_name, algo_name, param_name, param_values,
                          n_episodes=1000, n_seeds=10, max_steps=200,
                          base_params=None):
    """
    Sweeps one hyperparameter across the given values and returns performance.

    For each value, trains the algorithm with n_seeds seeds and records
    the mean reward over the last 100 episodes as the performance metric.

    Returns (mean_rewards, std_rewards), each of shape (len(param_values),).
    """
    if base_params is None:
        base_params = {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0,
                       'epsilon_min': 0.01, 'epsilon_decay': 0.995}

    env, _ = make_env(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    AgentClass = ALGO_CLASSES[algo_name]

    mean_rewards = np.zeros(len(param_values))
    std_rewards = np.zeros(len(param_values))

    for i, val in enumerate(param_values):
        params = base_params.copy()
        params[param_name] = val
        seed_rewards = []

        for seed in range(n_seeds):
            np.random.seed(seed)
            env, _ = make_env(env_name)
            env.reset(seed=seed)

            agent = AgentClass(n_states=n_states, n_actions=n_actions, **params)
            rewards = train_agent(env, agent, n_episodes, max_steps)
            seed_rewards.append(np.mean(rewards[-100:]))

        mean_rewards[i] = np.mean(seed_rewards)
        std_rewards[i] = np.std(seed_rewards)

    return mean_rewards, std_rewards


def run_sensitivity_2d(env_name, algo_name, param1_name, param1_values,
                       param2_name, param2_values,
                       n_episodes=1000, n_seeds=10, max_steps=200,
                       base_params=None):
    """
    Runs a 2D grid search over two hyperparameters for heatmap generation.

    For every combination of param1 and param2 values, trains n_seeds agents
    and records mean asymptotic reward (last 100 episodes).

    Returns performance_grid of shape (len(param1_values), len(param2_values)).
    """
    if base_params is None:
        base_params = {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0,
                       'epsilon_min': 0.01, 'epsilon_decay': 0.995}

    env, _ = make_env(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    AgentClass = ALGO_CLASSES[algo_name]

    grid = np.zeros((len(param1_values), len(param2_values)))

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            params = base_params.copy()
            params[param1_name] = v1
            params[param2_name] = v2
            seed_rewards = []

            for seed in range(n_seeds):
                np.random.seed(seed)
                env, _ = make_env(env_name)
                env.reset(seed=seed)
                agent = AgentClass(n_states=n_states, n_actions=n_actions, **params)
                rewards = train_agent(env, agent, n_episodes, max_steps)
                seed_rewards.append(np.mean(rewards[-100:]))

            grid[i, j] = np.mean(seed_rewards)

    return grid


def run_bias_experiment(env_name, n_episodes=500, n_seeds=30,
                        alpha=0.1, gamma=0.99, epsilon=0.1,
                        epsilon_min=0.1, epsilon_decay=1.0, max_steps=200):
    """
    Tracks Q value overestimation for Q learning and Double Q learning over time.

    Epsilon is fixed (no decay) so the exploration policy stays constant
    throughout, isolating the bias signal from exploration changes.

    After each episode records: mean over all states of (max Q(s) minus max Q*(s)).
    Positive values mean overestimation, negative means underestimation.

    Returns (q_overest, dq_overest, q_star), each seed matrix is (n_seeds, n_episodes).
    """
    env, _ = make_env(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_star, _, _ = value_iteration(env, gamma=gamma)

    q_overest = np.zeros((n_seeds, n_episodes))
    dq_overest = np.zeros((n_seeds, n_episodes))

    for seed in range(n_seeds):
        for algo_idx, AgentClass in enumerate([QLearning, DoubleQLearning]):
            np.random.seed(seed)
            env, _ = make_env(env_name)
            env.reset(seed=seed)

            agent = AgentClass(
                n_states=n_states, n_actions=n_actions,
                alpha=alpha, gamma=gamma,
                epsilon=epsilon, epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
            )

            for ep in range(n_episodes):
                state, _ = env.reset()
                for step in range(max_steps):
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
                agent.decay_epsilon()

                max_q = np.max(agent.get_q_table(), axis=1)
                max_q_star = np.max(q_star, axis=1)
                overest = np.mean(max_q - max_q_star)

                if algo_idx == 0:
                    q_overest[seed, ep] = overest
                else:
                    dq_overest[seed, ep] = overest

    return q_overest, dq_overest, q_star


def smooth(data, window=50):
    """
    Applies a moving average with the given window size to a 1D array.
    Returns data unchanged if it is shorter than the window.
    Output length is len(data) - window + 1 when smoothing is applied.
    """
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_convergence(results, env_name, window=50, save_path=None):
    """
    Plots smoothed reward curves and cumulative reward with 95% CI bands.
    Left panel shows per episode reward, right panel shows cumulative reward.
    Saves to save_path if provided.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for algo_name, data in results.items():
        mean, ci = compute_confidence_interval(data)
        sm_mean = smooth(mean, window)
        sm_ci = smooth(ci, window)
        x = np.arange(len(sm_mean))
        color = ALGO_COLORS.get(algo_name, 'gray')
        ax.plot(x, sm_mean, label=algo_name, color=color, linewidth=1.5)
        ax.fill_between(x, sm_mean - sm_ci, sm_mean + sm_ci, alpha=0.15, color=color)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title(f'{env_name}: Reward per Episode')
    ax.legend(frameon=True, fancybox=True)

    ax = axes[1]
    for algo_name, data in results.items():
        cum_rewards = np.cumsum(data, axis=1)
        mean, ci = compute_confidence_interval(cum_rewards)
        color = ALGO_COLORS.get(algo_name, 'gray')
        ax.plot(mean, label=algo_name, color=color, linewidth=1.5)
        ax.fill_between(range(len(mean)), mean - ci, mean + ci, alpha=0.15, color=color)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'{env_name}: Cumulative Reward')
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensitivity_heatmap(grid, param1_name, param1_values,
                             param2_name, param2_values,
                             algo_name, env_name, save_path=None):
    """
    Plots the 2D hyperparameter grid as a color coded heatmap.
    Rows are param1 values, columns are param2 values.
    Green cells are high performance, red cells are low performance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    p1_labels = [f'{v:.3g}' for v in param1_values]
    p2_labels = [f'{v:.3g}' for v in param2_values]

    sns.heatmap(grid, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=p2_labels, yticklabels=p1_labels,
                ax=ax, cbar_kws={'label': 'Mean Reward (last 100 eps)'})
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title(f'{algo_name} Sensitivity: {param1_name} vs {param2_name}\n({env_name})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensitivity_1d(all_results, param_name, param_values, env_name, save_path=None):
    """
    Plots 1D sensitivity curves for all algorithms on the same axes.
    Error bands show +/- 1 standard deviation across seeds.
    A flat curve means the algorithm is robust to that parameter.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo_name, (means, stds) in all_results.items():
        color = ALGO_COLORS.get(algo_name, 'gray')
        ax.plot(param_values, means, 'o-', label=algo_name, color=color, linewidth=1.5)
        ax.fill_between(param_values, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel(param_name)
    ax.set_ylabel('Mean Reward (last 100 eps)')
    ax.set_title(f'Sensitivity to {param_name} ({env_name})')
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_bias_comparison(q_overest, dq_overest, env_name, window=20, save_path=None):
    """
    Plots Q value overestimation over episodes for Q learning vs Double Q learning.
    The dashed line at y=0 represents perfect calibration (Q values match Q*).
    Values above the line mean the agent overestimates future reward.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for data, name, color in [(q_overest, 'Q-learning', ALGO_COLORS['Q-learning']),
                               (dq_overest, 'Double Q-learning', ALGO_COLORS['Double Q-learning'])]:
        mean, ci = compute_confidence_interval(data)
        sm_mean = smooth(mean, window)
        sm_ci = smooth(ci, window)
        x = np.arange(len(sm_mean))
        ax.plot(x, sm_mean, label=name, color=color, linewidth=1.5)
        ax.fill_between(x, sm_mean - sm_ci, sm_mean + sm_ci, alpha=0.15, color=color)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='True Q*')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Q-value Overestimation')
    ax.set_title(f'Maximization Bias: Q-learning vs Double Q-learning\n({env_name})')
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_policy_grid(q_table, env_name, nrow, ncol, title=None,
                     hole_states=None, goal_states=None, save_path=None):
    """
    Visualizes the greedy policy from a Q table as arrows on a grid.
    Hole states are shown in red, goal states in green, all others show an arrow.
    Arrow direction is argmax_a Q(s, a) for each state.
    """
    action_arrows = {0: '\u2191', 1: '\u2192', 2: '\u2193', 3: '\u2190'}
    policy = np.argmax(q_table, axis=1)

    fig, ax = plt.subplots(figsize=(ncol * 1.2, nrow * 1.2))
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_aspect('equal')

    if hole_states is None:
        hole_states = set()
    if goal_states is None:
        goal_states = set()

    for s in range(nrow * ncol):
        r = s // ncol
        c = s % ncol
        y = nrow - 1 - r

        if s in hole_states:
            ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor='#e74c3c', alpha=0.3))
            ax.text(c + 0.5, y + 0.5, 'H', ha='center', va='center', fontsize=14, fontweight='bold')
        elif s in goal_states:
            ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor='#2ecc71', alpha=0.3))
            ax.text(c + 0.5, y + 0.5, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            arrow = action_arrows.get(policy[s], '?')
            max_q = np.max(q_table[s])
            ax.text(c + 0.5, y + 0.5, arrow, ha='center', va='center', fontsize=18)

    for i in range(nrow + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(ncol + 1):
        ax.axvline(j, color='gray', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f'Learned Policy ({env_name})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_q_rmse(q_tables_dict, q_star, env_name, save_path=None):
    """
    Bar chart showing Q value RMSE vs Q* for each algorithm.
    Lower RMSE means the learned Q table is closer to the true optimal values.
    Error bars show standard deviation across seeds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    algo_names = []
    rmse_means = []
    rmse_stds = []

    for algo_name, q_list in q_tables_dict.items():
        rmses = [np.sqrt(np.mean((q - q_star) ** 2)) for q in q_list]
        algo_names.append(algo_name)
        rmse_means.append(np.mean(rmses))
        rmse_stds.append(np.std(rmses))

    colors = [ALGO_COLORS.get(n, 'gray') for n in algo_names]
    bars = ax.bar(algo_names, rmse_means, yerr=rmse_stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('RMSE vs Q*')
    ax.set_title(f'Q-value Accuracy ({env_name})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_final_performance_table(all_env_results, save_path=None):
    """
    Multi panel bar chart showing final performance of all algorithms on all environments.
    Each sub plot is one environment. Bars show mean reward over last 100 episodes
    with 95% CI error bars.
    """
    envs = list(all_env_results.keys())
    algos = list(all_env_results[envs[0]].keys())
    n_envs = len(envs)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_envs, figsize=(4 * n_envs, 5), sharey=False)
    if n_envs == 1:
        axes = [axes]

    for idx, env_name in enumerate(envs):
        ax = axes[idx]
        results = all_env_results[env_name]
        means = []
        cis = []
        colors = []
        for algo in algos:
            data = results[algo]
            final_rewards = np.mean(data[:, -100:], axis=1)
            means.append(np.mean(final_rewards))
            cis.append(stats.sem(final_rewards) * stats.t.ppf(0.975, len(final_rewards) - 1))
            colors.append(ALGO_COLORS.get(algo, 'gray'))

        bars = ax.bar(range(n_algos), means, yerr=cis, capsize=4,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(n_algos))
        ax.set_xticklabels([a.replace(' ', '\n') for a in algos], fontsize=8)
        ax.set_title(env_name, fontsize=10)
        ax.set_ylabel('Mean Reward (last 100 eps)' if idx == 0 else '')

    plt.suptitle('Final Performance Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_policy_optimality(q_table, optimal_policy):
    """
    Returns what percent of states have the correct greedy action compared to optimal policy.
    100% means every state's best learned action matches the value iteration result.
    """
    learned_policy = np.argmax(q_table, axis=1)
    return np.mean(learned_policy == optimal_policy) * 100


def statistical_comparison(results, env_name):
    """
    Prints a summary table and pairwise Welch t tests for all algorithm pairs.

    Summary table shows mean reward, 95% CI, and std over last 100 episodes.
    Pairwise table shows t statistic, p value with significance stars, and Cohen d.
    Cohen d is the effect size: around 0.2 is small, 0.5 is medium, 0.8 is large.
    Significance stars: *** p<0.001, ** p<0.01, * p<0.05.
    """
    algos = list(results.keys())
    print(f"\n{'='*60}")
    print(f"Statistical Comparison: {env_name}")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Mean Reward':>12} {'95% CI':>15} {'Std':>10}")
    print(f"{'-'*60}")

    final_rewards = {}
    for algo in algos:
        data = results[algo]
        fr = np.mean(data[:, -100:], axis=1)
        final_rewards[algo] = fr
        mean = np.mean(fr)
        ci = stats.sem(fr) * stats.t.ppf(0.975, len(fr) - 1)
        std = np.std(fr)
        print(f"{algo:<20} {mean:>12.4f} {f'[{mean-ci:.4f}, {mean+ci:.4f}]':>15} {std:>10.4f}")

    print(f"\nPairwise t-tests (Welch's):")
    print(f"{'Comparison':<35} {'t-stat':>8} {'p-value':>10} {'Cohen d':>10}")
    print(f"{'-'*65}")
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = final_rewards[algos[i]], final_rewards[algos[j]]
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
            d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"{algos[i]} vs {algos[j]:<20} {t_stat:>8.3f} {p_val:>10.4f}{sig} {d:>10.3f}")
