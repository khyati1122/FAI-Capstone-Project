"""
Runs experiments 2, 3, and 4 from the project root:
    python3 run_experiments.py

Experiment 2: Compares all four algorithms across seven environments with 30 seeds.
Experiment 3: Sweeps alpha, gamma, and epsilon decay to analyze hyperparameter sensitivity.
Experiment 4: Tracks Q value overestimation to demonstrate maximization bias in Q learning.

All figures are saved to the results/ folder. Total runtime is around 45 to 60 minutes.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

from src.algorithms import QLearning, SARSA, ExpectedSARSA, DoubleQLearning, value_iteration
from src.environments import make_env, ENV_CONFIGS
from src.utils import (
    run_experiment, run_sensitivity_sweep, run_sensitivity_2d, run_bias_experiment,
    plot_convergence, plot_sensitivity_heatmap, plot_sensitivity_1d,
    plot_bias_comparison, plot_policy_grid, plot_q_rmse,
    plot_final_performance_table, compute_policy_optimality,
    statistical_comparison, train_agent, compute_confidence_interval, smooth,
    ALGO_COLORS
)

sns.set_theme(style='whitegrid', font_scale=1.1)
os.makedirs('results', exist_ok=True)

ALGOS = ['Q-learning', 'SARSA', 'Expected SARSA', 'Double Q-learning']
N_SEEDS = 30

EXP2_CONFIGS = {
    'CustomGrid-4x4':      {'n_episodes': 1500, 'max_steps': 100,  'alpha': 0.1,  'gamma': 0.99, 'epsilon_decay': 0.995},
    'FrozenLake-4x4-det':  {'n_episodes': 1500, 'max_steps': 100,  'alpha': 0.1,  'gamma': 0.99, 'epsilon_decay': 0.995},
    'FrozenLake-4x4-slip': {'n_episodes': 3000, 'max_steps': 100,  'alpha': 0.05, 'gamma': 0.99, 'epsilon_decay': 0.999},
    'FrozenLake-8x8-det':  {'n_episodes': 3000, 'max_steps': 200,  'alpha': 0.1,  'gamma': 0.99, 'epsilon_decay': 0.998},
    'FrozenLake-8x8-slip': {'n_episodes': 5000, 'max_steps': 200,  'alpha': 0.05, 'gamma': 0.99, 'epsilon_decay': 0.9995},
    'CliffWalking':        {'n_episodes': 500,  'max_steps': 200,  'alpha': 0.1,  'gamma': 0.99, 'epsilon_decay': 0.99},
    'Taxi':                {'n_episodes': 2000, 'max_steps': 200,  'alpha': 0.1,  'gamma': 0.99, 'epsilon_decay': 0.998},
}


def run_exp2():
    """
    Runs all four algorithms on all seven environments with 30 seeds each.

    For each environment: trains all algorithms, saves convergence plots,
    saves Q RMSE bar charts, prints statistical comparison tables, saves
    policy grid plots for CliffWalking, and saves a final summary figure.

    Returns all_results, all_q_tables, and all_q_stars dictionaries.
    """
    print('\n' + '='*60)
    print('EXPERIMENT 2: Four-Algorithm Comparison')
    print('='*60)

    all_results  = {}
    all_q_tables = {}
    all_q_stars  = {}

    for env_name, cfg in EXP2_CONFIGS.items():
        t0 = time.time()
        print(f'\n  Running {env_name}...', flush=True)

        results, q_tables, q_star = run_experiment(
            env_name, ALGOS,
            n_episodes=cfg['n_episodes'],
            n_seeds=N_SEEDS,
            max_steps=cfg['max_steps'],
            alpha=cfg['alpha'],
            gamma=cfg['gamma'],
            epsilon_decay=cfg['epsilon_decay'],
        )
        all_results[env_name]  = results
        all_q_tables[env_name] = q_tables
        all_q_stars[env_name]  = q_star
        print(f'  Done in {time.time() - t0:.1f}s', flush=True)

        window = 100 if 'slip' in env_name or '8x8' in env_name else 50
        plot_convergence(results, env_name, window=window,
                         save_path=f'results/convergence_{env_name}.png')
        plt.close('all')

        if q_star is not None:
            plot_q_rmse(q_tables, q_star, env_name,
                        save_path=f'results/q_rmse_{env_name}.png')
            plt.close('all')

        statistical_comparison(results, env_name)

    cliff_q = all_q_tables.get('CliffWalking', {})
    if cliff_q:
        cliff_holes = set(range(37, 47))
        goal = {47}
        for algo_name in ['Q-learning', 'SARSA']:
            if algo_name in cliff_q:
                plot_policy_grid(
                    cliff_q[algo_name][0], 'CliffWalking',
                    nrow=4, ncol=12,
                    title=f'{algo_name} Policy (CliffWalking)',
                    hole_states=cliff_holes,
                    goal_states=goal,
                    save_path=f'results/policy_cliff_{algo_name.replace(" ","_")}.png',
                )
                plt.close('all')

    plot_final_performance_table(all_results, save_path='results/final_performance_summary.png')
    plt.close('all')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    summary_envs = [
        'FrozenLake-4x4-det', 'FrozenLake-4x4-slip', 'CliffWalking',
        'FrozenLake-8x8-det', 'Taxi', 'CustomGrid-4x4',
    ]
    for idx, env_name in enumerate(summary_envs):
        if env_name not in all_results:
            continue
        ax = axes[idx // 3, idx % 3]
        results = all_results[env_name]
        window = 100 if 'slip' in env_name or '8x8' in env_name else 50
        for algo_name, data in results.items():
            mean, ci = compute_confidence_interval(data)
            sm_mean = smooth(mean, window)
            x = np.arange(len(sm_mean))
            color = ALGO_COLORS.get(algo_name, 'gray')
            ax.plot(x, sm_mean, label=algo_name, color=color, linewidth=1.2)
        ax.set_title(env_name, fontsize=11)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        if idx == 0:
            ax.legend(fontsize=8, frameon=True)

    plt.suptitle('TD Control Algorithm Comparison Across Environments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/summary_all_envs.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"\n{'Environment':<25} {'Algorithm':<20} {'Policy Opt %':>12}")
    print('=' * 60)
    for env_name in all_results:
        q_star = all_q_stars[env_name]
        if q_star is not None:
            opt_policy = np.argmax(q_star, axis=1)
            for algo_name in ALGOS:
                q_list = all_q_tables[env_name][algo_name]
                opt_rates = [compute_policy_optimality(q, opt_policy) for q in q_list]
                print(f'{env_name:<25} {algo_name:<20} {np.mean(opt_rates):>9.1f}% +/- {np.std(opt_rates):.1f}')

    print('\n' + '='*90)
    print('SUMMARY: Mean Reward (last 100 episodes)')
    print('='*90)
    header = f'{"Environment":<25}'
    for algo in ALGOS:
        header += f'{algo:>16}'
    print(header)
    print('-'*90)
    for env_name, results in all_results.items():
        row = f'{env_name:<25}'
        for algo in ALGOS:
            final = np.mean(results[algo][:, -100:])
            row += f'{final:>16.4f}'
        print(row)

    return all_results, all_q_tables, all_q_stars


def run_exp3():
    """
    Sweeps alpha, gamma, and epsilon decay to analyze hyperparameter sensitivity.

    Generates 12 alpha x gamma heatmaps (4 algorithms x 3 environments) and
    1D sensitivity curves for alpha and epsilon decay on 3 environments.
    Uses 10 seeds per configuration to keep runtime reasonable.
    """
    print('\n' + '='*60)
    print('EXPERIMENT 3: Sensitivity Analysis')
    print('='*60)

    ALPHA_VALUES         = [0.01, 0.05, 0.1, 0.2, 0.5]
    GAMMA_VALUES         = [0.8, 0.9, 0.95, 0.99]
    EPSILON_DECAY_VALUES = [0.99, 0.995, 0.999, 0.9999]

    SENS_ENVS  = ['FrozenLake-4x4-det', 'FrozenLake-4x4-slip', 'CliffWalking']
    SENS_SEEDS = 10

    for env_name in SENS_ENVS:
        n_eps = min(EXP2_CONFIGS[env_name]['n_episodes'], 1500)
        max_s = EXP2_CONFIGS[env_name]['max_steps']

        for algo_name in ALGOS:
            t0 = time.time()
            print(f'  alpha x gamma: {algo_name} / {env_name}...', end='', flush=True)
            grid = run_sensitivity_2d(
                env_name, algo_name,
                'alpha', ALPHA_VALUES,
                'gamma', GAMMA_VALUES,
                n_episodes=n_eps, n_seeds=SENS_SEEDS, max_steps=max_s,
            )
            plot_sensitivity_heatmap(
                grid, 'alpha', ALPHA_VALUES, 'gamma', GAMMA_VALUES,
                algo_name, env_name,
                save_path=f'results/sens_alpha_gamma_{algo_name.replace(" ","_")}_{env_name}.png',
            )
            plt.close('all')
            print(f' {time.time()-t0:.0f}s', flush=True)

    for env_name in SENS_ENVS:
        n_eps = min(EXP2_CONFIGS[env_name]['n_episodes'], 1500)
        max_s = EXP2_CONFIGS[env_name]['max_steps']
        alpha_results = {}
        for algo_name in ALGOS:
            means, stds = run_sensitivity_sweep(
                env_name, algo_name, 'alpha', ALPHA_VALUES,
                n_episodes=n_eps, n_seeds=SENS_SEEDS, max_steps=max_s,
            )
            alpha_results[algo_name] = (means, stds)
        plot_sensitivity_1d(alpha_results, 'alpha', ALPHA_VALUES, env_name,
                            save_path=f'results/sens_alpha_{env_name}.png')
        plt.close('all')

    for env_name in SENS_ENVS:
        n_eps = min(EXP2_CONFIGS[env_name]['n_episodes'], 1500)
        max_s = EXP2_CONFIGS[env_name]['max_steps']
        decay_results = {}
        for algo_name in ALGOS:
            means, stds = run_sensitivity_sweep(
                env_name, algo_name, 'epsilon_decay', EPSILON_DECAY_VALUES,
                n_episodes=n_eps, n_seeds=SENS_SEEDS, max_steps=max_s,
            )
            decay_results[algo_name] = (means, stds)
        plot_sensitivity_1d(decay_results, 'epsilon_decay', EPSILON_DECAY_VALUES, env_name,
                            save_path=f'results/sens_eps_decay_{env_name}.png')
        plt.close('all')


def run_exp4():
    """
    Measures Q value overestimation for Q learning and Double Q learning.

    Q learning overestimates because max over noisy Q values is always higher
    than the true max. Double Q learning fixes this by using separate tables
    for action selection and evaluation. Epsilon is fixed at 0.1 throughout
    so exploration does not confound the bias signal.

    Prints final overestimation stats and p values, saves bias plot per environment.
    """
    print('\n' + '='*60)
    print('EXPERIMENT 4: Maximization Bias')
    print('='*60)

    BIAS_ENVS = ['FrozenLake-4x4-slip', 'FrozenLake-4x4-det', 'CliffWalking']

    for env_name in BIAS_ENVS:
        t0 = time.time()
        print(f'  Bias: {env_name}...', end='', flush=True)

        cfg = EXP2_CONFIGS[env_name]
        n_eps = min(cfg['n_episodes'], 1500)

        q_overest, dq_overest, q_star = run_bias_experiment(
            env_name,
            n_episodes=n_eps,
            n_seeds=N_SEEDS,
            alpha=cfg['alpha'],
            gamma=0.99,
            epsilon=0.1,
            epsilon_min=0.1,
            epsilon_decay=1.0,
            max_steps=cfg['max_steps'],
        )

        plot_bias_comparison(q_overest, dq_overest, env_name,
                             save_path=f'results/bias_{env_name}.png')
        plt.close('all')

        q_final  = np.mean(q_overest[:, -100:], axis=1)
        dq_final = np.mean(dq_overest[:, -100:], axis=1)
        t_stat, p_val = stats.ttest_ind(q_final, dq_final)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        print(f' {time.time()-t0:.0f}s')
        print(f'    Q learn overest:  {np.mean(q_final):.4f} +/- {np.std(q_final):.4f}')
        print(f'    Double Q overest: {np.mean(dq_final):.4f} +/- {np.std(dq_final):.4f}')
        print(f'    t={t_stat:.3f}, p={p_val:.6f} {sig}')


if __name__ == '__main__':
    total_start = time.time()

    print('CS5100 Capstone: Empirical Comparison of TD Control Algorithms')
    print('Ihika Narayana Reddy Gari and Khyati Nirenkumar Amin')
    print('='*60)

    run_exp2()
    run_exp3()
    run_exp4()

    total_time = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'ALL EXPERIMENTS COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)')
    print(f'Results saved to results/')
    print(f'{"="*60}')
