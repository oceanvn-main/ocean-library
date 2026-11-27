import numpy as np
import matplotlib.pyplot as plt
import time

def run_simulation(num_runs=2000, num_steps=10000, num_arms=10, epsilon=0.1, alpha=0.1):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Pre-allocate arrays for step metrics
    rewards_sample_average = np.zeros(num_steps)
    optimal_pct_sample_average = np.zeros(num_steps)
    
    rewards_constant = np.zeros(num_steps)
    optimal_pct_constant = np.zeros(num_steps)
    
    # ----------------------------------------------------
    # Agent 1: Sample-Average (alpha_n = 1/n)
    # ----------------------------------------------------
    print("Running Sample-Average Agent...")
    start_time = time.time()
    
    # Initialize true action values to 0
    q_true = np.zeros((num_runs, num_arms))
    # Initialize estimates Q and counts N
    Q = np.zeros((num_runs, num_arms))
    N = np.zeros((num_runs, num_arms))
    
    for t in range(num_steps):
        # Identify current optimal action for each run
        optimal_actions = np.argmax(q_true, axis=1)
        
        # Select action using epsilon-greedy policy
        # We add tiny noise to Q to break ties randomly
        Q_noise = Q + np.random.randn(num_runs, num_arms) * 1e-12
        greedy_actions = np.argmax(Q_noise, axis=1)
        random_actions = np.random.randint(0, num_arms, size=num_runs)
        explore = np.random.rand(num_runs) < epsilon
        actions = np.where(explore, random_actions, greedy_actions)
        
        # Get rewards from N(q_*(a), 1)
        rewards = np.random.normal(q_true[np.arange(num_runs), actions], 1.0)
        
        # Update estimates with sample-average step-size (1/n)
        N[np.arange(num_runs), actions] += 1
        n_counts = N[np.arange(num_runs), actions]
        Q[np.arange(num_runs), actions] += (rewards - Q[np.arange(num_runs), actions]) / n_counts
        
        # Record stats
        rewards_sample_average[t] = np.mean(rewards)
        optimal_pct_sample_average[t] = np.mean(actions == optimal_actions) * 100
        
        # Environment drift: q_*(a) <- q_*(a) + N(0, 0.01^2)
        q_true += np.random.normal(0, 0.01, size=(num_runs, num_arms))
        
    print(f"Sample-Average finished in {time.time() - start_time:.2f} seconds.")
    
    # ----------------------------------------------------
    # Agent 2: Constant Step-Size (alpha = 0.1)
    # ----------------------------------------------------
    print("Running Constant Step-Size Agent...")
    start_time = time.time()
    
    # Reset environment and estimates
    q_true = np.zeros((num_runs, num_arms))
    Q = np.zeros((num_runs, num_arms))
    
    for t in range(num_steps):
        # Identify current optimal action
        optimal_actions = np.argmax(q_true, axis=1)
        
        # Select action using epsilon-greedy policy
        Q_noise = Q + np.random.randn(num_runs, num_arms) * 1e-12
        greedy_actions = np.argmax(Q_noise, axis=1)
        random_actions = np.random.randint(0, num_arms, size=num_runs)
        explore = np.random.rand(num_runs) < epsilon
        actions = np.where(explore, random_actions, greedy_actions)
        
        # Get rewards
        rewards = np.random.normal(q_true[np.arange(num_runs), actions], 1.0)
        
        # Update estimates with constant step-size (alpha)
        Q[np.arange(num_runs), actions] += alpha * (rewards - Q[np.arange(num_runs), actions])
        
        # Record stats
        rewards_constant[t] = np.mean(rewards)
        optimal_pct_constant[t] = np.mean(actions == optimal_actions) * 100
        
        # Environment drift
        q_true += np.random.normal(0, 0.01, size=(num_runs, num_arms))
        
    print(f"Constant Step-Size finished in {time.time() - start_time:.2f} seconds.")
    
    return (rewards_sample_average, optimal_pct_sample_average,
            rewards_constant, optimal_pct_constant)

def plot_results(rewards_sa, optimal_sa, rewards_c, optimal_c):
    # Set up matplotlib style for clean, professional aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    
    # Colors matching a professional, modern palette
    color_sa = '#D95319'  # Rust Red for Sample-Average
    color_c = '#0072BD'   # Ocean Blue for Constant Step-Size
    grid_color = '#F0F0F0'
    
    # Subplot 1: Average Reward
    axes[0].plot(rewards_c, label='Constant (alpha = 0.1)', color=color_c, linewidth=1.5)
    axes[0].plot(rewards_sa, label='Sample-Average (alpha = 1/n)', color=color_sa, linewidth=1.5)
    axes[0].set_xlabel('Steps', fontsize=10, fontweight='bold', labelpad=8)
    axes[0].set_ylabel('Average Reward', fontsize=10, fontweight='bold', labelpad=8)
    axes[0].set_title('Average Reward', fontsize=11, fontweight='bold', pad=10)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=9)
    axes[0].grid(True, linestyle='--', color=grid_color)
    axes[0].tick_params(colors='#555555')
    
    # Subplot 2: % Optimal Action
    axes[1].plot(optimal_c, label='Constant (alpha = 0.1)', color=color_c, linewidth=1.5)
    axes[1].plot(optimal_sa, label='Sample-Average (alpha = 1/n)', color=color_sa, linewidth=1.5)
    axes[1].set_xlabel('Steps', fontsize=10, fontweight='bold', labelpad=8)
    axes[1].set_ylabel('% Optimal Action', fontsize=10, fontweight='bold', labelpad=8)
    axes[1].set_ylim(0, 100)
    axes[1].set_title('% Optimal Action', fontsize=11, fontweight='bold', pad=10)
    axes[1].legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=9)
    axes[1].grid(True, linestyle='--', color=grid_color)
    axes[1].tick_params(colors='#555555')
    
    fig.suptitle('Exercise 2.5: Nonstationary 10-Armed Bandit Testbed', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = 'nonstationary_bandit_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Results successfully plotted and saved to: {plot_path}")

if __name__ == '__main__':
    rewards_sa, optimal_sa, rewards_c, optimal_c = run_simulation()
    plot_results(rewards_sa, optimal_sa, rewards_c, optimal_c)
