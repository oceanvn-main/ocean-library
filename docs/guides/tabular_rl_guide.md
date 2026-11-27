# Tabular Reinforcement Learning & Dynamic Programming Guide

## Overview

The `OceanStats.RL` package provides a decoupled, modular framework for solving tabular Markov Decision Processes (MDPs) using Dynamic Programming (DP) algorithms such as Policy Iteration and Value Iteration.

The core architecture strictly separates domain physics from mathematical solver routines:
1. `TabularEnvironment`: Abstract base class specifying the environment contract (`get_all_states`, `get_valid_actions`, `get_transitions`).
2. `ModifiedCarRentalEnv`: Concrete subclass implementing Jack's Car Rental problem (Sutton & Barto Example 4.2 & Exercise 4.7).
3. `PolicyIterationSolver`: Pure mathematical solver executing exact Bellman expectation updates and greedy policy improvements.
4. `TabularRLStrategy` / `plot_tabular_rl`: OceanViz visualization engines rendering 3D state value surfaces and 2D policy heatmaps in dark theme aesthetics.

---

## Usage

### 1. Creating a Tabular MDP Environment

```matlab
% Create Jack's Car Rental environment (Exercise 4.7 configuration)
env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv(...
    'MaxCars', 20, ...
    'MaxMove', 5, ...
    'RentReward', 10, ...
    'MoveCost', 2, ...
    'ParkingThreshold', 10, ...
    'ParkingPenalty', 4, ...
    'FreeShuttle', true ...
);
```

### 2. Initializing & Running Policy Iteration

```matlab
% Create solver injecting the environment
solver = OceanStats.RL.DP.PolicyIterationSolver(env, ...
    'DiscountFactor', 0.9, ...
    'Theta', 1e-4 ...
);

% Run Policy Iteration loop (returns Value matrix and Policy matrix)
[V, pi, history] = solver.run();
```

### 3. Rendering OceanViz Visualizations

```matlab
% Plot 3D Value Surface and 2D Policy Heatmap
fig = plot_tabular_rl(solver, 'Title', 'Exercise 4.7 Solution');
```

---

## Examples

### Replicating Sutton & Barto Example 4.2 vs Exercise 4.7

```matlab
% Part A: Original Example 4.2 (Symmetric costs, no shuttle, no penalty)
env_orig = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('FreeShuttle', false, 'ParkingPenalty', 0);
solver_orig = OceanStats.RL.DP.PolicyIterationSolver(env_orig);
[V_orig, pi_orig] = solver_orig.run();

% Part B: Exercise 4.7 (Free shuttle 1->2, $4 penalty if cars > 10)
env_mod = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('FreeShuttle', true, 'ParkingPenalty', 4);
solver_mod = OceanStats.RL.DP.PolicyIterationSolver(env_mod);
[V_mod, pi_mod] = solver_mod.run();

% Compare policy changes
diff_matrix = pi_mod - pi_orig;
fprintf('Percentage of states with modified action: %.1f%%\n', 100 * sum(diff_matrix(:) ~= 0) / numel(pi_orig));
```

---

## See Also

* [Tabular RL Recipes](../recipes/tabular_rl_recipes.md)
* [Architecture Reference](../../.agents/architecture.md)
* [OceanViz Visualization Guide](visualization.md)
