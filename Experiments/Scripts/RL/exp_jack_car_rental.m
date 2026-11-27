% EXP_JACK_CAR_RENTAL Reinforcement Learning Experiment: Sutton & Barto Ex 4.2 & Ex 4.7
%
% This experiment solves Jack's Car Rental problem using Dynamic Programming
% Policy Iteration. It compares:
%   Part A: Example 4.2 (Original problem: symmetric costs, no free shuttle, no parking penalty)
%   Part B: Exercise 4.7 (Modified problem: employee free shuttle 1->2, $4 parking penalty for >10 cars)
%
% Verification & Research Philosophy:
%   - All policy matrices and value topographies are derived explicitly.
%   - Strict separation of environment physics from DP solver mathematics.

clear; clc; close all;

fprintf('=========================================================================\n');
fprintf('  OCEAN LIBRARY RL EXPERIMENT: JACK CAR RENTAL (EX 4.2 & EX 4.7)\n');
fprintf('=========================================================================\n\n');

%% PART A: Original Jack's Car Rental (Example 4.2)
fprintf('--- PART A: Solving Original Jack Car Rental (Example 4.2) ---\n');

env_orig = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv(...
    'MaxCars', 20, ...
    'MaxMove', 5, ...
    'RentReward', 10, ...
    'MoveCost', 2, ...
    'ParkingThreshold', 10, ...
    'ParkingPenalty', 0, ...
    'FreeShuttle', false ...
);

solver_orig = OceanStats.RL.DP.PolicyIterationSolver(env_orig, 'DiscountFactor', 0.9, 'Theta', 1e-4);

tic;
[V_orig, pi_orig, hist_orig] = solver_orig.run();
t_orig = toc;

fprintf('Example 4.2 Converged in %d policy iterations (%d evaluation sweeps total, %.3f s)\n', ...
    hist_orig.PolicySteps, sum(hist_orig.EvalSweeps), t_orig);
fprintf('  Max Value V(20,20) = %.2f\n', V_orig(21, 21));
fprintf('  Min Value V(0,0)   = %.2f\n\n', V_orig(1, 1));

% Render Visualization for Part A
fig_orig = plot_tabular_rl(V_orig, pi_orig, 'Title', "Example 4.2: Original Jack's Car Rental");

%% PART B: Modified Jack's Car Rental (Exercise 4.7)
fprintf('--- PART B: Solving Modified Jack Car Rental (Exercise 4.7) ---\n');

env_mod = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv(...
    'MaxCars', 20, ...
    'MaxMove', 5, ...
    'RentReward', 10, ...
    'MoveCost', 2, ...
    'ParkingThreshold', 10, ...
    'ParkingPenalty', 4, ...
    'FreeShuttle', true ...
);

solver_mod = OceanStats.RL.DP.PolicyIterationSolver(env_mod, 'DiscountFactor', 0.9, 'Theta', 1e-4);

tic;
[V_mod, pi_mod, hist_mod] = solver_mod.run();
t_mod = toc;

fprintf('Exercise 4.7 Converged in %d policy iterations (%d evaluation sweeps total, %.3f s)\n', ...
    hist_mod.PolicySteps, sum(hist_mod.EvalSweeps), t_mod);
fprintf('  Max Value V(20,20) = %.2f\n', V_mod(21, 21));
fprintf('  Min Value V(0,0)   = %.2f\n\n', V_mod(1, 1));

% Render Visualization for Part B
fig_mod = plot_tabular_rl(V_mod, pi_mod, 'Title', "Exercise 4.7: Modified Jack's Car Rental (Free Shuttle & Parking Penalty)");

%% PART C: Quantitative Policy Shift Analysis
fprintf('=========================================================================\n');
fprintf('  POLICY COMPARISON & METRICS SUMMARY\n');
fprintf('=========================================================================\n');

policy_diff = pi_mod - pi_orig;
num_actions_changed = sum(policy_diff(:) ~= 0);
pct_changed = (num_actions_changed / numel(pi_orig)) * 100;

fprintf('States with Policy Action Shift: %d / %d (%.1f%%)\n', ...
    num_actions_changed, numel(pi_orig), pct_changed);
fprintf('Average Action Shift: %.3f cars moved\n', mean(policy_diff(:)));
fprintf('Max Action Difference: %d cars\n', max(abs(policy_diff(:))));

% State Value Difference Summary
v_diff = V_mod - V_orig;
fprintf('Average Expected Value Reduction due to Parking Penalty: %.2f\n', mean(-v_diff(:)));
fprintf('Max Value Penalty at (20,20): %.2f\n', -v_diff(21, 21));

fprintf('\nExperiment Complete successfully!\n');
