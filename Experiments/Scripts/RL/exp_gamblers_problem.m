% EXP_GAMBLERS_PROBLEM Gambler's Problem Value Iteration Experiment (Ex 4.9)
%
% Solves the Gambler's Problem (Sutton & Barto Example 4.3 & Exercise 4.9) using
% Dynamic Programming Value Iteration.
%
% Evaluates three coin win probability scenarios:
%   Part A: ph = 0.40 (Example 4.3 benchmark: replicates Sutton & Barto Fig 4.3)
%   Part B: ph = 0.25 (Exercise 4.9 part 1: subfair coin, aggressive multi-tier betting)
%   Part C: ph = 0.55 (Exercise 4.9 part 2: favorable coin, conservative minimal stake a=1)
%
% Architecture & Design Principles:
%   - All problem logic, state vectors, sweep tracking, tie-break policy extraction,
%     and dark-mode OceanViz plots are implemented directly in this scientific script.
%   - Uses generic tabular Value Iteration concepts.

clc; close all; clear;
fprintf('=========================================================================\n');
fprintf('  OCEAN LIBRARY RL EXPERIMENT: GAMBLER''S PROBLEM VALUE ITERATION (EX 4.9)\n');
fprintf('=========================================================================\n\n');

%% Configuration & Environment Parameters
goal = 100;
theta = 1e-9;
tracked_sweeps = [1, 2, 3, 32];

% Initialize OceanViz Core for Dark Theme Styling
viz = OceanViz.Core('Theme', 'Dark');
theme = viz.Colors;

%% PART A: Benchmark Scenario (ph = 0.40 - Replicating Sutton & Barto Fig 4.3)
fprintf('--- PART A: Solving Benchmark Gambler''s Problem (ph = 0.40) ---\n');
res_A = run_gamblers_value_iteration(0.40, goal, theta, tracked_sweeps);

fprintf('ph = 0.40 Converged in %d sweeps (Max Delta < %.1e)\n', res_A.sweeps, theta);
fprintf('  Max Value V(99) = %.4f, V(50) = %.4f\n\n', res_A.V(100), res_A.V(51));

fig_A = render_gamblers_figure(res_A, "Gambler's Problem (ph = 0.40)", theme);

%% PART B: Subfair Coin Scenario (ph = 0.25 - Exercise 4.9 Part 1)
fprintf('--- PART B: Solving Subfair Coin Scenario (ph = 0.25) ---\n');
res_B = run_gamblers_value_iteration(0.25, goal, theta, tracked_sweeps);

fprintf('ph = 0.25 Converged in %d sweeps (Max Delta < %.1e)\n', res_B.sweeps, theta);
fprintf('  Max Value V(99) = %.4f, V(50) = %.4f\n\n', res_B.V(100), res_B.V(51));

fig_B = render_gamblers_figure(res_B, "Subfair Coin (ph = 0.25)", theme);

%% PART C: Favorable Coin Scenario (ph = 0.55 - Exercise 4.9 Part 2)
fprintf('--- PART C: Solving Favorable Coin Scenario (ph = 0.55) ---\n');
res_C = run_gamblers_value_iteration(0.55, goal, theta, tracked_sweeps);

fprintf('ph = 0.55 Converged in %d sweeps (Max Delta < %.1e)\n', res_C.sweeps, theta);
fprintf('  Max Value V(99) = %.4f, V(50) = %.4f\n\n', res_C.V(100), res_C.V(51));

fig_C = render_gamblers_figure(res_C, "Favorable Coin (ph = 0.55)", theme);

%% PART D: Comparative Quantitative Summary
fprintf('=========================================================================\n');
fprintf('  GAMBLER''S PROBLEM COMPARATIVE SUMMARY\n');
fprintf('=========================================================================\n');
fprintf('%-15s | %-12s | %-12s | %-15s | %-20s\n', ...
    'Coin Prob (ph)', 'Sweeps', 'V(50)', 'Max Stake a', 'Policy Characteristic');
fprintf('-------------------------------------------------------------------------\n');

desc_A = 'Pyramid Peaks (25, 50, 75)';
desc_B = 'Aggressive Multi-Tier';
desc_C = 'Conservative Minimal (a=1)';

fprintf('%-15.2f | %-12d | %-12.4f | %-15d | %-20s\n', 0.40, res_A.sweeps, res_A.V(51), max(res_A.policy), desc_A);
fprintf('%-15.2f | %-12d | %-12.4f | %-15d | %-20s\n', 0.25, res_B.sweeps, res_B.V(51), max(res_B.policy), desc_B);
fprintf('%-15.2f | %-12d | %-12.4f | %-15d | %-20s\n', 0.55, res_C.sweeps, res_C.V(51), max(res_C.policy), desc_C);

fprintf('\nExperiment Complete successfully!\n');

%% Helper Functions for Gambler's Problem Engine & Visualization

function res = run_gamblers_value_iteration(ph, goal, theta, tracked_sweeps)
    % RUN_GAMBLERS_VALUE_ITERATION Value Iteration engine for Gambler's Problem.
    
    num_states = goal + 1; % 0 to 100 capital states (1-based index: 1 to 101)
    V = zeros(num_states, 1);
    
    % Boundary conditions: V(0) = 0, V(goal) = 1.0
    V(1) = 0.0;
    V(num_states) = 1.0;
    
    sweep_history = struct('Sweeps', [], 'Labels', {{}}, 'Values', {{}});
    
    sweep_count = 0;
    while true
        sweep_count = sweep_count + 1;
        delta = 0;
        
        v_old = V;
        
        % Bellman Optimality Update over active capital states s in 1..99
        for s = 1:(goal - 1)
            s_idx = s + 1;
            
            % Valid action stakes a in 1..min(s, goal - s)
            max_stake = min(s, goal - s);
            actions = 1:max_stake;
            num_act = length(actions);
            
            q_vals = zeros(num_act, 1);
            for a_idx = 1:num_act
                a = actions(a_idx);
                % Expected return: ph * V(s + a) + (1 - ph) * V(s - a)
                q_vals(a_idx) = ph * v_old(s + a + 1) + (1 - ph) * v_old(s - a + 1);
            end
            
            v_best = max(q_vals);
            delta = max(delta, abs(v_best - v_old(s_idx)));
            V(s_idx) = v_best;
        end
        
        % Ensure terminal boundaries remain immutable
        V(1) = 0.0;
        V(num_states) = 1.0;
        
        % Cache milestone sweeps
        if ismember(sweep_count, tracked_sweeps)
            idx = numel(sweep_history.Sweeps) + 1;
            sweep_history.Sweeps(idx) = sweep_count;
            sweep_history.Labels{idx} = sprintf('sweep %d', sweep_count);
            sweep_history.Values{idx} = V;
        end
        
        if delta < theta
            break;
        end
    end
    
    % Cache final converged sweep
    idx = numel(sweep_history.Sweeps) + 1;
    sweep_history.Sweeps(idx) = sweep_count;
    sweep_history.Labels{idx} = 'Final value function';
    sweep_history.Values{idx} = V;
    
    % Deterministic Policy Extraction with floating-point tie tolerance
    policy = zeros(goal - 1, 1);
    tie_tol = 1e-9;
    
    for s = 1:(goal - 1)
        max_stake = min(s, goal - s);
        actions = 1:max_stake;
        num_act = length(actions);
        
        q_vals = zeros(num_act, 1);
        for a_idx = 1:num_act
            a = actions(a_idx);
            q_vals(a_idx) = ph * V(s + a + 1) + (1 - ph) * V(s - a + 1);
        end
        
        max_q = max(q_vals);
        % Find all action stakes within tolerance of max Q
        tied_actions = actions(abs(q_vals - max_q) <= tie_tol);
        
        % Minimum stake tie-breaker yields classic Figure 4.3 pyramid
        policy(s) = min(tied_actions);
    end
    
    res.Ph = ph;
    res.Goal = goal;
    res.V = V;
    res.policy = policy;
    res.sweeps = sweep_count;
    res.history = sweep_history;
end

function fig = render_gamblers_figure(res, title_str, theme)
    % RENDER_GAMBLERS_FIGURE Renders Figure 4.3 styled 2-panel dark-mode figure.
    
    fig = figure('Name', title_str, 'Color', theme.Background, 'Position', [150, 150, 900, 700]);
    
    capital_range = 1:(res.Goal - 1);
    
    % Panel 1: Value Function Sweeps
    ax1 = subplot(2, 1, 1, 'Parent', fig);
    set(ax1, 'Color', theme.Background, 'XColor', theme.Text, 'YColor', theme.Text, 'GridColor', theme.Grid);
    hold(ax1, 'on');
    
    neon_colors = [
        0.00, 0.90, 1.00; % Cyan (sweep 1)
        1.00, 0.20, 0.60; % Magenta (sweep 2)
        0.00, 1.00, 0.40; % Electric Green (sweep 3)
        1.00, 0.60, 0.00; % Orange (sweep 32)
        1.00, 1.00, 0.00  % Yellow (Final)
    ];
    
    sweeps_data = res.history;
    num_cached = length(sweeps_data.Sweeps);
    
    for k = 1:num_cached
        v_sweep = sweeps_data.Values{k};
        v_active = v_sweep(2:end-1);
        col_idx = mod(k - 1, size(neon_colors, 1)) + 1;
        
        if k == num_cached
            plot(ax1, capital_range, v_active, 'Color', neon_colors(col_idx, :), ...
                'LineWidth', 2.0, 'DisplayName', sweeps_data.Labels{k});
        else
            plot(ax1, capital_range, v_active, 'Color', [neon_colors(col_idx, :) 0.85], ...
                'LineWidth', 1.3, 'DisplayName', sweeps_data.Labels{k});
        end
    end
    
    grid(ax1, 'on');
    xlim(ax1, [1, res.Goal - 1]);
    ylim(ax1, [0, 1.05]);
    xlabel(ax1, 'Capital', 'Color', theme.Text, 'FontSize', 11);
    ylabel(ax1, 'Value Estimates', 'Color', theme.Text, 'FontSize', 11);
    title(ax1, title_str, 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');
    legend(ax1, 'Location', 'southeast', 'TextColor', theme.Text, 'Color', [theme.Background 0.8], 'EdgeColor', 'none');
    hold(ax1, 'off');
    
    % Panel 2: Final Policy (Stake)
    ax2 = subplot(2, 1, 2, 'Parent', fig);
    set(ax2, 'Color', theme.Background, 'XColor', theme.Text, 'YColor', theme.Text, 'GridColor', theme.Grid);
    hold(ax2, 'on');
    
    h_stem = stem(ax2, capital_range, res.policy, 'filled', ...
        'Color', [0.00, 0.90, 1.00], 'MarkerFaceColor', [1.00, 0.20, 0.60], ...
        'MarkerSize', 3.5, 'LineWidth', 1.0);
    h_stem.DisplayName = 'Optimal Stake a = \pi(s)';
    
    grid(ax2, 'on');
    xlim(ax2, [1, res.Goal - 1]);
    ylim(ax2, [0, max(max(res.policy) + 2, 5)]);
    xlabel(ax2, 'Capital', 'Color', theme.Text, 'FontSize', 11);
    ylabel(ax2, 'Final Policy (Stake)', 'Color', theme.Text, 'FontSize', 11);
    title(ax2, sprintf('Final Policy (Stake) for ph = %.2f', res.Ph), 'Color', theme.Text, 'FontSize', 12, 'FontWeight', 'bold');
    legend(ax2, 'Location', 'northeast', 'TextColor', theme.Text, 'Color', [theme.Background 0.8], 'EdgeColor', 'none');
    hold(ax2, 'off');
end
