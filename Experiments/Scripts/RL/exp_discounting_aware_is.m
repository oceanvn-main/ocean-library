% EXP_DISCOUNTING_AWARE_IS Scientific Experiment Script for Sutton & Barto Section 5.8
%
% Problem Statement:
%   Sutton & Barto (2nd Ed.), Section 5.8: Discounting-aware Importance Sampling.
%   Demonstrates variance reduction for off-policy prediction when gamma < 1
%   on long-horizon episodes (T = 100).
%
% MLOps Medallion Pipeline Stage: Gold / Experiment
% Package: OceanStats.RL.MonteCarlo / Experiments.Scripts.RL
% Reference: Sutton & Barto (2018) Section 5.8, Equations 5.9 - 5.10

clear; clc; close all;
addpath(pwd);
rehash path;

fprintf('========================================================================\n');
fprintf('  SECTION 5.8: DISCOUNTING-AWARE IMPORTANCE SAMPLING EXPERIMENT\n');
fprintf('========================================================================\n\n');

%% 1. Initialize Environment & Parameters
env = OceanStats.RL.Env.Classic.BlackjackEnv();
gamma = 0.90;
num_episodes = 50000;

target_pi   = @(s_idx, a_idx) double(a_idx == 1);   % Always stick (action 1)
behavior_b  = @(s_idx, a_idx) 0.5;                  % Uniform random (0.5, 0.5)

fprintf('[1/3] Running Standard Off-Policy Prediction (Gamma = %.2f)...\n', gamma);
solver_std = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        "OffPolicyPrediction", ...
    'Gamma',            gamma,                 ...
    'NumEpisodes',      num_episodes,          ...
    'TargetPolicy',     target_pi,             ...
    'BehaviorPolicy',   behavior_b,            ...
    'DiscountingAware', false,                 ...
    'CheckInterval',    5000,                  ...
    'Verbose',          true                   ...
);
[V_std, ~, ~, hist_std] = solver_std.run();

fprintf('[2/3] Running Discounting-Aware Off-Policy Prediction (Gamma = %.2f)...\n', gamma);
solver_da = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        "OffPolicyPrediction", ...
    'Gamma',            gamma,                 ...
    'NumEpisodes',      num_episodes,          ...
    'TargetPolicy',     target_pi,             ...
    'BehaviorPolicy',   behavior_b,            ...
    'DiscountingAware', true,                  ...
    'CheckInterval',    5000,                  ...
    'Verbose',          true                   ...
);
[V_da, ~, ~, hist_da] = solver_da.run();

%% 3. Visualization using OceanViz Engine (Dark Theme)
fprintf('\n[3/3] Generating Visualizations via OceanViz Engine...\n');
viz = OceanViz.Core('Theme', 'Dark');

fig = viz.createFigure('Section 5.8 - Discounting-Aware IS Variance Comparison');
ax  = axes(fig);
viz.styleAxes(ax);

checkpoints = hist_std.CheckPoints;
ret_std     = hist_std.AvgReturn;
ret_da      = hist_da.AvgReturn;

plot(ax, checkpoints, ret_std, '-o', ...
    'LineWidth', 2.0, ...
    'Color', [0.9, 0.3, 0.3], ...
    'MarkerSize', 5, ...
    'MarkerFaceColor', [0.9, 0.3, 0.3], ...
    'DisplayName', 'Standard Off-Policy IS');

hold(ax, 'on');

plot(ax, checkpoints, ret_da, '-s', ...
    'LineWidth', 2.0, ...
    'Color', viz.Colors.Primary, ...
    'MarkerSize', 5, ...
    'MarkerFaceColor', viz.Colors.Primary, ...
    'DisplayName', 'Discounting-Aware IS (S&B Eq 5.10)');

title(ax, sprintf('Discounting-Aware IS vs Standard IS (Gamma = %.2f)', gamma), ...
    'Color', viz.Colors.Text, 'FontSize', 14);
xlabel(ax, 'Training Episodes', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax, 'Average Episode Return', 'Color', viz.Colors.Text, 'FontSize', 12);

lgd = legend(ax, 'Location', 'northeast');
set(lgd, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

artifact_dir = "C:\Users\USER\.gemini\antigravity-ide\brain\228de5d4-0017-45c6-a208-2755b9e45fe2";
if exist(artifact_dir, 'dir')
    exportgraphics(fig, fullfile(artifact_dir, 'discounting_aware_is_variance.png'));
    fprintf('      Saved figures to artifact directory.\n');
end

fprintf('\n========================================================================\n');
fprintf('  DISCOUNTING-AWARE IS EXPERIMENT COMPLETE\n');
fprintf('========================================================================\n');
