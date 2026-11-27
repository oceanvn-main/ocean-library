% EXP_DISCOUNTING_AWARE_CONTROL Scientific experiment for Sutton & Barto Exercise 5.14
%
% Benchmarks Standard Off-Policy MC Control (S&B p.111) vs
% Discounting-Aware Off-Policy MC Control (Exercise 5.14) on the Racetrack MDP
% with gamma = 0.95, measuring learning performance and return curves.
%
% Visualizations generated via OceanViz Engine (Dark Theme).
%
% Package: Experiments.Scripts.RL
% Reference: Sutton & Barto (2018) Chapter 5, Exercise 5.14

clc; clear; close all;

fprintf('========================================================================\n');
fprintf('  EXERCISE 5.14: DISCOUNTING-AWARE OFF-POLICY MC CONTROL EXPERIMENT\n');
fprintf('========================================================================\n\n');

% -------------------------------------------------------------------------
% 1. Environment & Setup
% -------------------------------------------------------------------------
env = OceanStats.RL.Env.Classic.RacetrackEnv("Track1", 'FailProb', 0.10);

num_episodes   = 50000;
check_interval = 2500;
gamma_val      = 0.95;

% -------------------------------------------------------------------------
% 2. Standard Off-Policy MC Control (Gamma = 0.95)
% -------------------------------------------------------------------------
fprintf('[1/2] Running Standard Off-Policy MC Control (Gamma = %.2f)...\n', gamma_val);
solver_std = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        "OffPolicyControl", ...
    'Gamma',            gamma_val,          ...
    'NumEpisodes',      num_episodes,       ...
    'CheckInterval',    check_interval,     ...
    'Epsilon',          0.10,               ...
    'DiscountingAware', false,              ...
    'Verbose',          true                ...
);
rng(42);
[~, ~, ~, hist_std] = solver_std.run();

% -------------------------------------------------------------------------
% 3. Discounting-Aware Off-Policy MC Control (Exercise 5.14, Gamma = 0.95)
% -------------------------------------------------------------------------
fprintf('\n[2/2] Running Discounting-Aware Off-Policy MC Control (Gamma = %.2f)...\n', gamma_val);
solver_da = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',        "DiscountingAwareControl", ...
    'Gamma',            gamma_val,                 ...
    'NumEpisodes',      num_episodes,              ...
    'CheckInterval',    check_interval,            ...
    'Epsilon',          0.10,                      ...
    'DiscountingAware', true,                      ...
    'Verbose',          true                       ...
);
rng(42);
[~, ~, ~, hist_da] = solver_da.run();

% -------------------------------------------------------------------------
% 4. Visualization using OceanViz Engine
% -------------------------------------------------------------------------
fprintf('\n[3/3] Generating Visualizations via OceanViz Engine...\n');
viz = OceanViz.Core('Theme', 'Dark');

fig = figure('Name', 'Exercise 5.14 Discounting-Aware Control Comparison', ...
    'Position', [100, 100, 850, 500], 'Visible', 'off');
ax  = axes('Parent', fig);

x_pts = hist_std.CheckPoints(:);
y_std = hist_std.AvgReturn(:);
y_da  = hist_da.AvgReturn(:);

data = struct();
data.X = x_pts;
data.Lines = { ...
    struct('Y', y_std, 'Name', 'Standard Off-Policy Control (S&B p.111)'), ...
    struct('Y', y_da,  'Name', 'Discounting-Aware Control (Exercise 5.14)') ...
};

viz.plot(ax, 'LearningCurve', data, ...
    'Title',      'Exercise 5.14: Off-Policy MC Control on Racetrack (\gamma = 0.95)', ...
    'XLabel',     'Episodes', ...
    'YLabel',     'Average Sample Return', ...
    'ShowLegend', true ...
);

% Save figure to artifact directory
artifact_dir = 'C:\Users\USER\.gemini\antigravity-ide\brain\228de5d4-0017-45c6-a208-2755b9e45fe2';
if ~exist(artifact_dir, 'dir')
    mkdir(artifact_dir);
end
out_fig_path = fullfile(artifact_dir, 'discounting_aware_control_racetrack.png');
exportgraphics(fig, out_fig_path, 'Resolution', 300);
close(fig);

fprintf('      Saved comparison plot to: %s\n', out_fig_path);
fprintf('\n========================================================================\n');
fprintf('  DISCOUNTING-AWARE CONTROL EXPERIMENT COMPLETE\n');
fprintf('========================================================================\n');
