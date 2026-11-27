% EXP_RACETRACK_OFFPOLICY_MC Scientific Experiment Script for Exercise 5.12
%
% Problem Statement:
%   Sutton & Barto (2nd Ed.), Exercise 5.12: Racetrack with Off-Policy Monte Carlo Control.
%   Learn optimal driving policy on a discrete grid racetrack with momentum:
%     State:  S_t = (x_t, y_t, v_{x,t}, v_{y,t}), v_x, v_y in {0,1,2,3,4}
%     Action: A_t = (a_x, a_y) in {-1, 0, +1}^2  (9 discrete accelerations)
%     Reward: R_{t+1} = -1 per step until finish line is crossed
%     Noise:  10% probability of acceleration failure (realized a = [0,0])
%
% MLOps Medallion Pipeline Stage: Gold / Experiment
% Package: OceanStats.RL.MonteCarlo / Experiments.Scripts.RL
% Reference: Sutton & Barto (2018) Exercise 5.12, Sections 5.6 - 5.7

clear; clc; close all;
addpath(pwd);
rehash path;

fprintf('========================================================================\n');
fprintf('  EXERCISE 5.12: RACETRACK WITH OFF-POLICY MONTE CARLO CONTROL\n');
fprintf('========================================================================\n\n');

%% 1. Initialize Racetrack Environment & Monte Carlo Solver
track_name = "Track1";
fail_prob  = 0.1;   % 10% acceleration failure during training
num_episodes = 150000;

fprintf('[1/4] Initializing Environment (%s, FailProb = %.2f)...\n', track_name, fail_prob);
env = OceanStats.RL.Env.Classic.RacetrackEnv(track_name, 'FailProb', fail_prob);

fprintf('      Track Size: %d x %d | Total Valid States: %d | Start Cells: %d\n', ...
    env.Height, env.Width, env.TotalStates, size(env.StartCells, 1));

% Configure Off-Policy Monte Carlo Control Solver (Weighted Importance Sampling)
% Uses default Epsilon-Greedy behavior policy (Epsilon = 0.1) w.r.t Q-table
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',      "OffPolicyControl",     ...
    'Gamma',          1.0,                    ... % Undiscounted gamma = 1
    'NumEpisodes',    num_episodes,           ...
    'MaxEpisodeLen',  500,                    ...
    'Epsilon',        0.1,                    ... % Epsilon for epsilon-greedy behavior policy
    'InitialQValue',  -1000.0,                ... % Pessimistic initialization for negative return MDP
    'CheckInterval',  15000,                  ...
    'Verbose',        true                    ...
);

%% 2. Run Training Pipeline
fprintf('\n[2/4] Running Off-Policy MC Control Training (%d episodes)...\n', num_episodes);
tic;
[V, Q, pi_greedy, history] = solver.run();
elapsed_time = toc;
fprintf('      Training completed in %.2f seconds.\n', elapsed_time);

%% 3. Evaluate Optimal Target Policy (Noise Disabled P(fail) = 0)
fprintf('\n[3/4] Evaluating Learned Greedy Policy with Noise Disabled...\n');
env.set_fail_prob(0.0);  % Turn off noise for evaluation rollouts

n_starts = size(env.StartCells, 1);
trajectories = cell(n_starts, 1);
traj_lengths = zeros(n_starts, 1);

greedy_policy_fn = @(s, e) e.index_to_action(pi_greedy(e.state_to_index(s)));

for i = 1:n_starts
    start_pos = env.StartCells(i, :);
    env.reset();
    env.set_state([start_pos(1), start_pos(2), 0, 0]);

    ep = env.generate_episode(greedy_policy_fn, 'MaxLen', 500, 'ForceStartState', [start_pos(1), start_pos(2), 0, 0]);
    trajectories{i} = ep.States;  % (T x 4) matrix of [x, y, vx, vy]
    traj_lengths(i) = ep.Length;

    fprintf('      Start Cell (%2d, %2d) -> Finished in %3d steps | Final Velocity: (%d, %d)\n', ...
        start_pos(1), start_pos(2), ep.Length, ep.States(end, 3), ep.States(end, 4));
end

%% 4. Visualization using OceanViz Engine (Dark Theme)
fprintf('\n[4/4] Generating Visualizations via OceanViz Engine...\n');

viz = OceanViz.Core('Theme', 'Dark');

% Figure 1: Convergence History
fig1 = viz.createFigure('Exercise 5.12 - Convergence History');
ax1 = axes(fig1);
viz.styleAxes(ax1);

checkpoints = history.CheckPoints;
avg_returns = history.AvgReturn;

plot(ax1, checkpoints, avg_returns, '-o', ...
    'LineWidth', 2.0, ...
    'Color', viz.Colors.Primary, ...
    'MarkerSize', 5, ...
    'MarkerFaceColor', viz.Colors.Primary, ...
    'DisplayName', 'Avg Episode Return (WIS)');

title(ax1, 'Off-Policy MC Control Learning Curve (Racetrack Track 1)', 'Color', viz.Colors.Text, 'FontSize', 14);
xlabel(ax1, 'Training Episodes', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax1, 'Average Episode Return (G)', 'Color', viz.Colors.Text, 'FontSize', 12);
lgd1 = legend(ax1, 'Location', 'southeast');
set(lgd1, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

% Figure 2: Racetrack Grid & Optimal Driving Trajectories
fig2 = viz.createFigure('Exercise 5.12 - Optimal Driving Trajectories');
ax2 = axes(fig2);
viz.styleAxes(ax2);
hold(ax2, 'on');

% Render track grid
% 0=OffTrack (Dark Gray), 1=Track (Slate), 2=Start (Green), 3=Finish (Cyan)
grid_matrix = env.Grid;
H = env.Height;
W = env.Width;

% Custom RGB map
cmap = [ ...
    0.10, 0.10, 0.12;  % 0: Wall / OffTrack
    0.20, 0.22, 0.28;  % 1: Track
    0.15, 0.65, 0.35;  % 2: Start Line
    0.20, 0.70, 0.90   % 3: Finish Line
];

imagesc(ax2, [1 W], [1 H], grid_matrix);
colormap(ax2, cmap);
caxis(ax2, [0 3]);

% Plot trajectory lines from each starting cell
colors_palette = viz.getColorSequence(n_starts);

for i = 1:n_starts
    path = trajectories{i};
    px = path(:, 1);
    py = path(:, 2);

    plot(ax2, px, py, '-o', ...
        'LineWidth', 2.2, ...
        'MarkerSize', 6, ...
        'Color', colors_palette(i, :), ...
        'MarkerFaceColor', colors_palette(i, :), ...
        'DisplayName', sprintf('Start (%d,%d): %d steps', env.StartCells(i, 1), env.StartCells(i, 2), traj_lengths(i)));
end

title(ax2, sprintf('Optimal Learned Driving Trajectories (%s, Noise P=0)', track_name), 'Color', viz.Colors.Text, 'FontSize', 14);
xlabel(ax2, 'Grid Column (x)', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax2, 'Grid Row (y)', 'Color', viz.Colors.Text, 'FontSize', 12);
xlim(ax2, [0.5 W+0.5]);
ylim(ax2, [0.5 H+0.5]);
axis(ax2, 'equal');

lgd2 = legend(ax2, 'Location', 'northeastoutside');
set(lgd2, 'Color', viz.Colors.Background, 'TextColor', viz.Colors.Text, 'EdgeColor', viz.Colors.Grid);

artifact_dir = "C:\Users\USER\.gemini\antigravity-ide\brain\228de5d4-0017-45c6-a208-2755b9e45fe2";
if exist(artifact_dir, 'dir')
    exportgraphics(fig1, fullfile(artifact_dir, 'racetrack_learning_curve.png'));
    exportgraphics(fig2, fullfile(artifact_dir, 'racetrack_trajectories.png'));
    fprintf('      Saved figures to artifact directory.\n');
end

fprintf('\n========================================================================\n');
fprintf('  RACETRACK EXPERIMENT COMPLETE\n');
fprintf('========================================================================\n');
