% EXP_COMPARE_MC_VS_TD_RACETRACK Scientific Experiment Comparing MC vs TD Control
%
% Problem Statement:
%   Compare Off-Policy Monte Carlo Control (WIS) against Q-Learning TD Control
%   on the Racetrack task (Exercise 5.12):
%     1. Off-Policy Monte Carlo Control (MonteCarloSolver, 'OffPolicyControl')
%     2. Q-Learning Off-Policy TD Control (TDLearner, 'Target', 'qlearning')
%   over SEVERAL training seeds: one run is one sample of a random learning process,
%   so the result is a success rate and mean +- SE, not a single run.
%
% MLOps Medallion Pipeline Stage: Gold / Experiment
% Package: OceanStats.RL.MonteCarlo / OceanStats.RL.TD / Experiments.Scripts.RL
% Reference: Sutton & Barto (2018) Chapters 5 & 6
%
% 2026-09-27 fixes:
%  * Map: d_shape_with_shortcut declares movement_mode = bidirectional (speeds -1..1)
%    and max_episode_steps = 5000; RacetrackEnv now uses a map's declared properties.
%    Before, it drove forward-only and the finish was unreachable (return -500 always).
%    The script stops if the finish is unreachable and uses the map's episode cap.
%  * Off-policy MC breaks ties among greedy actions at random (behaviour AND target; the
%    importance weights use b(A|S) recorded when the action was chosen) and does not
%    learn from episodes cut off at the cap (no complete return).
%  * Multi-seed: with one (default, unseeded) run the MC result was down to luck -- its
%    greedy policy finished for most seeds but stood still / crash-looped for others
%    (high-variance WIS estimates). Each seed now trains both methods; the greedy
%    policies are evaluated with noise off, failures are classified.
%  * Learning curves: block means of per-episode returns, mean +- SE over seeds.

clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
root = fullfile(script_dir, '..', '..', '..');
addpath(root);
out_dir = fullfile(script_dir, 'Results'); if ~isfolder(out_dir), mkdir(out_dir); end

fprintf('========================================================================\n');
fprintf('  SCIENTIFIC BENCHMARK: MONTE CARLO VS Q-LEARNING (TD) ON RACETRACK\n');
fprintf('========================================================================\n\n');

%% 1. Configuration (all settings; no hard-coded results)
cfg = struct();
cfg.Track         = "d_shape_with_shortcut";
cfg.FailProb      = 0.1;
cfg.Seeds         = 1:8;            % independent training runs (both methods per seed)
cfg.Episodes      = 50000;
cfg.CheckInterval = 5000;           % block size of the learning curves
cfg.MaxSteps      = [];             % [] = the map's max_episode_steps (else 500)
cfg.MC_Epsilon    = 0.1;
cfg.TD_Epsilon    = 0.1;
cfg.TD_StepSize   = 0.1;
cfg.InitialValue  = -1000.0;
cfg.UseParallel   = true;           % parfor over seeds when the Parallel Computing Toolbox is present

probe = OceanStats.RL.Env.Classic.RacetrackEnv(cfg.Track, 'FailProb', cfg.FailProb);
fprintf('[1/4] %s: %d x %d, %d states, %d start cell(s), %s, speeds %d..%d, finish reachable: %d\n', cfg.Track, ...
    probe.Height, probe.Width, probe.TotalStates, size(probe.StartCells, 1), probe.MovementMode, ...
    probe.VelocityMin, probe.VelocityMax, probe.FinishReachable);
if ~probe.FinishReachable
    error('The finish of "%s" cannot be reached under %s movement -- no agent can learn. Check the map / MovementMode.', ...
        cfg.Track, probe.MovementMode);
end
if isempty(cfg.MaxSteps)
    cfg.MaxSteps = 500; if isfinite(probe.MaxEpisodeSteps), cfg.MaxSteps = probe.MaxEpisodeSteps; end
end
nS = numel(cfg.Seeds); nB = cfg.Episodes / cfg.CheckInterval;
usePar = cfg.UseParallel && ~isempty(ver('parallel')) && license('test', 'Distrib_Computing_Toolbox');
fprintf('      episode cap %d | %d seeds x %d episodes per method | parallel: %d\n', cfg.MaxSteps, nS, cfg.Episodes, usePar);

%% 2-3. Train both methods for every seed
res = cell(nS, 1);
tAll = tic;
if usePar
    parfor i = 1:nS, res{i} = runSeed(cfg, cfg.Seeds(i)); end
else
    for i = 1:nS
        res{i} = runSeed(cfg, cfg.Seeds(i));
        fprintf('      seed %d done (MC %.0f s, TD %.0f s)\n', cfg.Seeds(i), res{i}.mc.seconds, res{i}.td.seconds);
    end
end
fprintf('[2-3/4] all training: %.0f s\n', toc(tAll));

%% 4. Summary
R = [res{:}];
mc = [R.mc]; td = [R.td];
fprintf('\n========================================================================\n');
fprintf('  GREEDY POLICY EVALUATION (noise off), %d seeds\n', nS);
fprintf('========================================================================\n');
fprintf('  seed |  Monte Carlo (WIS)            |  Q-learning (TD)\n');
for i = 1:nS
    fprintf('  %4d |  %-28s |  %s\n', cfg.Seeds(i), outcomeText(mc(i)), outcomeText(td(i)));
end
fprintf('------------------------------------------------------------------------\n');
fprintf('  Monte Carlo : finished %d/%d seeds, mean steps (finished) %.1f\n', nnz([mc.finished]), nS, mean([mc([mc.finished]).steps]));
fprintf('  Q-learning  : finished %d/%d seeds, mean steps (finished) %.1f\n', nnz([td.finished]), nS, mean([td([td.finished]).steps]));
fprintf('========================================================================\n\n');
save(fullfile(out_dir, "compare_mc_vs_td_" + cfg.Track + ".mat"), 'cfg', 'R');

%% 5. Figure (OceanViz): learning curves mean +- SE over seeds, all greedy paths
viz = OceanViz.Core('Theme', 'Dark'); C = viz.Colors;
fig = viz.createFigure('Monte Carlo vs Q-Learning Comparison');
tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile(tl, 1); viz.styleAxes(ax1); hold(ax1, 'on');
x = (1:nB)' * cfg.CheckInterval;
bandPlot(ax1, x, reshape([mc.blocks], nB, nS), C.Primary, sprintf('Off-Policy Monte Carlo (WIS), %d seeds', nS), '-o');
bandPlot(ax1, x, reshape([td.blocks], nB, nS), C.Secondary, sprintf('Q-Learning (TD Control), %d seeds', nS), '-s');
title(ax1, 'Learning curves (mean \pm SE over seeds)', 'Color', C.Text, 'FontSize', 12);
xlabel(ax1, 'Training Episodes', 'Color', C.Text, 'FontSize', 11);
ylabel(ax1, 'Average Episode Return (G)', 'Color', C.Text, 'FontSize', 11);
grid(ax1, 'on');
legend(ax1, 'Location', 'southeast', 'TextColor', C.Text, 'Color', C.Background);

ax2 = nexttile(tl, 2); viz.styleAxes(ax2);
imagesc(ax2, probe.Grid);
colormap(ax2, [0.08 0.08 0.12; 0.25 0.25 0.30; 0.95 0.40 0.10; 0.10 0.85 0.30]);
axis(ax2, 'xy'); hold(ax2, 'on');
h = gobjects(0);
for i = 1:nS
    p = plot(ax2, td(i).path(:, 1), td(i).path(:, 2), '-', 'LineWidth', 2.2, 'Color', [C.Secondary 0.55]);
    if i == 1, p.DisplayName = 'Q-Learning greedy paths'; h(end+1) = p; end %#ok<SAGROW>
end
for i = 1:nS
    p = plot(ax2, mc(i).path(:, 1), mc(i).path(:, 2), '--', 'LineWidth', 1.6, 'Color', [C.Primary 0.8]);
    if i == 1, p.DisplayName = 'Monte Carlo greedy paths'; h(end+1) = p; end %#ok<SAGROW>
end
title(ax2, sprintf('Greedy paths (%s): MC finished %d/%d, TD %d/%d', cfg.Track, nnz([mc.finished]), nS, ...
    nnz([td.finished]), nS), 'Color', C.Text, 'FontSize', 12, 'Interpreter', 'none');
xlabel(ax2, 'Grid Column (X)', 'Color', C.Text, 'FontSize', 11);
ylabel(ax2, 'Grid Row (Y)', 'Color', C.Text, 'FontSize', 11);
legend(ax2, h, 'Location', 'northwest', 'TextColor', C.Text, 'Color', C.Background);

output_img = fullfile(script_dir, 'compare_mc_vs_td_racetrack.png');
saveas(fig, output_img);
fprintf('Saved comparison dashboard with learned trajectories to: %s\n', output_img);

%% --------------------------------------------------------------------------------
function r = runSeed(cfg, seed)
% Train MC and Q-learning with one seed; evaluate both greedy policies (noise off).
r = struct();
% Monte Carlo (uses the global random stream: seed it)
rng(seed, 'twister');
env = OceanStats.RL.Env.Classic.RacetrackEnv(cfg.Track, 'FailProb', cfg.FailProb);
solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, 'Algorithm', "OffPolicyControl", 'Gamma', 1.0, ...
    'NumEpisodes', cfg.Episodes, 'MaxEpisodeLen', cfg.MaxSteps, 'Epsilon', cfg.MC_Epsilon, ...
    'InitialQValue', cfg.InitialValue, 'TieBreak', "random", 'TruncatedEpisodes', "skip", ...
    'CheckInterval', cfg.CheckInterval, 'Verbose', false);
t = tic; [~, ~, pi_mc, h_mc] = solver.run(); secs = toc(t);
r.mc = evaluate(cfg, pi_mc, blocksOf(h_mc.EpisodeReturns, cfg), secs);
% Q-learning (own seeded streams)
env = OceanStats.RL.Env.Classic.RacetrackEnv(cfg.Track, 'FailProb', cfg.FailProb);
env.setSeed(seed);
learner = OceanStats.RL.TD.TDLearner('Env', env, 'Task', "control", 'Mode', "onestep", 'Target', "qlearning", ...
    'StepSize', cfg.TD_StepSize, 'Exploration', cfg.TD_Epsilon, 'Gamma', 1.0, 'Episodes', cfg.Episodes, ...
    'MaxSteps', cfg.MaxSteps, 'InitialValue', cfg.InitialValue, 'Seed', seed);
t = tic; [~, ~, pi_td, h_td] = learner.run(); secs = toc(t);
r.td = evaluate(cfg, pi_td, blocksOf(h_td.EpisodeReturns, cfg), secs);
end

function b = blocksOf(returns, cfg)
nB = cfg.Episodes / cfg.CheckInterval;
b = mean(reshape(returns(1:nB * cfg.CheckInterval), cfg.CheckInterval, nB), 1)';
end

function e = evaluate(cfg, pi_, blocks, secs)
% Greedy policy from the start line, noise off. Failure kinds: "stands still"
% (a repeated state without any crash) or "crash loop" (a crash, then the same again).
env = OceanStats.RL.Env.Classic.RacetrackEnv(cfg.Track, 'FailProb', 0);
env.reset(); env.set_state([env.StartCells(1, :) 0 0]);
path = env.StartCells(1, :); crashed = false; kind = "cap"; steps = 0;
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
while ~env.EpisodeDone && steps < cfg.MaxSteps
    s = env.CurrentState; key = mat2str(s);
    if isKey(seen, key)                                         % deterministic greedy: a repeat never ends
        if crashed, kind = "crash loop"; else, kind = "stands still"; end
        steps = cfg.MaxSteps; break;
    end
    seen(key) = true;
    k = env.state_to_index(s); a = 5; if k > 0 && pi_(k) > 0, a = pi_(k); end
    [ns, ~, term] = env.step(a); steps = steps + 1;
    if ~term && all(ns(3:4) == 0) && ~all(s(3:4) == 0)
        crashed = true; path = [path; NaN NaN]; %#ok<AGROW>   % break the line at a crash
    end
    path = [path; ns(1:2)]; %#ok<AGROW>
end
e = struct('finished', env.EpisodeDone, 'steps', steps, 'failure', kind, 'path', path, 'blocks', blocks, 'seconds', secs);
if e.finished, e.failure = ""; end
end

function t = outcomeText(e)
if e.finished, t = sprintf('finished in %d steps', e.steps); else, t = char("did not finish (" + e.failure + ")"); end
end

function bandPlot(ax, x, Y, color, name, style)
m = mean(Y, 2); se = std(Y, 0, 2) / sqrt(size(Y, 2));
fill(ax, [x; flipud(x)], [m - se; flipud(m + se)], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(ax, x, m, style, 'LineWidth', 2, 'Color', color, 'MarkerSize', 5, 'MarkerFaceColor', color, 'DisplayName', name);
end
