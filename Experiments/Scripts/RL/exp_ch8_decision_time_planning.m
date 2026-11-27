%% Decision-time planning and state-space search on Windy Gridworld
% References: Sutton & Barto (2nd ed.), Sections 8.8-8.11 and Example 6.5 / Exercise 6.10;
%             A*: Hart, Nilsson & Raphael (1968); IDA*: Korf (1985); beam search: Russell & Norvig ch. 4.
%
% PURPOSE:
%   Apply the general OceanAlgo.StateSearch engines to an RL task through the exact model
%   (OceanStats.RL.DecisionTime.WorldModelSearchModel) and compare what each one buys per unit of computation.
%
% PART A -- deterministic wind (S&B Example 6.5, 4 moves, reward -1 per step, optimal return -15):
%   plan once from the start with A* (h = 0 and h = column distance), weighted A* (w = 2), IDA*, and beam
%   search of several widths; report path cost against the optimum and the search work.
%   h(s) = |goal column - column of s| is admissible and consistent: every move changes the column by at
%   most one, and the wind only moves the agent vertically.
%
% PART B -- stochastic wind with king moves (S&B Exercise 6.10): act in the real environment with a
%   decision-time agent (plan at every state, act once, replan). Compare depth-limited search (8.9) at
%   several depths, the rollout algorithm (8.10) over a uniform random rollout policy, and MCTS (8.11) at
%   several iteration budgets; reference lines: the random policy itself and the optimal value V*(start)
%   from value iteration on the exact model. Leaf value for all three: -column distance (an imperfect,
%   hand-designed evaluation function, as in classical heuristic search).
%
% Not in the text: the experiment design (budgets, heuristic) is ours; MCTS chance nodes and the
% exploration constant scaling are documented in MonteCarloTreeSearch.

clearvars -except QuickMode; clc;
if ~exist('QuickMode', 'var'), QuickMode = false; end
if QuickMode
    cfg = struct('Episodes', 5, 'Depths', [1 3], 'MctsIters', [50 200], 'Rollouts', 8, 'BeamWidths', [1 4 16]);
else
    cfg = struct('Episodes', 30, 'Depths', [1 3 5], 'MctsIters', [50 200 800], 'Rollouts', 20, 'BeamWidths', [1 2 4 8 16 32]);
end

%% PART A -- path search on the deterministic Windy Gridworld
envA = OceanStats.RL.Env.Classic.WindyGridworldEnv();
pmA = OceanStats.RL.DecisionTime.WorldModelSearchModel(envA);
startA = envA.idx(4, 1); goalCol = 8;
hCost = @(s) abs(goalCol - (mod(s - 1, envA.Cols) + 1));        % remaining cost estimate (A*, IDA*)

enginesA = {
    "A*, h = 0",           OceanAlgo.StateSearch.AStarSearch()
    "A*, h = col. dist.",  OceanAlgo.StateSearch.AStarSearch('Heuristic', hCost)
    "weighted A*, w = 2",  OceanAlgo.StateSearch.AStarSearch('Heuristic', hCost, 'Weight', 2)
    "IDA*",                OceanAlgo.StateSearch.IDAStarSearch('Heuristic', hCost)
    };
for B = cfg.BeamWidths
    enginesA(end+1, :) = {sprintf("beam, B = %d", B), ...
        OceanAlgo.StateSearch.BeamSearch('BeamWidth', B, 'MaxDepth', 40, 'LeafValue', @(s) -hCost(s))}; %#ok<SAGROW>
end
nA = size(enginesA, 1); costA = nan(nA, 1); workA = nan(nA, 1); timeA = nan(nA, 1); reachedA = false(nA, 1);
fprintf('=== Part A: deterministic Windy Gridworld, plan from the start ===\n');
for k = 1:nA
    t0 = tic; r = enginesA{k, 2}.search(pmA, startA); timeA(k) = toc(t0);
    costA(k) = -sum(r.Path.Rewards); workA(k) = r.Work.Expansions; reachedA(k) = r.Path.Terminated;
    fprintf('  %-22s status %-8s plan length/cost %5.1f  reaches goal %d  expansions %6d  (%.3f s)\n', ...
        enginesA{k, 1}, r.Status, costA(k), reachedA(k), workA(k), timeA(k));
end

%% PART B -- decision-time agents on the stochastic Windy Gridworld (king moves)
envB = OceanStats.RL.Env.Classic.WindyGridworldEnv('KingMoves', true, 'StochasticWind', true);
pmB = OceanStats.RL.DecisionTime.WorldModelSearchModel(envB);
startB = envB.idx(4, 1);
vStar = valueIteration(pmB);
leaf = @(s) -hCost(s);
maxSteps = 100;

methods = {};
for d = cfg.Depths
    methods(end+1, :) = {sprintf("8.9 depth %d", d), @() OceanAlgo.StateSearch.DepthLimitedSearch('Depth', d, 'LeafValue', leaf)}; %#ok<SAGROW>
end
methods(end+1, :) = {sprintf("8.10 rollout, N = %d", cfg.Rollouts), @() OceanAlgo.StateSearch.RolloutSearch( ...
    'NumRollouts', cfg.Rollouts, 'Horizon', 30, 'LeafValue', leaf, 'Seed', 1)};
for it = cfg.MctsIters
    methods(end+1, :) = {sprintf("8.11 MCTS, %d it.", it), @() OceanAlgo.StateSearch.MonteCarloTreeSearch( ...
        'NumIterations', it, 'ExplorationConstant', 5, 'MaxDepth', 30, 'LeafValue', leaf, 'Seed', 1)}; %#ok<SAGROW>
end
nB = size(methods, 1); retB = nan(nB, cfg.Episodes); workB = nan(nB, cfg.Episodes);
fprintf('\n=== Part B: stochastic wind + king moves, %d episodes each, V*(start) = %.3f ===\n', cfg.Episodes, vStar(startB));
for k = 1:nB
    engine = methods{k, 2}();
    agent = OceanStats.RL.DecisionTime.DecisionTimeAgent(engine, pmB);
    for e = 1:cfg.Episodes
        ep = agent.runEpisode(envB, 'Seed', e, 'MaxSteps', maxSteps);
        retB(k, e) = ep.Return;
        workB(k, e) = (ep.Expansions + ep.SimulatedSteps) / max(ep.Steps, 1);
    end
    fprintf('  %-24s mean return %7.2f +- %5.2f   work per decision %8.1f\n', methods{k, 1}, ...
        mean(retB(k, :)), std(retB(k, :)) / sqrt(cfg.Episodes), mean(workB(k, :)));
end
randomRet = nan(1, cfg.Episodes);
for e = 1:cfg.Episodes
    randomRet(e) = randomEpisode(envB, e, maxSteps);
end
fprintf('  %-24s mean return %7.2f +- %5.2f   (episodes capped at %d steps)\n', 'random policy', ...
    mean(randomRet), std(randomRet) / sqrt(cfg.Episodes), maxSteps);

%% Plots (OceanViz theme)
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Decision-time planning on Windy Gridworld', 'Position', [100 100 1150 820]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact');
cols = viz.getColorSequence(max(nA, nB));

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on'); grid(ax, 'on');
catsA = categorical(string(enginesA(:, 1)), string(enginesA(:, 1)));
shown = costA; shown(~reachedA) = NaN;                     % a plan that never reaches the goal has no path cost
b = bar(ax, catsA, shown, 'FaceColor', 'flat');
b.CData = cols(1:nA, :);
for k = find(~reachedA)'
    text(ax, catsA(k), 1, 'goal not reached', 'Rotation', 90, 'Color', ax.XColor, 'FontSize', 8);
end
yline(ax, 15, '--', 'optimal (15)', 'Color', ax.XColor, 'LabelHorizontalAlignment', 'left');
ylim(ax, [0, 1.3 * max([15; shown])]);
ylabel(ax, 'path cost of the plan'); title(ax, 'A: plan quality (deterministic wind)');

ax = nexttile(tl); viz.styleAxes(ax); hold(ax, 'on'); grid(ax, 'on');
b = bar(ax, categorical(string(enginesA(:, 1)), string(enginesA(:, 1))), workA, 'FaceColor', 'flat');
b.CData = cols(1:nA, :); set(ax, 'YScale', 'log'); ylim(ax, [10, 3 * max(workA)]);
ylabel(ax, 'node expansions (log)'); title(ax, 'A: search work');

ax = nexttile(tl, [1 2]); viz.styleAxes(ax); hold(ax, 'on'); grid(ax, 'on');
mw = mean(workB, 2); mr = mean(retB, 2); se = std(retB, 0, 2) / sqrt(cfg.Episodes);
markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '>'};
for k = 1:nB
    errorbar(ax, mw(k), mr(k), se(k), markers{mod(k - 1, numel(markers)) + 1}, 'Color', cols(k, :), ...
        'MarkerFaceColor', cols(k, :), 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', methods{k, 1});
end
set(ax, 'XScale', 'log');
xlim(ax, [0.5, 2 * max(mw)]);
ylim(ax, [min([mr - se; mean(randomRet)]) - 8, vStar(startB) + 6]);
yline(ax, vStar(startB), '--', 'V^*(start), optimal', 'Color', ax.XColor, 'HandleVisibility', 'off', ...
    'LabelHorizontalAlignment', 'left');
yline(ax, mean(randomRet), ':', 'random policy', 'Color', ax.XColor, 'HandleVisibility', 'off', ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'top');
xlabel(ax, 'model calls per decision (expansions + simulated steps, log)');
ylabel(ax, 'return per episode (mean \pm s.e.)');
title(ax, 'B: decision-time planning, stochastic wind + king moves');
lgd = legend(ax, 'Location', 'eastoutside'); lgd.Color = ax.Color; lgd.TextColor = ax.XColor; lgd.EdgeColor = ax.GridColor;

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================
function V = valueIteration(pm)
    % Exact V* by synchronous Bellman optimality sweeps on the exact model (γ = 1, episodic).
    V = zeros(pm.numStates(), 1);
    for it = 1:10000
        Vn = V;
        for s = 1:pm.numStates()
            Vn(s) = OceanStats.RL.DP.BellmanOptimality.backup(pm.WorldModel, s, V, 1);
        end
        if max(abs(Vn - V)) < 1e-10, V = Vn; return; end
        V = Vn;
    end
end

function G = randomEpisode(env, seed, maxSteps)
    % Uniform random policy in the real environment (its own stream, seeded per episode).
    rs = RandStream('mrg32k3a', 'Seed', seed);
    [s, info] = env.reset('Seed', seed); G = 0;
    for t = 1:maxSteps
        acts = find(info.action_mask);
        [s, r, terminated, truncated, info] = env.step(acts(randi(rs, numel(acts)))); %#ok<ASGLU>
        G = G + r;
        if terminated || truncated, return; end
    end
end
