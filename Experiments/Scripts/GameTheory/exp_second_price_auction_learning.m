% EXP_SECOND_PRICE_AUCTION_LEARNING
%   Second-price (Vickrey) sealed-bid auction: verifies, both analytically
%   and empirically, that bidding one's true value is the best strategy.
%
%   ENVIRONMENT:     OceanStats.GameTheory.Core.RepeatedAuctionSimulator,
%                     rule = OceanStats.GameTheory.Core.SecondPriceRule
%                     (auction mechanism, decoupled from bidder strategies)
%   ANALYTIC CHECK:  OceanStats.GameTheory.Core.SingleItemAuctionGame builds
%                     each bidder's reduced 2-player game; verified against
%                     OceanStats.GameTheory.Solvers.IteratedDominanceSolver
%                     that truthful bidding is never weakly dominated
%                     (Vickrey 1961).
%   LEARNING:        OceanStats.GameTheory.Solvers.MonteCarloBidStrategy --
%                     independent epsilon-greedy bandit learners, one per
%                     bidder, discover their best bid purely empirically
%                     (never told their own value directly; only observes
%                     realized payoffs).
%   HYPERPARAMETERS:
%     NUM_SETTINGS  = 20    (different random items/valuations)
%     NUM_BIDDERS   = 3
%     BID_GRID      = 0:20  (fixed strategy set -- identical across every setting)
%     VALUE_RANGE   = [0,20] (private values drawn i.i.d. randi)
%     NUM_ROUNDS    = 3000   (repeated-play rounds per setting)
%     EPSILON       = 0.1
%     LAST_K_ROUNDS = 200    (window averaged for the "final learned bid")
%   ASSUMPTIONS:
%     - Ties broken in the bidder's own favor (SecondPriceRule convention)
%     - Independent-learners setup: each agent updates from its own
%       realized payoff only, unaware of others' value/strategy
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%   OUTPUTS:
%     - Per-setting analytic Vickrey check (pass/fail)
%     - Printed mean-absolute-deviation summary (learned bid vs true value)
%     - Convergence-curve plot for one representative setting
%     - Learned-bid-vs-true-value scatter across all settings/agents
%
%   Reference: Vickrey (1961); Sutton & Barto (2018) sec. 2.2/2.4 (bandits)
%
%   Package context: OceanStats.GameTheory.{Core,Solvers}

clc; clear; close all;
rng(42, 'twister');

% =========================================================================
% === 1. HYPERPARAMETERS ==================================================
% =========================================================================
NUM_SETTINGS  = 20;
NUM_BIDDERS   = 3;
BID_GRID      = 0:20;    % the fixed strategy set -- identical across every setting
VALUE_RANGE   = [0, 20];
NUM_ROUNDS    = 3000;
EPSILON       = 0.1;
LAST_K_ROUNDS = 200;

rule = OceanStats.GameTheory.Core.SecondPriceRule();
dominanceSolver = OceanStats.GameTheory.Solvers.IteratedDominanceSolver();
gen = OceanStats.GameTheory.Core.SingleItemAuctionGame( ...
    'NumBidders', NUM_BIDDERS, 'BidGrid', BID_GRID, 'Rule', rule);

allTrueValues  = zeros(NUM_SETTINGS, NUM_BIDDERS);
allLearnedBids = zeros(NUM_SETTINGS, NUM_BIDDERS);
representativeHistory = struct();
representativeValues = [];

% =========================================================================
% === 2. LOOP OVER RANDOM SETTINGS (different items / valuations) ========
% =========================================================================
for setting = 1:NUM_SETTINGS
    fprintf('[Setting %2d/%d] ', setting, NUM_SETTINGS);

    % --- generate a fresh auction instance (same BID_GRID, new values) ---
    [games, values] = gen.generate(VALUE_RANGE);
    allTrueValues(setting, :) = values;

    % --- analytic check: truthful bidding is never weakly dominated ------
    for i = 1:NUM_BIDDERS
        truthfulIdx = find(BID_GRID == values(i), 1);
        if dominanceSolver.isWeaklyDominated(games{i}, 1, truthfulIdx)
            error('exp_second_price_auction_learning:VickreyCheckFailed', ...
                'Truthful bidding was found weakly dominated for bidder %d in setting %d.', ...
                i, setting);
        end
    end
    fprintf('analytic check OK; ');

    % --- empirical learning: independent Monte Carlo bandit learners -----
    strategies = cell(1, NUM_BIDDERS);
    for i = 1:NUM_BIDDERS
        strategies{i} = OceanStats.GameTheory.Solvers.MonteCarloBidStrategy(BID_GRID, 'Epsilon', EPSILON);
    end
    sim = OceanStats.GameTheory.Core.RepeatedAuctionSimulator(rule, BID_GRID, values, strategies);
    history = sim.runRounds(NUM_ROUNDS);

    learnedBids = mean(history.Bids(end - LAST_K_ROUNDS + 1:end, :), 1);
    allLearnedBids(setting, :) = learnedBids;

    meanAbsDev = mean(abs(learnedBids - values));
    fprintf('mean |learned-true| = %.3f\n', meanAbsDev);

    if setting == 1
        representativeHistory = history;
        representativeValues = values;
    end
end

% =========================================================================
% === 3. SUMMARY ===========================================================
% =========================================================================
overallMeanAbsDev = mean(abs(allLearnedBids(:) - allTrueValues(:)));
fprintf('\n[Summary] Overall mean |learned bid - true value| across %d settings x %d bidders: %.3f\n', ...
    NUM_SETTINGS, NUM_BIDDERS, overallMeanAbsDev);
fprintf('          (bid grid spacing = 1; a value near 0 means agents learned to bid truthfully)\n');

% =========================================================================
% === 4. VISUALIZATION =====================================================
% =========================================================================
fprintf('\n[Viz] Rendering figures...\n');
viz = OceanViz.Core('Theme', 'Dark');
C = viz.Colors;
palette = viz.getColorSequence(NUM_BIDDERS);

% ----- FIGURE 1: bid convergence, smoothed + legend --------------------
% Raw per-round bids are shown faint (constant-epsilon exploration means
% ~10% of rounds are always a random pick, even late -- that's expected
% bandit behavior, not a bug, but it's illegible on its own); a moving
% average is overlaid solid to show the actual learning trend.
SMOOTH_WINDOW = 51;

fig1 = figure('Name', 'Second-Price Auction — Bid Convergence (representative setting)', ...
    'Color', C.Background, 'Position', [40, 40, 900, 520]);
ax1 = axes('Parent', fig1, 'Color', C.Background, 'XColor', C.Text, 'YColor', C.Text);
hold(ax1, 'on');
legendHandles = gobjects(1, NUM_BIDDERS);
legendLabels = strings(1, NUM_BIDDERS);
for i = 1:NUM_BIDDERS
    plot(ax1, 1:NUM_ROUNDS, representativeHistory.Bids(:, i), 'Color', [palette(i, :), 0.15], ...
        'LineWidth', 0.5, 'HandleVisibility', 'off');
    smoothed = movmean(representativeHistory.Bids(:, i), SMOOTH_WINDOW);
    legendHandles(i) = plot(ax1, 1:NUM_ROUNDS, smoothed, 'Color', palette(i, :), 'LineWidth', 2);
    plot(ax1, [1, NUM_ROUNDS], [representativeValues(i), representativeValues(i)], '--', ...
        'Color', palette(i, :), 'LineWidth', 1.2, 'HandleVisibility', 'off');
    legendLabels(i) = sprintf('Bidder %d (true value = %d)', i, representativeValues(i));
end
xlabel(ax1, 'Round', 'Color', C.Text);
ylabel(ax1, 'Bid', 'Color', C.Text);
title(ax1, 'Learned bid (smoothed) vs. true value (dashed) over rounds', 'Color', C.Text);
subtitle(ax1, sprintf('Faint = raw per-round bid; solid = %d-round moving average; dashed = true value', ...
    SMOOTH_WINDOW), 'Color', C.Text);
legend(ax1, legendHandles, legendLabels, 'TextColor', C.Text, 'Color', C.Background, ...
    'EdgeColor', C.Grid, 'Location', 'best');
grid(ax1, 'on');
ax1.GridColor = C.Grid;
hold(ax1, 'off');

% ----- FIGURE 2: learned bid vs true value scatter, with legend --------
fig2 = figure('Name', 'Second-Price Auction — Learned Bid vs True Value (all settings)', ...
    'Color', C.Background, 'Position', [980, 40, 600, 550]);
ax2 = axes('Parent', fig2, 'Color', C.Background, 'XColor', C.Text, 'YColor', C.Text);
hold(ax2, 'on');
hScatter = scatter(ax2, allTrueValues(:), allLearnedBids(:), 24, C.Primary, 'filled', 'MarkerFaceAlpha', 0.6);
hRef = plot(ax2, BID_GRID([1, end]), BID_GRID([1, end]), '--', 'Color', C.Secondary, 'LineWidth', 1.5);
xlabel(ax2, 'True value', 'Color', C.Text);
ylabel(ax2, sprintf('Learned bid (avg of last %d rounds)', LAST_K_ROUNDS), 'Color', C.Text);
title(ax2, 'Every agent, every setting: learned bid vs. true value', 'Color', C.Text);
legend(ax2, [hScatter, hRef], {'Agent (one per bidder x setting)', 'y = x (perfect truthful bidding)'}, ...
    'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', C.Grid, 'Location', 'best');
grid(ax2, 'on');
ax2.GridColor = C.Grid;
axis(ax2, 'equal');
hold(ax2, 'off');

% ----- FIGURE 3: expected payoff of ONE bidder under three strategies ----
% Fixes both issues with the earlier version: (1) one bidder only, so the
% comparison isn't confounded by which bidder happened to have the highest
% value in the representative setting; (2) averaged over NUM_TRIALS
% independent random opponent draws (Monte Carlo), not one noisy 3000-round
% realization -- with error bars showing the result is not chance. Opponents
% are modeled as bidding truthfully (the theoretical prediction), so this
% plot isolates the Vickrey payoff-ranking property cleanly, independent of
% the empirical bandit-learning noise/undershoot discussed for Figure 1.
STUDY_BIDDER_VALUE = representativeValues(1);   % the one bidder under study
NUM_TRIALS = 5000;                               % independent random-opponent draws
MARGIN = 5;                                      % Underbid/Overbid offset, clamped to BID_GRID bounds

opponentValues = randi(VALUE_RANGE, NUM_TRIALS, NUM_BIDDERS - 1);
maxOpponentBid = max(opponentValues, [], 2);     % opponents assumed truthful: bid = value

candidateBids = [max(BID_GRID(1), STUDY_BIDDER_VALUE - MARGIN), ...
    STUDY_BIDDER_VALUE, ...
    min(BID_GRID(end), STUDY_BIDDER_VALUE + MARGIN)];
stratNames = {'Underbid (v-5)', 'Truthful (v)', 'Overbid (v+5)'};

payoffSamples = zeros(NUM_TRIALS, 3);
for s = 1:3
    payoffSamples(:, s) = arrayfun(@(m) rule.resolveAgainstMax(candidateBids(s), m, STUDY_BIDDER_VALUE), ...
        maxOpponentBid);
end
meanPayoff = mean(payoffSamples, 1);
stderrPayoff = std(payoffSamples, 0, 1) / sqrt(NUM_TRIALS);

fig3 = figure('Name', 'Second-Price Auction — Expected Payoff, One Bidder, Three Strategies', ...
    'Color', C.Background, 'Position', [40, 600, 700, 500]);
ax3 = axes('Parent', fig3, 'Color', C.Background, 'XColor', C.Text, 'YColor', C.Text);
hold(ax3, 'on');
stratPalette = viz.getColorSequence(3);
hBar = bar(ax3, 1:3, meanPayoff, 0.5, 'FaceColor', 'flat');
hBar.CData = stratPalette;
errorbar(ax3, 1:3, meanPayoff, stderrPayoff, 'Color', C.Text, 'LineStyle', 'none', ...
    'LineWidth', 1.5, 'CapSize', 12);
set(ax3, 'XTick', 1:3, 'XTickLabel', stratNames);
xlabel(ax3, 'Strategy', 'Color', C.Text);
ylabel(ax3, 'Expected payoff', 'Color', C.Text);
title(ax3, sprintf('Bidder with true value = %d: expected payoff by strategy', STUDY_BIDDER_VALUE), ...
    'Color', C.Text);
subtitle(ax3, sprintf('Averaged over %d independent random-opponent trials (error bars = standard error of the mean)', ...
    NUM_TRIALS), 'Color', C.Text);
grid(ax3, 'on');
ax3.GridColor = C.Grid;
hold(ax3, 'off');
