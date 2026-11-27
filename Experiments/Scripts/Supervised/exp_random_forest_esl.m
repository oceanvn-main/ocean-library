% EXP_RANDOM_FOREST_ESL
% ESL Chapter 15 — Random Forests
%
% EXPERIMENT OVERVIEW:
%   Reproduces the key empirical results of Breiman (2001) / ESL §15:
%
%   PART A — Test error vs number of trees (ESL Fig 15.1 spirit):
%     Compares Random Forest vs Bagging vs single Decision Tree on a
%     synthetic binary classification dataset (ESL mixture generator).
%     Shows how RF converges faster and to a lower error floor.
%
%   PART B — Feature importance bar chart (ESL §15.3.2):
%     Trains RF on the same dataset, displays MDA importance.
%     Visual confirmation that signal features rank above noise features.
%
% ASSUMPTIONS:
%   - ESLMixtureGenerator produces a 2-class mixture with a well-defined
%     Bayes error around 0.05–0.10 depending on configuration.
%   - A single Decision Tree uses default depth=3 (high bias, low variance).
%   - Bagging and RF both use 100 trees as the maximum.
%
% REPRODUCIBILITY:
%   - rng(42) is set at the top. All models use Seed=42.
%
% GENERALIZATION:
%   - All parameters are derived from the data at fit-time (no hardcoded labels).
%
% Author: Ocean Library
% Date:   2026-05-12

clear; clc; close all;
rng(42);

% =========================================================================
%  DATA — ESL mixture generator
% =========================================================================
fprintf('[Setup] Generating ESL mixture data...\n');

gen = OceanStats.Data.Generators.ESLMixtureGenerator( ...
    'NumClasses', 2, 'NSamples', 300, 'Seed', 42);
[X_all, Y_all] = gen.generate();

% Train / test split (70/30)
N     = size(X_all, 1);
idx   = randperm(N);
n_tr  = floor(0.70 * N);
X_tr  = X_all(idx(1:n_tr), :);      Y_tr  = Y_all(idx(1:n_tr));
X_te  = X_all(idx(n_tr+1:end), :);  Y_te  = Y_all(idx(n_tr+1:end));
p     = size(X_tr, 2);

fprintf('[Setup] Train=%d  Test=%d  Features=%d\n', n_tr, numel(Y_te), p);

% =========================================================================
%  PART A — Error vs NumTrees
% =========================================================================
tree_counts = [1, 5, 10, 20, 30, 50, 75, 100];
nT          = numel(tree_counts);

err_rf  = zeros(1, nT);
err_bag = zeros(1, nT);

fprintf('\n[Part A] Sweeping NumTrees = %s\n', mat2str(tree_counts));

for i = 1:nT
    B = tree_counts(i);

    % Random Forest
    rf = OceanStats.Ensemble.RandomForest( ...
        'NumTrees', B, 'Criterion', 'gini', ...
        'ComputeOOB', false, 'ComputeImportance', false, ...
        'Verbose', false, 'Seed', 42);
    rf.fit(X_tr, Y_tr);
    err_rf(i) = mean(rf.predict(X_te) ~= Y_te);

    % Bagging (full-feature trees — no random feature selection)
    rf_bag = OceanStats.Ensemble.RandomForest( ...
        'NumTrees', B, 'Criterion', 'gini', ...
        'Mtry', p, ...                          % all features = bagging
        'ComputeOOB', false, 'ComputeImportance', false, ...
        'Verbose', false, 'Seed', 42);
    rf_bag.fit(X_tr, Y_tr);
    err_bag(i) = mean(rf_bag.predict(X_te) ~= Y_te);

    fprintf('  B=%3d  RF=%.4f  Bag=%.4f\n', B, err_rf(i), err_bag(i));
end

% Single Decision Tree (baseline)
single_tree = OceanStats.Supervised.Tree.DecisionTree( ...
    'Criterion', 'gini', 'MaxDepth', 5, 'MinLeafSize', 2);
single_tree.fit(X_tr, Y_tr);
Y_tree = single_tree.predict(X_te);
err_tree = mean(Y_tree ~= Y_te);

fprintf('  Single Tree (depth=5): %.4f\n', err_tree);

% --- Figure 1 ---
viz = OceanViz.Core('Theme', 'Dark');
fig1 = figure('Name', 'RF vs Bagging vs Tree — Test Error', ...
    'Color', [0.12 0.12 0.15], 'Position', [50 100 800 450]);
ax1 = axes(fig1);

hold(ax1, 'on');
plot(ax1, tree_counts, err_rf,  '-o', 'Color', [0.2 0.7 1.0], ...
    'LineWidth', 2.5, 'MarkerSize', 7, 'DisplayName', 'Random Forest');
plot(ax1, tree_counts, err_bag, '-s', 'Color', [1.0 0.6 0.2], ...
    'LineWidth', 2.0, 'MarkerSize', 7, 'DisplayName', 'Bagging');
yline(ax1, err_tree, '--', 'Color', [0.9 0.3 0.3], ...
    'LineWidth', 1.8, 'Label', sprintf('Single Tree %.3f', err_tree), ...
    'LabelHorizontalAlignment', 'left', 'DisplayName', 'Single Tree');
hold(ax1, 'off');

xlabel(ax1, 'Number of Trees',  'Color', 'w', 'FontSize', 12);
ylabel(ax1, 'Test Error Rate',  'Color', 'w', 'FontSize', 12);
title(ax1,  'ESL §15 — Random Forest vs Bagging', ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 14);
legend(ax1, 'Location', 'northeast', 'TextColor', 'w', 'Box', 'off');
set(ax1, 'Color', [0.12 0.12 0.15], 'XColor', 'w', 'YColor', 'w', ...
    'FontSize', 11, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.5);
grid(ax1, 'on');
xlim(ax1, [0, max(tree_counts) + 5]);
ylim(ax1, [0, max([err_rf, err_bag, err_tree]) * 1.15]);

% =========================================================================
%  PART B — Feature Importance
% =========================================================================
fprintf('\n[Part B] Training RF (B=100) for feature importance...\n');

rf_full = OceanStats.Ensemble.RandomForest( ...
    'NumTrees', 100, 'Criterion', 'gini', ...
    'ComputeOOB', true, 'ComputeImportance', true, ...
    'ImportanceMethod', 'MDA', 'Verbose', true, 'Seed', 42);
rf_full.fit(X_tr, Y_tr);
rf_full.printSummary();

imp    = rf_full.featureImportance();
[~, ord] = sort(imp, 'descend');

% Feature labels (generic)
feat_labels = arrayfun(@(k) sprintf('F%d', k), ord, 'UniformOutput', false);
imp_sorted  = imp(ord);

% --- Figure 2 ---
fig2 = figure('Name', 'Feature Importance (MDA)', ...
    'Color', [0.12 0.12 0.15], 'Position', [900 100 650 420]);
ax2  = axes(fig2);

colors = interp1([0;1], [0.2 0.5 1.0; 0.9 0.3 0.3], linspace(0,1,p));
barh(ax2, fliplr(imp_sorted), 'FaceColor', 'flat');
bar_h = get(ax2, 'Children');
if ~isempty(bar_h) && isprop(bar_h(1), 'CData')
    bar_h(1).CData = flipud(colors);
end

set(ax2, 'YTick', 1:p, 'YTickLabel', fliplr(feat_labels), ...
    'Color', [0.12 0.12 0.15], 'XColor', 'w', 'YColor', 'w', ...
    'FontSize', 11);
xlabel(ax2, 'Mean Decrease Accuracy (MDA)', 'Color', 'w', 'FontSize', 12);
title(ax2, sprintf('Feature Importance — RF (B=100, OOB err=%.3f)', rf_full.oobError()), ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 14);
grid(ax2, 'on');
set(ax2, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.5);

% =========================================================================
%  Save Results
% =========================================================================
resultsDir = fullfile('Results', 'Supervised');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

exportgraphics(fig1, fullfile(resultsDir, 'exp_rf_error_vs_trees.png'), ...
    'Resolution', 150, 'BackgroundColor', [0.12 0.12 0.15]);
exportgraphics(fig2, fullfile(resultsDir, 'exp_rf_feature_importance.png'), ...
    'Resolution', 150, 'BackgroundColor', [0.12 0.12 0.15]);

fprintf('\n[Done] Figures saved to %s\n', resultsDir);
