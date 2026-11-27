% EXP_ISLE_GBM_KNOBS
% ISLE Grand Unification — Two-Knob Ensemble Experiment
%
% EXPERIMENT OVERVIEW:
%   Demonstrates the Grand Unification Theory of Ensembles (Friedman & Hall
%   2007, ESL §16). The GradientBoostingMachine has two universal knobs:
%
%     eta (SubsampleFraction) — "Randomness knob": fraction of rows per tree
%     nu  (LearningRate)      — "Memory knob": shrinkage applied per tree
%
%   By sweeping both knobs across a grid, this experiment shows how the
%   test error surface continuously morphs through the ensemble spectrum:
%
%     eta=1.0, nu->0  : Bagging (no memory, full data)
%     eta<0.5, nu->0  : Subsampled Bagging (RF-style diversity, no boosting)
%     eta=1.0, nu>0   : Standard Gradient Boosting
%     eta<=0.5, nu=0.1: Stochastic GBM / ISLE optimal recipe
%
%   PART A — (eta x nu) Error Heatmap
%     Test error at each (eta, nu) grid cell. Visual proof of the 2D
%     hyperparameter continuum described in the ISLE paper.
%
%   PART B — Error vs NumTrees curves
%     Four representative (eta, nu) configs plotted on the same axes:
%     Bagging | Standard GBM | ISLE optimal | Aggressive subsampling
%     Shows convergence speed and final error floor per configuration.
%
%   PART C — Training loss curves
%     Training loss trajectory for the four configs. Demonstrates that
%     small nu (shrinkage) leads to slower but more stable descent.
%
% ASSUMPTIONS:
%   - ESLMixtureGenerator (2-class, N=600) provides a separable binary
%     problem with a well-defined Bayes error around 0.05.
%   - GBM uses CrossEntropyLoss (binary) throughout. OutputMode='binary'.
%   - All configs use the same random seed and train/test split.
%   - Bagging approximation: eta=1.0, nu=1e-6 (memory -> 0).
%
% REPRODUCIBILITY:  rng(42) fixed at top.
% GENERALIZATION:   No parameters hardcoded from expected results.
%
% Author: Ocean Library
% Date:   2026-05-14

clear; clc; close all;
rng(42);

% =========================================================================
%  CONFIGURATION
% =========================================================================
N_TOTAL   = 600;      % total samples
TRAIN_FRAC = 0.70;    % 70/30 split
NUM_TREES  = 100;     % trees per config (convergence analysis)
MAX_DEPTH  = 3;       % base tree depth (shallow = weak learner)
MIN_LEAF   = 5;       % minimum leaf size

% ISLE two-knob grid (Part A)
ETA_GRID = [1.0, 0.5, 0.2];   % SubsampleFraction values
NU_GRID  = [0.01, 0.1, 0.5, 1.0];  % LearningRate values

% Four representative configs for Parts B & C (name, eta, nu)
CONFIGS = {
    'Bagging approx  (eta=1.0, nu~0)',  1.0, 1e-4;
    'Standard GBM    (eta=1.0, nu=0.1)', 1.0, 0.10;
    'ISLE optimal    (eta=0.5, nu=0.1)', 0.5, 0.10;
    'Aggressive sub  (eta=0.2, nu=0.1)', 0.2, 0.10;
};

% Colors for the four configs (vibrant, distinct)
CFG_COLORS = [
    0.95 0.45 0.20;   % orange  — Bagging
    0.20 0.70 1.00;   % blue    — Standard GBM
    0.30 0.95 0.45;   % green   — ISLE optimal
    0.90 0.25 0.65;   % pink    — Aggressive
];

RESULTS_DIR = fullfile('Results', 'Supervised');
BG = [0.11 0.11 0.14];   % figure background

% =========================================================================
%  DATA
% =========================================================================
fprintf('[Setup] Generating ESL 2-class mixture (N=%d)...\n', N_TOTAL);

gen = OceanData.Generators.ESLMixtureGenerator('N', N_TOTAL, 'Seed', 42);
[X_all, Y_raw] = gen.generate();

% ESLMixtureGenerator returns labels 1..K — convert to binary +1/-1 style
% GBM binary path expects class labels {1,2} or {0,1}. Keep as {1,2}.
Y_all = Y_raw;

% Train / test split (fixed permutation from rng(42))
N     = size(X_all, 1);
idx   = randperm(N);
n_tr  = floor(TRAIN_FRAC * N);
X_tr  = X_all(idx(1:n_tr), :);      Y_tr = Y_all(idx(1:n_tr));
X_te  = X_all(idx(n_tr+1:end), :);  Y_te = Y_all(idx(n_tr+1:end));

fprintf('[Setup] Train=%d  Test=%d  Features=%d\n', ...
    n_tr, numel(Y_te), size(X_tr, 2));

% Loss function — CrossEntropyLoss (detects K=2 automatically)
loss = OceanStats.Loss.Classification.CrossEntropyLoss();

% =========================================================================
%  PART A — (eta x nu) Error Heatmap
% =========================================================================
fprintf('\n[Part A] Sweeping %d x %d (eta, nu) grid...\n', ...
    numel(ETA_GRID), numel(NU_GRID));

nEta   = numel(ETA_GRID);
nNu    = numel(NU_GRID);
err_grid = nan(nEta, nNu);   % rows=eta, cols=nu

for ie = 1:nEta
    for in_ = 1:nNu
        eta_i = ETA_GRID(ie);
        nu_i  = NU_GRID(in_);

        gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine( ...
            'Loss',              loss, ...
            'NumTrees',          NUM_TREES, ...
            'LearningRate',      nu_i, ...
            'SubsampleFraction', eta_i, ...
            'MaxDepth',          MAX_DEPTH, ...
            'MinLeafSize',       MIN_LEAF, ...
            'Verbose',           false);
        gbm.fit(X_tr, Y_tr);

        % Predict: max probability class (1 or 2)
        prob  = gbm.predict(X_te);
        [~, Y_hat] = max(prob, [], 2);
        err_grid(ie, in_) = mean(Y_hat ~= Y_te);

        fprintf('  eta=%.2f  nu=%.4f  TestErr=%.4f\n', eta_i, nu_i, err_grid(ie, in_));
    end
end

% --- Figure 1: Heatmap ---
fig1 = figure('Name', 'ISLE — (eta, nu) Error Surface', ...
    'Color', BG, 'Position', [50 80 820 520]);
ax1 = axes(fig1);

imagesc(ax1, err_grid);
colormap(ax1, parula);
cb = colorbar(ax1);
cb.Color = 'w';
cb.Label.String = 'Test Error Rate';
cb.Label.Color  = 'w';
cb.Label.FontSize = 11;

% Dynamically compute text contrast threshold
err_min = min(err_grid(:));
err_max = max(err_grid(:));
threshold = err_min + 0.5 * (err_max - err_min);

% Annotate each cell with the error value
for ie = 1:nEta
    for in_ = 1:nNu
        val = err_grid(ie, in_);
        % Use white text for dark cells (low values in parula), black for light cells
        if val < threshold
            txt_color = 'w';
        else
            txt_color = 'k';
        end
        text(ax1, in_, ie, sprintf('%.3f', val), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', txt_color, 'FontSize', 10, 'FontWeight', 'bold');
    end
end

set(ax1, 'XTick', 1:nNu, 'XTickLabel', arrayfun(@(v) sprintf('%.4g', v), NU_GRID, ...
        'UniformOutput', false), ...
    'YTick', 1:nEta, 'YTickLabel', arrayfun(@(v) sprintf('%.2f', v), ETA_GRID, ...
        'UniformOutput', false), ...
    'Color', BG, 'XColor', 'w', 'YColor', 'w', 'FontSize', 11, ...
    'XAxisLocation', 'bottom');

xlabel(ax1, 'Learning Rate  \nu  (memory knob)', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax1, 'Subsample Fraction  \eta  (randomness knob)', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');
title(ax1, 'ISLE Grand Unification — Test Error Surface (\eta \times \nu)', ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 14);

% =========================================================================
%  PART B — Error vs NumTrees (convergence curves)
% =========================================================================
fprintf('\n[Part B] Training %d configs across %d tree counts...\n', ...
    size(CONFIGS, 1), NUM_TREES);

tree_counts  = unique(round(linspace(1, NUM_TREES, 25)));
nTC          = numel(tree_counts);
nCfg         = size(CONFIGS, 1);
err_curves   = nan(nCfg, nTC);

for ic = 1:nCfg
    cfg_name = CONFIGS{ic, 1};
    eta_c    = CONFIGS{ic, 2};
    nu_c     = CONFIGS{ic, 3};
    fprintf('  Config: %s\n', cfg_name);

    for it = 1:nTC
        B = tree_counts(it);
        gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine( ...
            'Loss',              loss, ...
            'NumTrees',          B, ...
            'LearningRate',      nu_c, ...
            'SubsampleFraction', eta_c, ...
            'MaxDepth',          MAX_DEPTH, ...
            'MinLeafSize',       MIN_LEAF, ...
            'Verbose',           false);
        gbm.fit(X_tr, Y_tr);

        prob  = gbm.predict(X_te);
        [~, Y_hat] = max(prob, [], 2);
        err_curves(ic, it) = mean(Y_hat ~= Y_te);
    end
end

% --- Figure 2: Convergence curves ---
fig2 = figure('Name', 'ISLE — Error vs NumTrees', ...
    'Color', BG, 'Position', [900 80 760 470]);
ax2 = axes(fig2);

hold(ax2, 'on');
line_styles = {'-o', '-s', '-^', '-d'};
for ic = 1:nCfg
    plot(ax2, tree_counts, err_curves(ic, :), line_styles{ic}, ...
        'Color',      CFG_COLORS(ic, :), ...
        'LineWidth',  2.2, ...
        'MarkerSize', 5, ...
        'DisplayName', CONFIGS{ic, 1});
end
hold(ax2, 'off');

xlabel(ax2, 'Number of Trees  (M)', 'Color', 'w', 'FontSize', 12);
ylabel(ax2, 'Test Error Rate',      'Color', 'w', 'FontSize', 12);
title(ax2,  'ISLE — Convergence: Error vs NumTrees', ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 13);
legend(ax2, 'Location', 'northeast', 'TextColor', 'w', ...
    'Color', BG, 'EdgeColor', [0.4 0.4 0.5], 'FontSize', 9);
set(ax2, 'Color', BG, 'XColor', 'w', 'YColor', 'w', 'FontSize', 11, ...
    'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.5);
grid(ax2, 'on');
xlim(ax2, [0 NUM_TREES + 5]);

% =========================================================================
%  PART C — Training Loss Curves (full NUM_TREES)
% =========================================================================
fprintf('\n[Part C] Training loss curves (%d trees each)...\n', NUM_TREES);

fig3 = figure('Name', 'ISLE — Training Loss Curves', ...
    'Color', BG, 'Position', [50 580 760 420]);
ax3 = axes(fig3);
hold(ax3, 'on');

for ic = 1:nCfg
    eta_c = CONFIGS{ic, 2};
    nu_c  = CONFIGS{ic, 3};

    gbm = OceanStats.Boosting.Algorithms.GradientBoostingMachine( ...
        'Loss',              loss, ...
        'NumTrees',          NUM_TREES, ...
        'LearningRate',      nu_c, ...
        'SubsampleFraction', eta_c, ...
        'MaxDepth',          MAX_DEPTH, ...
        'MinLeafSize',       MIN_LEAF, ...
        'Verbose',           false);
    gbm.fit(X_tr, Y_tr);
    L = gbm.trainLoss();

    plot(ax3, 1:NUM_TREES, L, '-', ...
        'Color',       CFG_COLORS(ic, :), ...
        'LineWidth',   2.0, ...
        'DisplayName', CONFIGS{ic, 1});
end
hold(ax3, 'off');

xlabel(ax3, 'Boosting Round  (m)', 'Color', 'w', 'FontSize', 12);
ylabel(ax3, 'Training Loss',       'Color', 'w', 'FontSize', 12);
title(ax3,  'ISLE — Training Loss Descent per Configuration', ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 13);
legend(ax3, 'Location', 'northeast', 'TextColor', 'w', ...
    'Color', BG, 'EdgeColor', [0.4 0.4 0.5], 'FontSize', 9);
set(ax3, 'Color', BG, 'XColor', 'w', 'YColor', 'w', 'FontSize', 11, ...
    'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.5);
grid(ax3, 'on');
xlim(ax3, [1 NUM_TREES]);

% =========================================================================
%  SUMMARY TABLE
% =========================================================================
fprintf('\n============================================================\n');
fprintf('  ISLE Knob Summary — Final Test Error at M=%d\n', NUM_TREES);
fprintf('============================================================\n');
fprintf('  %-42s  TestErr\n', 'Config');
fprintf('  %s\n', repmat('-', 1, 52));
for ic = 1:nCfg
    fprintf('  %-42s  %.4f\n', CONFIGS{ic, 1}, err_curves(ic, end));
end

% Best cell in the heatmap
[best_err, best_lin] = min(err_grid(:));
[best_ie, best_in]   = ind2sub(size(err_grid), best_lin);
fprintf('\n  Heatmap best: eta=%.2f  nu=%.4f  TestErr=%.4f\n', ...
    ETA_GRID(best_ie), NU_GRID(best_in), best_err);
fprintf('============================================================\n');

% =========================================================================
%  SAVE RESULTS
% =========================================================================
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

exportgraphics(fig1, fullfile(RESULTS_DIR, 'exp_isle_heatmap.png'), ...
    'Resolution', 150, 'BackgroundColor', BG);
exportgraphics(fig2, fullfile(RESULTS_DIR, 'exp_isle_convergence.png'), ...
    'Resolution', 150, 'BackgroundColor', BG);
exportgraphics(fig3, fullfile(RESULTS_DIR, 'exp_isle_train_loss.png'), ...
    'Resolution', 150, 'BackgroundColor', BG);

fprintf('\n[Done] Figures saved to %s\n', RESULTS_DIR);
