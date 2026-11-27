% EXP_EXTRA_TREES_COMPARISON
% ESL §15 / Geurts et al. (2006) — RF vs Extra-Trees Empirical Comparison
%
% EXPERIMENT OVERVIEW:
%   PART A — Test error vs NumTrees:
%     RF vs ExtraTrees (default: no bootstrap, random thresholds) on a
%     synthetic ESL mixture dataset. Shows ExtraTrees converges faster.
%
%   PART B — Feature importance comparison:
%     Side-by-side MDI importance bars for RF (MDA) and ExtraTrees (MDI).
%
%   PART C — Proximity Matrix visualization:
%     Heatmap (sorted by class) + MDS scatter coloured by class.
%     Uses ExtraTreesForest to build the proximity matrix on training data.
%
% ASSUMPTIONS:
%   - ESLMixtureGenerator produces a separable 2-class mixture.
%   - ExtraTrees defaults (UseBootstrap=false, NumCandidates=1) are used.
%   - Proximity computed on N_sub=80 training samples for speed.
%
% REPRODUCIBILITY: rng(42) and Seed=42 for all models.
%
% Author: Ocean Library
% Date:   2026-05-12

clear; clc; close all;
rng(42);

% =========================================================================
%  Data
% =========================================================================
fprintf('[Setup] Generating ESL mixture data...\n');
gen = OceanStats.Data.Generators.ESLMixtureGenerator( ...
    'NumClasses', 2, 'NSamples', 300, 'Seed', 42);
[X_all, Y_all] = gen.generate();

N   = size(X_all, 1);
idx = randperm(N);
n_tr = floor(0.70 * N);
X_tr = X_all(idx(1:n_tr), :);    Y_tr = Y_all(idx(1:n_tr));
X_te = X_all(idx(n_tr+1:end),:); Y_te = Y_all(idx(n_tr+1:end));
p    = size(X_tr, 2);
fprintf('[Setup] Train=%d  Test=%d  p=%d\n', n_tr, numel(Y_te), p);

% =========================================================================
%  PART A — Error vs NumTrees
% =========================================================================
Bs = [1 5 10 20 30 50 75 100];
err_rf = zeros(1, numel(Bs));
err_et = zeros(1, numel(Bs));

fprintf('\n[Part A] Sweeping NumTrees...\n');
for i = 1:numel(Bs)
    B = Bs(i);

    rf = OceanStats.Ensemble.RandomForest('NumTrees', B, ...
        'ComputeOOB', false, 'ComputeImportance', false, 'Seed', 42);
    rf.fit(X_tr, Y_tr);
    err_rf(i) = mean(rf.predict(X_te) ~= Y_te);

    et = OceanStats.Ensemble.ExtraTreesForest('NumTrees', B, ...
        'ComputeImportance', false, 'Seed', 42);
    et.fit(X_tr, Y_tr);
    err_et(i) = mean(et.predict(X_te) ~= Y_te);

    fprintf('  B=%3d  RF=%.4f  ET=%.4f\n', B, err_rf(i), err_et(i));
end

fig1 = figure('Name', 'RF vs ExtraTrees Error', ...
    'Color', [0.10 0.10 0.13], 'Position', [30 100 820 440]);
ax1  = axes(fig1, 'Color', [0.10 0.10 0.13], 'XColor','w', 'YColor','w');
hold(ax1,'on');
plot(ax1, Bs, err_rf, '-o', 'Color', [0.30 0.70 1.00], ...
    'LineWidth', 2.5, 'MarkerSize', 7, 'DisplayName', 'Random Forest');
plot(ax1, Bs, err_et, '-s', 'Color', [0.95 0.55 0.15], ...
    'LineWidth', 2.5, 'MarkerSize', 7, 'DisplayName', 'Extra-Trees (K=1)');
hold(ax1,'off');
xlabel(ax1, 'Number of Trees', 'Color','w', 'FontSize',12);
ylabel(ax1, 'Test Error Rate',  'Color','w', 'FontSize',12);
title(ax1,  'RF vs Extra-Trees — Test Error Convergence', ...
    'Color','w','FontWeight','bold','FontSize',14);
legend(ax1, 'Location','northeast','TextColor','w','Box','off');
grid(ax1,'on'); set(ax1,'GridColor',[0.3 0.3 0.3],'GridAlpha',0.5);
xlim(ax1,[0, max(Bs)+5]); ylim(ax1,[0, max([err_rf,err_et])*1.15]);

% =========================================================================
%  PART B — Feature Importance (RF-MDA vs ET-MDI)
% =========================================================================
fprintf('\n[Part B] Training B=100 models for importance...\n');

rf_imp = OceanStats.Ensemble.RandomForest('NumTrees', 100, ...
    'ComputeOOB', true, 'ComputeImportance', true, 'ImportanceMethod', 'MDA', ...
    'Verbose', true, 'Seed', 42);
rf_imp.fit(X_tr, Y_tr);

et_imp = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 100, ...
    'ComputeImportance', true, 'ImportanceMethod', 'MDI', ...
    'Verbose', true, 'Seed', 42);
et_imp.fit(X_tr, Y_tr);

imp_rf = rf_imp.featureImportance();
imp_et = et_imp.featureImportance();

% Normalize ET to [0,1] for comparison (MDI is already normalized)
if max(imp_rf) > 0, imp_rf_n = imp_rf / max(imp_rf); else, imp_rf_n = imp_rf; end
if max(imp_et) > 0, imp_et_n = imp_et / max(imp_et); else, imp_et_n = imp_et; end

[~, ord_rf] = sort(imp_rf_n, 'descend');
labels = arrayfun(@(k) sprintf('F%d', k), ord_rf, 'UniformOutput', false);

fig2 = figure('Name', 'Feature Importance Comparison', ...
    'Color', [0.10 0.10 0.13], 'Position', [870 100 700 430]);
ax2  = axes(fig2, 'Color', [0.10 0.10 0.13], 'XColor','w', 'YColor','w');
hold(ax2,'on');
bar_x = 1:p;
bar(ax2, bar_x - 0.2, imp_rf_n(ord_rf), 0.35, ...
    'FaceColor', [0.30 0.70 1.00], 'DisplayName', 'RF (MDA, norm.)');
bar(ax2, bar_x + 0.2, imp_et_n(ord_rf), 0.35, ...
    'FaceColor', [0.95 0.55 0.15], 'DisplayName', 'ExtraTrees (MDI, norm.)');
hold(ax2,'off');
set(ax2, 'XTick', 1:p, 'XTickLabel', labels, 'FontSize', 11);
xlabel(ax2, 'Feature', 'Color','w','FontSize',12);
ylabel(ax2, 'Importance (normalized)', 'Color','w','FontSize',12);
title(ax2, 'Feature Importance: RF-MDA vs ExtraTrees-MDI', ...
    'Color','w','FontWeight','bold','FontSize',13);
legend(ax2,'Location','northeast','TextColor','w','Box','off');
grid(ax2,'on'); set(ax2,'GridColor',[0.3 0.3 0.3],'GridAlpha',0.5);

% =========================================================================
%  PART C — Proximity Matrix
% =========================================================================
fprintf('\n[Part C] Computing proximity matrix (N_sub=80)...\n');

N_sub  = min(80, n_tr);
sub_idx = 1:N_sub;
X_sub   = X_tr(sub_idx, :);
Y_sub   = Y_tr(sub_idx);

P_prox = OceanStats.Ensemble.ProximityMatrix.compute(et_imp, X_sub, 'Verbose', true);
out_sc = OceanStats.Ensemble.ProximityMatrix.outlierScores(P_prox);

% --- C1: Sorted heatmap
fig3 = figure('Name', 'Proximity Heatmap', ...
    'Color', [0.10 0.10 0.13], 'Position', [30 580 550 450]);
ax3  = axes(fig3, 'Color', [0.10 0.10 0.13]);
OceanStats.Ensemble.ProximityMatrix.visualize(P_prox, ax3, Y_sub, ...
    'Title', 'Proximity Matrix (sorted by class)', 'Colormap', 'parula');
set(ax3, 'XColor','w','YColor','w','FontSize',10);
title(ax3, 'ExtraTrees Proximity (N=80, sorted)', 'Color','w','FontWeight','bold','FontSize',13);

% --- C2: MDS scatter
[coords, stress] = OceanStats.Ensemble.ProximityMatrix.mdsCoords(P_prox, 2);
classes_sub = unique(Y_sub);
colors_map  = {[0.30 0.70 1.00], [0.95 0.40 0.30]};

fig4 = figure('Name', 'Proximity MDS', ...
    'Color', [0.10 0.10 0.13], 'Position', [600 580 600 450]);
ax4  = axes(fig4, 'Color', [0.10 0.10 0.13], 'XColor','w', 'YColor','w');
hold(ax4,'on');
for c = 1:numel(classes_sub)
    mask = (Y_sub == classes_sub(c));
    scatter(ax4, coords(mask,1), coords(mask,2), 60, ...
        'filled', 'MarkerFaceColor', colors_map{c}, ...
        'MarkerFaceAlpha', 0.8, ...
        'DisplayName', sprintf('Class %d', classes_sub(c)));
end
% Annotate top-3 outliers
[~, out_ord] = sort(out_sc, 'descend');
for k = 1:min(3, N_sub)
    oi = out_ord(k);
    text(ax4, coords(oi,1), coords(oi,2), sprintf(' #%d', oi), ...
        'Color', [1 1 0.4], 'FontSize', 8);
end
hold(ax4,'off');
xlabel(ax4,'MDS dim 1','Color','w','FontSize',11);
ylabel(ax4,'MDS dim 2','Color','w','FontSize',11);
title(ax4, sprintf('Proximity MDS (stress=%.3f)', stress), ...
    'Color','w','FontWeight','bold','FontSize',13);
legend(ax4,'Location','best','TextColor','w','Box','off');
grid(ax4,'on'); set(ax4,'GridColor',[0.3 0.3 0.3],'GridAlpha',0.5);

% =========================================================================
%  Save
% =========================================================================
resultsDir = fullfile('Results', 'Supervised');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
exportgraphics(fig1, fullfile(resultsDir, 'exp_et_error_vs_trees.png'),   'Resolution', 150, 'BackgroundColor', [0.10 0.10 0.13]);
exportgraphics(fig2, fullfile(resultsDir, 'exp_et_importance.png'),        'Resolution', 150, 'BackgroundColor', [0.10 0.10 0.13]);
exportgraphics(fig3, fullfile(resultsDir, 'exp_et_proximity_heatmap.png'), 'Resolution', 150, 'BackgroundColor', [0.10 0.10 0.13]);
exportgraphics(fig4, fullfile(resultsDir, 'exp_et_proximity_mds.png'),     'Resolution', 150, 'BackgroundColor', [0.10 0.10 0.13]);
fprintf('\n[Done] 4 figures saved to %s\n', resultsDir);

rf_imp.printSummary();
et_imp.printSummary();
