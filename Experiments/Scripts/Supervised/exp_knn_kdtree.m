% EXP_KNN_KDTREE
% KNN Decision Boundary & Bias-Variance Experiment
%
% OVERVIEW:
%   PART A — Decision boundary at K=1, 5, 15 on ESL mixture data (2D).
%   PART B — CV error vs K sweep (K=1..25) showing bias-variance tradeoff.
%   PART C — KNN vs RF test error comparison (B=100 RF, K=5 KNN).
%
% ASSUMPTIONS:
%   - ESLMixtureGenerator provides a 2D separable mixture (d=2 for plotting).
%   - The decision boundary is visualized on a grid (100x100 cells).
%
% REPRODUCIBILITY: rng(42), Seed=42.
%
% Author: Ocean Library
% Date:   2026-05-12

clear; clc; close all; rng(42);

% =========================================================================
%  Data
% =========================================================================
fprintf('[Setup] Generating ESL 2D mixture...\n');
gen = OceanStats.Data.Generators.ESLMixtureGenerator( ...
    'NumClasses', 2, 'NSamples', 200, 'Seed', 42);
[X_all, Y_all] = gen.generate();

% Use only first 2 features for boundary plot
X2 = X_all(:, 1:2);
N  = size(X2, 1);
idx = randperm(N);
n_tr = floor(0.70 * N);
X_tr2 = X2(idx(1:n_tr),:);     Y_tr = Y_all(idx(1:n_tr));
X_te2 = X2(idx(n_tr+1:end),:); Y_te = Y_all(idx(n_tr+1:end));

% =========================================================================
%  PART A — Decision Boundary
% =========================================================================
fprintf('\n[Part A] Decision boundaries for K = 1, 5, 15...\n');
Kvals_bound = [1, 5, 15];
colors_cls  = {[0.25 0.55 1.0], [1.0 0.40 0.30]};

% Build grid
x1r = linspace(min(X2(:,1))-0.5, max(X2(:,1))+0.5, 100);
x2r = linspace(min(X2(:,2))-0.5, max(X2(:,2))+0.5, 100);
[G1, G2] = meshgrid(x1r, x2r);
Xgrid = [G1(:), G2(:)];

fig1 = figure('Name','KNN Decision Boundaries', ...
    'Color',[0.10 0.10 0.13],'Position',[30 100 1100 370]);
classes = unique(Y_tr);

for ii = 1:3
    K_b = Kvals_bound(ii);
    knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_b);
    knn.fit(X_tr2, Y_tr);
    Z = knn.predict(Xgrid);
    Z = reshape(Z, size(G1));

    ax = subplot(1,3,ii, 'Parent', fig1);
    set(ax,'Color',[0.10 0.10 0.13],'XColor','w','YColor','w');
    hold(ax,'on');

    % Background decision regions
    contourf(ax, G1, G2, Z, [1.5 1.5], 'LineColor','none');
    colormap(ax, [0.15 0.35 0.65; 0.65 0.20 0.15]);
    caxis(ax,[1 2]);

    % Training points
    for c = 1:2
        mask = Y_tr == classes(c);
        scatter(ax, X_tr2(mask,1), X_tr2(mask,2), 22, ...
            'filled','MarkerFaceColor',colors_cls{c},'MarkerFaceAlpha',0.8);
    end
    hold(ax,'off');

    tr_err = mean(knn.predict(X_te2) ~= Y_te);
    title(ax, sprintf('K=%d  (test err=%.3f)', K_b, tr_err), ...
        'Color','w','FontSize',12,'FontWeight','bold');
    xlabel(ax,'x_1','Color','w'); ylabel(ax,'x_2','Color','w');
end
sgtitle(fig1,'KNN Decision Boundary vs K','Color','w','FontSize',14,'FontWeight','bold');

% =========================================================================
%  PART B — CV Error vs K
% =========================================================================
fprintf('[Part B] Sweeping K = 1..25 with 5-fold CV...\n');
Krange = 1:25;
cv_err = zeros(1, 25);

for K_i = Krange
    strat = OceanStats.Selection.Strategies.KNNStrategy('K', K_i);
    cv    = OceanStats.Selection.CrossValidation.CrossValidator( ...
        strat, OceanStats.Selection.CrossValidation.KFold(5));
    cv_err(K_i) = cv.evaluate(X_tr2, Y_tr);
    fprintf('  K=%2d  CV err=%.4f\n', K_i, cv_err(K_i));
end

[~, best_K] = min(cv_err);
fig2 = figure('Name','KNN CV Error vs K','Color',[0.10 0.10 0.13],'Position',[30 540 700 380]);
ax2  = axes(fig2,'Color',[0.10 0.10 0.13],'XColor','w','YColor','w');
plot(ax2, Krange, cv_err, '-o','Color',[0.30 0.70 1.0],'LineWidth',2.5,'MarkerSize',6);
xline(ax2, best_K,'--','Color',[0.95 0.55 0.15],'LineWidth',1.8,...
    'Label',sprintf('Best K=%d', best_K),'LabelColor',[0.95 0.55 0.15]);
xlabel(ax2,'K','Color','w','FontSize',12);
ylabel(ax2,'5-fold CV Error','Color','w','FontSize',12);
title(ax2,'KNN Bias-Variance Tradeoff','Color','w','FontWeight','bold','FontSize',14);
grid(ax2,'on'); set(ax2,'GridColor',[0.3 0.3 0.3]);

% =========================================================================
%  PART C — KNN vs RF
% =========================================================================
fprintf('\n[Part C] KNN (K=best) vs RF (B=100) test comparison...\n');

knn_best = OceanStats.Supervised.Neighbors.KNNClassifier('K', best_K);
knn_best.fit(X_tr2, Y_tr);
err_knn = mean(knn_best.predict(X_te2) ~= Y_te);

rf = OceanStats.Ensemble.RandomForest('NumTrees',100,'ComputeOOB',false, ...
    'ComputeImportance',false,'Seed',42);
rf.fit(X_tr2, Y_tr);
err_rf = mean(rf.predict(X_te2) ~= Y_te);

fprintf('  KNN (K=%d)  test error: %.4f\n', best_K, err_knn);
fprintf('  RF  (B=100) test error: %.4f\n', err_rf);

% =========================================================================
%  Save
% =========================================================================
resultsDir = fullfile('Results','Supervised');
if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
exportgraphics(fig1, fullfile(resultsDir,'exp_knn_boundaries.png'), ...
    'Resolution',150,'BackgroundColor',[0.10 0.10 0.13]);
exportgraphics(fig2, fullfile(resultsDir,'exp_knn_cv_error.png'), ...
    'Resolution',150,'BackgroundColor',[0.10 0.10 0.13]);
fprintf('\n[Done] Saved to %s\n', resultsDir);
