%% exp_figure10_2_adaboost.m
%
% PURPOSE:
%   Professional replication of ESL Figure 10.2 (Hastie, Tibshirani,
%   Friedman, 2009, p. 339) with an additional 2D decision boundary panel.
%   All visualizations rendered through OceanViz.Core (Dark theme).
%
% REFERENCE:
%   Hastie, T., Tibshirani, R., & Friedman, J. (2009).
%   The Elements of Statistical Learning (2nd ed.). Springer.
%
% DATASET:
%   N_train = 2000  |  N_test = 10000  |  p = 10 features
%   X_ij ~ N(0,1) i.i.d.
%   Y =  1  if  sum(X_j^2, j=1..10) > 9.34
%   Y = -1  otherwise.
%   Threshold 9.34 is the median of chi-squared(10).
%
% MODELS:
%   A  Single Decision Stump   (MaxNumSplits = 1)
%   B  Large Decision Tree     (MaxNumSplits = 244)
%   C  AdaBoost.M1             (T = 400 stumps, LearnRate = 1)
%
% OUTPUT:
%   Figure 1  Learning curve — test error vs. boosting round (ESL Fig 10.2)
%   Figure 2  1x3 decision boundary panel in (X1, X2) cross-section
%
% REQUIRES:
%   Statistics and Machine Learning Toolbox (fitctree, fitensemble, loss)
%   OceanViz package (OceanViz.Core, Dark theme)
%
% COMMAND:
%   cd('c:\Ocean Library\Matlab code practice\')
%   run Experiments/exp_figure10_2_adaboost.m
% =========================================================================

clearvars; close all; clc;
rng(2009, 'twister');   % ESL edition year as canonical seed

% =========================================================================
%  SECTION 1: OceanViz Engine — initialise once, derive all colours here
% =========================================================================
viz = OceanViz.Core('Theme', 'Dark');
C   = viz.Colors;   % C.Primary, C.Secondary, C.Boundary,
                    % C.Background, C.Text, C.Grid, C.Scatter

% Semantic colour assignments
C_ADA   = C.Primary;    % AdaBoost learning curve
C_STUMP = C.Secondary;  % Single stump baseline
C_TREE  = C.Boundary;   % Large tree baseline  (also used as true boundary)

% Decision region colormaps: dark two-tone, readable on dark axes
CMAP_A = [0.06 0.12 0.32;  0.35 0.06 0.10];   % navy / crimson
CMAP_B = [0.06 0.22 0.14;  0.30 0.10 0.04];   % teal  / amber
CMAP_C = [0.08 0.06 0.26;  0.28 0.10 0.04];   % indigo / amber

% =========================================================================
%  SECTION 2: Data Generation
% =========================================================================
N_TRAIN   = 2000;
N_TEST    = 10000;
P         = 10;
THRESHOLD = 9.34;   % Median of chi-squared(10)

label_fn = @(X) 2 * double(sum(X.^2, 2) > THRESHOLD) - 1;  % {-1, +1}

X_train = randn(N_TRAIN, P);   Y_train = label_fn(X_train);
X_test  = randn(N_TEST,  P);   Y_test  = label_fn(X_test);

fprintf('=== ESL Figure 10.2 — AdaBoost.M1 Replication ===\n\n');
fprintf('Training : %d samples  |  +1: %d  |  -1: %d\n', ...
    N_TRAIN, sum(Y_train == 1), sum(Y_train == -1));
fprintf('Test     : %d samples  |  +1: %d  |  -1: %d\n\n', ...
    N_TEST,  sum(Y_test  == 1), sum(Y_test  == -1));

% =========================================================================
%  SECTION 3: Model Training
% =========================================================================
T_BOOST = 400;

fprintf('[1/3] Training Model A: Single Stump...\n'); tic;
mdl_A = fitctree(X_train, Y_train, 'MaxNumSplits', 1, 'MinLeafSize', 1);
fprintf('      Done in %.3f s\n\n', toc);

fprintf('[2/3] Training Model B: Large Tree (244 splits)...\n'); tic;
mdl_B = fitctree(X_train, Y_train, 'MaxNumSplits', 244, 'MinLeafSize', 1);
fprintf('      Done in %.3f s\n\n', toc);

fprintf('[3/3] Training Model C: AdaBoost.M1 (T=%d stumps)...\n', T_BOOST); tic;
t_stump = templateTree('MaxNumSplits', 1, 'MinLeafSize', 1);
mdl_C   = fitensemble(X_train, Y_train, 'AdaBoostM1', T_BOOST, t_stump, ...
              'LearnRate', 1.0);
fprintf('      Done in %.3f s\n\n', toc);

% =========================================================================
%  SECTION 4: Error Rate Computation
% =========================================================================
err_A = loss(mdl_A, X_test, Y_test);
err_B = loss(mdl_B, X_test, Y_test);

fprintf('Computing cumulative test error (%d rounds)...\n', T_BOOST); tic;
cum_err_C = loss(mdl_C, X_test, Y_test, 'Mode', 'Cumulative');
% cum_err_C is a (T x 1) column vector; entry m = error using first m learners.
fprintf('Done in %.3f s\n\n', toc);

[min_err_C, min_t] = min(cum_err_C);

fprintf('--- Results --------------------------------------------------\n');
fprintf('Model A  Stump         Test Error : %.4f  (%.2f%%)\n', err_A, err_A*100);
fprintf('Model B  Large Tree    Test Error : %.4f  (%.2f%%)\n', err_B, err_B*100);
fprintf('Model C  AdaBoost T=%d Final Error: %.4f  (%.2f%%)\n', T_BOOST, cum_err_C(end), cum_err_C(end)*100);
fprintf('                       Min   Error: %.4f  (%.2f%%)  @ T=%d\n', min_err_C, min_err_C*100, min_t);
fprintf('--------------------------------------------------------------\n\n');

% =========================================================================
%  SECTION 5: Figure 1 — Learning Curve  (ESL Figure 10.2)
%
%  OceanViz usage:
%    viz.createFigure()  — theme-consistent figure window
%    viz.styleAxes()     — applies Dark background, grid, font colours
%    Raw plot() on the styled ax — the existing strategies do not cover
%    a "learning curve vs. iteration" plot type; direct drawing is correct.
% =========================================================================
fig1 = viz.createFigure('ESL Figure 10.2 — AdaBoost.M1 Test Error vs. Iterations');
fig1.Position = [60, 120, 920, 530];

ax1 = axes(fig1);
viz.styleAxes(ax1);
hold(ax1, 'on');

% --- AdaBoost learning curve ---
plot(ax1, 1:T_BOOST, cum_err_C * 100, ...
    'Color',       C_ADA, ...
    'LineWidth',   2.2, ...
    'DisplayName', sprintf('AdaBoost.M1  (min %.2f%%  T=%d)', min_err_C*100, min_t));

% Minimum error marker
plot(ax1, min_t, min_err_C * 100, 'o', ...
    'MarkerFaceColor', C.Boundary, ...
    'MarkerEdgeColor', C.Boundary, ...
    'MarkerSize',      9, ...
    'HandleVisibility', 'off');
text(ax1, min_t + 6, min_err_C * 100 + 0.5, ...
    sprintf('T=%d, %.2f%%', min_t, min_err_C*100), ...
    'Color', C.Boundary, 'FontSize', 9.5);

% --- Horizontal reference lines ---
yline(ax1, err_A * 100, '--', ...
    'Color', C_STUMP, 'LineWidth', 1.8, ...
    'DisplayName', sprintf('Single Stump (%.2f%%)', err_A*100));

yline(ax1, err_B * 100, '--', ...
    'Color', C_TREE, 'LineWidth', 1.8, ...
    'DisplayName', sprintf('Large Tree (%.2f%%)', err_B*100));

hold(ax1, 'off');

xlabel(ax1, 'Boosting Iterations  (T)', 'FontSize', 13, 'Color', C.Text);
ylabel(ax1, 'Test Error Rate (%)',       'FontSize', 13, 'Color', C.Text);
title(ax1, 'ESL Fig. 10.2  —  AdaBoost.M1 on 10-D Gaussian Problem', ...
    'FontSize', 13.5, 'Color', C.Text, 'FontWeight', 'normal');
legend(ax1, 'Location', 'northeast', ...
    'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', C.Grid, 'FontSize', 10);

xlim(ax1, [1, T_BOOST]);
% Compute y-axis bounds safely using scalar horizontal concatenation — no
% vertcat mixing of column/row vectors (that was the original bug).
all_refs = [cum_err_C(1), cum_err_C(2), cum_err_C(3), err_A, err_B];
y_lo = max(0.0, min([min_err_C, err_A, err_B]) * 100 - 2.0);
y_hi = min(60,  max(all_refs) * 100 + 3.0);
ylim(ax1, [y_lo, y_hi]);

% =========================================================================
%  SECTION 6: Figure 2 — Decision Boundaries in (X1, X2) Cross-Section
%
%  The true rule is sum_{j=1}^10 x_j^2 > 9.34.
%  Setting x_3 = ... = x_10 = 0 (E[X_j] = 0 for Gaussian features):
%    x_1^2 + x_2^2 = 9.34   =>   circle, radius = sqrt(9.34) ~ 3.056
% =========================================================================
GRID_RES = 350;
g        = linspace(-4, 4, GRID_RES);
[G1, G2] = meshgrid(g, g);

% 10-D prediction matrix: X3..X10 are exactly 0
X_grid = [G1(:), G2(:), zeros(GRID_RES^2, P - 2)];

% True circular boundary
theta_c = linspace(0, 2*pi, 1200);
r_true  = sqrt(THRESHOLD);     % ~3.056
cx1     = r_true * cos(theta_c);
cx2     = r_true * sin(theta_c);

fprintf('Computing decision boundaries (%d x %d grid)...\n', GRID_RES, GRID_RES);
tic; Z_A = reshape(double(predict(mdl_A, X_grid)), GRID_RES, GRID_RES); fprintf('  Model A: %.3f s\n', toc);
tic; Z_B = reshape(double(predict(mdl_B, X_grid)), GRID_RES, GRID_RES); fprintf('  Model B: %.3f s\n', toc);
tic; Z_C = reshape(double(predict(mdl_C, X_grid)), GRID_RES, GRID_RES); fprintf('  Model C: %.3f s\n\n', toc);

% Binary maps: 1 = class +1, 0 = class -1 (two-tone imagesc)
Z_A_bin = double(Z_A == 1);
Z_B_bin = double(Z_B == 1);
Z_C_bin = double(Z_C == 1);

fig2 = viz.createFigure('Decision Boundaries — (X1, X2) Cross-Section');
fig2.Position = [60, 60, 1420, 520];

panels = { ...
    Z_A_bin, CMAP_A, 'Model A  —  Single Stump',                   err_A; ...
    Z_B_bin, CMAP_B, 'Model B  —  Large Tree (244 splits)',         err_B; ...
    Z_C_bin, CMAP_C, sprintf('Model C  —  AdaBoost.M1  T=%d', T_BOOST), cum_err_C(end)};

for k = 1:3
    Z_k    = panels{k, 1};
    cmap_k = panels{k, 2};
    ttl_k  = panels{k, 3};
    err_k  = panels{k, 4};

    ax_k = subplot(1, 3, k, 'Parent', fig2);
    viz.styleAxes(ax_k);
    set(ax_k, 'YDir', 'normal');   % imagesc reverses Y by default; correct it
    hold(ax_k, 'on');

    % Decision regions
    % imagesc: Z(row,col) -> position (x=g(col), y=g(row)) — matches meshgrid
    imagesc(ax_k, g, g, Z_k, [0, 1]);
    colormap(ax_k, cmap_k);

    % Predicted boundary edge (subtle grey contour)
    % Only draw contour if there is actually a boundary (Z_k is not constant)
    if length(unique(Z_k)) > 1
        contour(ax_k, G1, G2, Z_k, [0.5, 0.5], ...
            'Color', C.Scatter, 'LineWidth', 0.9);
    end

    % True circular boundary (OceanViz Boundary colour)
    plot(ax_k, cx1, cx2, 'Color', C.Boundary, 'LineWidth', 2.5);

    hold(ax_k, 'off');
    axis(ax_k, 'equal');
    xlim(ax_k, [-4, 4]);
    ylim(ax_k, [-4, 4]);

    title(ax_k, sprintf('%s\nTest Error = %.2f%%', ttl_k, err_k*100), ...
        'Color', C.Text, 'FontSize', 10.5, 'FontWeight', 'normal');
    xlabel(ax_k, 'X_1', 'Color', C.Text, 'FontSize', 12);
    if k == 1
        ylabel(ax_k, 'X_2', 'Color', C.Text, 'FontSize', 12);
    end

    % Inline class annotations (avoids colorbar clutter)
    text(ax_k, -3.7,  3.65, 'Class +1', 'Color', min(cmap_k(2,:)+0.45,1), 'FontSize', 8.5);
    text(ax_k, -3.7, -3.75, 'Class -1', 'Color', min(cmap_k(1,:)+0.45,1), 'FontSize', 8.5);
end

sgtitle(fig2, ...
    sprintf('Decision Boundaries (X_1, X_2 cross-section)  |  Circle = True Boundary  r=%.3f', r_true), ...
    'Color', C.Text, 'FontSize', 12);

fprintf('Done. Two figures rendered.\n');
