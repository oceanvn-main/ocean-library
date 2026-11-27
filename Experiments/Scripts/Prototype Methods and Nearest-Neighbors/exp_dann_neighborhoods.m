% EXP_DANN_NEIGHBORHOODS  ESL Section 13.4.1 — DANN Adaptive Neighborhoods.
%
%   Reproduces ESL Figure 13.14:
%   Visualises the elliptical neighborhoods produced by the Discriminant
%   Adaptive Nearest Neighbors (DANN) procedure on a two-class concentric
%   rings dataset.
%
%   DATASET:        ConcentricRings (N=400, p=2) via ShapesGenerator
%                   Inner ring radius=0.5, outer ring radius=1.5, noise=0.15
%   MODEL:          DANN: Knn=50 (metric estimation), K=15 (final vote), epsilon=1
%   ASSUMPTIONS:    - Two concentric ring-shaped classes mimic the ESL setup
%                     where one class surrounds the other.
%                   - Knn=50 provides enough local samples for stable
%                     covariance estimation.
%                   - Epsilon=1 controls rounding from infinite strip to
%                     ellipsoid (ESL recommends =1).
%   REPRODUCIBILITY: rng(42, 'twister')
%
% Pipeline: 8-step Ocean Library ML Pipeline
% Author: Ocean Library   Date: 2026-04-06
% Ref: Hastie & Tibshirani (1996); Hastie et al. (2009), ESL §13.4.1, Fig 13.14
clc; clear; close all; rng(42, 'twister');

%% === 1. Data Generation ===
fprintf('=== Step 1: Data Generation ===\n');

gen = OceanData.Generators.ShapesGenerator( ...
    'N', 400, 'Shape', 'ConcentricRings', ...
    'Noise', 0.15, 'R_inner', 0.5, 'R_outer', 1.5);
gen.Seed = 42;
[X, Y] = gen.generate();

fprintf('  N=%d, p=%d, Classes=%s\n', size(X,1), size(X,2), mat2str(unique(Y)'));

%% === 2-3. No Feature Engineering / No Split (visualization task) ===
fprintf('\n[Steps 2-5] [SKIP] Direct visualization task.\n');

%% === 4. Model Configuration ===
fprintf('\n=== Step 4: Model Configuration ===\n');

KNN_METRIC = 50;   % neighbors for local metric estimation
K_FINAL    = 15;    % neighbors for final vote
EPS_DANN   = 1.0;   % regularization

fprintf('  Knn=%d (metric), K=%d (vote), epsilon=%.1f\n', KNN_METRIC, K_FINAL, EPS_DANN);

dann = OceanStats.Supervised.Neighbors.DANNClassifier( ...
    'K', K_FINAL, 'Knn', KNN_METRIC, 'Epsilon', EPS_DANN);
dann.fit(X, Y);

%% === 5-6. Classification & Evaluation ===
fprintf('\n=== Step 6: Evaluation ===\n');

y_pred = dann.predict(X);
trainErr = mean(y_pred ~= Y);
fprintf('  DANN Train Error: %.3f (%.1f%%)\n', trainErr, trainErr*100);

% Compare with standard KNN
knn_std = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_FINAL);
knn_std.fit(X, Y);
y_pred_knn = knn_std.predict(X);
trainErr_knn = mean(y_pred_knn ~= Y);
fprintf('  KNN  Train Error: %.3f (%.1f%%)\n', trainErr_knn, trainErr_knn*100);

%% === 7. Visualization (ESL Figure 13.14) ===
fprintf('\n=== Step 7: Visualization ===\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ESL Figure 13.14 — DANN Adaptive Neighborhoods');
fig.Position = [50, 50, 800, 750];
ax  = axes('Parent', fig);
hold(ax, 'on'); viz.styleAxes(ax);

% ---- Scatter the two classes ----
colInner = [0.30 0.65 0.95];   % blue (inner ring, class 0)
colOuter = [1.00 0.65 0.15];   % orange (outer ring, class 1)

mask0 = Y == 0;
mask1 = Y == 1;
scatter(ax, X(mask0,1), X(mask0,2), 25, colInner, 'o', ...
    'LineWidth', 1.0, 'MarkerEdgeAlpha', 0.7, ...
    'DisplayName', 'Class 0 (inner)');
scatter(ax, X(mask1,1), X(mask1,2), 25, colOuter, 'o', ...
    'LineWidth', 1.0, 'MarkerEdgeAlpha', 0.7, ...
    'DisplayName', 'Class 1 (outer)');

% ---- Select query points at diverse locations ----
% Near boundary (between rings), in pure inner, in pure outer
angles_query = [0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4];
radii_query  = [1.0, 1.0, 1.0, 1.0,   ... % boundary region (between rings)
                0.3, 0.0, 1.8, 0.5];       % mixed: inner pure, center, outer pure, inner
nQ = numel(angles_query);

queryPts = zeros(nQ, 2);
for q = 1:nQ
    queryPts(q,:) = radii_query(q) * [cos(angles_query(q)), sin(angles_query(q))];
end

% ---- Compute DANN metric at each query and draw ellipses ----
colEllipse = [0.30 0.90 0.50];  % green ellipses

nEllipsePoints = 100;
theta_ell = linspace(0, 2*pi, nEllipsePoints)';

for q = 1:nQ
    x0 = queryPts(q, :);
    
    % Get local metric
    [Sigma, nn_idx] = dann.getLocalMetric(x0);
    
    % Compute ellipse from Sigma^{-1} (the "neighborhood shape")
    % The neighborhood is {x : (x-x0)' Sigma (x-x0) <= r^2}
    % We need eigendecomposition of Sigma^{-1} to get the ellipse axes
    Sigma_inv = inv(Sigma);
    Sigma_inv = (Sigma_inv + Sigma_inv') / 2;  % symmetrize
    [V, D] = eig(Sigma_inv);
    d_eig = diag(D);
    d_eig = max(d_eig, 1e-10);
    
    % Scale: use the K-th neighbor's adaptive distance as the radius
    % This makes the ellipse enclose exactly the K nearest adaptive neighbors
    diffs_nn = X(nn_idx, :) - x0;
    d2_adapt = sum((diffs_nn * Sigma) .* diffs_nn, 2);
    r2 = max(d2_adapt);  % radius² enclosing all K neighbors
    
    % Unit circle → ellipse
    circle = [cos(theta_ell), sin(theta_ell)];
    ellipse = circle * diag(sqrt(d_eig * r2)) * V';
    ellipse = ellipse + x0;
    
    plot(ax, ellipse(:,1), ellipse(:,2), '-', ...
        'Color', [colEllipse, 0.8], 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
    
    % Mark query point with +
    plot(ax, x0(1), x0(2), '+', 'Color', [1 1 1], ...
        'MarkerSize', 10, 'LineWidth', 2.0, ...
        'HandleVisibility', 'off');
end

% ---- Styling ----
axis(ax, 'equal');
xlabel(ax, 'X_1', 'Color', viz.Colors.Text, 'FontSize', 12);
ylabel(ax, 'X_2', 'Color', viz.Colors.Text, 'FontSize', 12);
title(ax, 'DANN Neighborhoods (ESL Figure 13.14)', ...
    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
legend(ax, 'Location', 'northeast', 'FontSize', 10, ...
    'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
hold(ax, 'off');

% Save
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig, fullfile(resultsDir, 'fig13_14_dann_neighborhoods.png'));
fprintf('[Step 7] Saved: fig13_14_dann_neighborhoods.png\n');

%% === 8. Summary ===
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ConcentricRings (N=%d, p=%d)\n', size(X,1), size(X,2));
fprintf('  Model         : DANN (Knn=%d, K=%d, eps=%.1f)\n', KNN_METRIC, K_FINAL, EPS_DANN);
fprintf('  DANN Train Err: %.4f\n', trainErr);
fprintf('  KNN  Train Err: %.4f (standard %d-NN baseline)\n', trainErr_knn, K_FINAL);
fprintf('  Query Points  : %d (diverse radii/angles)\n', nQ);
fprintf('==========================\n');
