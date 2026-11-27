% EXP_KNN_VS_PROTOTYPE  ESL Section 13.3.1 — A Comparative Study
%
% Reproduces ESL Figure 13.5:
%   4-panel comparison of nearest-neighbors, K-means, and LVQ
%   on "easy" (linear boundary) and "difficult" (3D checkerboard) problems.
%
%   - 10 independent Uniform[0,1] features
%   - N_train = 100, N_test = 1000
%   - 10 realizations, report mean ± 1 SE
%   - Bayes error rate = 0 for both problems
%
% Pipeline: 8-step Ocean Library ML Pipeline
% Author: Ocean Library   Date: 2026-03-26
% Ref: Hastie et al. (2009), ESL Section 13.3.1, Figure 13.5
clc; clear; close all;

%% === Configuration ===
N_TRAIN     = 100;
N_TEST      = 1000;
P           = 10;       % feature dimensionality
N_REAL      = 10;       % number of realizations
RNG_SEED    = 42;

% Tuning parameter grids
Ks_knn   = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70];
Rs_proto = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30];

nK = numel(Ks_knn);
nR = numel(Rs_proto);

%% === Label Rules ===
% Easy:      Y = I(X1 > 0.5)
% Difficult: Y = I(sign(prod_{j=1}^3 (Xj - 0.5)) > 0)
labelEasy = @(X) double(X(:,1) > 0.5);
labelDiff = @(X) double(sign(prod(X(:,1:3) - 0.5, 2)) > 0);

%% === Preallocate Results ===
% KNN errors: (nK x N_REAL) for each problem
knnErr_easy = zeros(nK, N_REAL);
knnErr_diff = zeros(nK, N_REAL);

% K-means errors: (nR x N_REAL) for each problem
kmErr_easy  = zeros(nR, N_REAL);
kmErr_diff  = zeros(nR, N_REAL);

% LVQ errors: (nR x N_REAL) for each problem
lvqErr_easy = zeros(nR, N_REAL);
lvqErr_diff = zeros(nR, N_REAL);

%% === 1-6. Data Generation + Sweep (per realization) ===
fprintf('=== ESL 13.5: Comparative Study (%d realizations) ===\n', N_REAL);

rng(RNG_SEED);

for rep = 1:N_REAL
    fprintf('\n--- Realization %d/%d ---\n', rep, N_REAL);
    
    % Generate fresh train/test data
    X_train = rand(N_TRAIN, P);
    X_test  = rand(N_TEST,  P);
    
    y_train_easy = labelEasy(X_train);
    y_test_easy  = labelEasy(X_test);
    y_train_diff = labelDiff(X_train);
    y_test_diff  = labelDiff(X_test);
    
    % ------ KNN sweep ------
    for ki = 1:nK
        K_val = Ks_knn(ki);
        
        % Easy
        knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', K_val);
        knn.fit(X_train, y_train_easy);
        knnErr_easy(ki, rep) = mean(knn.predict(X_test) ~= y_test_easy);
        
        % Difficult
        knn.fit(X_train, y_train_diff);
        knnErr_diff(ki, rep) = mean(knn.predict(X_test) ~= y_test_diff);
    end
    
    % ------ K-means & LVQ sweep ------
    for ri = 1:nR
        R_val = Rs_proto(ri);
        
        % === K-means (per-class, then 1-NN on centroids) ===
        km = OceanStats.Unsupervised.KMeans();
        classes = [0, 1];
        
        % Easy
        protos_e = []; plabels_e = [];
        for c = 1:2
            X_c = X_train(y_train_easy == classes(c), :);
            R_use = min(R_val, size(X_c, 1));  % safety: can't exceed class size
            if R_use > 0
                km.fit(X_c, R_use);
                protos_e  = [protos_e; km.Centroids];
                plabels_e = [plabels_e; repmat(classes(c), R_use, 1)];
            end
        end
        % 1-NN on centroids
        knn1 = OceanStats.Supervised.Neighbors.KNNClassifier('K', 1);
        knn1.fit(protos_e, plabels_e);
        kmErr_easy(ri, rep) = mean(knn1.predict(X_test) ~= y_test_easy);
        
        % Difficult
        protos_d = []; plabels_d = [];
        for c = 1:2
            X_c = X_train(y_train_diff == classes(c), :);
            R_use = min(R_val, size(X_c, 1));
            if R_use > 0
                km.fit(X_c, R_use);
                protos_d  = [protos_d; km.Centroids];
                plabels_d = [plabels_d; repmat(classes(c), R_use, 1)];
            end
        end
        knn1.fit(protos_d, plabels_d);
        kmErr_diff(ri, rep) = mean(knn1.predict(X_test) ~= y_test_diff);
        
        % === LVQ ===
        % Easy
        lvq = OceanStats.Supervised.Neighbors.LVQ1('R', R_val, ...
            'MaxEpochs', 50, 'LearningRate', 0.1, 'Decay', 0.01);
        lvq.fit(X_train, y_train_easy);
        lvqErr_easy(ri, rep) = mean(lvq.predict(X_test) ~= y_test_easy);
        
        % Difficult
        lvq2 = OceanStats.Supervised.Neighbors.LVQ1('R', R_val, ...
            'MaxEpochs', 50, 'LearningRate', 0.1, 'Decay', 0.01);
        lvq2.fit(X_train, y_train_diff);
        lvqErr_diff(ri, rep) = mean(lvq2.predict(X_test) ~= y_test_diff);
    end
    
    fprintf('  KNN(1)  Easy=%.3f  Diff=%.3f\n', knnErr_easy(1,rep), knnErr_diff(1,rep));
    fprintf('  KNN(15) Easy=%.3f  Diff=%.3f\n', ...
        knnErr_easy(Ks_knn==15,rep), knnErr_diff(Ks_knn==15,rep));
    fprintf('  KM(5)   Easy=%.3f  Diff=%.3f\n', ...
        kmErr_easy(Rs_proto==5,rep), kmErr_diff(Rs_proto==5,rep));
end

%% === 7. Aggregate Results ===
fprintf('\n=== Step 7: Aggregated Results (mean ± SE) ===\n');

knn_mean_e = mean(knnErr_easy, 2);  knn_se_e = std(knnErr_easy, 0, 2) / sqrt(N_REAL);
knn_mean_d = mean(knnErr_diff, 2);  knn_se_d = std(knnErr_diff, 0, 2) / sqrt(N_REAL);
km_mean_e  = mean(kmErr_easy, 2);   km_se_e  = std(kmErr_easy, 0, 2) / sqrt(N_REAL);
km_mean_d  = mean(kmErr_diff, 2);   km_se_d  = std(kmErr_diff, 0, 2) / sqrt(N_REAL);
lvq_mean_e = mean(lvqErr_easy, 2);  lvq_se_e = std(lvqErr_easy, 0, 2) / sqrt(N_REAL);
lvq_mean_d = mean(lvqErr_diff, 2);  lvq_se_d = std(lvqErr_diff, 0, 2) / sqrt(N_REAL);

fprintf('\nKNN (selected K values):\n');
for ki = [1, find(Ks_knn==15), find(Ks_knn==25), nK]
    fprintf('  K=%2d  Easy=%.3f±%.3f  Diff=%.3f±%.3f\n', ...
        Ks_knn(ki), knn_mean_e(ki), knn_se_e(ki), knn_mean_d(ki), knn_se_d(ki));
end

fprintf('\nK-means / LVQ (selected R values):\n');
for ri = [1, find(Rs_proto==5), find(Rs_proto==10), nR]
    fprintf('  R=%2d  KM_Easy=%.3f±%.3f  LVQ_Easy=%.3f±%.3f  KM_Diff=%.3f±%.3f  LVQ_Diff=%.3f±%.3f\n', ...
        Rs_proto(ri), km_mean_e(ri), km_se_e(ri), lvq_mean_e(ri), lvq_se_e(ri), ...
        km_mean_d(ri), km_se_d(ri), lvq_mean_d(ri), lvq_se_d(ri));
end

%% === 8. Visualization ===
fprintf('\n=== Step 8: Visualization ===\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ESL Figure 13.5 — Comparative Study');
fig.Position = [50, 30, 1100, 900];

% Colors
colKNN = [0.25 0.55 0.95];   % blue
colKM  = [0.25 0.55 0.95];   % blue (same as ESL)
colLVQ = [0.90 0.25 0.25];   % red

% --- Panel positions: [left bottom width height] ---
panels = {
    [0.08  0.56  0.40  0.38],   % top-left:     KNN / Easy
    [0.56  0.56  0.40  0.38],   % top-right:    KM+LVQ / Easy
    [0.08  0.08  0.40  0.38],   % bottom-left:  KNN / Difficult
    [0.56  0.08  0.40  0.38],   % bottom-right: KM+LVQ / Difficult
};

% ---- Top-Left: KNN / Easy ----
ax1 = axes('Parent', fig, 'Position', panels{1});
hold(ax1, 'on'); viz.styleAxes(ax1);
errorbar(ax1, Ks_knn, knn_mean_e, knn_se_e, '-o', ...
    'Color', colKNN, 'MarkerFaceColor', colKNN, 'MarkerSize', 4, ...
    'LineWidth', 1.5, 'CapSize', 4);
xlabel(ax1, 'Number of Neighbors', 'Color', viz.Colors.Text);
ylabel(ax1, 'Misclassification Error', 'Color', viz.Colors.Text);
title(ax1, 'Nearest Neighbors / Easy', 'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold');
ylim(ax1, [0.1 0.5]);
hold(ax1, 'off');

% ---- Top-Right: K-means + LVQ / Easy ----
ax2 = axes('Parent', fig, 'Position', panels{2});
hold(ax2, 'on'); viz.styleAxes(ax2);
errorbar(ax2, Rs_proto, km_mean_e, km_se_e, '-o', ...
    'Color', colKM, 'MarkerFaceColor', colKM, 'MarkerSize', 4, ...
    'LineWidth', 1.5, 'CapSize', 4, 'DisplayName', 'K-means');
errorbar(ax2, Rs_proto, lvq_mean_e, lvq_se_e, '-s', ...
    'Color', colLVQ, 'MarkerFaceColor', colLVQ, 'MarkerSize', 4, ...
    'LineWidth', 1.5, 'CapSize', 4, 'DisplayName', 'LVQ');
xlabel(ax2, 'Number of Prototypes per Class', 'Color', viz.Colors.Text);
ylabel(ax2, 'Misclassification Error', 'Color', viz.Colors.Text);
title(ax2, 'K-means & LVQ / Easy', 'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold');
legend(ax2, 'Location', 'northeast', 'FontSize', 9, ...
    'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
ylim(ax2, [0.1 0.5]);
hold(ax2, 'off');

% ---- Bottom-Left: KNN / Difficult ----
ax3 = axes('Parent', fig, 'Position', panels{3});
hold(ax3, 'on'); viz.styleAxes(ax3);
errorbar(ax3, Ks_knn, knn_mean_d, knn_se_d, '-o', ...
    'Color', colKNN, 'MarkerFaceColor', colKNN, 'MarkerSize', 4, ...
    'LineWidth', 1.5, 'CapSize', 4);
xlabel(ax3, 'Number of Neighbors', 'Color', viz.Colors.Text);
ylabel(ax3, 'Misclassification Error', 'Color', viz.Colors.Text);
title(ax3, 'Nearest Neighbors / Difficult', 'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold');
ylim(ax3, [0.4 0.6]);
hold(ax3, 'off');

% ---- Bottom-Right: K-means + LVQ / Difficult ----
ax4 = axes('Parent', fig, 'Position', panels{4});
hold(ax4, 'on'); viz.styleAxes(ax4);
errorbar(ax4, Rs_proto, km_mean_d, km_se_d, '-o', ...
    'Color', colKM, 'MarkerFaceColor', colKM, 'MarkerSize', 4, ...
    'LineWidth', 1.5, 'CapSize', 4, 'DisplayName', 'K-means');
errorbar(ax4, Rs_proto, lvq_mean_d, lvq_se_d, '-s', ...
    'Color', colLVQ, 'MarkerFaceColor', colLVQ, 'MarkerSize', 4, ...
    'LineWidth', 1.5, 'CapSize', 4, 'DisplayName', 'LVQ');
xlabel(ax4, 'Number of Prototypes per Class', 'Color', viz.Colors.Text);
ylabel(ax4, 'Misclassification Error', 'Color', viz.Colors.Text);
title(ax4, 'K-means & LVQ / Difficult', 'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold');
legend(ax4, 'Location', 'northeast', 'FontSize', 9, ...
    'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
ylim(ax4, [0.4 0.6]);
hold(ax4, 'off');

% Save
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig, fullfile(resultsDir, 'fig13_5_comparative_study.png'));
fprintf('[Step 8] Saved: fig13_5_comparative_study.png\n');

fprintf('\n=== Done ===\n');
