% EXP_PROTOTYPE_VS_GMM  ESL Figure 13.2 — K-means vs Gaussian Mixtures
%
% Reproduces Figure 13.2: 2-class ESL mixture data (same as Fig 2.1),
% R=5 prototypes/subclasses per class.
%
% Upper panel: K-means prototypes — piecewise-linear decision boundary
% Lower panel: Gaussian mixture (MDA pooled) — smooth decision boundary
% Dashed purple curve: Bayes optimal decision boundary
%
% Author: Ocean Library   Date: 2026-03-24
% Ref: Hastie et al. (2009), ESL Section 13.2, Figure 13.2
clc; clear; rng(42, 'twister');
RNG_SEED = 42;  DATASET = 'esl_mixture_2class';

%% === 1. Data Generation ===
fprintf('=== Step 1: Data Generation ===\n');
gen = OceanData.Generators.ESLMixtureGenerator('N', 200, 'Seed', RNG_SEED, ...
    'NumClasses', 2, 'CentersPerClass', 10);
[X, y] = gen.generate();
fprintf('[DATA] N = %d | p = %d | K = %d classes\n', size(X,1), size(X,2), numel(unique(y)));
for k = 1:2
    fprintf('  Class %d: %d samples\n', k, sum(y == k));
end

%% === 1.5. Train/Test Split ===
fprintf('\n=== Step 1.5: Train/Test Split ===\n');
splitter = OceanData.DataArch.DataSplitter('Strategy', 'holdout', 'Ratio', 0.5);
split_result = splitter.split(X, y);
X_train = X(split_result.trainIdx, :);  y_train = y(split_result.trainIdx);
X_test  = X(split_result.testIdx, :);   y_test  = y(split_result.testIdx);
fprintf('[SPLIT] Train: %d | Test: %d\n', sum(split_result.trainIdx), sum(split_result.testIdx));

%% === 2-6. DA Layers ===
fprintf('\n[SKIP] DataLake / DataWarehouse / CachingWarehouse / FeatureStore / FeatureCache\n');

%% === 6.5b. Model Training (R=5 per class) ===
fprintf('\n=== Step 6.5: Model Training (R=5 per class) ===\n');
R = 5;
K = 2;

% --- (A) K-means prototypes (per-class) ---
km = OceanStats.Unsupervised.KMeans();
km_prototypes = zeros(K * R, 2);
km_labels     = zeros(K * R, 1);
for k = 1:K
    X_k = X_train(y_train == k, :);
    km.fit(X_k, R);
    s = (k-1)*R + 1;  e = k * R;
    km_prototypes(s:e, :) = km.Centroids;
    km_labels(s:e) = k;
end
fprintf('[K-MEANS] Fitted %d prototypes per class (%d total)\n', R, K*R);

% --- (B) Gaussian Mixtures via MDA (pooled covariance) ---
mda = OceanStats.Supervised.Discriminant.MDA( ...
    'NumSubclasses', R, 'Mode', 'pooled', 'MaxIter', 100, ...
    'Tol', 1e-5, 'Verbose', false);
mda.fit(X_train, y_train);
% Extract subclass centers for visualization
mda_prototypes = vertcat(mda.SubclassMeans{:});         % (R*K) x p
mda_labels     = [ones(R,1); 2*ones(R,1)];              % class labels
fprintf('[MDA]     Fitted Gaussian mixture model (%d subclasses/class, pooled cov)\n', R);

%% === 7. Final Evaluation ===
fprintf('\n=== Step 7: Final Evaluation ===\n');

% K-means: nearest-prototype on train and test
y_train_km = nearestPrototypePredict(X_train, km_prototypes, km_labels);
y_test_km  = nearestPrototypePredict(X_test,  km_prototypes, km_labels);
err_train_km = mean(y_train_km ~= y_train);
err_test_km  = mean(y_test_km  ~= y_test);
fprintf('[K-MEANS] Train Error = %.3f | Test Error = %.3f\n', err_train_km, err_test_km);

% MDA: on train and test
y_train_mda = mda.predict(X_train);
y_test_mda  = mda.predict(X_test);
err_train_mda = mean(y_train_mda ~= y_train);
err_test_mda  = mean(y_test_mda  ~= y_test);
fprintf('[MDA    ] Train Error = %.3f | Test Error = %.3f\n', err_train_mda, err_test_mda);

% Bayes optimal
[~, err_bayes_test]  = gen.bayesPredict(X_test, y_test);
[~, err_bayes_train] = gen.bayesPredict(X_train, y_train);
fprintf('[BAYES  ] Train Error = %.3f | Test Error = %.3f (irreducible)\n', ...
    err_bayes_train, err_bayes_test);

%% === 8. Visualization ===
fprintf('\n=== Step 8: Visualization ===\n');

% --- Build grid ---
margin = 0.8;
x1_range = [min(X(:,1))-margin, max(X(:,1))+margin];
x2_range = [min(X(:,2))-margin, max(X(:,2))+margin];
gridRes  = 250;
[xx, yy] = meshgrid(linspace(x1_range(1), x1_range(2), gridRes), ...
                     linspace(x2_range(1), x2_range(2), gridRes));
X_grid = [xx(:), yy(:)];

% K-means decision regions
zz_km = nearestPrototypePredict(X_grid, km_prototypes, km_labels);
zz_km = reshape(zz_km, size(xx));

% MDA decision regions
y_mda_grid = mda.predict(X_grid);
zz_mda = reshape(y_mda_grid, size(xx));

% Bayes decision boundary
D = gen.bayesDensity(X_grid);
[~, zz_bayes] = max(D, [], 2);
zz_bayes = reshape(zz_bayes, size(xx));

% --- Create figure ---
viz = OceanViz.Core('Theme', 'Scientific');
fig = viz.createFigure('ESL 13.2 - K-means vs Gaussian Mixtures');
fig.Position = [100, 50, 700, 900];

% ===== Panel 1: K-means =====
ax1 = subplot(2,1,1, 'Parent', fig);
data_km = struct();
data_km.X = X_train;
data_km.Y = y_train;
data_km.Prototypes  = km_prototypes;
data_km.ProtoLabels = km_labels;
data_km.Grid      = struct('xx', xx, 'yy', yy, 'zz', zz_km);
data_km.BayesGrid = struct('xx', xx, 'yy', yy, 'zz', zz_bayes);
viz.plot(ax1, 'PrototypeClassifier', data_km, ...
    'Title', 'K-means - 5 Prototypes per Class', ...
    'ShowLegend', true);
% Error annotation (bottom-left)
xl = xlim(ax1); yl = ylim(ax1);
txtX = xl(1) + 0.03*(xl(2)-xl(1));
txtY = yl(1) + 0.05*(yl(2)-yl(1));
text(ax1, txtX, txtY + 0.10*(yl(2)-yl(1)), ...
    sprintf('Training Error:  %.3f', err_train_km), ...
    'Color', viz.Colors.Text, 'FontSize', 9, 'FontName', 'Consolas');
text(ax1, txtX, txtY + 0.05*(yl(2)-yl(1)), ...
    sprintf('Test Error:      %.3f', err_test_km), ...
    'Color', viz.Colors.Text, 'FontSize', 9, 'FontName', 'Consolas');
text(ax1, txtX, txtY, ...
    sprintf('Bayes Error:     %.3f', err_bayes_test), ...
    'Color', viz.Colors.Text, 'FontSize', 9, 'FontName', 'Consolas');

% ===== Panel 2: Gaussian Mixtures (MDA) =====
ax2 = subplot(2,1,2, 'Parent', fig);
data_mda = struct();
data_mda.X = X_train;
data_mda.Y = y_train;
data_mda.Prototypes  = mda_prototypes;
data_mda.ProtoLabels = mda_labels;
data_mda.Grid      = struct('xx', xx, 'yy', yy, 'zz', zz_mda);
data_mda.BayesGrid = struct('xx', xx, 'yy', yy, 'zz', zz_bayes);
viz.plot(ax2, 'PrototypeClassifier', data_mda, ...
    'Title', 'Gaussian Mixtures - 5 Subclasses per Class', ...
    'ShowLegend', true);
% Error annotation
xl = xlim(ax2); yl = ylim(ax2);
txtX = xl(1) + 0.03*(xl(2)-xl(1));
txtY = yl(1) + 0.05*(yl(2)-yl(1));
text(ax2, txtX, txtY + 0.10*(yl(2)-yl(1)), ...
    sprintf('Training Error:  %.3f', err_train_mda), ...
    'Color', viz.Colors.Text, 'FontSize', 9, 'FontName', 'Consolas');
text(ax2, txtX, txtY + 0.05*(yl(2)-yl(1)), ...
    sprintf('Test Error:      %.3f', err_test_mda), ...
    'Color', viz.Colors.Text, 'FontSize', 9, 'FontName', 'Consolas');
text(ax2, txtX, txtY, ...
    sprintf('Bayes Error:     %.3f', err_bayes_test), ...
    'Color', viz.Colors.Text, 'FontSize', 9, 'FontName', 'Consolas');

fprintf('\n[DONE] Figure 13.2 reproduced successfully.\n');

%% === Local Helper ===
function y_pred = nearestPrototypePredict(X, prototypes, labels)
    N = size(X, 1);
    y_pred = zeros(N, 1);
    for i = 1:N
        dists = sum((prototypes - X(i,:)).^2, 2);
        [~, nearest] = min(dists);
        y_pred(i) = labels(nearest);
    end
end
