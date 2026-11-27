% EXP_PROTOTYPE_METHODS  ESL Chapter 13 — Prototype Methods & Nearest-Neighbors
%
% Reproduces Figure 13.1: Simulated 3-class data (mixture of Gaussians),
% R=5 prototypes per class. Compares K-means vs LVQ prototype classifiers.
%4
% Upper panel: K-means prototypes + 1-NN decision boundary
% Lower panel: LVQ-refined prototypes + 1-NN decision boundary
% Dashed purple curve: Bayes optimal decision boundary
%
% Author: Ocean Library   Date: 2026-03-24
% Ref: Hastie et al. (2009), ESL Section 13.2, Figure 13.1
clc; clear; rng(42, 'twister');
RNG_SEED = 42;  DATASET = 'esl_mixture_3class';

%% === 1. Data Generation ===
fprintf('=== Step 1: Data Generation ===\n');
gen = OceanData.Generators.ESLMixtureGenerator('N', 300, 'Seed', RNG_SEED, ...
    'NumClasses', 3, 'CentersPerClass', 10);
[X, y] = gen.generate();
fprintf('[DATA] N = %d | p = %d | K = %d classes\n', size(X,1), size(X,2), numel(unique(y)));
for k = 1:3
    fprintf('  Class %d: %d samples\n', k, sum(y == k));
end

%% === 1.5. Train/Test Split (Anti-Leakage) ===
fprintf('\n=== Step 1.5: Train/Test Split ===\n');
splitter = OceanData.DataArch.DataSplitter('Strategy', 'holdout', 'Ratio', 0.375);
split_result = splitter.split(X, y);
X_train = X(split_result.trainIdx, :);  y_train = y(split_result.trainIdx);
X_test  = X(split_result.testIdx, :);   y_test  = y(split_result.testIdx);
fprintf('[SPLIT] Train: %d | Test: %d\n', sum(split_result.trainIdx), sum(split_result.testIdx));

%% === 2. DataLake (DA-1) ===
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    lake.put('raw/esl_ch13/X_train', X_train, struct('source', DATASET, 'seed', RNG_SEED));
    lake.put('raw/esl_ch13/X_test',  X_test,  struct('source', DATASET, 'seed', RNG_SEED));
    fprintf('[PASS] DataLake: raw train/test saved.\n');
catch, fprintf('[SKIP] DataLake unavailable.\n'); end

%% === 3. DataWarehouse (DA-2/DA-3) ===
try
    dw = OceanData.DataArch.DataWarehouse('rootDir', './dw/');
    colNames = {'x1', 'x2', 'class'};
    colTypes = {'double', 'double', 'double'};
    schema = OceanData.DataArch.DWSchema(colNames, colTypes);
    dw.createTable(DATASET, schema);
    dw.insertBatch(DATASET, array2table([X_train, y_train], 'VariableNames', colNames));
    fprintf('[PASS] DataWarehouse: %d rows inserted.\n', size(X_train, 1));
catch, fprintf('[SKIP] DataWarehouse unavailable.\n'); end

%% === 4. CachingWarehouse (DA-4) ===
try
    redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
    cdw   = OceanData.Adapters.CachingWarehouse(dw, redis);
    fprintf('[PASS] CachingWarehouse connected.\n');
catch, fprintf('[SKIP] Redis unavailable.\n'); end

%% === 5 & 6. FeatureStore & FeatureCache (DA-5/DA-6) ===
try 
    %Store on disk
    fs = OceanData.DataArch.FeatureStore('rootDir', './fs/');
    feat_key = sprintf('%s_seed%d', DATASET, RNG_SEED);
    if ~fs.exists(feat_key)
        fs.saveFeatures(feat_key, X_train);
    else
        X_train = fs.loadFeatures(feat_key);
    end
    %Store on Cache. Next time. Get from cache
    cache = OceanData.DataArch.FeatureCache();
    cache.put('X_train_feat', X_train);
    X_batch = cache.get('X_train_feat');
    fprintf('[PASS] FeatureStore + FeatureCache ready.\n');
catch, fprintf('[SKIP] FeatureStore/Cache unavailable.\n'); end

%% === 6.5a. Feature Selection (training data only) ===
% No feature selection needed — only 2 features, both required for 2D viz.
fprintf('\n[SKIP] Feature selection: p=2, all features retained.\n');

%% === 6.5b. Cross-Validation + Hyperparameter Tuning ===
fprintf('\n=== Step 6.5b: Prototype Training (R=5 per class) ===\n');
R = 5;  % Number of prototypes per class
K = 3;

% --- K-means prototypes (per-class) ---
km = OceanStats.Unsupervised.KMeans();
km_prototypes = zeros(K * R, 2);
km_labels     = zeros(K * R, 1);
for k = 1:K
    X_k = X_train(y_train == k, :);
    km.fit(X_k, R);
    startIdx = (k-1)*R + 1;
    endIdx   = k * R;
    km_prototypes(startIdx:endIdx, :) = km.Centroids;
    km_labels(startIdx:endIdx) = k;
end
fprintf('[K-MEANS] Fitted %d prototypes per class (%d total)\n', R, K*R);

% --- LVQ1 refinement from K-means initialization ---
lvq = OceanStats.Supervised.Neighbors.LVQ1('R', R, ...
    'LearningRate', 0.1, 'MaxEpochs', 50, 'Decay', 0.02, 'Verbose', false);
lvq.fit(X_train, y_train, 'InitPrototypes', km_prototypes, 'InitLabels', km_labels);
fprintf('[LVQ1] Refined prototypes via competitive learning (50 epochs)\n');

%% === 6.5c. Model Comparison ===
% Deferred to Step 7 (we compare on test set).

%% === 7. Final Model Evaluation (test set: ONE time only) ===
fprintf('\n=== Step 7: Final Evaluation on Test Set ===\n');

% --- K-means 1-NN prototype classifier ---
y_pred_km = nearestPrototypePredict(X_test, km_prototypes, km_labels);
err_km = mean(y_pred_km ~= y_test);
fprintf('[K-MEANS] Test Error = %.3f\n', err_km);

% --- LVQ 1-NN prototype classifier ---
y_pred_lvq = lvq.predict(X_test);
err_lvq = mean(y_pred_lvq ~= y_test);
fprintf('[LVQ1  ] Test Error = %.3f\n', err_lvq);

% --- Bayes optimal ---
[~, err_bayes] = gen.bayesPredict(X_test, y_test);
fprintf('[BAYES ] Test Error = %.3f (irreducible)\n', err_bayes);

%% === 8. Visualization ===
fprintf('\n=== Step 8: Visualization ===\n');
close all;

% --- Build grid for decision boundaries ---
margin = 0.5;
x1_range = [min(X(:,1)) - margin, max(X(:,1)) + margin];
x2_range = [min(X(:,2)) - margin, max(X(:,2)) + margin];
gridRes = 200;
[xx, yy] = meshgrid(linspace(x1_range(1), x1_range(2), gridRes), ...
                     linspace(x2_range(1), x2_range(2), gridRes));
X_grid = [xx(:), yy(:)];

% K-means decision regions
zz_km = nearestPrototypePredict(X_grid, km_prototypes, km_labels);
zz_km = reshape(zz_km, size(xx));

% LVQ decision regions
zz_lvq = lvq.predict(X_grid);
zz_lvq = reshape(zz_lvq, size(xx));

% Bayes decision boundary
D = gen.bayesDensity(X_grid);  % N x K
[~, zz_bayes] = max(D, [], 2);
zz_bayes = reshape(zz_bayes, size(xx));

% --- Create figure with OceanViz ---
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ESL 13.1 — Prototype Methods');
fig.Position = [50, 30, 750, 900];

% Manual axes positions [left bottom width height] in normalized units
% Leaves room for title, labels, legend
axPos1 = [0.07, 0.54, 0.78, 0.42];   % top panel
axPos2 = [0.07, 0.05, 0.78, 0.42];   % bottom panel

% --- Panel 1: K-means ---
ax1 = axes('Parent', fig, 'Position', axPos1);
data_km.X = X;
data_km.Y = y;
data_km.Prototypes = km_prototypes;
data_km.ProtoLabels = km_labels;
data_km.Grid = struct('xx', xx, 'yy', yy, 'zz', zz_km);
data_km.BayesGrid = struct('xx', xx, 'yy', yy, 'zz', zz_bayes);
data_km.DataPointSize = 12;    % passed through data, not options (options has fixed allowlist)
data_km.ProtoSize     = 65;
data_km.StippleDensity = 80;
viz.plot(ax1, 'PrototypeClassifier', data_km, ...
    'Title', sprintf('K-means - %d Prototypes per Class  |  Test Error = %.3f', R, err_km), ...
    'ShowLegend', true);

% --- Panel 2: LVQ ---
ax2 = axes('Parent', fig, 'Position', axPos2);
data_lvq.X = X;
data_lvq.Y = y;
data_lvq.Prototypes = lvq.Prototypes;
data_lvq.ProtoLabels = lvq.PrototypeLabels;
data_lvq.Grid = struct('xx', xx, 'yy', yy, 'zz', zz_lvq);
data_lvq.BayesGrid = struct('xx', xx, 'yy', yy, 'zz', zz_bayes);
data_lvq.DataPointSize = 12;
data_lvq.ProtoSize     = 65;
data_lvq.StippleDensity = 80;
viz.plot(ax2, 'PrototypeClassifier', data_lvq, ...
    'Title', sprintf('LVQ - %d Prototypes per Class  |  Test Error = %.3f', R, err_lvq), ...
    'ShowLegend', true);

fprintf('\n[DONE] Figure 13.1 reproduced successfully.\n');

%% === Local Helper ===
function y_pred = nearestPrototypePredict(X, prototypes, labels)
    % 1-NN classification using prototype locations.
    N = size(X, 1);
    y_pred = zeros(N, 1);
    for i = 1:N
        dists = sum((prototypes - X(i,:)).^2, 2);
        [~, nearest] = min(dists);
        y_pred(i) = labels(nearest);
    end
end
