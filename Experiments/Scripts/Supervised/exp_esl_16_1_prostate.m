% EXP_ESL_16_1_PROSTATE  Demonstrates the 8-step data science pipeline.
%
%   DATASET:        Prostate Cancer Data (8 predictors, target = lpsa)
%   MODEL:          Lasso vs Forward Stagewise Linear Regression (FS0)
%   REPRODUCIBILITY: Ref: ESL Figure 16.1
%
%   Goal: Produce the coefficient profiles for Lasso (vs L1 norm) and 
%         Forward Stagewise (vs Iteration step M), using the Pipeline Architecture.

clc; clear; close all; rng(42, 'twister');
RNG_SEED = 42;  DATASET = 'prostate';

fprintf('=== ESL 16.1: Boosting and Regularization Paths (Pipeline) ===\n\n');

%% === 1. Parquet Conversion & Ingestion ===
% Locate the root of the project to load data robustly
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fullfile(scriptDir, '..', '..', '..');
dataPath = fullfile(rootDir, 'Data', 'Prostate', 'data.data');
T = readtable(dataPath, 'FileType', 'text', 'Delimiter', '\t');

% In data.data, Col 1 is Index (Var1), Cols 2:9 are features, Col 10 is lpsa (target)
X_all = table2array(T(:, 2:9));
y_all = T.lpsa;
is_train = strcmp(T.train, 'T') | strcmp(T.train, 'TRUE');

featureNames = T.Properties.VariableNames(2:9);
colNames = [featureNames, {'lpsa'}];
colTypes = repmat({'double'}, 1, 9);

%% === 2. Train/Test Split (MANDATORY — Anti-Leakage) ===
% We use the predefined 'train' flag from the dataset instead of DataSplitter
isTrain = strcmp(T.train, 'T') | strcmp(T.train, 'TRUE');
X_train = X_all(isTrain, :);   y_train = y_all(isTrain);
X_test  = X_all(~isTrain, :);  y_test  = y_all(~isTrain);

%% === 3 & 4. Bronze Layer (Parquet DataLake) & Tier 2 Metadata ===
try
    lake = OceanData.DataArch.DataLake('rootDir', fullfile(rootDir, 'lake'));
    trainTable = array2table([X_train, y_train], 'VariableNames', colNames);
    testTable  = array2table([X_test, y_test], 'VariableNames', colNames);
    
    % Write pristine Parquet files
    lake.put(sprintf('raw/%s/train/X', DATASET), trainTable, struct('source', DATASET), 'asParquet', true);
    lake.put(sprintf('raw/%s/test/X', DATASET),  testTable,  struct('source', DATASET), 'asParquet', true);
    fprintf('[PASS] Bronze Layer: Raw Parquet train/test saved.\n');
    
    % Register their existence in the Metadata Store (Try Postgres first, then fallback)
    try
        cfg = OceanData.Adapters.DbConfig();
        tracker = OceanData.Adapters.PostgresRegistry(cfg);
        tracker.connect();
    catch
        fprintf('[INFO] Postgres connection failed. Falling back to local MetadataRegistry.\n');
        tracker = OceanData.DataArch.MetadataRegistry('rootDir', fullfile(rootDir, 'meta'));
    end
    
    tracker.registerDataset(DATASET, 'v1', 'train', sprintf('lake/raw/%s/train/X.parquet', DATASET), height(trainTable));
    tracker.registerDataset(DATASET, 'v1', 'test',  sprintf('lake/raw/%s/test/X.parquet', DATASET),  height(testTable));
catch ME
    fprintf('[SKIP] Bronze/Metadata tracking unavailable: %s\n', ME.message);
end

%% === 5. Silver Layer (Feature Engineering & FeatureStore Parquet) ===
% We calculate Z-scores using ONLY training data
[X_tr_std, mu, sig] = zscore(X_train);
X_te_std = (X_test - mu) ./ sig;
y_tr_std = y_train - mean(y_train);
y_te_std = y_test - mean(y_train);

try
    fs = OceanData.DataArch.DataLake('rootDir', fullfile(rootDir, 'fs'));
    featTrain = array2table([X_tr_std, y_tr_std], 'VariableNames', colNames);
    
    % Write Gold features as Parquet, embedding the standardisation parameters in the metadata
    meta = struct('mu', mu, 'sigma', sig);
    fs.put(sprintf('%s/train_feat', DATASET), featTrain, meta, 'asParquet', true);
    tracker.registerDataset(DATASET + "_feat", 'v1', 'train', sprintf('fs/%s/train_feat.parquet', DATASET), height(featTrain));
    fprintf('[PASS] Silver Layer: Parquet features engineered and registered.\n');
catch
    fprintf('[SKIP] FeatureStore unavailable. Proceeding with in-memory arrays.\n');
end

% We assign arrays to be used for model training
X_train_std = X_tr_std;
Y_train_std = y_tr_std;

%% === 8a. Final Model Training (Path Generation) ===
fprintf('Computing Lasso Path...\n');
[B_lasso, ~] = lasso(X_train_std, Y_train_std, 'Standardize', false, 'NumLambda', 100);
L1_lasso = sum(abs(B_lasso), 1);

fprintf('Computing Infinitesimal Forward Stagewise (FS0) Path...\n');
fs_model = OceanStats.Supervised.Regression.ForwardStagewiseRegression(...
    'NumIterations', 250, 'LearningRate', 0.01, 'Standardize', false);
fs_fit = fs_model.train(X_train_std, Y_train_std);
B_fs = fs_fit.BetaPath;

%% === 8b. Experiment Evaluation ===
% Calculate RMSE for both models on the held-out test set
rmse_lasso = sqrt(mean((y_te_std - X_te_std * B_lasso(:, end)).^2));
rmse_fs = sqrt(mean((y_te_std - X_te_std * B_fs(:, end)).^2));

%% === 9 & 10. Experiment Tracking (Tier 3) & Gold Layer (Data Warehouse) ===
try
    % Tier 3: Log run metadata
    tracker.logRun('r_lasso_01', 'v1', 'Lasso', 'rmse', rmse_lasso, '{"lambda": "min"}');
    tracker.logRun('r_fs_01', 'v1', 'ForwardStagewise', 'rmse', rmse_fs, '{"iters": 250}');
    
    % Tier 4: Push to Gold OLAP Data Warehouse
    dw = OceanData.DataArch.DataWarehouse('rootDir', fullfile(rootDir, 'datawarehouse'));
    perfTable = table(["r_lasso_01"; "r_fs_01"], ["v1"; "v1"], ["Lasso"; "FS0"], ...
        [rmse_lasso; rmse_fs], [datetime('now'); datetime('now')], ...
        'VariableNames', {'run_id', 'dataset_version', 'model', 'rmse', 'created_at'});
    
    dw.insertBatch('model_performance', perfTable);
    fprintf('[PASS] Gold Layer: Analytical aggregates pushed to Data Warehouse.\n');
catch ME
    fprintf('[SKIP] Gold Layer unavailable: %s\n', ME.message);
end

%% === 11. Visualization ===
fprintf('Generating Plots using OceanViz...\n');
viz = OceanViz.Core('Theme', 'Light');

fig = viz.createFigure('ESL 16.1: Regularization Paths');
fig.Position = [100, 100, 1000, 500];

% Get beautiful distinct colors from OceanViz
colors = viz.getColorSequence(size(X_train, 2));

% --- Left Panel: Lasso ---
ax1 = subplot(1, 2, 1);
colororder(ax1, colors);
plot(ax1, L1_lasso, B_lasso', 'LineWidth', 1.5);
title(ax1, 'Lasso');
xlabel(ax1, 't = \Sigma |\beta_k|'); ylabel(ax1, 'Coefficients');
yline(ax1, 0, 'k-', 'LineWidth', 1);

[max_t, max_idx] = max(L1_lasso);
for j = 1:size(X_train, 2)
    % Align text nicely and color-match the corresponding line
    text(ax1, max_t * 1.02, B_lasso(j, max_idx), featureNames{j}, ...
        'FontSize', 10, 'Color', colors(j,:), 'VerticalAlignment', 'middle');
end
xlim(ax1, [0, max_t * 1.25]);

% --- Right Panel: Forward Stagewise ---
ax2 = subplot(1, 2, 2);
colororder(ax2, colors);
iterations = 1:size(B_fs, 2);
plot(ax2, iterations, B_fs', 'LineWidth', 1.5);
title(ax2, 'Forward Stagewise');
xlabel(ax2, 'Iteration'); ylabel(ax2, 'Coefficients');
yline(ax2, 0, 'k-', 'LineWidth', 1);

max_iter = size(B_fs, 2);
for j = 1:size(X_train, 2)
    % Align text nicely and color-match the corresponding line
    text(ax2, max_iter * 1.02, B_fs(j, end), featureNames{j}, ...
        'FontSize', 10, 'Color', colors(j,:), 'VerticalAlignment', 'middle');
end
xlim(ax2, [0, max_iter * 1.25]);

viz.styleAxes(ax1);
viz.styleAxes(ax2);
fprintf('Experiment complete.\n');
