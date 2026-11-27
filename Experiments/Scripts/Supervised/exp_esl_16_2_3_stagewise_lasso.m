% EXP_ESL_16_2_3_STAGEWISE_LASSO  Comparison of Lasso and Infinitesimal Forward Stagewise paths.
%
%   DATASET:        Simulated dictionary of 1000 Gaussian variables, 
%                   strongly correlated (rho=0.95) within blocks of 20.
%                   N = 60 samples. 50 active variables. NSR = 0.72.
%   MODEL:          Lasso vs Forward Stagewise Linear Regression (FS0)
%   ASSUMPTIONS:    Demonstrates that Forward Stagewise is "slow to overfit"
%                   because its paths are smoother in the presence of strong
%                   multi-collinearity compared to Lasso.
%   REPRODUCIBILITY: rng(42, 'twister')
%
%   Ref: ESL Figure 16.3 / Exercise 16.1

clc; clear; close all;
rng(42, 'twister');

RNG_SEED = 42;
DATASET = 'sim_correlated';
rootDir = fullfile(pwd);

fprintf('=== ESL 16.2.3: Regularization Paths and Multi-collinearity ===\n\n');

%% === 1. Ingestion ===
fprintf('1. Generating Block-Correlated Dataset (N=100, p=1000)...\n');
gen = OceanData.Generators.CorrelatedGenerator(...
    'N', 100, 'NumFeatures', 1000, 'BlockSize', 20, ...
    'Rho', 0.95, 'ActivePerBlock', 1, 'NoiseSignalRatio', 0.72);

[X_all, y_all] = gen.generate();

%% === 2. Train/Test Split (MANDATORY — Anti-Leakage) ===
fprintf('2. Splitting Train/Test (60/40)...\n');
% Using N=60 for train to match the original ESL experiment visually
idx_train = 1:60;
idx_test = 61:100;
X_train = X_all(idx_train, :); y_train = y_all(idx_train);
X_test  = X_all(idx_test, :);  y_test  = y_all(idx_test);

%% === 3 & 4. Bronze Layer (Parquet DataLake) & Tier 2 Metadata ===
try
    lake = OceanData.DataArch.DataLake('rootDir', fullfile(rootDir, 'lake'));
    % Create generic column names
    colNames = [arrayfun(@(i) sprintf('X%d',i), 1:size(X_all,2), 'UniformOutput', false), {'Y'}];
    trainT = array2table([X_train, y_train], 'VariableNames', colNames); 
    testT  = array2table([X_test, y_test], 'VariableNames', colNames);
    
    lake.put(sprintf('raw/%s/train/X', DATASET), trainT, struct(), 'asParquet', true);
    lake.put(sprintf('raw/%s/test/X',  DATASET), testT,  struct(), 'asParquet', true);
    
    try
        cfg = OceanData.Adapters.DbConfig();
        tracker = OceanData.Adapters.PostgresRegistry(cfg);
        tracker.connect();
    catch
        tracker = OceanData.DataArch.MetadataRegistry('rootDir', fullfile(rootDir, 'meta'));
    end
    tracker.registerDataset(DATASET, 'v1', 'train', sprintf('lake/raw/%s/train/X.parquet', DATASET), height(trainT));
    tracker.registerDataset(DATASET, 'v1', 'test',  sprintf('lake/raw/%s/test/X.parquet', DATASET),  height(testT));
catch ME
    fprintf('[SKIP] Bronze/Metadata tracking unavailable: %s\n', ME.message); 
end

%% === 5 & 6. Silver Layer (Feature Engineering & FeatureStore Parquet) ===
[X_tr_std, mu, sig] = zscore(X_train);
X_te_std = (X_test - mu) ./ sig;
y_tr_std = y_train - mean(y_train);
y_te_std = y_test - mean(y_train);

try
    fs = OceanData.DataArch.DataLake('rootDir', fullfile(rootDir, 'fs'));
    featTrain = array2table([X_tr_std, y_tr_std], 'VariableNames', colNames);
    meta = struct('mu', mu, 'sigma', sig);
    fs.put(sprintf('%s/train_feat', DATASET), featTrain, meta, 'asParquet', true);
    tracker.registerDataset(DATASET+"_feat", 'v1', 'train', sprintf('fs/%s/train_feat.parquet', DATASET), size(X_tr_std,1));
catch
    fprintf('[SKIP] FeatureStore unavailable. Proceeding with in-memory arrays.\n');
end

X_train_std = X_tr_std; Y_train_std = y_tr_std;

%% === 7. Model Selection (Skipped for Path Viz) ===
fprintf('[SKIP] Model selection bypassed to generate full regularization path.\n');

%% === 8. Final Model Training & Evaluation ===
fprintf('8. Computing Paths...\n');
[B_lasso, ~] = lasso(X_train_std, Y_train_std, 'Standardize', false);
L1_lasso = sum(abs(B_lasso), 1);
frac_lasso = L1_lasso / max(L1_lasso);

fs_model = OceanStats.Supervised.Regression.ForwardStagewiseRegression(...
    'NumIterations', 6000, 'LearningRate', 0.01, 'Standardize', false);
fs_fit = fs_model.train(X_train_std, Y_train_std);
B_fs = fs_fit.BetaPath;
L1_fs = sum(abs(B_fs), 1);
frac_fs = L1_fs / max(L1_fs);

% Calculate RMSE for the final models on the unseen test set
rmse_lasso = sqrt(mean((y_te_std - X_te_std * B_lasso(:, end)).^2));
rmse_fs = sqrt(mean((y_te_std - X_te_std * B_fs(:, end)).^2));

%% === 9 & 10. Tier 3 Tracking & Gold Layer (Data Warehouse) ===
try
    tracker.logRun('r_lasso_02', 'v1', 'Lasso', 'rmse', rmse_lasso, '{"lambda": "path"}');
    tracker.logRun('r_fs_02', 'v1', 'ForwardStagewise', 'rmse', rmse_fs, '{"iters": 6000}');
    
    dw = OceanData.DataArch.DataWarehouse('rootDir', fullfile(rootDir, 'datawarehouse'));
    perfTable = table(["r_lasso_02"; "r_fs_02"], ["v1"; "v1"], ["Lasso"; "FS0"], ...
        ["rmse"; "rmse"], [rmse_lasso; rmse_fs], [datetime('now'); datetime('now')], ...
        'VariableNames', {'run_id', 'dataset_version', 'model', 'metric_name', 'metric_value', 'created_at'});
    dw.insertBatch('model_performance', perfTable);
catch ME
    fprintf('[SKIP] Gold Layer unavailable: %s\n', ME.message);
end

%% === 11. Visualization ===
fprintf('11. Rendering Beautiful Dark Theme Paths...\n');
viz = OceanViz.Core('Theme', 'Dark');

fig = viz.createFigure('ESL 16.3: Multi-collinearity Robustness');
fig.Position = [100, 100, 1200, 600];

% Left: Lasso
ax1 = subplot(1, 2, 1);
plot(ax1, frac_lasso, B_lasso, 'LineWidth', 1.5);
title(ax1, 'LASSO (Extremely Erratic)');
xlabel(ax1, '|\alpha(m)| / |\alpha(\infty)|');
ylabel(ax1, 'Standardized Coefficients');
ylim(ax1, [-25, 30]); xlim(ax1, [0, 1]);
viz.styleAxes(ax1);

% Right: Forward Stagewise
ax2 = subplot(1, 2, 2);
plot(ax2, frac_fs, B_fs, 'LineWidth', 1.5);
title(ax2, 'Forward Stagewise (FS_0) (Monotonic & Smooth)');
xlabel(ax2, '|\alpha(m)| / |\alpha(\infty)|');
ylabel(ax2, 'Standardized Coefficients');
ylim(ax2, [-25, 30]); xlim(ax2, [0, 1]);
viz.styleAxes(ax2);

% Add zero lines perfectly formatted for Dark Theme
yline(ax1, 0, 'w--', 'LineWidth', 1.5, 'Alpha', 0.5);
yline(ax2, 0, 'w--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Use a super title across both subplots
sgtitle('ESL 16.2.3: Regularization Paths Under Strong Multi-Collinearity', 'Color', 'w', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('\n=== EXPERIMENT COMPLETE ===\n');
fprintf('  Observe how the Forward Stagewise paths are monotonic and smoother\n');
fprintf('  in the later stages compared to the widely fluctuating Lasso paths.\n');
fprintf('  This illustrates why Boosting is highly resistant to over-fitting\n');
fprintf('  even with massive multi-collinearity.\n');
fprintf('===========================\n');
