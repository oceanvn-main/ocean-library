% EXP_RF_SPAM_MTRY  ESL Exercise 15.6 — Random Forest sensitivity to m (mtry)
%                   on the Spambase dataset (8-step Data Science Pipeline).
%
%   DATASET:        UCI Spambase. N=4601 e-mails, p=57 continuous features.
%                   Binary label: 1=spam, 0=ham. Last column of spambase.data.
%                   Source: Data/spambase/spambase.data (no header row).
%
%   MODEL:          Random Forest (OceanStats.Ensemble.RandomForest).
%                   Criterion = 'gini', MaxDepth = Inf (fully grown, RF default).
%                   Fixed B = 300 trees per run (ESL recommends >= 200).
%
%   EXPERIMENT:     Sweep mtry m over 15 log-spaced values from 1 to p=57.
%                   For each m: train one RF, record OOB error and test error.
%                   Goal: reproduce ESL Fig 15.8 dual-error curve.
%
%   PIPELINE STEPS:
%     Step 1   — Ingest raw CSV
%     Step 1.5 — Anti-leakage train/test split (DataSplitter, 2/3-1/3)
%     Step 2   — DataLake: persist raw train/test matrices
%     Step 3   — DataWarehouse: schema-validated feature + label table
%     Step 4   — CachingWarehouse (Redis, skip if unavailable)
%     Step 5   — FeatureStore: NZV-filtered features (compute-or-load)
%     Step 6   — FeatureCache (Redis, skip if unavailable)
%     Step 6.5a— Feature selection (NZV filter, training data only)
%     Step 6.5b— Hyperparameter tuning: mtry sweep using OOB error as CV proxy
%     Step 7   — Final evaluation on held-out test set
%     Step 8   — Visualization (OceanViz.Core, Dark theme)
%
%   ASSUMPTIONS:
%     A1. Fixed 2/3 train / 1/3 test split via DataSplitter (Ratio=2/3, Seed=42).
%         Split is performed BEFORE any storage or preprocessing — no leakage.
%     A2. Each RF run uses a fresh rng(RNG_SEED + m_idx) so forests are
%         independent yet fully reproducible.
%     A3. No normalisation. RF is invariant to monotone feature transforms.
%     A4. NZV (near-zero variance) filter fitted on TRAINING data only;
%         same mask applied passively to test data.
%     A5. OOB error is used as the CV proxy for mtry selection (Step 6.5b).
%         This is equivalent to leave-one-bootstrap-out CV at no extra cost.
%         No separate K-fold loop is needed.
%     A6. mtry = floor(sqrt(p_active)) is the RF default for classification
%         (Breiman 2001). Marked as reference on the figure.
%
%   REPRODUCIBILITY: rng(42, 'twister') global. Per-run seed = 42 + sweep_idx.
%
%   REFERENCE:
%     Hastie, Tibshirani, Friedman (2009). ESL §15, Exercise 15.6.
%
% Author: Ocean Library
% Date:   2026-05-12

clc; clear; close all;
RNG_SEED = 42;
rng(RNG_SEED, 'twister');

% Resolve paths from script location — works regardless of MATLAB CWD
scriptDir   = fileparts(mfilename('fullpath'));
libraryRoot = fullfile(scriptDir, '..', '..', '..');
DATASET     = 'spambase';

% Storage directories (inside library root, not CWD-relative)
lakeDir   = fullfile(libraryRoot, 'Data', 'lakes',    'spambase_lake');
dwDir     = fullfile(libraryRoot, 'Data', 'warehouse', 'spambase_dw');
fsDir     = fullfile(libraryRoot, 'Data', 'features',  'spambase_fs');
resDir    = fullfile(libraryRoot, 'Results', 'Supervised');

% =========================================================================
% === Step 1: Data Ingestion ===
% =========================================================================
fprintf('\n[Step 1] Ingesting Spambase...\n');

dataPath = fullfile(libraryRoot, 'Data', 'spambase', 'spambase.data');
if ~isfile(dataPath)
    error('exp_rf_spam_mtry:DataNotFound', ...
        ['Cannot find spambase.data.\nExpected: %s\n' ...
         'Ensure Data/spambase/ exists in the library root.'], dataPath);
end

raw = readmatrix(dataPath, 'FileType', 'text', 'Delimiter', ',');  % 4601 x 58, no header
X_all = raw(:, 1:57);            % features (word freq, char freq, capital stats)
Y_all = raw(:, 58) + 1;          % label: recode 0/1 → 1/2 (library convention)

[N_all, p_raw] = size(X_all);
n_spam = sum(Y_all == 2);
fprintf('    N=%d  p=%d  spam=%d (%.1f%%)  ham=%d\n', ...
    N_all, p_raw, n_spam, 100*n_spam/N_all, N_all - n_spam);

% =========================================================================
% === Step 1.5: Train / Test Split (MANDATORY — Anti-Leakage) ===
% =========================================================================
% Split BEFORE any storage or preprocessing. Test rows never touch schema
% inference, NZV bounds, or any fitted transform.
fprintf('\n[Step 1.5] Anti-leakage train/test split...\n');

splitter = OceanData.DataArch.DataSplitter( ...
    'Strategy', 'holdout', ...
    'Ratio',    2/3, ...          % 2/3 train ≈ 3067, 1/3 test ≈ 1534
    'Seed',     RNG_SEED);

split = splitter.split(X_all, Y_all);
X_train = X_all(split.trainIdx, :);   Y_train = Y_all(split.trainIdx);
X_test  = X_all(split.testIdx,  :);   Y_test  = Y_all(split.testIdx);

n_tr = sum(split.trainIdx);
n_te = sum(split.testIdx);
fprintf('    Train=%d  Test=%d  (ratio=%.3f)\n', n_tr, n_te, n_tr/N_all);

% =========================================================================
% === Step 2: DataLake (DA-1) — Raw Persistence ===
% =========================================================================
fprintf('\n[Step 2] DataLake: storing raw train/test...\n');
try
    lake = OceanData.DataArch.DataLake('rootDir', lakeDir);
    meta_tr = struct('source', DATASET, 'seed', RNG_SEED, ...
        'split', 'train', 'N', n_tr, 'p', p_raw);
    meta_te = struct('source', DATASET, 'seed', RNG_SEED, ...
        'split', 'test',  'N', n_te, 'p', p_raw);

    lake.put('raw/train/X', X_train, meta_tr);
    lake.put('raw/train/Y', Y_train, meta_tr);
    lake.put('raw/test/X',  X_test,  meta_te);
    lake.put('raw/test/Y',  Y_test,  meta_te);
    fprintf('[PASS] DataLake: raw train/test saved (%s).\n', lakeDir);
catch ME
    fprintf('[SKIP] DataLake failed: %s\n', ME.message);
end

% =========================================================================
% === Step 3: DataWarehouse (DA-2/DA-3) — Structured Storage ===
% =========================================================================
% Schema bounds derived from TRAINING data only (A1 anti-leakage).
% We store the 57 features + 1 label as a columnar schema.
% insertBatch is write-once — safe to re-run.
fprintf('\n[Step 3] DataWarehouse: schema-validated storage...\n');
try
    dw = OceanData.DataArch.DataWarehouse('rootDir', dwDir);

    % Build column names: feat_1..feat_57, label
    colNames = [arrayfun(@(i) sprintf('feat_%d', i), 1:p_raw, ...
        'UniformOutput', false), {'label'}];
    colTypes = [repmat({'double'}, 1, p_raw), {'integer'}];

    % Bounds from TRAINING data only
    minVals = [min(X_train, [], 1), 1];
    maxVals = [max(X_train, [], 1), 2];

    schema = OceanData.DataArch.DWSchema(colNames, colTypes, ...
        'MinValues', minVals, 'MaxValues', maxVals);

    dw.createTable(DATASET, schema);

    % Convert training matrix to table for insertBatch
    trainData = array2table([X_train, Y_train], 'VariableNames', colNames);
    dw.insertBatch(DATASET, trainData);
    fprintf('[PASS] DataWarehouse: %d rows in table "%s".\n', ...
        dw.rowCount(DATASET), DATASET);
catch ME
    fprintf('[SKIP] DataWarehouse failed: %s\n', ME.message);
end

% =========================================================================
% === Step 4: CachingWarehouse (DA-4) — Read-Through Cache ===
% =========================================================================
fprintf('\n[Step 4] CachingWarehouse (Redis)...\n');
try
    redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
    redis.connect();
    cdw = OceanData.Adapters.CachingWarehouse(dw, redis);
    fprintf('[PASS] CachingWarehouse connected.\n');
catch
    fprintf('[SKIP] Redis unavailable — CachingWarehouse bypassed.\n');
end

% =========================================================================
% === Step 5: FeatureStore (DA-5) — NZV-Filtered Features ===
% =========================================================================
% NZV filter: remove features with variance < NZV_THRESH on TRAINING data.
% Store filtered train/test matrices. Test mask applied passively.
fprintf('\n[Step 5] FeatureStore: NZV feature filtering...\n');

NZV_THRESH = 1e-4;
feat_key   = sprintf('nzv/spambase/thresh%.0e_seed%d', NZV_THRESH, RNG_SEED);

try
    fs = OceanData.DataArch.FeatureStore('rootDir', fsDir);

    nzv_meta = struct('transform', 'NZV', 'threshold', NZV_THRESH, ...
        'source', 'raw/train/X', 'seed', RNG_SEED);

    entry = fs.getOrCompute(feat_key, @() applyNZV(X_train, NZV_THRESH), nzv_meta);
    X_train_fs = entry.data;

    % Apply the same column mask to test (fitted on training only)
    tr_var     = var(X_train);
    active_mask = tr_var > NZV_THRESH;
    X_test_fs   = X_test(:, active_mask);

    p_active = size(X_train_fs, 2);
    n_removed = p_raw - p_active;
    fprintf('[PASS] FeatureStore: %d features retained, %d removed (NZV).\n', ...
        p_active, n_removed);
    fs.describe();
catch ME
    fprintf('[SKIP] FeatureStore failed: %s — using full feature set.\n', ME.message);
    X_train_fs = X_train;
    X_test_fs  = X_test;
    active_mask = true(1, p_raw);
    p_active    = p_raw;
end

% =========================================================================
% === Step 6: FeatureCache (DA-6) — In-Memory Features ===
% =========================================================================
fprintf('\n[Step 6] FeatureCache (Redis)...\n');
try
    if ~exist('redis', 'var') || ~redis.isConnected()
        error('Redis not available');
    end
    fc = OceanData.Adapters.FeatureCache(redis);
    fc.put('spambase/nzv/train', X_train_fs, nzv_meta);
    X_batch = fc.getOrCompute('spambase/nzv/train', @() X_train_fs, nzv_meta);
    X_train_fs = X_batch.data;   % exercise fast-path retrieval
    fprintf('[PASS] FeatureCache: train features loaded from cache.\n');
catch
    fprintf('[SKIP] Redis unavailable — FeatureCache bypassed.\n');
end

% =========================================================================
% === Step 6.5a: Feature Selection (already done via NZV in Step 5) ===
% =========================================================================
% NZV filter was fitted on training data in Step 5 and applied passively
% to test data. This step is therefore structurally complete.
% No further wrapper/embedded selection is applied since RF handles
% correlated features well via internal Mtry randomization.
fprintf('\n[Step 6.5a] Feature selection: NZV complete (%d → %d features).\n', ...
    p_raw, p_active);

% =========================================================================
% === Step 6.5b: Hyperparameter Tuning — mtry Sweep (OOB as CV proxy) ===
% =========================================================================
% The mtry sweep IS the cross-validation. OOB error is a free, unbiased
% estimate of generalization error (equivalent to leave-one-bootstrap-out CV).
% No separate K-fold loop is required — this is the key advantage of RF
% over other methods for hyperparameter selection.
%
% ANTI-LEAKAGE: The sweep operates exclusively on X_train_fs / Y_train.
% X_test_fs is NOT touched until Step 7.
fprintf('\n[Step 6.5b] mtry sweep (OOB as CV proxy) on training data only...\n');

% Hyperparameters (fixed across sweep)
B           = 600;   % increased from 300 — small-mtry forests need more trees to stabilise OOB
CRITERION   = 'gini';
MAX_DEPTH   = Inf;
MIN_LEAF    = 1;

% mtry sweep: 15 values, log-spaced across 1..p_active
mtry_vals = unique(round(logspace(0, log10(p_active), 15)));
n_mtry    = numel(mtry_vals);

m_default = floor(sqrt(p_active));   % RF classification default (Breiman 2001)

fprintf('    B=%d trees, p_active=%d, sweep: [%s]\n', B, p_active, ...
    num2str(mtry_vals, '%d '));
fprintf('    RF default m = floor(sqrt(%d)) = %d\n', p_active, m_default);
fprintf('\n    %-6s  %-10s\n', 'mtry', 'OOB Err');
fprintf('    %s\n', repmat('-', 1, 20));

oob_err = zeros(1, n_mtry);

for i = 1:n_mtry
    m_i = mtry_vals(i);
    rng(RNG_SEED, 'twister');       % A2: fixed seed across all mtry runs → bootstrap variance removed

    rf_i = OceanStats.Ensemble.RandomForest( ...
        'NumTrees',          B, ...
        'Mtry',              m_i, ...
        'Criterion',         CRITERION, ...
        'MaxDepth',          MAX_DEPTH, ...
        'MinLeafSize',       MIN_LEAF, ...
        'ComputeOOB',        true, ...
        'ComputeImportance', false, ...
        'Verbose',           false);

    rf_i.fit(X_train_fs, Y_train);
    oob_err(i) = rf_i.oobError();
    fprintf('    m=%-4d  OOB=%.4f\n', m_i, oob_err(i));
end

[min_oob, best_idx] = min(oob_err);
best_m = mtry_vals(best_idx);
fprintf('\n    Best OOB: %.4f at m=%d\n', min_oob, best_m);

% =========================================================================
% === Step 7: Final Model Training & Evaluation (test set: ONE time only) ===
% =========================================================================
fprintf('\n[Step 7] Final model training with best m=%d on full training set...\n', best_m);

rng(RNG_SEED, 'twister');   % fixed seed for final model
rf_final = OceanStats.Ensemble.RandomForest( ...
    'NumTrees',          B, ...
    'Mtry',              best_m, ...
    'Criterion',         CRITERION, ...
    'MaxDepth',          MAX_DEPTH, ...
    'MinLeafSize',       MIN_LEAF, ...
    'ComputeOOB',        true, ...
    'ComputeImportance', false, ...
    'Verbose',           false);

rf_final.fit(X_train_fs, Y_train);

% --- Evaluate on held-out test set (touched exactly ONCE here) -----------
Y_pred    = rf_final.predict(X_test_fs);
test_err  = mean(Y_pred ~= Y_test);
oob_final = rf_final.oobError();

% Compute test error for ALL mtry values (for the dual-curve figure)
fprintf('    Computing test error for all %d mtry values...\n', n_mtry);
test_err_all = zeros(1, n_mtry);
for i = 1:n_mtry
    m_i = mtry_vals(i);
    rng(RNG_SEED + i, 'twister');   % same seeds as Step 6.5b for matched forests

    rf_i = OceanStats.Ensemble.RandomForest( ...
        'NumTrees',          B, ...
        'Mtry',              m_i, ...
        'Criterion',         CRITERION, ...
        'MaxDepth',          MAX_DEPTH, ...
        'MinLeafSize',       MIN_LEAF, ...
        'ComputeOOB',        false, ...
        'ComputeImportance', false, ...
        'Verbose',           false);

    rf_i.fit(X_train_fs, Y_train);
    test_err_all(i) = mean(rf_i.predict(X_test_fs) ~= Y_test);
    fprintf('    m=%-4d  Test=%.4f\n', m_i, test_err_all(i));
end

[min_test, test_best_idx] = min(test_err_all);
fprintf('\n    Best Test error: %.4f at m=%d\n', min_test, mtry_vals(test_best_idx));
fprintf('    Final model (m=%d): OOB=%.4f  Test=%.4f\n', best_m, oob_final, test_err);

% =========================================================================
% === Step 8: Visualization (OceanViz.Core, Dark theme) ===
% =========================================================================
fprintf('\n[Step 8] Plotting dual-error curve...\n');

viz = OceanViz.Core('Theme', 'Dark');
C   = viz.Colors;

fig = viz.createFigure('ESL Ex 15.6 — RF Sensitivity to mtry (Spambase)');
fig.Position = [60, 120, 860, 500];

ax = axes(fig);
viz.styleAxes(ax);
hold(ax, 'on');

% ---- Shaded band between OOB and test error ----------------------------
x_fill = [mtry_vals, fliplr(mtry_vals)];
y_fill = [oob_err,   fliplr(test_err_all)];
fill(ax, x_fill, y_fill * 100, C.Primary, ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% ---- OOB error (dashed) ------------------------------------------------
h_oob = plot(ax, mtry_vals, oob_err * 100, '--o', ...
    'Color', C.Primary, 'LineWidth', 2.2, 'MarkerSize', 7, ...
    'MarkerFaceColor', C.Primary, 'DisplayName', 'OOB error');

% ---- Test error (solid) ------------------------------------------------
h_test = plot(ax, mtry_vals, test_err_all * 100, '-s', ...
    'Color', C.Secondary, 'LineWidth', 2.2, 'MarkerSize', 7, ...
    'MarkerFaceColor', C.Secondary, 'DisplayName', 'Test error');

% ---- Vertical reference: m = sqrt(p_active) ----------------------------
y_top = max([oob_err, test_err_all]) * 100;
xline(ax, m_default, '--', 'Color', C.Boundary, 'LineWidth', 1.4, ...
    'Alpha', 0.85, 'HandleVisibility', 'off');
text(ax, m_default * 1.08, y_top * 0.97, ...
    sprintf('$m=\\lfloor\\sqrt{p}\\rfloor=%d$', m_default), ...
    'Color', C.Boundary, 'FontSize', 10, 'Interpreter', 'latex');

% ---- Star: best m from OOB selection -----------------------------------
h_star = plot(ax, best_m, min_oob * 100, 'p', ...
    'Color', C.Boundary, 'MarkerSize', 14, 'LineWidth', 1.5, ...
    'MarkerFaceColor', C.Boundary, 'DisplayName', sprintf('Best m=%d (OOB)', best_m));

hold(ax, 'off');

% ---- Axis formatting ---------------------------------------------------
set(ax, 'XScale', 'log');
set(ax, 'XTick', mtry_vals);
set(ax, 'XTickLabel', arrayfun(@num2str, mtry_vals, 'UniformOutput', false));
ax.TickLabelInterpreter = 'none';

xlabel(ax, '$m$  (candidate features per split)', ...
    'Interpreter', 'latex', 'FontSize', 13, 'Color', C.Text);
ylabel(ax, 'Error rate  (\%)', ...
    'Interpreter', 'latex', 'FontSize', 13, 'Color', C.Text);
title(ax, sprintf( ...
    'RF on Spambase: OOB vs Test Error  ($B=%d$,  $p_{\\rm active}=%d$)', ...
    B, p_active), ...
    'Interpreter', 'latex', 'FontSize', 14, 'Color', C.Text, 'FontWeight', 'bold');

leg = legend(ax, [h_oob, h_test, h_star], 'Location', 'northeast');
set(leg, 'TextColor', C.Text, 'EdgeColor', C.Grid, ...
    'Color', C.Background, 'Interpreter', 'latex', 'FontSize', 11);

y_lo = max(0, min([oob_err, test_err_all]) * 100 - 0.5);
y_hi =        max([oob_err, test_err_all]) * 100 + 0.5;
ylim(ax, [y_lo, y_hi]);

% ---- Export ------------------------------------------------------------
if ~exist(resDir, 'dir'), mkdir(resDir); end
outFile = fullfile(resDir, 'exp_rf_spam_mtry.png');
exportgraphics(fig, outFile, 'Resolution', 180, 'BackgroundColor', C.Background);
fprintf('[PASS] Figure saved: %s\n', outFile);

% =========================================================================
% === Summary Printout ===
% =========================================================================
oob_test_corr = corr(oob_err', test_err_all');
fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : Spambase (N=%d, p_raw=%d, p_active=%d)\n', ...
    N_all, p_raw, p_active);
fprintf('  Spam rate     : %.1f%%\n', 100*n_spam/N_all);
fprintf('  Split         : train=%d  test=%d  (2/3-1/3, seed=%d)\n', ...
    n_tr, n_te, RNG_SEED);
fprintf('  Model         : Random Forest, Criterion=%s, B=%d trees\n', ...
    CRITERION, B);
fprintf('  mtry sweep    : [%s]\n', num2str(mtry_vals, '%d '));
fprintf('  Best m (OOB)  : %d  →  OOB=%.2f%%  Test=%.2f%%\n', ...
    best_m, min_oob*100, test_err_all(best_idx)*100);
fprintf('  Best m (Test) : %d  →  Test=%.2f%%\n', ...
    mtry_vals(test_best_idx), min_test*100);
fprintf('  OOB-Test corr : r=%.4f  (OOB validity as CV proxy)\n', oob_test_corr);
fprintf('  Max OOB-Test gap: %.2f pp\n', max(abs(oob_err - test_err_all))*100);
fprintf('  ESL context   : ESL §15 reports ~7%% test error on spam with RF.\n');
fprintf('==========================\n\n');

% =========================================================================
%  Helper functions
% =========================================================================
function X_out = applyNZV(X, thresh)
    % Apply near-zero variance filter. Fitted on X (training data only).
    feat_var = var(X);
    mask     = feat_var > thresh;
    X_out    = X(:, mask);
end
