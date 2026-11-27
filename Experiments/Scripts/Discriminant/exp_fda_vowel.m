% EXP_FDA_VOWEL  Flexible Discriminant Analysis on the Vowel Dataset
%
%   DATASET:        Vowel (N_train=528, N_test=462, p=10, K=11 classes)
%   MODELS:         1. LDA (Linear Discriminant Analysis)
%                   2. FDA (Flexible Discriminant Analysis via MARS)
%   ASSUMPTIONS:    Using MARS (Degree 2) to approximate S-PLUS 'bruto'. 
%                   The goal is to cleanly separate the 11 vowel sounds
%                   in the 2D canonical subspace.
%   REPRODUCIBILITY: rng(42)
%
%   Ref: ESL Figure 12.10 and Table 12.3

clc; clear; close all;
rng(42, 'twister');
RNG_SEED = 42;
DATASET  = 'vowel';

ROOT = 'c:\Ocean Library\Matlab code practice';
addpath(ROOT);

fprintf('=== EXP_FDA_VOWEL: ESL Figure 12.10 Reproduction ===\n\n');

%% === 1. Ingestion ===
fprintf('[1] Loading Vowel Dataset...\n');
trainFile = fullfile(ROOT, 'Data', 'vowel', 'vowel.train.txt');
testFile  = fullfile(ROOT, 'Data', 'vowel', 'vowel.test.txt');

trainData = readtable(trainFile, 'Format', repmat('%f', 1, 12), 'Delimiter', ',', 'HeaderLines', 1);
testData  = readtable(testFile, 'Format', repmat('%f', 1, 12), 'Delimiter', ',', 'HeaderLines', 1);

% Extract features (columns 3 to 12) and labels (column 2)
X_train = table2array(trainData(:, 3:12));
y_train = table2array(trainData(:, 2));

X_test = table2array(testData(:, 3:12));
y_test = table2array(testData(:, 2));

[N_tr, p] = size(X_train);
K_classes = length(unique(y_train));
fprintf('  Train N = %d, Test N = %d, p = %d, Classes = %d\n\n', N_tr, size(X_test,1), p, K_classes);

%% === 1.5. Train / Test Split (MANDATORY — Anti-Leakage) ===
% [SKIP - 1.5] Reason: Vowel dataset has a CANONICAL ESL train/test split
% provided in separate files (vowel.train.txt / vowel.test.txt).
fprintf('[1.5] [SKIP] Train/test split provided by canonical ESL files.\n');

%% === 2. DataLake (DA-1) ===
fprintf('\n[2] DataLake (DA-1) - Raw Persistence...\n');
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    meta = struct('source', DATASET, 'seed', RNG_SEED);
    lake.put('raw/vowel/X_train', X_train, meta);
    lake.put('raw/vowel/y_train', y_train, meta);
    lake.put('raw/vowel/X_test',  X_test,  meta);
    lake.put('raw/vowel/y_test',  y_test,  meta);
    fprintf('  [PASS] DataLake ready.\n');
catch e
    fprintf('  [SKIP] DataLake: %s\n', e.message);
end

%% === 3. DataWarehouse (DA-2/DA-3) ===
fprintf('\n[3] DataWarehouse - Structured Storage...\n');
try
    dw = OceanData.DataArch.DataWarehouse('rootDir', './dw/');
    colNames = [arrayfun(@(i) sprintf('x%d',i), 1:p, 'UniformOutput', false), {'y'}];
    colTypes = [repmat({'double'}, 1, p), {'double'}];
    schema = OceanData.DataArch.DWSchema(colNames, colTypes);
    dw.createTable(DATASET, schema);
    trainTable = array2table([X_train, y_train], 'VariableNames', colNames);
    dw.insertBatch(DATASET, trainTable);
    fprintf('  [PASS] DataWarehouse ready.\n');
catch e
    fprintf('  [SKIP] DataWarehouse: %s\n', e.message);
end

%% === 4. CachingWarehouse (DA-4) ===
fprintf('\n[4] CachingWarehouse (DA-4) - Read-Through Cache...\n');
try
    redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
    cdw   = OceanData.Adapters.CachingWarehouse(dw, redis);  %#ok<NASGU>
    fprintf('  [PASS] CachingWarehouse connected to Redis.\n');
catch
    fprintf('  [SKIP] Redis unavailable.\n');
end

%% === 5 & 6. FeatureStore & FeatureCache (DA-5/DA-6) ===
fprintf('\n[5&6] FeatureStore & FeatureCache...\n');
try
    fs       = OceanData.DataArch.FeatureStore('rootDir', './fs/');
    feat_key = sprintf('%s_raw_seed%d', DATASET, RNG_SEED);
    if ~fs.exists(feat_key)
        fs.put(feat_key, X_train);
    end
    entry  = fs.get(feat_key);
    X_feat = entry.data; %#ok<NASGU>
    cache  = OceanData.Adapters.FeatureCache();
    cache.put('X_train_feat', X_train);
    X_batch = cache.get('X_train_feat'); %#ok<NASGU>
    fprintf('  [PASS] FeatureStore + FeatureCache ready.\n');
catch e
    fprintf('  [SKIP] FeatureStore/Cache: %s\n', e.message);
end

%% === 6.5. Feature Selection + Model Selection ===
% [SKIP] LDA and FDA have no hyperparameters to tune for this reproduction.
fprintf('\n[6.5] [SKIP] Feature selection / tuning not needed (reproduction experiment).\n');

%% === 7. Model Training & Evaluation ===
fprintf('\n[7] Model Training & Evaluation...\n');

% Model cache: train-once, load-on-rerun
modelStore = OceanData.DataArch.FeatureStore('rootDir', './cache/models/');
cachePrefix = sprintf('%s/seed%d', DATASET, RNG_SEED);

% ── 7a. LDA ──────────────────────────────────────────────────────────────
fprintf('  Training (1) LDA...\n');
ldaModel = OceanTools.cachedFit(modelStore, sprintf('%s/LDA', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.LDA(), X_train, y_train);
ldaErr_tr = mean(ldaModel.predict(X_train) ~= y_train);
ldaErr_te = mean(ldaModel.predict(X_test)  ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', ldaErr_tr, ldaErr_te);

% ── 7b. QDA ──────────────────────────────────────────────────────────────
fprintf('  Training (2) QDA...\n');
qdaModel = OceanTools.cachedFit(modelStore, sprintf('%s/QDA', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.QDA(), X_train, y_train);
qdaErr_tr = mean(qdaModel.predict(X_train) ~= y_train);
qdaErr_te = mean(qdaModel.predict(X_test)  ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', qdaErr_tr, qdaErr_te);

% ── 7c. CART ─────────────────────────────────────────────────────────────
fprintf('  Training (3) CART...\n');
cartModel = OceanTools.cachedFit(modelStore, sprintf('%s/CART_d15_gini', cachePrefix), ...
    @() OceanStats.Supervised.Tree.DecisionTree('MaxDepth', 15, 'Criterion', 'gini'), X_train, y_train);
cartErr_tr = mean(cartModel.predict(X_train) ~= y_train);
cartErr_te = mean(cartModel.predict(X_test)  ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', cartErr_tr, cartErr_te);

% ── 7d. Nearest Neighbor (1-NN) ──────────────────────────────────────────
fprintf('  Training (8) Nearest Neighbor (1-NN)...\n');
% Non-parametric: no model object to cache — distance computed inline
D = pdist2(X_train, X_train);
D(logical(eye(size(D)))) = Inf;  % exclude self
[~, nn_tr] = min(D, [], 2);
knnPred_tr = y_train(nn_tr);
knnErr_tr = mean(knnPred_tr ~= y_train);

D_te = pdist2(X_test, X_train);
[~, nn_te] = min(D_te, [], 2);
knnPred_te = y_train(nn_te);
knnErr_te = mean(knnPred_te ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', knnErr_tr, knnErr_te);

% ── 7e. FDA/BRUTO (smoothing splines) ────────────────────────────────────
fprintf('  Training (9) FDA/BRUTO (additive smoothing splines)...\n');
brutoBuilder = @() OceanStats.Supervised.Discriminant.BrutoAdapter();
fdaBruto = OceanTools.cachedFit(modelStore, sprintf('%s/FDA_BRUTO', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', brutoBuilder), X_train, y_train);
brutoErr_tr = mean(fdaBruto.predict(X_train) ~= y_train);
brutoErr_te = mean(fdaBruto.predict(X_test)  ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', brutoErr_tr, brutoErr_te);

% Reduced dimension evaluation for FDA/BRUTO
fprintf('    Evaluating reduced-dimension projections...\n');
best_bruto_dim = 0; best_bruto_te = 1.0;
K_minus_1 = K_classes - 1;
for d = 1:min(K_minus_1, p)
    Z_tr = fdaBruto.project(X_train, d);
    Z_te = fdaBruto.project(X_test,  d);
    sub_lda = OceanStats.Supervised.Discriminant.LDA();
    sub_lda.fit(Z_tr, y_train);
    sub_te = mean(sub_lda.predict(Z_te) ~= y_test);
    if sub_te < best_bruto_te
        best_bruto_te = sub_te;
        best_bruto_dim = d;
    end
end
fprintf('    Best reduced dim = %d → Test: %.2f\n', best_bruto_dim, best_bruto_te);

% ── 7f. FDA/MARS (degree = 1) ────────────────────────────────────────────
fprintf('  Training (10) FDA/MARS (degree = 1)...\n');
marsBuilder1 = @() OceanStats.Supervised.MARS.MARS('MaxInteractionDegree', 1, 'MaxTerms', 50, ...
    'Encoder', OceanStats.Data.RawEncoder());
fdaModel1 = OceanTools.cachedFit(modelStore, sprintf('%s/FDA_MARS_deg1_mt50', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', marsBuilder1), X_train, y_train);
fda1Err_tr = mean(fdaModel1.predict(X_train) ~= y_train);
fda1Err_te = mean(fdaModel1.predict(X_test)  ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', fda1Err_tr, fda1Err_te);

% Reduced dimension evaluation for FDA/MARS (degree = 1)
fprintf('    Evaluating reduced-dimension projections...\n');
best_fda1_dim = 0; best_fda1_te = 1.0;
for d = 1:min(K_minus_1, p)
    Z_tr = fdaModel1.project(X_train, d);
    Z_te = fdaModel1.project(X_test,  d);
    sub_lda = OceanStats.Supervised.Discriminant.LDA();
    sub_lda.fit(Z_tr, y_train);
    sub_te = mean(sub_lda.predict(Z_te) ~= y_test);
    if sub_te < best_fda1_te
        best_fda1_te = sub_te;
        best_fda1_dim = d;
    end
end
fprintf('    Best reduced dim = %d → Test: %.2f\n', best_fda1_dim, best_fda1_te);

% ── 7g. FDA/MARS (degree = 2) ────────────────────────────────────────────
fprintf('  Training (11) FDA/MARS (degree = 2)...\n');
marsBuilder2 = @() OceanStats.Supervised.MARS.MARS('MaxInteractionDegree', 2, 'MaxTerms', 50, ...
    'Encoder', OceanStats.Data.RawEncoder());
fdaModel2 = OceanTools.cachedFit(modelStore, sprintf('%s/FDA_MARS_deg2_mt50', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', marsBuilder2), X_train, y_train);
fda2Err_tr = mean(fdaModel2.predict(X_train) ~= y_train);
fda2Err_te = mean(fdaModel2.predict(X_test)  ~= y_test);
fprintf('    Train: %.2f  Test: %.2f\n', fda2Err_tr, fda2Err_te);

% Reduced dimension evaluation for FDA/MARS (degree = 2)
fprintf('    Evaluating reduced-dimension projections...\n');
best_fda2_dim = 0; best_fda2_te = 1.0;
for d = 1:min(K_minus_1, p)
    Z_tr = fdaModel2.project(X_train, d);
    Z_te = fdaModel2.project(X_test,  d);
    sub_lda = OceanStats.Supervised.Discriminant.LDA();
    sub_lda.fit(Z_tr, y_train);
    sub_te = mean(sub_lda.predict(Z_te) ~= y_test);
    if sub_te < best_fda2_te
        best_fda2_te = sub_te;
        best_fda2_dim = d;
    end
end
fprintf('    Best reduced dim = %d → Test: %.2f\n', best_fda2_dim, best_fda2_te);


%% === Table 12.3 Reproduction ═════════════════════════════════════════════
fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║       TABLE 12.3 — Vowel Recognition Performance            ║\n');
fprintf('║       Ref: ESL Section 12.5, p. 444                         ║\n');
fprintf('╠════╦════════════════════════════════════════╦═════════════════╣\n');
fprintf('║    ║ Technique                              ║ Training  Test  ║\n');
fprintf('╠════╬════════════════════════════════════════╬═════════════════╣\n');
fprintf('║ 1  ║ LDA                                    ║  %.2f    %.2f  ║\n', ldaErr_tr, ldaErr_te);
fprintf('║ 2  ║ QDA                                    ║  %.2f    %.2f  ║\n', qdaErr_tr, qdaErr_te);
fprintf('║ 3  ║ CART                                   ║  %.2f    %.2f  ║\n', cartErr_tr, cartErr_te);
fprintf('║ 5  ║ Single-layer perceptron                ║   —      —     ║\n');
fprintf('║ 6  ║ Multi-layer perceptron                 ║   —      —     ║\n');
fprintf('║ 7  ║ Gaussian node network                  ║   —      —     ║\n');
fprintf('║ 8  ║ Nearest neighbor (1-NN)                ║  %.2f    %.2f  ║\n', knnErr_tr, knnErr_te);
fprintf('║ 9  ║ FDA/BRUTO                              ║  %.2f    %.2f  ║\n', brutoErr_tr, brutoErr_te);
fprintf('║    ║   Best reduced dimension (=%d)          ║          %.2f  ║\n', best_bruto_dim, best_bruto_te);
fprintf('║ 10 ║ FDA/MARS (degree = 1)                  ║  %.2f    %.2f  ║\n', fda1Err_tr, fda1Err_te);
fprintf('║    ║   Best reduced dimension (=%d)          ║          %.2f  ║\n', best_fda1_dim, best_fda1_te);
fprintf('║ 11 ║ FDA/MARS (degree = 2)                  ║  %.2f    %.2f  ║\n', fda2Err_tr, fda2Err_te);
fprintf('║    ║   Best reduced dimension (=%d)          ║          %.2f  ║\n', best_fda2_dim, best_fda2_te);
fprintf('╠════╬════════════════════════════════════════╬═════════════════╣\n');
fprintf('║    ║ [ESL Reference Values]                 ║                 ║\n');
fprintf('║    ║ LDA (book)                             ║  0.32    0.56  ║\n');
fprintf('║    ║ QDA (book)                             ║  0.01    0.53  ║\n');
fprintf('║    ║ CART (book)                            ║  0.05    0.56  ║\n');
fprintf('║    ║ Nearest neighbor (book)                ║  0.00    0.44  ║\n');
fprintf('║    ║ FDA/BRUTO (book)                       ║  0.06    0.44  ║\n');
fprintf('║    ║ FDA/MARS deg=1 (book)                  ║  0.09    0.45  ║\n');
fprintf('║    ║   Best reduced dim=2 (book)            ║  0.18    0.42  ║\n');
fprintf('║    ║ FDA/MARS deg=2 (book)                  ║  0.02    0.42  ║\n');
fprintf('║    ║   Best reduced dim=6 (book)            ║  0.13    0.39  ║\n');
fprintf('╚════╩════════════════════════════════════════╩═════════════════╝\n');
fprintf('\nNote: Models 4-7 (perceptrons, Gaussian nets) not yet in library.\n');
fprintf('      Softmax variants omitted (not applicable to discriminant models).\n\n');


%% === 8. Visualization — ESL Figure 12.10 ═════════════════════════════════
fprintf('[8] Visualization — ESL Figure 12.10 (LDA vs FDA canonical variates)...\n');

% Extract the 2D canonical coordinates for the training data
coords_lda = ldaModel.project(X_train, 2);
coords_fda = fdaBruto.project(X_train, 2);

% 11 distinct colors matching textbook (vibrant, distinguishable)
colors11 = [
    0.90 0.10 0.10;  % 1 Red
    0.10 0.70 0.10;  % 2 Green
    0.10 0.10 0.90;  % 3 Blue
    1.00 0.65 0.00;  % 4 Orange
    0.60 0.10 0.80;  % 5 Purple
    0.00 0.80 0.80;  % 6 Cyan
    0.90 0.10 0.90;  % 7 Magenta
    0.60 0.60 0.10;  % 8 Olive/Yellow-green
    0.45 0.45 0.45;  % 9 Gray
    0.55 0.27 0.07;  % 10 Brown
    0.95 0.45 0.65   % 11 Pink
];

try
    viz = OceanViz.Core('Theme', 'Dark');
    fig = viz.createFigure('ESL Figure 12.10 — LDA vs FDA');
    set(fig, 'Position', [50, 100, 1100, 500]);

    % ── Left panel: LDA ──────────────────────────────────────────────────
    ax1 = subplot(1, 2, 1, 'Parent', fig);
    hold(ax1, 'on');
    for k = 1:11
        idx = (y_train == k);
        scatter(ax1, coords_lda(idx, 1), coords_lda(idx, 2), 30, ...
            'MarkerEdgeColor', colors11(k,:), 'MarkerFaceColor', 'none', 'LineWidth', 1.0);
    end
    hold(ax1, 'off');
    title(ax1, 'Linear Discriminant Analysis', 'FontWeight', 'bold', 'FontSize', 12, 'Color', viz.Colors.Text);
    xlabel(ax1, 'Coordinate 1 for Training Data', 'Color', viz.Colors.Text);
    ylabel(ax1, 'Coordinate 2 for Training Data', 'Color', viz.Colors.Text);
    viz.styleAxes(ax1);
    axis(ax1, 'padded');

    % ── Right panel: FDA ─────────────────────────────────────────────────
    ax2 = subplot(1, 2, 2, 'Parent', fig);
    hold(ax2, 'on');
    for k = 1:11
        idx = (y_train == k);
        scatter(ax2, coords_fda(idx, 1), coords_fda(idx, 2), 30, ...
            'MarkerEdgeColor', colors11(k,:), 'MarkerFaceColor', 'none', 'LineWidth', 1.0);
    end
    hold(ax2, 'off');
    title(ax2, 'Flexible Discriminant Analysis -- Bruto', 'FontWeight', 'bold', 'FontSize', 12, 'Color', viz.Colors.Text);
    xlabel(ax2, 'Coordinate 1 for Training Data', 'Color', viz.Colors.Text);
    ylabel(ax2, 'Coordinate 2 for Training Data', 'Color', viz.Colors.Text);
    viz.styleAxes(ax2);
    axis(ax2, 'padded');

    fprintf('  [PASS] Figure 12.10 displayed.\n');
catch plotEx
    fprintf('  [ERROR] Visualization: %s\n', plotEx.getReport());
end

fprintf('\n=== Experiment Complete ===\n');

