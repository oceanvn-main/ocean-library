% EXP_WAVEFORM_TABLE_12_4  ESL Table 12.4, Figure 12.14 & 12.15
%
% Ref: Hastie, Tibshirani, Friedman — ESL, Section 12.7.1
% Pipeline: Ocean Library 8-Step ML Pipeline
%
% Reproduces:
%   Table  12.4  — Classification performance on waveform data (8 models)
%   Figure 12.14 — Base waveform shapes (3 classes)
%   Figure 12.15 — PMDA canonical variate scatter plot

clc; clear; close all;
rng(42, 'twister');
RNG_SEED = 42;
DATASET  = 'waveform';
ROOT     = 'c:\Ocean Library\Matlab code practice';
addpath(ROOT);

fprintf('=== ESL Table 12.4 & Figures 12.14–12.15 — Waveform Data ===\n');
fprintf('==============================================================\n');

%% === 1. Ingestion ===
fprintf('\n[1] Ingestion — Generating Waveform Dataset...\n');
% Pure (no noise) for Figure 12.14 base waveform visualization
gen_pure = OceanData.Generators.WaveformGenerator('N', 300, 'IncludeNoise', false);
[X_pure, y_pure] = gen_pure.generate();

% Noisy data for training (N=300) and testing (N=500), per ESL Table 12.4
gen_train = OceanData.Generators.WaveformGenerator('N', 300, 'IncludeNoise', true);
[X_train, y_train] = gen_train.generate();

gen_test = OceanData.Generators.WaveformGenerator('N', 500, 'IncludeNoise', true);
[X_test, y_test] = gen_test.generate();

[N_tr, p] = size(X_train);
K_classes = numel(unique(y_train));
fprintf('  Train: %d samples, Test: %d samples, p=%d, K=%d\n', ...
    N_tr, size(X_test,1), p, K_classes);

%% === 1.5. Train / Test Split (MANDATORY — Anti-Leakage) ===
% [SKIP] WaveformGenerator called separately ⇒ independent random draws.
fprintf('\n[1.5] [SKIP] Train/test split handled by separate generators.\n');

%% === 2. DataLake (DA-1) ===
fprintf('\n[2] DataLake (DA-1) — Raw Persistence...\n');
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    meta = struct('source', DATASET, 'seed', RNG_SEED);
    lake.put('raw/waveform/X_train', X_train, meta);
    lake.put('raw/waveform/y_train', y_train, meta);
    lake.put('raw/waveform/X_test',  X_test,  meta);
    lake.put('raw/waveform/y_test',  y_test,  meta);
    fprintf('  [PASS] DataLake ready.\n');
catch e
    fprintf('  [SKIP] DataLake: %s\n', e.message);
end

%% === 3. DataWarehouse (DA-2/DA-3) ===
fprintf('\n[3] DataWarehouse — Structured Storage...\n');
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
fprintf('\n[4] CachingWarehouse (DA-4) — Read-Through Cache...\n');
try
    redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
    cdw   = OceanData.Adapters.CachingWarehouse(dw, redis); %#ok<NASGU>
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
% [SKIP] p=21 all informative. Table 12.4 uses default configs.
fprintf('\n[6.5] [SKIP] Feature selection / tuning not needed (p=21, default configs).\n');

%% === 7. Model Training & Evaluation ===
fprintf('\n[7] Model Training & Evaluation...\n');

% Model cache: train-once, load-on-rerun
modelStore = OceanData.DataArch.FeatureStore('rootDir', './cache/models/');
cachePrefix = sprintf('%s/seed%d', DATASET, RNG_SEED);

% 1D roughness penalty for PDA/PMDA on waveform's 21 ordered features
Omega = OceanMath.LinearAlgebra.Optimization.generateRoughnessPenalty1D(p);
pdaBuilder = @() OceanStats.Supervised.Regression.PenalizedRegression( ...
    'PenaltyMatrix', Omega, 'Lambda', 0.5);

% ── 7a. LDA ──────────────────────────────────────────────────────────────
fprintf('  Training (1) LDA...\n');
ldaModel = OceanTools.cachedFit(modelStore, sprintf('%s/LDA', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.LDA(), X_train, y_train);

% ── 7b. QDA ──────────────────────────────────────────────────────────────
fprintf('  Training (2) QDA...\n');
qdaModel = OceanTools.cachedFit(modelStore, sprintf('%s/QDA', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.QDA(), X_train, y_train);

% ── 7c. CART ─────────────────────────────────────────────────────────────
fprintf('  Training (3) CART...\n');
cartModel = OceanTools.cachedFit(modelStore, sprintf('%s/CART_d10_gini', cachePrefix), ...
    @() OceanStats.Supervised.Tree.DecisionTree('MaxDepth', 10, 'Criterion', 'gini'), X_train, y_train);

% ── 7d. FDA/MARS (degree = 1) ───────────────────────────────────────────
fprintf('  Training (4) FDA/MARS (degree = 1)...\n');
marsBuilder1 = @() OceanStats.Supervised.MARS.MARS('MaxTerms', 15, ...
    'MaxInteractionDegree', 1, 'Encoder', OceanStats.Data.RawEncoder());
fdaMars1 = OceanTools.cachedFit(modelStore, sprintf('%s/FDA_MARS_deg1_mt15', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', marsBuilder1), X_train, y_train);

% ── 7e. FDA/MARS (degree = 2) ───────────────────────────────────────────
fprintf('  Training (5) FDA/MARS (degree = 2)...\n');
marsBuilder2 = @() OceanStats.Supervised.MARS.MARS('MaxTerms', 15, ...
    'MaxInteractionDegree', 2, 'Encoder', OceanStats.Data.RawEncoder());
fdaMars2 = OceanTools.cachedFit(modelStore, sprintf('%s/FDA_MARS_deg2_mt15', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', marsBuilder2), X_train, y_train);

% ── 7f. MDA (3 subclasses) ──────────────────────────────────────────────
fprintf('  Training (6) MDA (3 subclasses)...\n');
mdaModel = OceanTools.cachedFit(modelStore, sprintf('%s/MDA_3sub', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.MDA('NumSubclasses', 3), X_train, y_train);

% ── 7g. PMDA (3 subclasses, penalized 4 df) ─────────────────────────────
fprintf('  Training (7) PMDA (3 subclasses, penalized 4 df)...\n');
pmdaModel = OceanTools.cachedFit(modelStore, sprintf('%s/PMDA_3sub_pen4df', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.MDA('NumSubclasses', 3, 'BaseModelConstructor', pdaBuilder), X_train, y_train);

% ── 7h. PDA (penalized 4 df) ────────────────────────────────────────────
fprintf('  Training (8) PDA (penalized 4 df)...\n');
pdaModel = OceanTools.cachedFit(modelStore, sprintf('%s/PDA_pen4df', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', pdaBuilder), X_train, y_train);

% ── Evaluation ───────────────────────────────────────────────────────────
models      = {ldaModel, qdaModel, cartModel, fdaMars1, fdaMars2, mdaModel, pmdaModel, pdaModel};
model_names = {'LDA', 'QDA', 'CART', 'FDA/MARS (degree = 1)', 'FDA/MARS (degree = 2)', ...
               'MDA (3 subclasses)', 'MDA (3 sub, penalized 4 df)', 'PDA (penalized 4 df)'};

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║       TABLE 12.4 — Waveform Classification Performance      ║\n');
fprintf('║       Ref: ESL Section 12.7.1                               ║\n');
fprintf('╠════╦════════════════════════════════════════╦═════════════════╣\n');
fprintf('║    ║ Technique                              ║ Training  Test  ║\n');
fprintf('╠════╬════════════════════════════════════════╬═════════════════╣\n');

for i = 1:length(models)
    try
        tr_err = mean(models{i}.predict(X_train) ~= y_train);
        te_err = mean(models{i}.predict(X_test)  ~= y_test);
        fprintf('║ %2d ║ %-38s ║  %.2f    %.2f  ║\n', i, model_names{i}, tr_err, te_err);
    catch e
        fprintf('║ %2d ║ %-38s ║  ERR     ERR   ║\n', i, model_names{i});
        fprintf('  [ERROR] %s\n', e.message);
    end
end

fprintf('╠════╬════════════════════════════════════════╬═════════════════╣\n');
fprintf('║    ║ Bayes (theoretical)                    ║  —       0.14  ║\n');
fprintf('╚════╩════════════════════════════════════════╩═════════════════╝\n');
fprintf('\nNote: ESL Table 12.4 averages over 10 simulations.\n');
fprintf('      This script runs a single simulation for reproducibility.\n');

%% === 8. Visualization ===
fprintf('\n[8] Visualization...\n');

% ── Figure 12.14: Base Waveform Shapes ───────────────────────────────────
fprintf('  Plotting Figure 12.14 (base waveforms)...\n');
try
    viz = OceanViz.Core('Theme', 'Dark');
    base_waveforms  = gen_pure.MetaData.BaseWaveforms;
    samplesPerClass = 5;
    fig1 = viz.createFigure('ESL Fig 12.14 — Waveform Bases');
    set(fig1, 'Position', [100 100 700 750]);
    class_titles = {'Class 1 (h1+h2)', 'Class 2 (h1+h3)', 'Class 3 (h2+h3)'};
    templates    = {[1,2], [1,3], [2,3]};
    sampleColor   = viz.Colors.Primary;
    templateColor = viz.Colors.Secondary;
    for k = 1:3
        ax = subplot(3, 1, k, 'Parent', fig1);
        hold(ax, 'on');
        idx  = find(y_pure == k);
        pick = idx(randi(numel(idx), samplesPerClass, 1));
        for i = 1:samplesPerClass
            plot(ax, 1:p, X_pure(pick(i), :), '--', 'Color', [sampleColor 0.35], 'LineWidth', 1.2);
        end
        for t = templates{k}
            plot(ax, 1:p, base_waveforms(t, :), '-', 'Color', templateColor, 'LineWidth', 2.2);
        end
        hold(ax, 'off');
        title(ax, class_titles{k}, 'Color', viz.Colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
        viz.styleAxes(ax);
        xlim(ax, [1, p]); ylim(ax, [0, 6.5]);
        if k < 3; set(ax, 'XTickLabel', {}); end
        ylabel(ax, 'Amplitude', 'Color', viz.Colors.Text);
    end
    xlabel(ax, 'Feature Index j', 'Color', viz.Colors.Text);
    fprintf('  [PASS] Figure 12.14 displayed.\n');
catch e
    fprintf('  [ERROR] Figure 12.14: %s\n', e.message);
end

% ── Figure 12.15: PMDA Canonical Variate Scatter ─────────────────────────
fprintf('  Plotting Figure 12.15 (PMDA canonical variates)...\n');
try
    X_test_proj = pmdaModel.project(X_test, 4);

    % Extract subclass centers and project them
    centers = []; center_labels = [];
    for k = 1:pmdaModel.NumClasses
        for r = 1:size(pmdaModel.SubclassMeans{k}, 1)
            centers = [centers; pmdaModel.SubclassMeans{k}(r, :)]; %#ok<AGROW>
            center_labels = [center_labels; k]; %#ok<AGROW>
        end
    end
    centers_proj = pmdaModel.project(centers, 4);

    cols = viz.getColorSequence(3);           % 3 class colors from theme
    light_cols = cols * 0.5 + 0.5;            % lightened for scatter text

    pad = @(v) [min(v)-0.1*range(v), max(v)+0.1*range(v)];

    fig2 = viz.createFigure('ESL Fig 12.15 — PMDA Canonical Variates');
    set(fig2, 'Position', [100 100 900 450]);

    % Left: Canonical Var 1 vs 2
    ax1 = subplot(1, 2, 1, 'Parent', fig2);
    hold(ax1, 'on');
    for k = 1:3
        for j = find(y_test == k)'
            text(ax1, X_test_proj(j,1), X_test_proj(j,2), num2str(k), ...
                'Color', light_cols(k,:), 'FontSize', 7, 'HorizontalAlignment', 'center', 'Clipping', 'on');
        end
    end
    for i = 1:size(centers_proj, 1)
        k = center_labels(i);
        scatter(ax1, centers_proj(i,1), centers_proj(i,2), 250, ...
            'MarkerFaceColor', viz.Colors.Background, 'MarkerEdgeColor', cols(k,:), 'LineWidth', 1.5);
        text(ax1, centers_proj(i,1), centers_proj(i,2), num2str(k), ...
            'Color', cols(k,:), 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
    hold(ax1, 'off');
    title(ax1, '3 Subclasses, Penalized 4 df', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    xlabel(ax1, 'Discriminant Var 1'); ylabel(ax1, 'Discriminant Var 2');
    viz.styleAxes(ax1);
    set(ax1, 'XLim', pad(X_test_proj(:,1)), 'YLim', pad(X_test_proj(:,2)));

    % Right: Canonical Var 3 vs 4
    ax2 = subplot(1, 2, 2, 'Parent', fig2);
    hold(ax2, 'on');
    for k = 1:3
        for j = find(y_test == k)'
            text(ax2, X_test_proj(j,3), X_test_proj(j,4), num2str(k), ...
                'Color', light_cols(k,:), 'FontSize', 7, 'HorizontalAlignment', 'center', 'Clipping', 'on');
        end
    end
    for i = 1:size(centers_proj, 1)
        k = center_labels(i);
        scatter(ax2, centers_proj(i,3), centers_proj(i,4), 250, ...
            'MarkerFaceColor', viz.Colors.Background, 'MarkerEdgeColor', cols(k,:), 'LineWidth', 1.5);
        text(ax2, centers_proj(i,3), centers_proj(i,4), num2str(k), ...
            'Color', cols(k,:), 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
    hold(ax2, 'off');
    title(ax2, '3 Subclasses, Penalized 4 df', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    xlabel(ax2, 'Discriminant Var 3'); ylabel(ax2, 'Discriminant Var 4');
    viz.styleAxes(ax2);
    set(ax2, 'XLim', pad(X_test_proj(:,3)), 'YLim', pad(X_test_proj(:,4)));

    fprintf('  [PASS] Figure 12.15 displayed.\n');
catch e
    fprintf('  [ERROR] Figure 12.15: %s\n', e.message);
end

fprintf('\n=== Experiment Complete ===\n');

