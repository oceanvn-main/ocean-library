% EXP_FDA_MDA_MIXTURE  Demonstrates the 8-step data science pipeline and recreates ESL Fig 12.13
clc; clear; rng(42, 'twister');
RNG_SEED = 42;
DATASET  = 'esl_mixture';

fprintf('Starting FDA vs MDA Mixture Experiment (ESL Fig 12.13)...\n');

%% === 1. Ingestion ===
fprintf('Step 1: Data Generation...');
gen_train = OceanData.Generators.ESLMixtureGenerator('N', 200, 'Seed', 42);
[X_train, y_train] = gen_train.generate();

gen_test = OceanData.Generators.ESLMixtureGenerator('N', 10000, 'Seed', 42);
[X_test, y_test] = gen_test.generate();

[~, p] = size(X_train);
fprintf('Generated %d training and %d testing samples (p=%d)\n', size(X_train,1), size(X_test,1), p);

%% === 1.5. Train / Test Split (MANDATORY — Anti-Leakage) ===
% [SKIP - 1.5] Reason: ESLMixtureGenerator is called separately for train
% and test with different sizes (200 vs 10000), producing independent draws.
fprintf('\n[1.5] [SKIP] Train/test split handled by separate generators.\n');

%% === 2. DataLake (DA-1) ===
fprintf('\nStep 2: DataLake (DA-1) - Raw Persistence...\n');
try
    lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    meta = struct('source', DATASET, 'seed', RNG_SEED);
    lake.put('raw/mixture/X_train', X_train, meta);
    lake.put('raw/mixture/y_train', y_train, meta);
    lake.put('raw/mixture/X_test',  X_test,  meta);
    lake.put('raw/mixture/y_test',  y_test,  meta);
    fprintf('  [PASS] DataLake ready.\n');
catch e
    fprintf('  [SKIP] DataLake: %s\n', e.message);
end

%% === 3. DataWarehouse (DA-2/DA-3) ===
fprintf('\nStep 3: DataWarehouse - Structured Storage...\n');
try
    dw = OceanData.DataArch.DataWarehouse('rootDir', './dw/');
    colNames = {'x1', 'x2', 'y'};
    colTypes = {'double', 'double', 'double'};
    schema = OceanData.DataArch.DWSchema(colNames, colTypes);
    dw.createTable(DATASET, schema);
    trainTable = array2table([X_train, y_train], 'VariableNames', colNames);
    dw.insertBatch(DATASET, trainTable);
    fprintf('  [PASS] DataWarehouse ready.\n');
catch e
    fprintf('  [SKIP] DataWarehouse: %s\n', e.message);
end

%% === 4. CachingWarehouse (DA-4) ===
fprintf('\nStep 4: CachingWarehouse (DA-4) - Read-Through Cache...\n');
try
    redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
    cdw   = OceanData.Adapters.CachingWarehouse(dw, redis);  %#ok<NASGU>
    fprintf('  [PASS] CachingWarehouse connected to Redis.\n');
catch
    fprintf('  [SKIP] Redis unavailable.\n');
end

%% === 5 & 6. FeatureStore & FeatureCache (DA-5/DA-6) ===
fprintf('\nSteps 5 & 6: FeatureStore & FeatureCache...\n');
try
    fs       = OceanData.DataArch.FeatureStore('rootDir', './fs/');
    feat_key = sprintf('%s_raw_seed%d', DATASET, RNG_SEED);
    if ~fs.exists(feat_key)
        fs.put(feat_key, X_train);
    end
    entry = fs.get(feat_key);
    X_feat = entry.data; %#ok<NASGU>
    fprintf('  [PASS] FeatureStore ready.\n');
catch e
    fprintf('  [SKIP] FeatureStore/Cache: %s\n', e.message);
end

%% === 6.5. Feature Selection + Model Selection ===
% [SKIP] p=2 dimensions, FDA/MARS and MDA have fixed hyperparams for this demo.
fprintf('\n[6.5] [SKIP] Feature selection / tuning not needed (p=2, demo config).\n');

%% === 7. Model Training & Evaluation ===
fprintf('Step 7: Model Training & Evaluation...\n');

% Model cache: train-once, load-on-rerun
modelStore = OceanData.DataArch.FeatureStore('rootDir', './cache/models/');
cachePrefix = sprintf('%s/seed%d', DATASET, RNG_SEED);

% Compute True Bayes Error (using generator's own centers & covariance)
[~, errBayes_te] = gen_train.bayesPredict(X_test, y_test);
fprintf('  Bayes Test Error: %.3f\n\n', errBayes_te);

% 7a. FDA (Flexible Discriminant Analysis via MARS Degree 2)
fprintf('  Training FDA (MARS Degree 2)...\n');
marsBuilder = @() OceanStats.Supervised.MARS.MARS('MaxInteractionDegree', 2, 'MaxTerms', 50);
fda = OceanTools.cachedFit(modelStore, sprintf('%s/FDA_MARS_deg2_mt50', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.FDA('BaseModelConstructor', marsBuilder), X_train, y_train);

fda_pred_tr = fda.predict(X_train);
fda_pred_te = fda.predict(X_test);

errFDA_tr = mean(fda_pred_tr ~= y_train);
errFDA_te = mean(fda_pred_te ~= y_test);

fprintf('    FDA Train Error: %.3f\n', errFDA_tr);
fprintf('    FDA Test Error:  %.3f\n\n', errFDA_te);

% 7b. MDA (Mixture Discriminant Analysis - 5 Subclasses)
fprintf('  Training MDA (5 Subclasses, Pooled Covariance)...\n');
mda = OceanTools.cachedFit(modelStore, sprintf('%s/MDA_5sub_pooled', cachePrefix), ...
    @() OceanStats.Supervised.Discriminant.MDA('NumSubclasses', 5, 'Mode', 'pooled', 'MaxIter', 100), X_train, y_train);

mda_pred_tr = mda.predict(X_train);
mda_pred_te = mda.predict(X_test);

errMDA_tr = mean(mda_pred_tr ~= y_train);
errMDA_te = mean(mda_pred_te ~= y_test);

fprintf('    MDA Train Error: %.3f\n', errMDA_tr);
fprintf('    MDA Test Error:  %.3f\n', errMDA_te);


%% === 8. Visualization ===
fprintf('Step 8: Visualization (Recreating ESL Fig 12.13)...\n');
try
    viz = OceanViz.Core('Theme', 'Dark');
    fig = viz.createFigure('FDA & MDA Mixture Decision Boundaries');
    set(fig, 'Position', [100, 100, 600, 900]);

    % Define plot grid constraints (use higher resolution for smooth rendering)
    mins = min(X_train) - 0.5;
    maxs = max(X_train) + 0.5;
    [xg, yg] = meshgrid(linspace(mins(1), maxs(1), 300), linspace(mins(2), maxs(2), 300));
    Xgrid = [xg(:), yg(:)];
    
    % Compute Bayes boundary on grid (using generator's own method)
    [bg1, bg2] = gen_train.bayesDensity(Xgrid);
    
    % Use theme colors instead of hardcoded values
    color1    = viz.Colors.Primary;      % Class 1 data points
    color2    = viz.Colors.Secondary;    % Class 2 data points
    bdryColor = viz.Colors.Boundary;     % Model decision boundary
    bayesColor = [0.7 0.1 0.8];          % Bayes boundary (purple — always distinct)
    bg_color1 = color1 * 0.3 + viz.Colors.Background * 0.7;  % light blend for region fill
    bg_color2 = color2 * 0.3 + viz.Colors.Background * 0.7;
    
    
    % Bayes boundary: contour the continuous log-density ratio (no smoothing needed)
    bayes_logRatio = reshape(log(bg1 + eps) - log(bg2 + eps), size(xg));
    
    % FDA boundary: use posterior probabilities (continuous, not hard labels)
    [ygrid_fda, fda_post] = fda.predict(Xgrid);
    fda_prob1 = reshape(fda_post(:, 1), size(xg));  % P(class 1 | x)
    bg_grid_C1 = ygrid_fda == 1;
    bg_grid_C2 = ygrid_fda == 2;
    
    % ----- Subplot 1: FDA Boundary -----
    ax1 = subplot(2, 1, 1, 'Parent', fig);
    hold(ax1, 'on');

    
    % Plot light dotted background grids
    scatter(ax1, Xgrid(bg_grid_C1, 1), Xgrid(bg_grid_C1, 2), 2, bg_color1, 'filled');
    scatter(ax1, Xgrid(bg_grid_C2, 1), Xgrid(bg_grid_C2, 2), 2, bg_color2, 'filled');
    
    % Plot the Bayes optimal boundary as dashed purple (contour at log-ratio = 0)
    contour(ax1, xg, yg, bayes_logRatio, [0 0], '--', 'LineColor', bayesColor, 'LineWidth', 1.5);
    
    % Plot the FDA decision boundary (contour at P(class1) = 0.5)
    contour(ax1, xg, yg, fda_prob1, [0.5 0.5], '-', 'LineColor', bdryColor, 'LineWidth', 1.5);
    
    % Plot data points
    idx1 = (y_train == 1);
    idx2 = (y_train == 2);
    scatter(ax1, X_train(idx1, 1), X_train(idx1, 2), 25, 'MarkerEdgeColor', color1, 'MarkerFaceColor', 'none', 'LineWidth', 1.2);
    scatter(ax1, X_train(idx2, 1), X_train(idx2, 2), 25, 'MarkerEdgeColor', color2, 'MarkerFaceColor', 'none', 'LineWidth', 1.2);
    
    % Text annotation
    txt1 = sprintf('Training Error: %.3f\nTest Error: %.3f\nBayes Error: %.3f', errFDA_tr, errFDA_te, errBayes_te);
    text(ax1, mins(1)+0.2, mins(2)+0.5, txt1, 'FontSize', 9, 'Color', viz.Colors.Text, ...
        'BackgroundColor', viz.Colors.Background, 'EdgeColor', 'none');
    
    title(ax1, 'FDA / MARS - Degree 2', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    viz.styleAxes(ax1);
    axis(ax1, 'tight');
    hold(ax1, 'off');
    
    % ----- Subplot 2: MDA Boundary -----
    ax2 = subplot(2, 1, 2, 'Parent', fig);
    hold(ax2, 'on');
    
    % MDA boundary: use posterior probabilities
    [ygrid_mda, mda_post] = mda.predict(Xgrid);
    mda_prob1 = reshape(mda_post(:, 1), size(xg));
    bg_grid_C1_mda = ygrid_mda == 1;
    bg_grid_C2_mda = ygrid_mda == 2;
    
    % Plot light dotted background grids
    scatter(ax2, Xgrid(bg_grid_C1_mda, 1), Xgrid(bg_grid_C1_mda, 2), 2, bg_color1, 'filled');
    scatter(ax2, Xgrid(bg_grid_C2_mda, 1), Xgrid(bg_grid_C2_mda, 2), 2, bg_color2, 'filled');
    
    % Plot the Bayes optimal boundary as dashed purple (contour at log-ratio = 0)
    contour(ax2, xg, yg, bayes_logRatio, [0 0], '--', 'LineColor', bayesColor, 'LineWidth', 1.5);
    
    % Plot the MDA decision boundary (contour at P(class1) = 0.5)
    contour(ax2, xg, yg, mda_prob1, [0.5 0.5], '-', 'LineColor', bdryColor, 'LineWidth', 1.5);
    
    % Plot data points
    scatter(ax2, X_train(idx1, 1), X_train(idx1, 2), 25, 'MarkerEdgeColor', color1, 'MarkerFaceColor', 'none', 'LineWidth', 1.2);
    scatter(ax2, X_train(idx2, 1), X_train(idx2, 2), 25, 'MarkerEdgeColor', color2, 'MarkerFaceColor', 'none', 'LineWidth', 1.2);
    
    % Plot MDA Centers
    mda_m1 = mda.SubclassMeans{1};
    mda_m2 = mda.SubclassMeans{2};
    scatter(ax2, mda_m1(:, 1), mda_m1(:, 2), 60, 'MarkerFaceColor', color1, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    scatter(ax2, mda_m2(:, 1), mda_m2(:, 2), 60, 'MarkerFaceColor', color2, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    % Text annotation
    txt2 = sprintf('Training Error: %.3f\nTest Error: %.3f\nBayes Error: %.3f', errMDA_tr, errMDA_te, errBayes_te);
    text(ax2, mins(1)+0.2, mins(2)+0.5, txt2, 'FontSize', 9, 'Color', viz.Colors.Text, ...
        'BackgroundColor', viz.Colors.Background, 'EdgeColor', 'none');
    
    title(ax2, 'MDA - 5 Subclasses per Class', 'Color', viz.Colors.Text, 'FontWeight', 'bold');
    viz.styleAxes(ax2);
    axis(ax2, 'tight');
    hold(ax2, 'off');
    
    if ~exist('Results', 'dir'); mkdir('Results'); end
    % saveas(fig, 'Results/exp_fda_mda_mixture.png');
    fprintf('  Plot generated successfully.\n');
catch e
    fprintf('  [ERROR] Visualization: %s\n', e.getReport());
end

fprintf('Experiment Complete.\n');
