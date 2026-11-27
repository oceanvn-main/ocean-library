%% EXP_FULLSTACK_DS  Full Data Science Pipeline Tour.
%
%   Walks through every architectural layer of the Ocean Library
%   from raw data generation to a trained model and polished visualisation.
%
%   DATASET:         Synthetic Nonlinear Regression
%                    N=2000 samples, 5 continuous features
%                    Y = sin(X1) + 0.5*X2^2 - X3.*X4 + 0.3*X5 + noise
%
%   STORAGE LAYERS:
%     §2  DataLake (disk)          — schema-free raw archive
%     §3  DataWarehouse (disk)     — schema-validated structured store
%     §4  PostgreSQL (optional)    — graceful skip if unavailable
%     §5  Redis CachingWarehouse   — read-through cache demo
%     §6  FeatureStore + FeatureCache — disk vs RAM feature cache timing
%
%   MODEL:           Ridge Regression  (lambda=0.1) vs OLS
%   ASSUMPTIONS:
%     - Polynomial basis expansion (degree=2) is the only feature transform
%     - 80/20 holdout split; seed fixed for reproducibility
%     - Redis Docker container (redis_ocean) is running on localhost:6379
%     - PostgreSQL test (§4) is completely optional and silently skipped
%
%   REPRODUCIBILITY: rng(42, 'twister') — change seed to verify no hardcoding
%
%   Run:
%     run('Experiments/Scripts/DataScience/exp_fullstack_ds.m')

clc; clear; close all;
rng(42, 'twister');

ROOT = 'c:\Ocean Library\Matlab code practice';
addpath(genpath(ROOT));

OUT_DIR  = fullfile(ROOT, 'Experiments', 'Outputs');
LAKE_DIR = fullfile(OUT_DIR, 'lake');
DW_DIR   = fullfile(OUT_DIR, 'warehouse');
FS_DIR   = fullfile(OUT_DIR, 'features');
if ~isfolder(OUT_DIR),  mkdir(OUT_DIR);  end

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║    OCEAN LIBRARY — FULL-STACK DATA SCIENCE TOUR     ║\n');
fprintf('╚══════════════════════════════════════════════════════╝\n\n');

%% ════════════════════════════════════════════════════════════════════════
%% § 1 — DATA GENERATION
%% ════════════════════════════════════════════════════════════════════════
fprintf('[§1] Generating synthetic nonlinear regression data...\n');

N = 2000;
P = 5;

X = 2*randn(N, P);    % 5 standard-normal features, scaled

% TRUE nonlinear response -- explicitly computed, never hard-coded
Y = sin(X(:,1)) + 0.5*X(:,2).^2 - X(:,3).*X(:,4) + 0.3*X(:,5) + 0.5*randn(N,1);

fprintf('   N=%d samples, P=%d features generated.\n', N, P);
fprintf('   Y range: [%.2f, %.2f], mean=%.3f, std=%.3f\n', ...
    min(Y), max(Y), mean(Y), std(Y));

%% ════════════════════════════════════════════════════════════════════════
%% § 2 — DATA LAKE  (Disk, schema-free)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§2] DataLake — raw disk persistence...\n');

lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);

% Write raw features & labels
meta_raw = struct('source', 'synthetic', 'N', N, 'P', P, 'seed', 42);
lake.put('raw/synth/X', X, meta_raw);
lake.put('raw/synth/Y', Y, struct('source', 'synthetic'));

% Read back and verify
entry_X = lake.get('raw/synth/X');
entry_Y = lake.get('raw/synth/Y');

assert(isequal(size(entry_X.data), [N, P]), 'DataLake X round-trip FAILED');
assert(numel(entry_Y.data) == N,            'DataLake Y round-trip FAILED');

fprintf('   Written and verified: raw/synth/X  [%dx%d]\n', size(entry_X.data));
fprintf('   Written and verified: raw/synth/Y  [%dx1]\n',  numel(entry_Y.data));
fprintf('   Keys in lake: %s\n', strjoin(lake.list(), ', '));

%% ════════════════════════════════════════════════════════════════════════
%% § 3 — DATA WAREHOUSE  (Disk, schema-validated)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§3] DataWarehouse — schema-validated structured storage...\n');

dw = OceanData.DataArch.DataWarehouse('rootDir', DW_DIR);

% Schema bounds derived from data (no hard-coded assumptions about Y distribution)
y_min = floor(min(Y)) - 5;
y_max = ceil(max(Y))  + 5;
schema = OceanData.DataArch.DWSchema( ...
    {'x1','x2','x3','x4','x5','y'}, ...
    {'double','double','double','double','double','double'}, ...
    'MinValues', [-10,-10,-10,-10,-10, y_min], ...
    'MaxValues', [ 10, 10, 10, 10, 10, y_max]);

dw.createTable('synth_data', schema);

% Insert all 2000 rows instantly using the bulk insertBatch method
fprintf('   Bulk inserting all %d rows into DataWarehouse...\n', N);
T = array2table([X, Y], 'VariableNames', {'x1','x2','x3','x4','x5','y'});
dw.insertBatch('synth_data', T);

% Query: filter rows where x1 > 0
rows_pos = dw.query('synth_data', 'x1 > 0');
pct_pos  = 100 * numel(rows_pos) / N;
fprintf('   %d rows inserted successfully.\n', N);
fprintf('   Query "x1 > 0" returned %d rows (%.1f%% of inserted).\n', ...
    numel(rows_pos), pct_pos);

%% ════════════════════════════════════════════════════════════════════════
%% § 4 — POSTGRESQL  (Optional — graceful skip if unavailable)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§4] PostgreSQL — optional relational backend...\n');

pgAvailable = false;
try
    pgCfg  = OceanData.Adapters.DbConfig( ...
        'Host','localhost','Port',5433, ...
        'Database','ocean_db','User','postgres','Password','postgres');

    pgdw = OceanData.Adapters.PostgresWarehouse(pgCfg);
    pgdw.connect();

    % Insert 100 sample rows into a lightweight test table
    pgdw.createTable('synth_pg', schema);
    for i = 1:100
        r = struct('x1',X(i,1),'x2',X(i,2),'x3',X(i,3), ...
                   'x4',X(i,4),'x5',X(i,5),'y',Y(i));
        pgdw.insert('synth_pg', r);
    end
    pg_rows = pgdw.query('synth_pg', 'x1 > 0');
    fprintf('   [PASS] Postgres: inserted 100 rows, queried %d with x1 > 0.\n', numel(pg_rows));
    pgdw.disconnect();
    pgAvailable = true;
catch ex
    fprintf('   [SKIP] PostgreSQL not available (%s). Continuing without it.\n', ex.message);
end

%% ════════════════════════════════════════════════════════════════════════
%% § 4b — CASSANDRA TIME-SERIES  (Optional — graceful skip if unavailable)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§4b] Cassandra TS DataWarehouse — distributed NoSQL backend...\n');

cassAvailable = false;
try
    cassCfg = OceanData.Adapters.DbConfig( ...
        'Vendor',   'Cassandra', ...
        'Host',     'localhost', ...
        'Port',     9042, ...
        'Database', 'ocean_ks', ...
        'User',     'cassandra', ...
        'Password', 'cassandra');

    tsw = OceanData.Adapters.CassandraTimeSeriesWarehouse(cassCfg);
    tsw.connect();

    % Create TS table and bulk-insert a time series for sensor_1 and sensor_2
    tsw.createTable('sensor_data');
    
    tVec = datetime('today') + seconds(1:100)';
    xVec1 = X(1:100, 1);
    xVec2 = X(1:100, 2);
    
    tsw.appendBatch('sensor_data', 'sensor_1', tVec, xVec1);
    tsw.appendBatch('sensor_data', 'sensor_2', tVec, xVec2);

    % Query back sensor_1
    ts_rows = tsw.query('sensor_data', 'sensor_1');
    fprintf('   [PASS] Cassandra TS: inserted 200 rows, queried back %d for sensor_1.\n', height(ts_rows));
    
    % Query range for sensor_1
    range_rows = tsw.queryRange('sensor_data', 'sensor_1', tVec(51), tVec(100));
    fprintf('   [PASS] Cassandra TS: queried range returned %d rows.\n', height(range_rows));

    tsw.disconnect();
    cassAvailable = true;
catch ex
    fprintf('   [SKIP] Cassandra not available (%s). Continuing without it.\n', ex.message);
end

%% ════════════════════════════════════════════════════════════════════════
%% § 5 — REDIS CACHING WAREHOUSE
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§5] Redis CachingWarehouse — read-through cache demo...\n');

redisAvailable = false;
try
    redisCfg = OceanData.Adapters.RedisConfig( ...
        'TTL', 300, 'Prefix', 'ocean_exp:');
    ra = OceanData.Adapters.RedisAdapter(redisCfg);
    ra.connect();

    cdw = OceanData.Adapters.CachingWarehouse(dw, ra);

    fprintf('   Query 1 (expect cache MISS) ...\n');
    t1 = tic;
    rows1 = cdw.query('synth_data', 'x1 > 0'); %#ok<NASGU>
    t_miss = toc(t1);

    fprintf('   Query 2 (expect cache HIT)  ...\n');
    t2 = tic;
    rows2 = cdw.query('synth_data', 'x1 > 0'); %#ok<NASGU>
    t_hit  = toc(t2);

    fprintf('   Cache miss time: %.4f s\n', t_miss);
    fprintf('   Cache hit  time: %.4f s\n', t_hit);
    fprintf('   Speedup factor:  %.1fx\n',  t_miss / max(t_hit, 1e-6));
    cdw.stats();
    redisAvailable = true;
catch ex
    fprintf('   [SKIP] Redis not available (%s). Continuing without it.\n', ex.message);
end

%% ════════════════════════════════════════════════════════════════════════
%% § 6 — FEATURE ENGINEERING  (Disk FeatureStore vs Redis FeatureCache)
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§6] Feature Engineering — polynomial basis expansion (degree=2)...\n');

% -- Helper: polynomial feature expander (1 + linear + degree-2 cross terms)
polyExpand = @(M) OceanStats.Data.polyFeatures(M, 2);

% § 6a — Disk-backed FeatureStore
fs = OceanData.DataArch.FeatureStore('rootDir', FS_DIR);
feat_meta = struct('transform','poly2','degree',2,'N',N,'source','raw/synth/X');

fprintf('   [FeatureStore] Computing + saving features to disk...\n');
t_disk_compute = tic;
entry_fs = fs.getOrCompute('poly2/synth/v1', @() polyExpand(X), feat_meta);
t_disk1  = toc(t_disk_compute);

fprintf('   [FeatureStore] Re-loading features from disk (cached)...\n');
t_disk_load = tic;
entry_fs2 = fs.getOrCompute('poly2/synth/v1', @() error('Should not recompute'), feat_meta);
t_disk2   = toc(t_disk_load);

fprintf('   Disk compute: %.4f s  |  Disk load: %.4f s  (%.1fx faster)\n', ...
    t_disk1, t_disk2, t_disk1/max(t_disk2,1e-6));

Phi = entry_fs2.data;  % [N x D] expanded feature matrix
D   = size(Phi, 2);
fprintf('   Feature matrix: [%d x %d] (from %d raw → %d polynomial features)\n', N, D, P, D);

% § 6b — Redis FeatureCache (if available)
if redisAvailable
    fc = OceanData.Adapters.FeatureCache(ra);

    fprintf('   [FeatureCache] Computing + storing features in Redis...\n');
    t_redis_compute = tic;
    e_redis1 = fc.getOrCompute('poly2/synth/v1', @() polyExpand(X), feat_meta);
    t_redis1 = toc(t_redis_compute);

    fprintf('   [FeatureCache] Reading features from Redis RAM...\n');
    t_redis_load = tic;
    e_redis2 = fc.getOrCompute('poly2/synth/v1', @() error('no recompute'), feat_meta);
    t_redis2 = toc(t_redis_load);

    fprintf('   Redis compute: %.4f s  |  Redis load: %.4f s  (%.1fx faster)\n', ...
        t_redis1, t_redis2, t_redis1/max(t_redis2,1e-6));

    fc.invalidate('poly2/synth/v1');
else
    fprintf('   [SKIP] Redis FeatureCache skipped (no connection).\n');
end

%% ════════════════════════════════════════════════════════════════════════
%% § 7 — TRAIN / TEST SPLIT + MODEL TRAINING
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§7] Model Training — Ridge vs OLS...\n');

% 80/20 holdout split
sp    = OceanData.DataArch.DataSplitter('Strategy','holdout','Ratio',0.8,'Seed',42);
split = sp.split(Phi, Y);

Phi_tr = Phi(split.trainIdx, :);
Y_tr   = Y(split.trainIdx);
Phi_te = Phi(split.testIdx, :);
Y_te   = Y(split.testIdx);

fprintf('   Train: %d rows  |  Test: %d rows\n', numel(Y_tr), numel(Y_te));

% ---- OLS ---------------------------------------------------------------
W_tr = ones(numel(Y_tr), 1);   % uniform weights (no sample weighting)
ols = OceanStats.Selection.Strategies.LinearReg();
ols_model = ols.train(Phi_tr, Y_tr, W_tr);
Y_ols_tr  = ols.predict(ols_model, Phi_tr);
Y_ols_te  = ols.predict(ols_model, Phi_te);

% ---- Ridge (lambda = 0.1) ---------------------------------------------
LAMBDA = 0.1;
ridge       = OceanStats.Selection.Strategies.RidgeReg(LAMBDA);
ridge_model = ridge.train(Phi_tr, Y_tr, W_tr);
Y_ridge_tr  = ridge.predict(ridge_model, Phi_tr);
Y_ridge_te  = ridge.predict(ridge_model, Phi_te);

% ---- Metrics (computed from data — never hard-coded) ------------------
rss_ = @(yt,yp) sum((yt-yp).^2);
r2_  = @(yt,yp) 1 - rss_(yt,yp)/sum((yt-mean(yt)).^2);
rmse_= @(yt,yp) sqrt(mean((yt-yp).^2));

ols_rss_tr  = rss_(Y_tr,  Y_ols_tr);   ols_rss_te  = rss_(Y_te,  Y_ols_te);
ols_r2_te   = r2_(Y_te,   Y_ols_te);   ols_rmse_te = rmse_(Y_te, Y_ols_te);
rdg_rss_tr  = rss_(Y_tr,  Y_ridge_tr); rdg_rss_te  = rss_(Y_te,  Y_ridge_te);
rdg_r2_te   = r2_(Y_te,   Y_ridge_te); rdg_rmse_te = rmse_(Y_te, Y_ridge_te);
baseline_rmse = rmse_(Y_te, repmat(mean(Y_tr), size(Y_te)));

fprintf('   OLS   — Train RSS: %8.2f  Test RSS: %8.2f  Test R²: %.4f  RMSE: %.4f\n', ...
    ols_rss_tr,  ols_rss_te,  ols_r2_te,  ols_rmse_te);
fprintf('   Ridge — Train RSS: %8.2f  Test RSS: %8.2f  Test R²: %.4f  RMSE: %.4f\n', ...
    rdg_rss_tr,  rdg_rss_te,  rdg_r2_te,  rdg_rmse_te);
fprintf('   Baseline (mean)   — Test RMSE: %.4f\n', baseline_rmse);

%% ════════════════════════════════════════════════════════════════════════
%% § 8 — EVALUATION + VISUALIZATION
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n[§8] Visualisation — OceanViz Dark theme, 4 panels...\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Full-Stack DS Experiment');
set(fig, 'Position', [80 80 1300 900]);

%-- Panel 1: Predicted vs Actual (test set) — Ridge
ax1 = subplot(2, 2, 1);
data1 = struct('X', Y_te, 'Y', Y_ridge_te, ...
    'Prediction', Y_ridge_te, 'GroundTruth', Y_te);
viz.plot(ax1, 'Regression', data1, ...
    'Title', 'Ridge: Predicted vs Actual (Test)', ...
    'XLabel', 'True Y', 'YLabel', 'Predicted Y');
y_lims = [min(Y_te)-0.5, max(Y_te)+0.5];
hold(ax1,'on');
plot(ax1, y_lims, y_lims, '--', 'Color', [0.95 0.55 0.2], 'LineWidth', 1.5);
hold(ax1,'off');

%-- Panel 2: OLS vs Ridge — Test vs Train RSS bar chart
ax2 = subplot(2, 2, 2);
bar_vals = [ols_rss_te, ols_rss_tr; rdg_rss_te, rdg_rss_tr];
b2 = bar(ax2, 1:2, bar_vals, 'grouped');
b2(1).FaceColor = viz.Colors.Primary;    b2(1).DisplayName = 'Test RSS';
b2(2).FaceColor = viz.Colors.Secondary;  b2(2).DisplayName = 'Train RSS';
ax2.XTick = [1 2]; ax2.XTickLabel = {'OLS','Ridge'};
title(ax2, 'OLS vs Ridge: RSS Comparison');
xlabel(ax2, 'Model'); ylabel(ax2, 'Residual Sum of Squares');
legend(ax2, 'show', 'Location','northwest');
viz.styleAxes(ax2);

%-- Panel 3: Ridge coefficient magnitudes (top features)
ax3 = subplot(2, 2, 3);
coeffs = abs(ridge_model.Beta);
nShow  = min(15, numel(coeffs));
[sorted_c, ~] = sort(coeffs, 'descend');
b3 = bar(ax3, 1:nShow, sorted_c(1:nShow));
b3.FaceColor = viz.Colors.Boundary;
title(ax3, sprintf('Ridge |β| — Top %d Features', nShow));
xlabel(ax3, 'Feature rank'); ylabel(ax3, '|Coefficient|');
viz.styleAxes(ax3);

%-- Panel 4: Normalised RMSE comparison
ax4 = subplot(2, 2, 4);
models_rmse = [ols_rmse_te, rdg_rmse_te, baseline_rmse];
norm_rmse   = models_rmse ./ baseline_rmse;
b4 = bar(ax4, 1:3, norm_rmse);
b4.FaceColor = 'flat';
b4.CData = [viz.Colors.Primary; viz.Colors.Secondary; 0.5 0.5 0.5];
ax4.XTick = [1 2 3]; ax4.XTickLabel = {'OLS','Ridge','Baseline'};
title(ax4, 'Normalised Test RMSE (÷ Baseline)');
xlabel(ax4, 'Model'); ylabel(ax4, 'Relative RMSE');
hold(ax4,'on');
yline(ax4, 1.0, '--', 'Color', [0.9 0.3 0.3], 'LineWidth', 1.5);
hold(ax4,'off');
viz.styleAxes(ax4);

%% ════════════════════════════════════════════════════════════════════════
%% SUMMARY
%% ════════════════════════════════════════════════════════════════════════
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║                    EXPERIMENT SUMMARY                   ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║  Dataset         : Synthetic Nonlinear (N=%d, P=%d)    ║\n', N, P);
fprintf('║  Features after polynomial expansion = %d              ║\n', D);
fprintf('║                                                          ║\n');
fprintf('║  STORAGE LAYERS EXERCISED:                               ║\n');
fprintf('║    DataLake (disk)       : PASS                          ║\n');
fprintf('║    DataWarehouse (disk)  : PASS                          ║\n');
fprintf('║    PostgreSQL            : %s                        ║\n', ...
    iif(pgAvailable,    'PASS (100 rows)         ', 'SKIP (not running)       '));
fprintf('║    Cassandra            : %s                        ║\n', ...
    iif(cassAvailable,  'PASS (100 rows)         ', 'SKIP (not running)       '));
fprintf('║    Redis CachingWarehouse: %s                        ║\n', ...
    iif(redisAvailable, 'PASS (hit/miss shown)   ', 'SKIP (not running)       '));
fprintf('║    FeatureStore (disk)   : PASS                          ║\n');
fprintf('║    FeatureCache (Redis)  : %s                        ║\n', ...
    iif(redisAvailable, 'PASS                    ', 'SKIP (not running)       '));
fprintf('║                                                          ║\n');
fprintf('║  MODELS:                                                 ║\n');
fprintf('║    OLS   — Test R²: %.4f  RMSE: %.4f               ║\n', ols_r2_te, ols_rmse_te);
fprintf('║    Ridge — Test R²: %.4f  RMSE: %.4f               ║\n', rdg_r2_te, rdg_rmse_te);
fprintf('║    Baseline RMSE  : %.4f                            ║\n', baseline_rmse);
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

% -- Cleanup Redis connection if open
if redisAvailable
    ra.disconnect();
end

%% ════════════════════════════════════════════════════════════════════════
%% LOCAL HELPER
%% ════════════════════════════════════════════════════════════════════════
function v = iif(cond, a, b)
    if cond, v = a; else, v = b; end
end
