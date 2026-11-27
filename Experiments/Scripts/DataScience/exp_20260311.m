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

%% Data Generation
N = 2000;
P = 5;

X = 2*randn(N, P);    % 5 standard-normal features, scaled

% TRUE nonlinear response -- explicitly computed, never hard-coded
Y = sin(X(:,1)) + 0.5*X(:,2).^2 - X(:,3).*X(:,4) + 0.3*X(:,5) + 0.5*randn(N,1);

fprintf('   N=%d samples, P=%d features generated.\n', N, P);
fprintf('   Y range: [%.2f, %.2f], mean=%.3f, std=%.3f\n', ...
    min(Y), max(Y), mean(Y), std(Y));
%% Data Lake (Disk, schema-free)

lake = OceanData.DataArch.DataLake('rootDir', LAKE_DIR);

%Write raw features & labels
meta_raw = struct('source', 'synthetic', 'N', N, 'P', P, 'seed', 24);
lake.put('raw/synth/X', X, meta_raw);
lake.put('raw/synth/Y', Y, struct('source', 'synthetic'));
%Read back and verify
entry_X = lake.get('raw/synth/X');
entry_Y = lake.get('raw/synth/Y');

%% Data warehouse (Disk, schema-validated)
dw = OceanData.DataArch.DataWarehouse("rootDir", DW_DIR);

%Schema boudns derived from data
y_min = floor(min(Y)) - 5;
y_max = ceil(max(Y))  + 5;
schema = OceanData.DataArch.DWSchema( ...
    {'x1','x2','x3','x4','x5','y'}, ...
    {'double','double','double','double','double','double'}, ...
    'MinValues', [-10,-10,-10,-10,-10, y_min], ...
    'MaxValues', [ 10, 10, 10, 10, 10, y_max]);

dw.createTable('synth_data', schema);

T = array2table([X, Y], 'VariableNames', {'x1', 'x2', 'x3', 'x4', 'x5', 'y'});
dw.insertBatch('synth_data', T);

%Query 
% Query: filter rows where x1 > 0
rows_pos = dw.query('synth_data', 'x1 > 0');
pct_pos  = 100 * numel(rows_pos) / N;
fprintf('   %d rows inserted successfully.\n', N);
fprintf('   Query "x1 > 0" returned %d rows (%.1f%% of inserted).\n', ...
    numel(rows_pos), pct_pos);

%% POSTGRESQL
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

% %% REDIS CACHING WAREHOUSE

fprintf('\n[§5] Redis CachingWarehouse — read-through cache demo...\n');

redisAvailable = false;

try
    redisCfg = OceanData.Adapters.RedisConfig('TTL', 300, 'Prefix', 'ocean_exp:');
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
%docker exec -it redis_ocean redis-cli

%%  FEATURE ENGINEERING 

fprintf('\n[§6] Feature Engineering — polynomial basis expansion (degree=2)...\n');
polyExpand = @(M) OceanStats.Data.polyFeatures(M, 2);

%Disk-backed FeatureStore
fs = OceanData.DataArch.FeatureStore('rootDir', FS_DIR);
feat_meta = struct('transform', 'poly2', 'degree', 2, 'N', N, 'source', 'raw/synth/X');
fprintf('   [FeatureStore] Computing + saving features to disk...\n');
t_dist_compute = tic;
entry_fs = fs.getOrCompute('poly2/synth/v1', @()polyExpand(X), feat_meta);
t_disk1 = toc(t_dist_compute);

fprintf('   [FeatureStore] Re-loading features from disk (cached)...\n');
t_disk_load = tic;
entry_fs2 = fs.getOrCompute('poly2/synth/v1', @() error('Should not recompute'), feat_meta);
t_disk2   = toc(t_disk_load);

fprintf('   Disk compute: %.4f s  |  Disk load: %.4f s  (%.1fx faster)\n', ...
    t_disk1, t_disk2, t_disk1/max(t_disk2,1e-6));

Phi = entry_fs2.data;  % [N x D] expanded feature matrix
D   = size(Phi, 2);
fprintf('   Feature matrix: [%d x %d] (from %d raw → %d polynomial features)\n', N, D, P, D);

%% Redis FeatureCache
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

%% Train/ test split + model training

fprintf('\n[§7] Model Training — Ridge vs OLS...\n');
%80/20 hold out split
sp = OceanData.DataArch.DataSplitter('Strategy', 'holdout', 'Ratio', 0.8, 'Seed', 42);
split = sp.split(Phi, Y);

%Slice
Phi_tr = Phi(split.trainIdx, :);
Y_tr = Phi(split.trainIdx);
Phi_te = Phi(split.testIdx, :);
Y_te = Y(split.testIdx);

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
