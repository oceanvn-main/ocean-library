classdef ExperimentContext < handle
    % EXPERIMENTCONTEXT  Reusable ML pipeline infrastructure for all experiments.
    %
    % Encapsulates Steps 3-4 of the Ocean Library ML Pipeline:
    %   - Password loading (env var → MATLAB pref → error)
    %   - PostgresWarehouse connection + experiment_results table
    %   - Redis CachingWarehouse setup
    %   - DataLake construction
    %   - Run context: git_hash, hostname, UUID-based run_id
    %   - Versioned result INSERT + historical query
    %
    % USAGE:
    %   ctx = OceanTools.ExperimentContext('exp_waveform_data', 'waveform', 42);
    %   ctx.connect('./lake_waveform/');   % Steps 2-4 in one call
    %   ctx.setNTrain(size(X_train, 1));   % call after split
    %   ctx.saveResult('LDA', 0.227, 'baseline');
    %   ctx.queryHistory();
    %   ctx.close();
    %
    % Run-ID format:  {expName}_{dataset}_{yyyyMMddTHHMM}_{6hex}_{modelName}
    %   e.g.  exp_waveform_data_waveform_20260317T0530_a3f2b1_LDA
    %
    % Inspect results:
    %   pgAdmin:      SELECT * FROM experiment_results ORDER BY run_ts DESC;
    %   RedisInsight: localhost:6379 → Browser tab
    %
    % See also: OceanData.Adapters.PostgresWarehouse, OceanData.DataArch.DataLake

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Reproducibility context (read-only from outside)
        GitHash     (1,1) string   % short git commit hash of current HEAD
        Hostname    (1,1) string   % machine name
        BaseRunId   (1,1) string   % {exp}_{dataset}_{ts}_{6hex}

        % Infrastructure handles (public read-only for custom queries in exp scripts)
        Pg          = []           % OceanData.Adapters.PostgresWarehouse (or [])
        Cdw         = []           % OceanData.Adapters.CachingWarehouse (Redis, or [])
        DataLake    = []           % OceanData.DataArch.DataLake (or [])
        ModelStore  = []           % OceanData.DataArch.FeatureStore (model cache)
    end

    properties (Access = private)
        expName_    (1,1) string
        dataset_    (1,1) string
        rngSeed_    (1,1) double
        nTrain_     (1,1) double = 0
    end

    % ------------------------------------------------------------------ %
    methods

        function obj = ExperimentContext(expName, dataset, rngSeed)
            % EXPERIMENTCONTEXT  Create context and collect run metadata.
            %   Does NOT connect to Postgres yet — call connect() after construction.
            %
            % Inputs:
            %   expName  - experiment script name (e.g. 'exp_waveform_data')
            %   dataset  - dataset identifier    (e.g. 'waveform', 'zip')
            %   rngSeed  - random seed used for reproducibility
            arguments
                expName  (1,1) string
                dataset  (1,1) string
                rngSeed  (1,1) double = 42
            end

            obj.expName_ = expName;
            obj.dataset_ = dataset;
            obj.rngSeed_ = rngSeed;

            % --- Git hash: identifies exact code version ---
            [~, gh] = system('git rev-parse --short HEAD');
            gh = strtrim(gh);
            if isempty(gh) || contains(gh,'error') || contains(gh,'not a git')
                gh = 'no-git';
            end
            obj.GitHash = gh;

            % --- Hostname: machine identifier ---
            h = strtrim(getenv('COMPUTERNAME'));
            if isempty(h); h = computer(); end
            obj.Hostname = h;

            % --- Base run-id: shared across all models in one run ---
            ts         = datestr(now, 'yyyymmddTHHMM');
            uuid_full  = char(java.util.UUID.randomUUID());
            uuid_short = uuid_full(1:6);
            obj.BaseRunId = sprintf('%s_%s_%s_%s', expName, dataset, ts, uuid_short);

            fprintf('[ExperimentContext] Run context ready.\n');
            fprintf('  expr    : %s\n', expName);
            fprintf('  dataset : %s  seed=%d\n', dataset, rngSeed);
            fprintf('  git     : %s\n', obj.GitHash);
            fprintf('  host    : %s\n', obj.Hostname);
            fprintf('  base_id : %s\n', obj.BaseRunId);
        end

        % -------------------------------------------------------------- %
        function connect(obj, lakeDir)
            % CONNECT  Steps 2-4: DataLake + PostgresWarehouse + Redis CachingWarehouse.
            %
            % Inputs:
            %   lakeDir - (optional) root directory for DataLake, e.g. './lake_waveform/'
            %             Pass '' or omit to skip DataLake.
            arguments
                obj
                lakeDir (1,1) string = ""
            end

            % --- Step 2: DataLake (optional) ---
            if lakeDir ~= ""
                try
                    obj.DataLake = OceanData.DataArch.DataLake('rootDir', char(lakeDir));
                    fprintf('[ExperimentContext][2] DataLake ready at %s\n', lakeDir);
                catch e
                    fprintf('[ExperimentContext][2] DataLake SKIP: %s\n', e.message);
                end
            end

            % --- Step 4 FIRST: Redis/Jedis JAR must load before Postgres JAR ---
            % In MATLAB batch mode: once Postgres JAR creates JVM objects, javaaddpath
            % for Jedis silently fails (classloader conflict). Loading Jedis first avoids this.
            redis = [];
            try
                redis = OceanData.Adapters.RedisAdapter(OceanData.Adapters.RedisConfig());
                redis.connect();
                fprintf('[ExperimentContext][4] Redis connected.\n');
            catch e
                fprintf('[ExperimentContext][4] Redis SKIP: %s\n', e.message);
                redis = [];
            end

            % --- Step 3: PostgresWarehouse (after Jedis is on the classpath) ---
            pg_pass = obj.loadPassword_();
            try
                cfg = OceanData.Adapters.DbConfig('Password', pg_pass);
                obj.Pg = OceanData.Adapters.PostgresWarehouse(cfg);
                obj.Pg.connect();
                fprintf('[ExperimentContext][3] PostgresWarehouse: %s\n', cfg.describe());
                obj.ensureResultsTable_();
            catch e
                fprintf('[ExperimentContext][3] PostgresWarehouse SKIP: %s\n', e.message);
                obj.Pg = [];
            end

            % --- Bind CachingWarehouse once both sides are live ---
            if ~isempty(redis) && ~isempty(obj.Pg)
                try
                    obj.Cdw = OceanData.Adapters.CachingWarehouse(obj.Pg, redis);
                    fprintf('[ExperimentContext][4] CachingWarehouse ready (Redis wrapping Postgres).\n');
                    fprintf('  After queryHistory(): RedisInsight -> Browser -> Scan -> "query:*"\n');
                catch e
                    fprintf('[ExperimentContext][4] CachingWarehouse SKIP: %s\n', e.message);
                end
            end
        end

        % -------------------------------------------------------------- %
        function setNTrain(obj, n)
            % SETNTRAIN  Provide training set size (for INSERT into experiment_results).
            %   Call this after the train/test split is complete.
            obj.nTrain_ = n;
        end

        % -------------------------------------------------------------- %
        function run_id = saveResult(obj, modelName, testError, notes, force)
            % SAVERESULT  INSERT one model result row into experiment_results.
            %
            % Duplicate prevention: if (dataset, model_name, rng_seed, git_hash) already
            % exists, INSERT is SKIPPED. Pass force=true to override.
            %
            % Usage:
            %   ctx.saveResult('LDA', 0.227)                 % skips if duplicate
            %   ctx.saveResult('LDA', 0.227, 'my note')      % with note
            %   ctx.saveResult('LDA', 0.227, 'note', true)   % force insert
            arguments
                obj
                modelName (1,1) string
                testError (1,1) double
                notes     (1,1) string  = ""
                force     (1,1) logical = false
            end

            run_id = sprintf('%s_%s', obj.BaseRunId, modelName);

            if isempty(obj.Pg)
                fprintf('[ExperimentContext] saveResult SKIP: Postgres not connected.\n');
                return;
            end

            % --- Duplicate detection ---
            % Skip if (dataset, model_name, rng_seed, git_hash) already in table.
            % Rationale: same code + same data + same model = identical result.
            % Override with: ctx.saveResult('LDA', err, 'Force', true);
            if ~force
                try
                    existing = obj.Pg.query('experiment_results');
                    if ~isempty(existing)
                        match = strcmp({existing.dataset},    char(obj.dataset_)) & ...
                                strcmp({existing.model_name}, char(modelName))    & ...
                                [existing.rng_seed] == obj.rngSeed_               & ...
                                strcmp({existing.git_hash},   char(obj.GitHash));
                        if any(match)
                            prev = existing(find(match, 1, 'last'));
                            fprintf('[ExperimentContext] SKIP duplicate: %s (prev err=%.4f, %s)\n', ...
                                modelName, prev.test_error, prev.run_ts);
                            fprintf('  Same dataset/model/seed/git_hash already in experiment_results.\n');
                            fprintf('  To force insert: ctx.saveResult(model, err, notes, Force=true)\n');
                            return;
                        end
                    end
                catch
                    % If duplicate check fails, proceed with insert anyway
                end
            end

            row = struct( ...
                'run_id',     char(run_id), ...
                'run_ts',     datestr(now), ...
                'dataset',    char(obj.dataset_), ...
                'model_name', char(modelName), ...
                'test_error', testError, ...
                'n_train',    double(obj.nTrain_), ...
                'rng_seed',   double(obj.rngSeed_), ...
                'git_hash',   char(obj.GitHash), ...
                'hostname',   char(obj.Hostname), ...
                'notes',      char(notes));
            try
                obj.Pg.insert('experiment_results', row);
                fprintf('[ExperimentContext] Saved: run_id=%s\n', run_id);
            catch e
                fprintf('[ExperimentContext] saveResult ERROR: %s\n', e.message);
            end
        end

        % -------------------------------------------------------------- %
        function queryHistory(obj, filterExpr)
            % QUERYHISTORY  Print all rows in experiment_results (newest last).
            %
            % Optional input:
            %   filterExpr - SQL WHERE fragment (e.g. 'dataset == waveform')
            %                Passed through to PostgresWarehouse.query().
            arguments
                obj
                filterExpr (1,1) string = ""
            end

            if isempty(obj.Pg)
                fprintf('[ExperimentContext] queryHistory SKIP: Postgres not connected.\n');
                return;
            end

            try
                % Route through CachingWarehouse (Redis) if available.
                % Key stored in Redis: "query:experiment_results:"
                % 1st call → Redis MISS → Postgres → fills Redis
                % 2nd call → Redis HIT  → instant, no Postgres round-trip
                if ~isempty(obj.Cdw)
                    try
                        hist = obj.Cdw.query('experiment_results');
                        obj.Cdw.stats();   % prints "Hits: X / Y (Z%)"
                    catch e
                        % Print actual error so we can diagnose — then fall back
                        fprintf('[ExperimentContext] CachingWarehouse.query failed: %s\n', e.message);
                        fprintf('  Falling back to direct Postgres query (no Redis cache).\n');
                        hist = obj.Pg.query('experiment_results');
                    end
                elseif ~isempty(obj.Pg)
                    hist = obj.Pg.query('experiment_results');
                else
                    hist = [];
                end

                if isempty(hist)
                    fprintf('\n  [experiment_results: no rows yet]\n');
                    return;
                end

                fprintf('\n--- experiment_results (%d row(s)) ---\n', numel(hist));
                fprintf('  %-46s  %-8s  %-10s  %-8s  %-10s  %s\n', ...
                    'run_id', 'model', 'test_err', 'git', 'host', 'run_ts');
                fprintf('  %s\n', repmat('-', 1, 105));
                for i = 1:numel(hist)
                    % Truncate run_id to last 46 chars if long
                    rid = char(hist(i).run_id);
                    if length(rid) > 46; rid = ['...' rid(end-42:end)]; end
                    fprintf('  %-46s  %-8s  %.3f      %-8s  %-10s  %s\n', ...
                        rid, hist(i).model_name, hist(i).test_error, ...
                        hist(i).git_hash, hist(i).hostname, hist(i).run_ts);
                end
            catch e
                fprintf('[ExperimentContext] queryHistory ERROR: %s\n', e.message);
            end
        end

        % -------------------------------------------------------------- %
        function model = cachedFit(obj, modelName, constructorFn, X, y, options)
            % CACHEDFIT  Train-once, load-on-rerun convenience wrapper.
            %
            %   model = ctx.cachedFit('LDA', @() LDA(), X_train, y_train)
            %
            % The cache key is auto-generated as:
            %   {dataset}/seed{rngSeed}/{modelName}
            %   e.g. 'vowel/seed42/LDA'
            %
            % Override with CacheDir to use a custom store location.
            %
            % See also: OceanTools.cachedFit
            arguments
                obj
                modelName      (1,1) string
                constructorFn  (1,1) function_handle
                X              double
                y
                options.CacheDir (1,1) string = "./cache/models/"
            end

            % Lazy-initialize ModelStore on first call
            if isempty(obj.ModelStore)
                obj.ModelStore = OceanData.DataArch.FeatureStore('rootDir', char(options.CacheDir));
            end

            key = sprintf('%s/seed%d/%s', obj.dataset_, obj.rngSeed_, modelName);
            model = OceanTools.cachedFit(obj.ModelStore, key, constructorFn, X, y);
        end

        % -------------------------------------------------------------- %
        function close(obj)
            % CLOSE  Disconnect from PostgresWarehouse cleanly.
            if ~isempty(obj.Pg)
                try
                    obj.Pg.disconnect();
                    fprintf('[ExperimentContext] Postgres disconnected.\n');
                catch
                end
            end
        end

    end

    % ------------------------------------------------------------------ %
    methods (Access = private)

        function pw = loadPassword_(~)
            % Priority: env var OCEAN_PG_PASS → MATLAB pref OceanDB.Password → error
            %
            % Set env var (Windows PowerShell, per session):
            %   $env:OCEAN_PG_PASS = 'yourpassword'
            %
            % Or store permanently in MATLAB prefs (run once in Command Window):
            %   setpref('OceanDB', 'Password', 'yourpassword')
            pw = getenv('OCEAN_PG_PASS');
            if isempty(pw) && ispref('OceanDB', 'Password')
                pw = getpref('OceanDB', 'Password');
            end
            if isempty(pw)
                error('OceanTools:ExperimentContext:NoPassword', ...
                    ['PostgreSQL password not found.\n' ...
                     'Option 1 (session):   setenv(''OCEAN_PG_PASS'', ''yourpassword'')\n' ...
                     'Option 2 (permanent): setpref(''OceanDB'', ''Password'', ''yourpassword'')']);
            end
        end

        function ensureResultsTable_(obj)
            % Create experiment_results if it does not exist. If it does exist
            % (from a previous run with fewer columns), automatically add any
            % missing columns via ALTER TABLE … ADD COLUMN IF NOT EXISTS.
            names = {'run_id','run_ts','dataset','model_name','test_error', ...
                     'n_train','rng_seed','git_hash','hostname','notes'};
            types = {'string','string','string','string','double', ...
                     'double','double','string','string','string'};
            schema = OceanData.DataArch.DWSchema(names, types);
            obj.Pg.createTable('experiment_results', schema);

            % SQL type map (mirrors PostgresWarehouse.buildColumnDDL_)
            typeMap = struct('double','DOUBLE PRECISION','integer','INTEGER', ...
                             'logical','BOOLEAN','string','TEXT','category','TEXT');

            % ALTER TABLE ADD COLUMN IF NOT EXISTS for each new column
            newCols = {'git_hash','hostname','notes'};
            for i = 1:numel(newCols)
                col     = newCols{i};
                colIdx  = find(strcmp(names, col), 1);
                sqlType = typeMap.(types{colIdx});
                sql = sprintf('ALTER TABLE experiment_results ADD COLUMN IF NOT EXISTS "%s" %s;', ...
                              col, sqlType);
                try; obj.Pg.Adapter_.execute(sql); catch; end
            end

            % --- Column descriptions (visible in pgAdmin: column → Properties) ---
            comments = { ...
                'run_id',     '{exp}_{dataset}_{yyyyMMddTHHMM}_{6hex}_{model}: unique run ID — shared prefix across all models in the same run'; ...
                'run_ts',     'Wall-clock timestamp when this row was inserted'; ...
                'dataset',    'Dataset identifier, e.g. waveform, zip, mixture'; ...
                'model_name', 'Model identifier, e.g. LDA, PDA, MDA, SVM'; ...
                'test_error', 'Misclassification rate (or RMSE) on the held-out test set — lower is better'; ...
                'n_train',    'Number of training samples that this model was fitted on'; ...
                'rng_seed',   'Random seed used for data split — same seed = same train/test partition'; ...
                'git_hash',   'Short git commit hash (git rev-parse --short HEAD): identifies which version of code produced this result. git diff <hashA> <hashB> shows what changed.'; ...
                'hostname',   'Machine name where the experiment ran (COMPUTERNAME env var)'; ...
                'notes',      'Free-text: what was changed in this run (hyperparams, preprocessing, ...)'; ...
            };
            try
                obj.Pg.Adapter_.execute('COMMENT ON TABLE experiment_results IS ''Versioned model results — one row per model per run. Duplicates (same dataset/model/seed/git_hash) are skipped automatically.'';');
                for i = 1:size(comments, 1)
                    sql = sprintf('COMMENT ON COLUMN experiment_results."%s" IS ''%s'';', ...
                                  comments{i,1}, strrep(comments{i,2}, '''', ''''''));
                    try; obj.Pg.Adapter_.execute(sql); catch; end
                end
            catch
            end
            fprintf('[ExperimentContext] experiment_results table ready (schema up to date).\n');
        end

    end
end
