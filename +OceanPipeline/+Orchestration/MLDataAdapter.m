classdef MLDataAdapter < handle
    % MLDATAADAPTER  Prepare ML-ready train/test packages from a DataPipeline.
    %
    %   Orchestrates the "final mile":
    %     DataPipeline.run() → clean data → DataSplitter → {X_tr, X_te, Y_tr, Y_te}
    %
    %   OPTIONAL: Cache the clean feature matrix in a FeatureStore to avoid
    %             re-running the full pipeline on repeated experiments.
    %
    %   USAGE (holdout):
    %     adapter = OceanPipeline.Orchestration.MLDataAdapter(pipeline, splitter);
    %     pkg = adapter.prepare(X_raw, Y);
    %     model.fit(pkg.X_train, pkg.Y_train);
    %     model.evaluate(pkg.X_test,  pkg.Y_test);
    %
    %   USAGE (k-fold):
    %     splitter = OceanData.DataArch.DataSplitter('Strategy','kfold','K',5);
    %     adapter  = OceanPipeline.Orchestration.MLDataAdapter(pipeline, splitter);
    %     pkg = adapter.prepare(X_raw, Y);
    %     % pkg.folds{k}.X_train, pkg.folds{k}.X_test, etc.
    %
    %   USAGE (with cache):
    %     fs      = OceanData.DataArch.FeatureStore('rootDir','./features/');
    %     adapter = OceanPipeline.Orchestration.MLDataAdapter(pipeline, splitter, ...
    %                   'FeatureStore', fs, 'CacheKey', 'spam/clean/v1');
    %
    %   See also: OceanPipeline.Orchestration.DataPipeline
    %             OceanData.DataArch.DataSplitter
    %             OceanData.DataArch.FeatureStore

    properties (SetAccess = private)
        Pipeline    OceanPipeline.Orchestration.DataPipeline
        Splitter    OceanData.DataArch.DataSplitter
        Store                  % OceanData.DataArch.FeatureStore (optional)
        CacheKey    (1,1) string = ""
    end

    methods
        function obj = MLDataAdapter(pipeline, splitter, opts)
            arguments
                pipeline OceanPipeline.Orchestration.DataPipeline
                splitter OceanData.DataArch.DataSplitter
                opts.FeatureStore = []
                opts.CacheKey     (1,1) string = ""
            end
            obj.Pipeline = pipeline;
            obj.Splitter = splitter;
            obj.Store    = opts.FeatureStore;
            obj.CacheKey = opts.CacheKey;
        end

        % ---------------------------------------------------------------- %
        function pkg = prepare(obj, X_raw, Y)
            % PREPARE  Clean X_raw, split, and package for ML.
            %
            %   X_raw — (N x p) raw feature matrix  (before pipeline)
            %   Y     — (N x q) target matrix or vector
            %   pkg   — struct with fields depending on splitter strategy:
            %
            %   holdout strategy:
            %     pkg.X_train, pkg.X_test, pkg.Y_train, pkg.Y_test
            %     pkg.trainIdx, pkg.testIdx
            %
            %   kfold/stratified strategy:
            %     pkg.folds   — (1 x K) cell of structs, each with:
            %                   .X_train, .X_test, .Y_train, .Y_test
            %     pkg.K
            arguments
                obj
                X_raw double
                Y     double = []
            end

            % Step 1: Fit pipeline on all raw data (will be re-split after)
            if ~obj.Pipeline.IsFitted
                obj.Pipeline.fit(X_raw);
            end

            % Step 2: Run pipeline to get clean data (or load from cache)
            X_clean = obj.getClean_(X_raw);

            % Step 3: Split and package
            pkg = obj.buildPackage_(X_clean, Y);
        end

    end

    % ================================================================== %
    methods (Access = private)

        function X_clean = getClean_(obj, X_raw)
            % Try cache first, compute if miss
            useCache = ~isempty(obj.Store) && strlength(obj.CacheKey) > 0;

            if useCache && obj.Store.exists(obj.CacheKey)
                entry   = obj.Store.get(obj.CacheKey);
                X_clean = entry.data;
                fprintf('[MLDataAdapter] Cache hit: %s\n', obj.CacheKey);
            else
                result  = obj.Pipeline.run(X_raw);
                X_clean = result.data;
                if useCache
                    obj.Store.put(obj.CacheKey, X_clean, ...
                        struct('source','MLDataAdapter','timestamp',datetime('now')));
                    fprintf('[MLDataAdapter] Cached: %s\n', obj.CacheKey);
                end
            end
        end

        function pkg = buildPackage_(obj, X_clean, Y)
            % Delegate splitting to DataSplitter, then index into X and Y
            hasY = ~isempty(Y);

            split = obj.Splitter.split(X_clean, Y);

            if isstruct(split)
                % Holdout mode
                pkg.X_train  = X_clean(split.trainIdx, :);
                pkg.X_test   = X_clean(split.testIdx,  :);
                pkg.trainIdx = split.trainIdx;
                pkg.testIdx  = split.testIdx;
                if hasY
                    pkg.Y_train = Y(split.trainIdx, :);
                    pkg.Y_test  = Y(split.testIdx,  :);
                end

            elseif iscell(split)
                % K-fold / stratified mode
                K   = numel(split);
                pkg.folds = cell(1, K);
                pkg.K     = K;
                for k = 1:K
                    f = split{k};
                    fold.X_train = X_clean(f.trainIdx, :);
                    fold.X_test  = X_clean(f.testIdx,  :);
                    fold.trainIdx = f.trainIdx;
                    fold.testIdx  = f.testIdx;
                    if hasY
                        fold.Y_train = Y(f.trainIdx, :);
                        fold.Y_test  = Y(f.testIdx,  :);
                    end
                    pkg.folds{k} = fold;
                end
            end

            pkg.nSamples  = size(X_clean, 1);
            pkg.nFeatures = size(X_clean, 2);
            pkg.timestamp = datetime('now');
        end

    end
end
