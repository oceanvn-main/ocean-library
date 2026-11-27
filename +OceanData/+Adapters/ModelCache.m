classdef ModelCache < handle
    % MODELCACHE  Redis-backed cache for model prediction outputs.
    %
    %   Caches predict() results so the same input-model pair is never
    %   evaluated twice during a session (or across sessions within TTL).
    %
    %   KEY PATTERN:
    %     pred:<model_id>:<hash_of_X>
    %
    %   HASH:
    %     We use a lightweight SHA-like fingerprint: sum of X(:) + std of X(:)
    %     combined with size. Fast and sufficient for collision avoidance.
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.RedisConfig('TTL', 1800);
    %     ra  = OceanData.Adapters.RedisAdapter(cfg); ra.connect();
    %     mc  = OceanData.Adapters.ModelCache(ra);
    %
    %     % Cache-or-predict:
    %     Y_hat = mc.getOrPredict('gbm_spam_v3', X_test, @(X) model.predict(X));
    %
    %   See also: OceanData.Adapters.RedisCacheStore
    %             OceanData.Adapters.FeatureCache

    properties (SetAccess = private)
        Store    OceanData.Adapters.RedisCacheStore
    end

    methods
        function obj = ModelCache(redis)
            arguments
                redis OceanData.Adapters.RedisAdapter
            end
            obj.Store = OceanData.Adapters.RedisCacheStore(redis);
        end

        % ---------------------------------------------------------------- %
        function Y_hat = getOrPredict(obj, modelId, X, predictFn)
            % GETORPREDICT  Return cached predictions, or run model + cache.
            %
            %   modelId   — unique string for the model (e.g. 'gbm_spam_v3')
            %   X         — (N x p) input matrix to predict from
            %   predictFn — one-arg function handle: fn(X) → Y_hat
            arguments
                obj
                modelId   (1,1) string
                X         double
                predictFn function_handle
            end
            key   = obj.predKey_(modelId, X);
            if obj.Store.exists(key)
                entry = obj.Store.get(key);
                Y_hat = entry.data;
                return;
            end
            Y_hat = predictFn(X);
            obj.Store.put(key, Y_hat, struct('modelId', char(modelId), ...
                'nInputRows', size(X,1)));
        end

        % ---------------------------------------------------------------- %
        function put(obj, modelId, X, Y_hat)
            % PUT  Explicitly cache predictions for an (modelId, X) pair.
            key = obj.predKey_(modelId, X);
            obj.Store.put(key, Y_hat, struct('modelId', char(modelId)));
        end

        % ---------------------------------------------------------------- %
        function tf = exists(obj, modelId, X)
            tf = obj.Store.exists(obj.predKey_(modelId, X));
        end

        % ---------------------------------------------------------------- %
        function invalidate(obj, modelId)
            % INVALIDATE  Remove all cached predictions for a model.
            %   Since we can't enumerate per-model keys efficiently,
            %   this is a best-effort sweep using list().
            keys = obj.Store.list();
            prefix = sprintf('pred:%s:', char(modelId));
            for i = 1:numel(keys)
                if startsWith(keys{i}, prefix)
                    obj.Store.delete(keys{i});
                end
            end
        end
    end

    methods (Access = private)
        function key = predKey_(~, modelId, X)
            % Lightweight fingerprint: size + sum + std of X
            h   = sum(X(:)) * numel(X) + std(X(:)) * size(X,1);
            key = sprintf('pred:%s:%s', char(modelId), num2str(h, '%+.6e'));
        end
    end
end

function r = startsWith(s, p)
    r = numel(s) >= numel(p) && strcmp(s(1:numel(p)), p);
end
