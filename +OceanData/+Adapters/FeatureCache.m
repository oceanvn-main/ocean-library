classdef FeatureCache < handle
    % FEATURECACHE  Redis-backed cache for computed feature vectors.
    %
    %   Caches expensive transforms (Fourier Feature Maps, PCA, etc.)
    %   keyed by dataset identity + transform parameters.
    %   Access time: ~0.1–1ms (in-memory) vs 50–200ms (disk FeatureStore).
    %
    %   KEY PATTERN:
    %     feature:<dataset>/<transform>/<version>
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.RedisConfig('TTL', 3600);
    %     ra  = OceanData.Adapters.RedisAdapter(cfg); ra.connect();
    %     fc  = OceanData.Adapters.FeatureCache(ra);
    %
    %     % Compute-or-load:
    %     Phi = fc.getOrCompute('spam/fourier/v3', @() ffm.transform(X), meta);
    %
    %     % Explicit put/get:
    %     fc.put('spam/pca/v1', X_pca, struct('nComponents', 20));
    %     entry = fc.get('spam/pca/v1');
    %
    %   See also: OceanData.Adapters.RedisCacheStore
    %             OceanData.DataArch.FeatureStore   (disk-backed alternative)

    properties (SetAccess = private)
        Store  OceanData.Adapters.RedisCacheStore
    end

    methods
        function obj = FeatureCache(redis)
            arguments
                redis OceanData.Adapters.RedisAdapter
            end
            obj.Store = OceanData.Adapters.RedisCacheStore(redis);
        end

        % ---------------------------------------------------------------- %
        function put(obj, key, data, meta)
            % PUT  Cache a feature matrix.
            arguments
                obj
                key  (1,1) string
                data
                meta struct = struct()
            end
            obj.Store.put(obj.featureKey_(key), data, meta);
        end

        % ---------------------------------------------------------------- %
        function entry = get(obj, key)
            % GET  Load cached feature entry (throws if not cached).
            entry = obj.Store.get(obj.featureKey_(key));
        end

        % ---------------------------------------------------------------- %
        function result = getOrCompute(obj, key, computeFn, meta)
            % GETORCOMPUTE  Return cached features, or compute + cache.
            %
            %   result = fc.getOrCompute('spam/fourier/v3', @() ffm.transform(X), meta)
            %
            %   computeFn  — zero-arg function handle that returns a matrix
            %   result     — struct(data, meta, timestamp, key)
            arguments
                obj
                key       (1,1) string
                computeFn function_handle
                meta      struct = struct()
            end
            fk = obj.featureKey_(key);
            result = obj.Store.getOrCompute(fk, computeFn, meta);
        end

        % ---------------------------------------------------------------- %
        function tf = exists(obj, key)
            tf = obj.Store.exists(obj.featureKey_(key));
        end

        % ---------------------------------------------------------------- %
        function invalidate(obj, key)
            % INVALIDATE  Remove a cached feature entry.
            obj.Store.delete(obj.featureKey_(key));
        end

        % ---------------------------------------------------------------- %
        function ks = list(obj)
            % LIST  Return all cached feature keys.
            ks = obj.Store.list();
        end
    end

    methods (Access = private)
        function fk = featureKey_(~, key)
            fk = sprintf('feature:%s', char(key));
        end
    end
end
