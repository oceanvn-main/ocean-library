classdef RedisCacheStore < handle
    % REDISCACHESTORE  In-memory key-value cache implementing IDataStore.
    %
    %   Serializes MATLAB structs/arrays to JSON for Redis storage.
    %   Deserializes on read with jsondecode.
    %
    %   USAGE:
    %     cfg   = OceanData.Adapters.RedisConfig('TTL', 600);
    %     redis = OceanData.Adapters.RedisAdapter(cfg);
    %     redis.connect();
    %     store = OceanData.Adapters.RedisCacheStore(redis);
    %
    %     store.put('spam/clean/v1', X_clean, struct('nFeatures', 57));
    %     entry = store.get('spam/clean/v1');
    %     X     = entry.data;
    %
    %   NOTE:
    %     jsonencode/jsondecode handles numeric arrays correctly.
    %     Complex MATLAB objects (classes, function handles) cannot be stored.
    %
    %   See also: OceanData.Adapters.RedisAdapter
    %             OceanData.DataArch.FeatureStore

    properties (SetAccess = private)
        Redis  OceanData.Adapters.RedisAdapter
    end

    methods
        function obj = RedisCacheStore(redis)
            arguments
                redis OceanData.Adapters.RedisAdapter
            end
            obj.Redis = redis;
        end

        % ---------------------------------------------------------------- %
        function put(obj, key, data, meta)
            % PUT  Serialize and store data + metadata in Redis.
            arguments
                obj
                key  (1,1) string
                data
                meta struct = struct()
            end
            entry.data      = data;
            entry.meta      = meta;
            entry.timestamp = char(datetime('now'));
            entry.key       = char(key);

            json = jsonencode(entry);
            obj.Redis.set(char(key), json);
        end

        % ---------------------------------------------------------------- %
        function entry = get(obj, key)
            % GET  Load and deserialize an entry from Redis.
            %   Throws KeyNotFound if key does not exist.
            json = obj.Redis.get(char(key));
            if isempty(json)
                error('OceanData:Adapters:RedisCacheStore:KeyNotFound', ...
                    'No cache entry found for key: "%s"', key);
            end
            entry = jsondecode(json);
        end

        % ---------------------------------------------------------------- %
        function entry = getOrCompute(obj, key, computeFn, meta)
            % GETORCOMPUTE  Return cached entry or compute, store, return.
            arguments
                obj
                key       (1,1) string
                computeFn function_handle
                meta      struct = struct()
            end
            if obj.exists(key)
                entry = obj.get(key);
                return;
            end
            data  = computeFn();
            obj.put(key, data, meta);
            entry = obj.get(key);
        end

        % ---------------------------------------------------------------- %
        function tf = exists(obj, key)
            tf = obj.Redis.exists(char(key));
        end

        % ---------------------------------------------------------------- %
        function delete(obj, key)
            % DELETE  Remove a key from Redis.
            obj.Redis.del(char(key));
        end

        % ---------------------------------------------------------------- %
        function ks = list(obj)
            % LIST  Return all keys in this store's namespace.
            ks = obj.Redis.keys('*');
        end
    end
end
