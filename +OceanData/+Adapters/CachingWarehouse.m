classdef CachingWarehouse < handle
    % CACHINGWAREHOUSE  Read-through cache wrapper for any DataWarehouse.
    %
    %   Places a Redis cache in front of a DataWarehouse (file-backed or
    %   PostgreSQL). Queries hit Redis first; only on a cache miss is the
    %   underlying warehouse queried, with the result then written to Redis.
    %   Inserts write to the warehouse AND invalidate the affected cache key.
    %
    %   USAGE:
    %     pg  = OceanData.DataArch.DataWarehouse(reg, 'rootDir', './dw/');
    %     cfg = OceanData.Adapters.RedisConfig('TTL', 600);
    %     ra  = OceanData.Adapters.RedisAdapter(cfg); ra.connect();
    %     dw  = OceanData.Adapters.CachingWarehouse(pg, ra);
    %
    %     % First query — hits Postgres, result stored in Redis
    %     rows = dw.query('users', 'age > 30');
    %
    %     % Same query again — Redis hit (instant)
    %     rows = dw.query('users', 'age > 30');
    %
    %     % Insert — writes to Postgres + invalidates Redis key for 'users'
    %     dw.insert('users', newRow);
    %
    %   See also: OceanData.Adapters.RedisAdapter
    %             OceanData.DataArch.DataWarehouse

    properties (SetAccess = private)
        Backend    % underlying DataWarehouse (file or Postgres)
        Cache      OceanData.Adapters.RedisCacheStore
        HitCount   (1,1) double = 0
        MissCount  (1,1) double = 0
    end

    methods
        function obj = CachingWarehouse(backend, redis)
            arguments
                backend           % OceanData.DataArch.DataWarehouse or similar
                redis OceanData.Adapters.RedisAdapter
            end
            obj.Backend = backend;
            obj.Cache   = OceanData.Adapters.RedisCacheStore(redis);
        end

        % ---------------------------------------------------------------- %
        function rows = query(obj, tableName, filter)
            % QUERY  Return rows matching filter (Redis read-through).
            arguments
                obj
                tableName (1,1) string
                filter    (1,1) string = ""
            end
            key = obj.queryKey_(tableName, filter);

            if obj.Cache.exists(key)
                entry = obj.Cache.get(key);
                rows  = entry.data;
                obj.HitCount = obj.HitCount + 1;
                return;
            end

            % Cache miss — query the warehouse
            obj.MissCount = obj.MissCount + 1;
            if strlength(filter) > 0
                rows = obj.Backend.query(tableName, filter);
            else
                rows = obj.Backend.query(tableName);
            end
            obj.Cache.put(key, rows, struct('table', char(tableName), 'filter', char(filter)));
        end

        % ---------------------------------------------------------------- %
        function insert(obj, tableName, data)
            % INSERT  Write to backend + invalidate all cached queries for this table.
            obj.Backend.insert(tableName, data);
            obj.invalidateTable_(tableName);
        end

        % ---------------------------------------------------------------- %
        function stats(obj)
            % STATS  Print cache hit/miss statistics.
            total = obj.HitCount + obj.MissCount;
            if total == 0
                fprintf('[CachingWarehouse] No queries yet.\n');
                return;
            end
            fprintf('[CachingWarehouse] Hits: %d / %d (%.1f%%)\n', ...
                obj.HitCount, total, 100*obj.HitCount/total);
        end
    end

    % ================================================================== %
    methods (Access = private)

        function key = queryKey_(~, tableName, filter)
            % Deterministic key from table + filter string.
            key = sprintf('query:%s:%s', char(tableName), char(filter));
        end

        function invalidateTable_(obj, tableName)
            % Remove all cached query results for a table.
            prefix = sprintf('query:%s:', char(tableName));
            keys   = obj.Cache.list();
            for i = 1:numel(keys)
                if startsWith(keys{i}, prefix)
                    obj.Cache.delete(keys{i});
                end
            end
        end
    end
end

function r = startsWith(s, p)
    r = numel(s) >= numel(p) && strcmp(s(1:numel(p)), p);
end
