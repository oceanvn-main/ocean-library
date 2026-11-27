classdef RedisAdapter < handle
    % REDISADAPTER  Low-level Redis client using the Jedis Java library.
    %
    %   Implements the same interface pattern as PostgresAdapter.
    %   Requires jedis-3.x.jar in +Adapters/ or on the Java classpath.
    %
    %   SETUP:
    %     1. Download Jedis jar:
    %        Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/redis/clients/jedis/3.10.0/jedis-3.10.0.jar" `
    %            -OutFile "+OceanData\+Adapters\jedis-3.10.0.jar"
    %     2. Connect:
    %        cfg = OceanData.Adapters.RedisConfig();
    %        rc  = OceanData.Adapters.RedisAdapter(cfg);
    %        rc.connect();
    %
    %   KEY METHODS:
    %     connect()           — open connection
    %     disconnect()        — close connection
    %     set(key, value)     — store a string value (with TTL if configured)
    %     get(key)            — retrieve string value ('' if missing)
    %     del(key)            — delete a key
    %     exists(key)         — logical
    %     keys(pattern)       — cell array of matching keys
    %     isConnected()       — logical
    %
    %   See also: OceanData.Adapters.RedisConfig
    %             OceanData.Adapters.RedisCacheStore

    properties (SetAccess = private)
        Config     OceanData.Adapters.RedisConfig
        Jedis_     % Java Jedis object (set after connect)
    end

    methods
        function obj = RedisAdapter(cfg)
            arguments
                cfg OceanData.Adapters.RedisConfig
            end
            obj.Config = cfg;
            obj.Jedis_ = [];
            obj.ensureJedisOnPath_();
        end

        % ---------------------------------------------------------------- %
        function connect(obj)
            % CONNECT  Open a connection to the Redis server.
            try
                obj.Jedis_ = javaObject('redis.clients.jedis.Jedis', ...
                    char(obj.Config.Host), int32(obj.Config.Port));

                if strlength(obj.Config.Password) > 0
                    obj.Jedis_.auth(char(obj.Config.Password));
                end
                if obj.Config.Db > 0
                    obj.Jedis_.select(int32(obj.Config.Db));
                end
                fprintf('[RedisAdapter] Connected: %s\n', obj.Config.describe());
            catch e
                error('OceanData:Adapters:RedisAdapter:ConnectFailed', ...
                    'Cannot connect to Redis at %s:%d — %s', ...
                    obj.Config.Host, obj.Config.Port, e.message);
            end
        end

        % ---------------------------------------------------------------- %
        function disconnect(obj)
            if ~isempty(obj.Jedis_)
                try, obj.Jedis_.close(); catch, end
                obj.Jedis_ = [];
            end
        end

        % ---------------------------------------------------------------- %
        function set(obj, key, value)
            % SET  Store a string value under a key (with configured TTL).
            obj.requireConnected_();
            fullKey = obj.fullKey_(key);
            obj.Jedis_.set(fullKey, value);
            if obj.Config.TTL > 0
                obj.Jedis_.expire(fullKey, int32(obj.Config.TTL));
            end
        end

        % ---------------------------------------------------------------- %
        function value = get(obj, key)
            % GET  Retrieve a string value. Returns '' if key does not exist.
            obj.requireConnected_();
            raw   = obj.Jedis_.get(obj.fullKey_(key));
            if isempty(raw) || raw == java.lang.String('')
                value = '';
            else
                value = char(raw);
            end
        end

        % ---------------------------------------------------------------- %
        function del(obj, key)
            % DEL  Delete a key (idempotent).
            obj.requireConnected_();
            obj.Jedis_.del(obj.fullKey_(key));
        end

        % ---------------------------------------------------------------- %
        function tf = exists(obj, key)
            % EXISTS  True if the key exists in Redis.
            obj.requireConnected_();
            tf = obj.Jedis_.exists(obj.fullKey_(key)).booleanValue();
        end

        % ---------------------------------------------------------------- %
        function ks = keys(obj, pattern)
            % KEYS  Return all matching keys (glob pattern, no prefix).
            arguments
                obj
                pattern (1,1) string = "*"
            end
            obj.requireConnected_();
            fullPat  = obj.fullKey_(pattern);
            javaKeys = obj.Jedis_.keys(fullPat);
            ks       = {};
            it       = javaKeys.iterator();
            pLen     = strlength(obj.Config.Prefix);
            while it.hasNext()
                k = char(it.next());
                % Strip prefix back for caller convenience
                if startsWith(k, char(obj.Config.Prefix))
                    k = k(pLen+1:end);
                end
                ks{end+1} = k; %#ok<AGROW>
            end
        end

        % ---------------------------------------------------------------- %
        function tf = isConnected(obj)
            % ISCONNECTED  True if the Jedis connection is open and responsive.
            tf = false;
            if ~isempty(obj.Jedis_)
                try
                    res = char(obj.Jedis_.ping());
                    tf = strcmp(res, 'PONG');
                catch
                    tf = false;
                end
            end
        end

        function delete(obj)
            obj.disconnect();
        end
    end

    % ================================================================== %
    methods (Access = private)

        function requireConnected_(obj)
            if ~obj.isConnected()
                error('OceanData:Adapters:RedisAdapter:NotConnected', ...
                    'Not connected. Call connect() first.');
            end
        end

        function fk = fullKey_(obj, key)
            fk = [char(obj.Config.Prefix), char(key)];
        end

        function ensureJedisOnPath_(~)
            % Add jedis jar to Java classpath if not already loaded.
            adapterDir = fileparts(mfilename('fullpath'));
            jars       = dir(fullfile(adapterDir, 'jedis-*.jar'));
            if isempty(jars)
                warning('OceanData:Adapters:RedisAdapter:NoJedisJar', ...
                    ['Jedis jar not found in +Adapters/. ' ...
                     'Download jedis-3.x.jar and place it there.']);
                return;
            end
            jarPath = fullfile(jars(end).folder, jars(end).name);
            cp      = javaclasspath('-dynamic');
            if ~any(strcmp(cp, jarPath))
                javaaddpath(jarPath);
                fprintf('[RedisAdapter] Loaded: %s\n', jars(end).name);
            end
        end
    end
end

function r = startsWith(s, p)
    r = numel(s) >= numel(p) && strcmp(s(1:numel(p)), p);
end
