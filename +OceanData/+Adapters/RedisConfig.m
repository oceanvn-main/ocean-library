classdef RedisConfig < handle
    % REDISCONFIG  Connection parameters for a Redis server.
    %
    %   Equivalent of DbConfig.m for Redis. Pass to RedisAdapter constructor.
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.RedisConfig('Host','localhost','TTL',600);
    %
    %   See also: OceanData.Adapters.RedisAdapter

    properties (SetAccess = private)
        Host     (1,1) string  = "localhost"
        Port     (1,1) double  = 6379
        Password (1,1) string  = ""          % empty → no AUTH
        Db       (1,1) double  = 0           % Redis database index (0-15)
        TTL      (1,1) double  = 300         % default time-to-live, seconds (0 = no expiry)
        Prefix   (1,1) string  = "ocean:"    % key namespace prefix
    end

    methods
        function obj = RedisConfig(opts)
            arguments
                opts.Host     (1,1) string = "localhost"
                opts.Port     (1,1) double = 6379
                opts.Password (1,1) string = ""
                opts.Db       (1,1) double = 0
                opts.TTL      (1,1) double = 300
                opts.Prefix   (1,1) string = "ocean:"
            end
            obj.Host     = opts.Host;
            obj.Port     = opts.Port;
            obj.Password = opts.Password;
            obj.Db       = opts.Db;
            obj.TTL      = opts.TTL;
            obj.Prefix   = opts.Prefix;
        end

        function str = describe(obj)
            str = sprintf('redis://%s:%d (db=%d, ttl=%ds, prefix=%s)', ...
                obj.Host, obj.Port, obj.Db, obj.TTL, obj.Prefix);
            fprintf('%s\n', str);
        end
    end
end
