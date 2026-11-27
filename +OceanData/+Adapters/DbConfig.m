classdef DbConfig
    % DBCONFIG  Connection configuration value object.
    %
    %   Immutable struct-like class that bundles all database
    %   connection parameters in one place. Pass to any IDbAdapter
    %   constructor. Store once; reuse everywhere.
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.DbConfig( ...
    %         'Vendor',   'PostgreSQL', ...
    %         'Host',     'localhost', ...
    %         'Port',     5432, ...
    %         'Database', 'OceanLibrary', ...
    %         'User',     'ocean', ...
    %         'Password', '891999A@');
    %
    %   See also: OceanData.Adapters.PostgresAdapter

    properties (SetAccess = immutable)
        Vendor   (1,1) string = "PostgreSQL"   % MATLAB Database Toolbox vendor name
        Host     (1,1) string = "localhost"
        Port     (1,1) double = 5432
        Database (1,1) string = "OceanLibrary"
        User     (1,1) string = "ocean"
        Password (1,1) string = ""             % stored in-memory only; never persisted
    end

    methods
        function obj = DbConfig(opts)
            arguments
                opts.Vendor   (1,1) string = "PostgreSQL"
                opts.Host     (1,1) string = "localhost"
                opts.Port     (1,1) double = 5432
                opts.Database (1,1) string = "OceanLibrary"
                opts.User     (1,1) string = "ocean"
                opts.Password (1,1) string = ""
            end
            obj.Vendor   = opts.Vendor;
            obj.Host     = opts.Host;
            obj.Port     = opts.Port;
            obj.Database = opts.Database;
            obj.User     = opts.User;
            obj.Password = opts.Password;
        end

        function s = describe(obj)
            % DESCRIBE  Return a safe (no-password) summary string.
            s = sprintf('%s://%s@%s:%d/%s', ...
                obj.Vendor, obj.User, obj.Host, obj.Port, obj.Database);
        end
    end
end
