classdef CassandraAdapter < handle
    % CASSANDRAADAPTER  Apache Cassandra via Python cassandra-driver bridge.
    %
    %   Uses the official DataStax Python cassandra-driver (3.x) through
    %   MATLAB's built-in py.* interface.  This avoids JDBC, which requires
    %   Java 11+ while MATLAB ships with Java 8.
    %
    %   PRE-REQUISITE (one-time):
    %     pip install cassandra-driver
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.DbConfig( ...
    %               'Vendor','Cassandra','Host','localhost','Port',9042, ...
    %               'Database','ocean_ks');
    %     ca  = OceanData.Adapters.CassandraAdapter(cfg);
    %     ca.connect();
    %     ca.execute('CREATE TABLE IF NOT EXISTS ...');
    %     T = ca.fetch('SELECT * FROM ocean_ks.t');
    %     ca.disconnect();
    %
    %   See also: OceanData.Adapters.CassandraWarehouse
    %             OceanData.Adapters.DbConfig

    properties (SetAccess = private)
        Config    OceanData.Adapters.DbConfig
        Bridge_   % Python module handle (cass_bridge)
    end

    methods
        function obj = CassandraAdapter(config)
            arguments
                config OceanData.Adapters.DbConfig
            end
            obj.Config  = config;
            obj.Bridge_ = obj.loadBridge_();
        end

        function connect(obj)
            % CONNECT  Open a connection to the Cassandra cluster.
            host = char(obj.Config.Host);
            port = double(obj.Config.Port);
            ks   = char(obj.Config.Database);
            try
                result = obj.Bridge_.connect(host, port, ks);
                if ~strcmp(char(result), 'OK')
                    error('CassandraAdapter:connect', 'Bridge returned: %s', char(result));
                end
                fprintf('[CassandraAdapter] Connected: %s\n', obj.Config.describe());
            catch ex
                error('CassandraAdapter:connect', 'Connection failed: %s', ex.message);
            end
        end

        function disconnect(obj)
            % DISCONNECT  Close the connection cleanly.
            try; obj.Bridge_.disconnect(); catch; end
            fprintf('[CassandraAdapter] Disconnected.\n');
        end

        function execute(obj, cql)
            % EXECUTE  Run a CQL DDL or DML statement (no result).
            try
                obj.Bridge_.execute(cql);
            catch ex
                error('CassandraAdapter:execute', 'CQL error: %s\nQuery: %s', ...
                      ex.message, cql);
            end
        end

        function T = fetch(obj, cql)
            % FETCH  Run a SELECT and return results as a MATLAB table.
            try
                jsonStr = char(obj.Bridge_.fetch(cql));
                rows    = jsondecode(jsonStr);
                if isempty(rows)
                    T = table();
                else
                    T = struct2table(rows);
                end
            catch ex
                error('CassandraAdapter:fetch', 'CQL fetch error: %s', ex.message);
            end
        end

        function insertBatch(obj, keyspace, tableName, rowsJson)
            % INSERTBATCH  Bulk insert rows provided as a JSON string.
            try
                obj.Bridge_.insert_batch(keyspace, tableName, rowsJson);
            catch ex
                error('CassandraAdapter:insertBatch', ...
                      'Batch insert failed: %s', ex.message);
            end
        end

        function ensureKeyspace(obj, ksName) %#ok<INUSD>
            % ENSUREKEYSPACE  No-op: connect() already creates the keyspace.
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function bridge = loadBridge_(obj)
            % Add the +Adapters folder to Python's module search path,
            % then import our cass_bridge module.
            adapterDir = fileparts(mfilename('fullpath'));

            % Insert adapterDir into Python sys.path if not already there
            pyPath  = py.sys.path;
            pyList  = cell(pyPath);          % convert py.list → MATLAB cell
            already = any(cellfun(@(p) strcmp(char(p), adapterDir), pyList));
            if ~already
                py.sys.path().insert(int32(0), adapterDir);
            end

            try
                % Import (or reload) the bridge module
                bridge = py.importlib.import_module('cass_bridge');
                py.importlib.reload(bridge);
            catch ex
                error('CassandraAdapter:loadBridge', ...
                      ['Failed to load cass_bridge.py.\n' ...
                       'Make sure Python is configured in MATLAB and\n' ...
                       '"pip install cassandra-driver" has been run.\n' ...
                       'Error: %s'], ex.message);
            end
        end
    end
end
