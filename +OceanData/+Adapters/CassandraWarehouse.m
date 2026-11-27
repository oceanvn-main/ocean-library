classdef CassandraWarehouse < OceanData.DataArch.IDataStore
    % CASSANDRAWAREHOUSE  Schema-enforced DataWarehouse backed by Apache Cassandra.
    %
    %   Implements the same IDataStore interface as DataWarehouse.m and
    %   PostgresWarehouse.m, but persists data into Cassandra tables.
    %
    %   CASSANDRA KEY DIFFERENCES vs SQL:
    %     - "Database" in DbConfig = Cassandra Keyspace
    %     - Every table MUST have a PRIMARY KEY  (auto-injected as row_id uuid)
    %     - No WHERE filtering without ALLOW FILTERING (safe for small tables)
    %     - Supports insertBatch() for high-throughput ingestion
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.DbConfig( ...
    %               'Vendor','Cassandra','Host','localhost','Port',9042, ...
    %               'Database','ocean_ks','User','cassandra','Password','cassandra');
    %     dw  = OceanData.Adapters.CassandraWarehouse(cfg);
    %     dw.connect();
    %     dw.createTable('observations', obsSchema);
    %     dw.insert('observations', struct('x1',0.5,'x2',1.2,'label',true));
    %     rows = dw.query('observations', 'x1 > 0');
    %     dw.disconnect();
    %
    %   See also: OceanData.DataArch.DataWarehouse (file-based)
    %             OceanData.Adapters.PostgresWarehouse (SQL)

    properties (SetAccess = private)
        Adapter_  OceanData.Adapters.CassandraAdapter
        schemas_  containers.Map   % tableName → DWSchema
        keyspace_ string
    end

    methods
        function obj = CassandraWarehouse(config)
            arguments
                config OceanData.Adapters.DbConfig
            end
            obj.Adapter_  = OceanData.Adapters.CassandraAdapter(config);
            obj.schemas_  = containers.Map('KeyType','char','ValueType','any');
            obj.keyspace_ = config.Database;
        end

        function connect(obj)
            % CONNECT  Open the connection.
            %   cass_bridge.connect() automatically ensures the keyspace
            %   exists before returning, so no manual bootstrap is needed.
            obj.Adapter_.connect();
        end

        function disconnect(obj)
            % DISCONNECT  Close the connection.
            obj.Adapter_.disconnect();
        end

        % ---------------------------------------------------------------- %
        % IDataStore CONTRACT — key-value interface
        % ---------------------------------------------------------------- %
        function put(obj, key, data, meta) %#ok<INUSD>
            % PUT  Store data struct under a Cassandra table named `key`.
            %   For CassandraWarehouse, tables must be created first with
            %   createTable(). This method inserts a single row struct.
            if ~obj.schemas_.isKey(char(key))
                error('CassandraWarehouse:put', ...
                      'Table "%s" not created. Call createTable() first.', key);
            end
            obj.insert(key, data);
        end

        function entry = get(obj, key)
            % GET  Retrieve all rows from table `key` as entry.data.
            if ~obj.tableExists_(key)
                error('OceanData:DataArch:CassandraWarehouse:KeyNotFound', ...
                      'Table "%s" not found in keyspace "%s".', key, obj.keyspace_);
            end
            rows = obj.query(key, '');
            entry.data      = rows;
            entry.meta      = struct();
            entry.timestamp = datetime('now');
            entry.key       = char(key);
        end

        function keys = list(obj)
            % LIST  Return all table names in this keyspace.
            cql = sprintf([ ...
                'SELECT table_name FROM system_schema.tables ' ...
                'WHERE keyspace_name = ''%s'''], obj.keyspace_);
            T = obj.Adapter_.fetch(cql);
            if isempty(T) || height(T) == 0
                keys = {};
            else
                keys = table2cell(T(:,1));
            end
        end

        function tf = exists(obj, key)
            % EXISTS  True if a table named `key` exists in the keyspace.
            tf = obj.tableExists_(key);
        end

        function delete(obj, key)
            % DELETE  Truncate a table by key name (idempotent).
            if obj.tableExists_(key)
                obj.delete_(key);
            end
        end

        function createTable(obj, name, schema)
            % CREATETABLE  Create a Cassandra table from a DWSchema.
            %
            %   A synthetic PRIMARY KEY (row_id uuid) is always injected
            %   because Cassandra requires one. The schema fields become
            %   regular columns.
            arguments
                obj
                name   (1,1) string
                schema OceanData.DataArch.DWSchema
            end
            obj.schemas_(char(name)) = schema;
            colDefs = obj.schemaToColumnDefs_(schema);
            fullTable = sprintf('%s.%s', obj.keyspace_, name);
            cql = sprintf([ ...
                'CREATE TABLE IF NOT EXISTS %s (' ...
                'row_id uuid PRIMARY KEY, %s)'], fullTable, colDefs);
            obj.Adapter_.execute(cql);
            fprintf('[CassandraWarehouse] Table ready: %s\n', name);
        end

        function insert(obj, name, row)
            % INSERT  Validate and insert a single row struct.
            if obj.schemas_.isKey(char(name))
                schema = obj.schemas_(char(name));
                schema.validate(row);
            end
            fullTable = sprintf('%s.%s', obj.keyspace_, name);
            cql = obj.buildInsertCql_(fullTable, row);
            obj.Adapter_.execute(cql);
        end

        function insertBatch(obj, name, data)
            % INSERTBATCH  Bulk insert a struct array or MATLAB table via Python bridge.
            if istable(data)
                data = table2struct(data);
            end
            if obj.schemas_.isKey(char(name))
                schema = obj.schemas_(char(name));
                for k = 1:numel(data)
                    schema.validate(data(k));
                end
            end
            rowsJson = jsonencode(data);
            obj.Adapter_.insertBatch(char(obj.keyspace_), char(name), rowsJson);
        end

        function rows = query(obj, name, filter)
            % QUERY  Select rows; filter is appended as WHERE clause.
            arguments
                obj
                name   (1,1) string
                filter (1,1) string = ""
            end
            fullTable = sprintf('%s.%s', obj.keyspace_, name);
            cql = sprintf('SELECT * FROM %s', fullTable);
            if strlength(filter) > 0
                cql = sprintf('%s WHERE %s ALLOW FILTERING', cql, filter);
            end
            T = obj.Adapter_.fetch(cql);
            if isempty(T) || height(T) == 0
                rows = struct([]);
            else
                rows = table2struct(T);
            end
        end

        function n = rowCount(obj, name)
            % ROWCOUNT  Return the number of rows in a table.
            fullTable = sprintf('%s.%s', obj.keyspace_, name);
            T = obj.Adapter_.fetch(sprintf('SELECT COUNT(*) FROM %s', fullTable));
            if isempty(T) || height(T) == 0
                n = 0;
            else
                v = T{1,1};
                if iscell(v), v = v{1}; end
                n = str2double(char(v));
            end
        end

        function delete_(obj, name)
            % DELETE_  Truncate all rows from the table (keep structure).
            fullTable = sprintf('%s.%s', obj.keyspace_, name);
            obj.Adapter_.execute(sprintf('TRUNCATE %s', fullTable));
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function defs = schemaToColumnDefs_(~, schema)
            % Map DWSchema types to CQL column types.
            % DWSchema types: 'double','integer','logical','string','category'
            typeMap = containers.Map( ...
                {'double','integer','logical','string','category'}, ...
                {'double','int','boolean','text','text'});
            parts = {};
            for k = 1:schema.NumColumns
                colName = schema.ColumnNames{k};
                colType = schema.ColumnTypes{k};
                if typeMap.isKey(colType)
                    cqlType = typeMap(colType);
                else
                    cqlType = 'text';
                end
                parts{end+1} = sprintf('%s %s', colName, cqlType); %#ok<AGROW>
            end
            defs = strjoin(parts, ', ');
        end

        function cql = buildInsertCql_(~, fullTable, row)
            % Build a single INSERT INTO ... VALUES(...) CQL string.
            fnames = fieldnames(row);
            cols   = strjoin(fnames, ', ');
            vals   = cell(1, numel(fnames));
            for k = 1:numel(fnames)
                v = row.(fnames{k});
                if islogical(v)
                    vals{k} = mat2str(v);           % 'true'/'false'
                elseif isnumeric(v)
                    vals{k} = sprintf('%.17g', v);
                else
                    safe    = strrep(char(v), '''', '''''');
                    vals{k} = sprintf('''%s''', safe);
                end
            end
            valStr = strjoin(vals, ', ');
            uuid   = lower(char(java.util.UUID.randomUUID().toString()));
            cql    = sprintf( ...
                'INSERT INTO %s (row_id, %s) VALUES (%s, %s)', ...
                fullTable, cols, uuid, valStr);
        end

        function tf = tableExists_(obj, name)
            % Check Cassandra system_schema to see if table name exists.
            cql = sprintf([ ...
                'SELECT table_name FROM system_schema.tables ' ...
                'WHERE keyspace_name = ''%s'' AND table_name = ''%s'''], ...
                obj.keyspace_, name);
            try
                T  = obj.Adapter_.fetch(cql);
                tf = ~isempty(T) && height(T) > 0;
            catch
                tf = false;
            end
        end
    end
end

