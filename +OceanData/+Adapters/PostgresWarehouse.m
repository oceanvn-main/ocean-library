classdef PostgresWarehouse < OceanData.DataArch.IDataStore
    % POSTGRESWAREHOUSE  Schema-enforced DataWarehouse backed by PostgreSQL.
    %
    %   Implements the same IDataStore interface as DataWarehouse.m but
    %   persists data into real PostgreSQL tables instead of .mat files.
    %
    %   DESIGN (mirrors DataWarehouse.m exactly):
    %     createTable(name, schema) → CREATE TABLE IF NOT EXISTS
    %     insert(name, row)         → DWSchema.validate() then INSERT
    %     query(name, filter)       → SELECT ... WHERE (filter → SQL)
    %     delete(name)              → DELETE FROM (keeps table structure)
    %     rowCount(name)            → SELECT COUNT(*)
    %
    %   CO-EXISTENCE:
    %     Identical API to DataWarehouse — swap by changing one line:
    %       dw = OceanData.DataArch.DataWarehouse(...)      % file-based
    %       dw = OceanData.Adapters.PostgresWarehouse(...)  % Postgres
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.DbConfig();
    %     dw  = OceanData.Adapters.PostgresWarehouse(cfg);
    %     dw.connect();
    %     dw.createTable('users', usersSchema);
    %     dw.insert('users', struct('age',35,'income',60000,'label',true));
    %     rows = dw.query('users', 'age > 30');
    %     dw.disconnect();
    %
    %   See also: OceanData.DataArch.DataWarehouse (file-based equivalent)
    %             OceanData.Adapters.PostgresAdapter

    properties (SetAccess = private)
        Adapter_    OceanData.Adapters.PostgresAdapter
        schemas_    containers.Map   % tableName → DWSchema
    end

    methods
        function obj = PostgresWarehouse(config)
            arguments
                config OceanData.Adapters.DbConfig
            end
            obj.Adapter_  = OceanData.Adapters.PostgresAdapter(config);
            obj.schemas_  = containers.Map('KeyType','char','ValueType','any');
        end

        function connect(obj)
            obj.Adapter_.connect();
        end

        function disconnect(obj)
            obj.Adapter_.disconnect();
        end

        % -------------------------------------------------------------- %
        function createTable(obj, tableName, schema)
            % CREATETABLE  Register schema and CREATE TABLE IF NOT EXISTS in Postgres.
            arguments
                obj
                tableName (1,1) string
                schema    OceanData.DataArch.DWSchema
            end
            key = char(tableName);
            obj.schemas_(key) = schema;

            colDefs = obj.buildColumnDDL_(schema);
            sql = sprintf('CREATE TABLE IF NOT EXISTS %s (%s);', ...
                obj.quoteIdent_(key), colDefs);
            obj.Adapter_.execute(sql);
            fprintf('[PostgresWarehouse] Table ready: %s\n', key);
        end

        % -------------------------------------------------------------- %
        function insert(obj, tableName, row)
            % INSERT  Validate row against DWSchema, then INSERT into Postgres.
            key    = char(tableName);
            schema = obj.getSchema_(key);
            schema.validate(row);               % reject before touching DB

            fields    = fieldnames(row);
            colList   = strjoin(cellfun(@obj.quoteIdent_, fields, ...
                'UniformOutput', false), ', ');
            valList   = strjoin(obj.rowToSqlValues_(row, schema), ', ');

            sql = sprintf('INSERT INTO %s (%s) VALUES (%s);', ...
                obj.quoteIdent_(key), colList, valList);
            obj.Adapter_.execute(sql);
        end

        % -------------------------------------------------------------- %
        function rows = query(obj, tableName, filterExpr)
            % QUERY  SELECT * FROM table [WHERE filter].
            %   filterExpr — same syntax as DataWarehouse: 'age > 30'
            %   Returns MATLAB struct array (empty if no rows).
            arguments
                obj
                tableName  (1,1) string
                filterExpr (1,1) string = ""
            end
            key    = char(tableName);
            schema = obj.getSchema_(key);
            if filterExpr == ""
                sql = sprintf('SELECT * FROM %s;', obj.quoteIdent_(key));
            else
                wherePart = obj.filterToSql_(filterExpr, schema);
                sql = sprintf('SELECT * FROM %s WHERE %s;', ...
                    obj.quoteIdent_(key), wherePart);
            end
            T    = obj.Adapter_.fetch(sql);
            rows = obj.tableToStructArray_(T);
        end

        % -------------------------------------------------------------- %
        function n = rowCount(obj, tableName)
            key = char(tableName);
            sql = sprintf('SELECT COUNT(*) AS n FROM %s;', obj.quoteIdent_(key));
            rows = obj.Adapter_.fetch(sql);
            n   = rows(1).n;
        end

        % -------------------------------------------------------------- %
        function schema = getSchema(obj, tableName)
            schema = obj.getSchema_(char(tableName));
        end

        % ================================================================
        % IDataStore interface
        % ================================================================

        function put(obj, key, data, ~)
            if ~obj.schemas_.isKey(char(key))
                error('OceanData:Adapters:PostgresWarehouse:TableNotFound', ...
                    'No schema for "%s". Call createTable() first.', key);
            end
            obj.insert(key, data);
        end

        function entry = get(obj, key)
            rows  = obj.query(key);
            entry = struct('data', rows, 'meta', struct(), ...
                'timestamp', datetime('now'), 'key', key);
        end

        function keys = list(obj)
            keys = obj.schemas_.keys();
        end

        function tf = exists(obj, key)
            sql = sprintf( ...
                "SELECT to_regclass('%s') IS NOT NULL AS ex;", char(key));
            rows = obj.Adapter_.fetch(sql);
            tf   = logical(rows(1).ex);
        end

        function delete(obj, key)
            % DELETE  Removes all rows but keeps the table structure.
            sql = sprintf('DELETE FROM %s;', obj.quoteIdent_(char(key)));
            obj.Adapter_.execute(sql);
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            keys = obj.schemas_.keys();
            fprintf('PostgresWarehouse (%d tables):\n', numel(keys));
            for i = 1:numel(keys)
                n = obj.rowCount(keys{i});
                fprintf('  %-20s  %d rows\n', keys{i}, n);
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function schema = getSchema_(obj, key)
            if ~obj.schemas_.isKey(key)
                error('OceanData:Adapters:PostgresWarehouse:TableNotFound', ...
                    'No schema for "%s". Call createTable() first.', key);
            end
            schema = obj.schemas_(key);
        end

        function ddl = buildColumnDDL_(~, schema)
            % Creates "col1 TYPE1 [NOT NULL], col2 TYPE2, ..."
            parts = cell(1, schema.NumColumns);
            typeMap = struct( ...
                'double',  'DOUBLE PRECISION', ...
                'integer', 'INTEGER', ...
                'logical', 'BOOLEAN', ...
                'string',  'TEXT', ...
                'category','TEXT');
            for i = 1:schema.NumColumns
                sqlType  = typeMap.(schema.ColumnTypes{i});
                nullable = '';
                if ~schema.Nullable(i)
                    nullable = ' NOT NULL';
                end
                parts{i} = sprintf('"%s" %s%s', ...
                    schema.ColumnNames{i}, sqlType, nullable);
            end
            ddl = strjoin(parts, ', ');
        end

        function vals = rowToSqlValues_(~, row, schema)
            % Converts a struct row → cell array of SQL literal strings
            vals = cell(1, schema.NumColumns);
            for i = 1:schema.NumColumns
                name = schema.ColumnNames{i};
                val  = row.(name);
                typ  = schema.ColumnTypes{i};
                switch typ
                    case {'double','integer'}
                        if isnan(val)
                            vals{i} = 'NULL';
                        else
                            vals{i} = num2str(val, '%.17g');
                        end
                    case 'logical'
                        if val, vals{i} = 'TRUE'; else, vals{i} = 'FALSE'; end
                    case {'string','category'}
                        safeVal = strrep(char(val), '''', '''''');
                        vals{i} = sprintf('''%s''', safeVal);
                end
            end
        end

        function whereSql = filterToSql_(~, filterExpr, schema)
            % Translate 'colname OP value' → SQL WHERE fragment.
            % Schema is used to detect logical columns → TRUE/FALSE literals.
            toks = regexp(char(filterExpr), ...
                '^\s*(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)\s*$', 'tokens');
            if isempty(toks)
                error('OceanData:Adapters:PostgresWarehouse:InvalidFilter', ...
                    'Cannot parse filter: "%s"', filterExpr);
            end
            col = toks{1}{1};
            op  = toks{1}{2};
            rhs = strtrim(toks{1}{3});

            % Operator translation
            switch op
                case '==', sqlOp = '=';
                case '!=', sqlOp = '<>';
                otherwise, sqlOp = op;
            end

            % Detect if filtered column is logical (BOOLEAN in Postgres)
            colIdx = find(strcmp(schema.ColumnNames, col), 1);
            isLogicalCol = ~isempty(colIdx) && ...
                strcmp(schema.ColumnTypes{colIdx}, 'logical');

            % RHS translation
            rhsNum = str2double(rhs);
            if ~isnan(rhsNum)
                if isLogicalCol
                    % PostgreSQL BOOLEAN needs TRUE/FALSE, not 1/0
                    if rhsNum ~= 0
                        rhsSql = 'TRUE';
                    else
                        rhsSql = 'FALSE';
                    end
                else
                    rhsSql = num2str(rhsNum, '%.17g');
                end
            else
                rhsSql = sprintf('''%s''', strrep(rhs, '''', ''''''));
            end

            whereSql = sprintf('"%s" %s %s', col, sqlOp, rhsSql);
        end

        function s = tableToStructArray_(~, rows)
            % fetch() already returns a struct array — just pass through.
            s = rows;
        end

        function q = quoteIdent_(~, name)
            q = sprintf('"%s"', name);
        end

    end
end
