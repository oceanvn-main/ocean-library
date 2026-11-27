classdef SchemaRegistry < handle
    % SCHEMAREGISTRY  Central registry of named DWSchema objects.
    %   Stores the Rules
    %   Stores and retrieves named schemas so that DataWarehouse, DataValidator,
    %   and pipeline components can look up column specs by table name without
    %   passing schema objects manually through the call stack.
    %
    %   DESIGN:
    %     Singleton-friendly handle class (one per pipeline context).
    %     Schemas are registered once at pipeline setup and never mutated.
    %
    %   USAGE:
    %     reg = OceanData.DataArch.SchemaRegistry();
    %
    %     % Register a schema
    %     schema = OceanData.DataArch.DWSchema({'age','label'}, {'double','logical'});
    %     reg.register('demographics', schema);
    %
    %     % Look up later
    %     s = reg.get('demographics');
    %     s.validate(row);
    %
    %     % List all registered tables
    %     reg.list()
    %
    %   See also: OceanData.DataArch.DWSchema
    %             OceanData.DataArch.DataWarehouse

    % ================================================================== %
    properties (Access = private)
        store_   containers.Map
    end

    % ================================================================== %
    methods

        function obj = SchemaRegistry()
            obj.store_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        % -------------------------------------------------------------- %
        function register(obj, tableName, schema)
            % REGISTER  Store a schema under a table name.
            %
            %   reg.register('demographics', schema)
            %
            %   Overwrites silently if already registered.
            arguments
                obj
                tableName (1,1) string
                schema    OceanData.DataArch.DWSchema
            end
            obj.store_(char(tableName)) = schema;
        end

        % -------------------------------------------------------------- %
        function schema = get(obj, tableName)
            % GET  Retrieve schema by table name.
            %
            %   schema = reg.get('demographics')
            %
            %   Throws SchemaNotFound if not registered.
            key = char(tableName);
            if ~obj.store_.isKey(key)
                error('OceanData:DataArch:SchemaRegistry:SchemaNotFound', ...
                    'No schema registered for table "%s". Call register() first.', key);
            end
            schema = obj.store_(key);
        end

        % -------------------------------------------------------------- %
        function tf = has(obj, tableName)
            % HAS  Return true if a schema is registered for this table.
            tf = obj.store_.isKey(char(tableName));
        end

        % -------------------------------------------------------------- %
        function remove(obj, tableName)
            % REMOVE  Unregister a schema.
            key = char(tableName);
            if obj.store_.isKey(key)
                obj.store_.remove(key);
            end
        end

        % -------------------------------------------------------------- %
        function names = list(obj)
            % LIST  Return all registered table names.
            names = obj.store_.keys();
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            % DESCRIBE  Print a summary of all registered schemas.
            names = obj.list();
            if isempty(names)
                fprintf('SchemaRegistry: (empty)\n');
                return;
            end
            fprintf('SchemaRegistry (%d tables):\n', numel(names));
            for i = 1:numel(names)
                s = obj.get(names{i});
                fprintf('  %-20s  %d columns: %s\n', names{i}, ...
                    s.NumColumns, strjoin(s.ColumnNames, ', '));
            end
        end

    end

end
