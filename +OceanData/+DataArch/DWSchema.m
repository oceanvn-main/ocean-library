classdef DWSchema
    % DWSCHEMA  Immutable column specification for a DataWarehouse table.
    %   Defines the Rules
    %   Defines the names, data types, nullability, and range constraints
    %   for every column in a structured table. Used by DataWarehouse to
    %   validate each row before committing to disk.
    %
    %   DESIGN:
    %     Value class — copyable, comparable, fully immutable after construction.
    %     Intended to be built once (at pipeline design time) and shared.
    %
    %   SUPPORTED TYPES:
    %     'double'   — numeric (scalar per cell)
    %     'integer'  — integer numeric (enforced via floor comparison)
    %     'logical'  — 0/1 boolean
    %     'string'   — char or MATLAB string
    %     'category' — string constrained to a fixed set of allowed values
    %
    %   USAGE:
    %     schema = OceanData.DataArch.DWSchema( ...
    %         {'age', 'income', 'label', 'gender'}, ...
    %         {'double', 'double', 'logical', 'category'}, ...
    %         'Nullable',  [false, false, false, true], ...
    %         'MinValues', [0, 0, NaN, NaN], ...
    %         'MaxValues', [120, 1e7, NaN, NaN], ...
    %         'Categories', struct('gender', {{'M','F','Other'}}));
    %
    %     schema.validate(row)     % struct → throws if violation
    %     schema.describe()        % prints column summary
    %
    %   See also: OceanData.DataArch.SchemaRegistry
    %             OceanData.DataArch.DataWarehouse

    % ================================================================== %
    properties (SetAccess = immutable)
        ColumnNames   (1,:) cell    % column name strings
        ColumnTypes   (1,:) cell    % 'double'|'integer'|'logical'|'string'|'category'
        Nullable      (1,:) logical % true = NaN/empty allowed
        MinValues     (1,:) double  % NaN = no minimum
        MaxValues     (1,:) double  % NaN = no maximum
        Categories    struct        % struct of allowed category sets, by column name
        NumColumns    (1,1) double
    end

    % ================================================================== %
    methods

        function obj = DWSchema(names, types, opts)
            % DWSCHEMA  Construct a column schema specification.
            arguments
                names   (1,:) cell
                types   (1,:) cell
                opts.Nullable   (1,:) logical = false(1, numel(names))
                opts.MinValues  (1,:) double  = nan(1, numel(names))
                opts.MaxValues  (1,:) double  = nan(1, numel(names))
                opts.Categories struct        = struct()
            end
            assert(numel(names) == numel(types), ...
                'OceanData:DataArch:DWSchema:DimMismatch', ...
                'names and types must have the same length.');
            assert(numel(opts.Nullable) == numel(names), ...
                'OceanData:DataArch:DWSchema:DimMismatch', ...
                'Nullable must have length == number of columns.');

            validTypes = {'double','integer','logical','string','category'};
            for i = 1:numel(types)
                assert(ismember(types{i}, validTypes), ...
                    'OceanData:DataArch:DWSchema:UnknownType', ...
                    'Column "%s": unknown type "%s".', names{i}, types{i});
            end

            obj.ColumnNames = names;
            obj.ColumnTypes = types;
            obj.Nullable    = opts.Nullable;
            obj.MinValues   = opts.MinValues;
            obj.MaxValues   = opts.MaxValues;
            obj.Categories  = opts.Categories;
            obj.NumColumns  = numel(names);
        end

        % -------------------------------------------------------------- %
        function validate(obj, row)
            % VALIDATE  Check a single row struct against this schema.
            %
            %   row — struct with one field per column name
            %
            %   Throws OceanData:DataArch:DWSchema:ValidationFailed on any violation.
            for i = 1:obj.NumColumns
                name = obj.ColumnNames{i};
                typ  = obj.ColumnTypes{i};

                % Field must exist
                if ~isfield(row, name)
                    error('OceanData:DataArch:DWSchema:MissingColumn', ...
                        'Row is missing required column "%s".', name);
                end
                val = row.(name);

                % Nullability check
                isNull = obj.isNullValue_(val, typ);
                if isNull && ~obj.Nullable(i)
                    error('OceanData:DataArch:DWSchema:NullViolation', ...
                        'Column "%s" is NOT nullable but received a null value.', name);
                end
                if isNull, continue; end  % remaining checks skip for null

                % Type check
                obj.checkType_(val, typ, name);

                % Range check (numeric only)
                if ismember(typ, {'double','integer'})
                    if ~isnan(obj.MinValues(i)) && val < obj.MinValues(i)
                        error('OceanData:DataArch:DWSchema:RangeViolation', ...
                            'Column "%s": value %g < minimum %g.', name, val, obj.MinValues(i));
                    end
                    if ~isnan(obj.MaxValues(i)) && val > obj.MaxValues(i)
                        error('OceanData:DataArch:DWSchema:RangeViolation', ...
                            'Column "%s": value %g > maximum %g.', name, val, obj.MaxValues(i));
                    end
                end

                % Category check
                if strcmp(typ, 'category')
                    if isfield(obj.Categories, name)
                        allowed = obj.Categories.(name);
                        valStr  = char(val);
                        if ~ismember(valStr, allowed)
                            error('OceanData:DataArch:DWSchema:CategoryViolation', ...
                                'Column "%s": "%s" is not in allowed set {%s}.', ...
                                name, valStr, strjoin(allowed, ', '));
                        end
                    end
                end
            end
        end

        % -------------------------------------------------------------- %
        function tf = isCompatible(obj, otherSchema)
            % ISCOMPATIBLE  Check if another schema has identical structure.
            tf = isequal(obj.ColumnNames, otherSchema.ColumnNames) && ...
                 isequal(obj.ColumnTypes, otherSchema.ColumnTypes);
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            % DESCRIBE  Print a formatted column summary.
            fprintf('DWSchema (%d columns):\n', obj.NumColumns);
            fprintf('  %-20s %-10s %-9s %-8s %-8s\n', ...
                'Column', 'Type', 'Nullable', 'Min', 'Max');
            fprintf('  %s\n', repmat('-', 1, 60));
            for i = 1:obj.NumColumns
                minStr = 'N/A'; maxStr = 'N/A';
                if ~isnan(obj.MinValues(i)), minStr = num2str(obj.MinValues(i)); end
                if ~isnan(obj.MaxValues(i)), maxStr = num2str(obj.MaxValues(i)); end
                fprintf('  %-20s %-10s %-9s %-8s %-8s\n', ...
                    obj.ColumnNames{i}, obj.ColumnTypes{i}, ...
                    mat2str(obj.Nullable(i)), minStr, maxStr);
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function tf = isNullValue_(~, val, typ)
            if ismember(typ, {'double','integer'})
                tf = isscalar(val) && isnan(val);
            elseif strcmp(typ, 'logical')
                tf = isempty(val);
            else
                tf = isempty(val) || (isstring(val) && val == "");
            end
        end

        function checkType_(~, val, typ, name)
            switch typ
                case 'double'
                    if ~isnumeric(val) || ~isscalar(val)
                        error('OceanData:DataArch:DWSchema:TypeViolation', ...
                            'Column "%s": expected numeric scalar, got %s.', name, class(val));
                    end
                case 'integer'
                    if ~isnumeric(val) || ~isscalar(val)
                        error('OceanData:DataArch:DWSchema:TypeViolation', ...
                            'Column "%s": expected integer scalar, got %s.', name, class(val));
                    end
                    if val ~= floor(val)
                        error('OceanData:DataArch:DWSchema:TypeViolation', ...
                            'Column "%s": value %g is not an integer.', name, val);
                    end
                case 'logical'
                    if ~islogical(val) && ~(isnumeric(val) && ismember(val,[0 1]))
                        error('OceanData:DataArch:DWSchema:TypeViolation', ...
                            'Column "%s": expected logical (0/1), got %s.', name, class(val));
                    end
                case {'string','category'}
                    if ~ischar(val) && ~isstring(val)
                        error('OceanData:DataArch:DWSchema:TypeViolation', ...
                            'Column "%s": expected string/char, got %s.', name, class(val));
                    end
            end
        end

    end

end
