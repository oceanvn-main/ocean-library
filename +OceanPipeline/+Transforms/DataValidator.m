classdef DataValidator < OceanPipeline.Transforms.IDataTransform
    % DATAVALIDATOR  Validate data against a SchemaRegistry column spec.
    %
    %   Returns a ValidationReport struct — does NOT modify data.
    %   apply() returns data unchanged (pass-through); use report for gating.
    %
    %   CHECKS PERFORMED:
    %     - Column count matches schema
    %     - No NaN in non-nullable columns
    %     - Values within [Min, Max] range (if specified in DWSchema)
    %     - Type compatibility (numeric expected)
    %
    %   USAGE:
    %     reg = OceanData.DataArch.SchemaRegistry();
    %     reg.register('users', schema);
    %
    %     val   = OceanPipeline.Transforms.DataValidator(reg, 'users');
    %     state = val.fit(X_train);          % learns nothing — stateless
    %     [~, report] = val.apply(X_test, state);
    %     if ~report.isValid
    %         disp(report.summary);
    %     end
    %
    %   See also: OceanData.DataArch.SchemaRegistry
    %             OceanData.DataArch.DWSchema
    %             OceanPipeline.Transforms.IDataTransform

    properties (SetAccess = private)
        Registry  OceanData.DataArch.SchemaRegistry
        TableName (1,1) string
    end

    methods
        function obj = DataValidator(registry, tableName)
            arguments
                registry  OceanData.DataArch.SchemaRegistry
                tableName (1,1) string
            end
            obj.Registry  = registry;
            obj.TableName = tableName;
        end

        % -------------------------------------------------------------- %
        function state = fit(obj, data) %#ok<INUSD>
            % FIT  Stateless — just captures schema for apply().
            schema     = obj.Registry.get(obj.TableName);
            state.schema    = schema;
            state.tableName = char(obj.TableName);
        end

        % -------------------------------------------------------------- %
        function [output, report] = apply(~, data, state)
            % APPLY  Validate data against schema. Returns data unchanged.
            %   output — same as input data (pass-through)
            %   report — struct(isValid, violations, summary)
            arguments
                ~
                data  double
                state struct
            end
            schema     = state.schema;
            violations = {};

            % 1. Column count
            expectedCols = numel(schema.ColumnNames);
            if size(data, 2) ~= expectedCols
                violations{end+1} = sprintf( ...
                    'Column count mismatch: expected %d, got %d.', ...
                    expectedCols, size(data, 2));
            end

            % 2. Per-column checks (only if column count matches)
            if size(data, 2) == expectedCols
                for c = 1:expectedCols
                    col  = data(:, c);
                    name = schema.ColumnNames{c};

                    % Nullable check
                    if ~schema.Nullable(c) && any(isnan(col))
                        violations{end+1} = sprintf( ...
                            'Column "%s": contains NaN but is non-nullable.', name);
                    end

                    % Range check (if schema provides bounds)
                    if isfield(schema, 'Min') && ~isnan(schema.Min(c))
                        badLow = col(~isnan(col)) < schema.Min(c);
                        if any(badLow)
                            violations{end+1} = sprintf( ...
                                'Column "%s": %d value(s) below Min=%.4g.', ...
                                name, sum(badLow), schema.Min(c));
                        end
                    end
                    if isfield(schema, 'Max') && ~isnan(schema.Max(c))
                        badHigh = col(~isnan(col)) > schema.Max(c);
                        if any(badHigh)
                            violations{end+1} = sprintf( ...
                                'Column "%s": %d value(s) above Max=%.4g.', ...
                                name, sum(badHigh), schema.Max(c));
                        end
                    end
                end
            end

            % Build report
            report.isValid    = isempty(violations);
            report.violations = violations;
            if report.isValid
                report.summary = sprintf('[%s] VALID — %d rows, %d cols.', ...
                    state.tableName, size(data,1), size(data,2));
            else
                report.summary = sprintf('[%s] INVALID — %d violation(s):\n  %s', ...
                    state.tableName, numel(violations), ...
                    strjoin(violations, '\n  '));
            end

            output = data;  % pass-through — validator never modifies data
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            fprintf('[DataValidator] table="%s"\n', obj.TableName);
        end
    end
end
