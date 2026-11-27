classdef MissingValueImputer < OceanPipeline.Transforms.IDataTransform
    % MISSINGVALUEIMPUTER  Fill NaN values using training-data statistics.
    %
    %   STRATEGIES:
    %     'mean'   — fill with per-column mean (default)
    %     'median' — fill with per-column median
    %     'zero'   — fill with 0
    %     'drop'   — remove rows containing any NaN (fit only)
    %
    %   ANTI-LEAKAGE:
    %     fit()   computes statistics from training data only.
    %     apply() uses those pre-computed values on any split.
    %
    %   USAGE:
    %     imp   = OceanPipeline.Transforms.MissingValueImputer('Strategy','median');
    %     state = imp.fit(X_train);
    %     X_train_clean = imp.apply(X_train, state);
    %     X_test_clean  = imp.apply(X_test,  state);
    %
    %   See also: OceanPipeline.Transforms.IDataTransform

    properties (SetAccess = private)
        Strategy (1,1) string = "mean"
    end

    methods
        function obj = MissingValueImputer(opts)
            arguments
                opts.Strategy (1,1) string ...
                    {mustBeMember(opts.Strategy, ...
                    {'mean','median','zero','drop'})} = "mean"
            end
            obj.Strategy = opts.Strategy;
        end

        % -------------------------------------------------------------- %
        function state = fit(obj, data)
            % FIT  Learn fill values from training data.
            % state.strategy   — string
            % state.fillValues — (1 × p) learned fill value per column
            % state.nCols      — p (for validation in apply)
            arguments
                obj
                data double
            end
            state.strategy = char(obj.Strategy);
            state.nCols    = size(data, 2);

            switch obj.Strategy
                case 'mean'
                    state.fillValues = mean(data, 1, 'omitnan');
                case 'median'
                    state.fillValues = median(data, 1, 'omitnan');
                case 'zero'
                    state.fillValues = zeros(1, size(data, 2));
                case 'drop'
                    state.fillValues = [];   % no fill — rows will be removed
            end
        end

        % -------------------------------------------------------------- %
        function output = apply(obj, data, state) %#ok<INUSL>
            % APPLY  Fill NaN using state learned during fit().
            arguments
                obj
                data  double
                state struct
            end
            if size(data, 2) ~= state.nCols
                error('OceanPipeline:Transforms:MissingValueImputer:DimMismatch', ...
                    'Expected %d columns, got %d.', state.nCols, size(data,2));
            end

            if strcmp(state.strategy, 'drop')
                % Remove rows that contain any NaN
                output = data(~any(isnan(data), 2), :);
            else
                output = data;
                for col = 1:size(data, 2)
                    mask = isnan(output(:, col));
                    if any(mask)
                        output(mask, col) = state.fillValues(col);
                    end
                end
            end
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            fprintf('[MissingValueImputer] strategy="%s"\n', obj.Strategy);
        end
    end
end
