classdef WelfordVarianceAccumulator < OceanEngine.Accumulators.AbstractAccumulator
    % WELFORDVARIANCEACCUMULATOR  Online numerically stable mean & variance accumulator.
    %
    %   Implements Welford's algorithm (1962) with weighted sample extension:
    %     C_n  = C_{n-1} + W_n
    %     d1   = X_n - mu_{n-1}
    %     mu_n = mu_{n-1} + (W_n / C_n) * d1
    %     d2   = X_n - mu_n
    %     M2_n = M2_{n-1} + W_n * d1 * d2
    %
    %   Provides O(1) memory online estimation of Mean, Variance, and Std Dev.
    %
    %   Package: OceanEngine.Accumulators
    %   Reference: Welford (1962), "Note on a method for calculating corrected sums of squares"
    %
    %   See also: OceanEngine.Accumulators.AbstractAccumulator
    %             OceanEngine.Accumulators.IAccumulator

    properties (Access = private)
        M2Table double   % Sum of squared differences M2
    end

    methods
        function obj = WelfordVarianceAccumulator(shape)
            % WELFORDVARIANCEACCUMULATOR Constructor.
            %
            % Inputs:
            %   shape - Array dimensions [rows, cols] or scalar N (default [1 1])
            arguments
                shape double = [1 1]
            end
            obj@OceanEngine.Accumulators.AbstractAccumulator(shape);
            obj.reset();
        end

        function var_val = getVariance(obj, opts)
            % GETVARIANCE  Returns online variance table.
            %   opts.Type - 'population' (divide by C) or 'sample' (divide by C - 1)
            arguments
                obj
                opts.Type (1,1) string {mustBeMember(opts.Type, ["population", "sample"])} = "population"
            end
            var_val = zeros(obj.Shape);
            if opts.Type == "population"
                valid = obj.WeightTable > 0;
                var_val(valid) = obj.M2Table(valid) ./ obj.WeightTable(valid);
            else
                valid = obj.WeightTable > 1;
                var_val(valid) = obj.M2Table(valid) ./ (obj.WeightTable(valid) - 1);
            end
        end

        function std_val = getStd(obj, opts)
            % GETSTD  Returns online standard deviation table.
            arguments
                obj
                opts.Type (1,1) string = "population"
            end
            std_val = sqrt(obj.getVariance('Type', opts.Type));
        end

        function reset(obj)
            % RESET  Resets mean, M2, and weight tables to zero.
            reset@OceanEngine.Accumulators.AbstractAccumulator(obj);
            obj.M2Table = zeros(obj.Shape);
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Returns pedagogical reflection metadata.
            meta = struct(...
                'Name', 'Welford Online Variance Accumulator', ...
                'SymbolicLatex', 'M_{2,n} = M_{2,n-1} + W_n (X_n - \mu_{n-1})(X_n - \mu_n)', ...
                'Formula', 'M_{2,n} = M_{2,n-1} + W_n (X_n - \mu_{n-1})(X_n - \mu_n)', ...
                'TextbookRef', 'Welford (1962), Technometrics 4(3)', ...
                'Shape', obj.Shape, ...
                'Description', 'Numerically stable single-pass online mean and variance accumulator.', ...
                'Symbols', [ ...
                    struct('Symbol', 'M_{2,n}', 'Description', 'Sum of squared differences from mean at step n'), ...
                    struct('Symbol', 'M_{2,n-1}', 'Description', 'Previous sum of squared differences at step n-1'), ...
                    struct('Symbol', '\mu_n', 'Description', 'Sample mean table at step n'), ...
                    struct('Symbol', '\mu_{n-1}', 'Description', 'Previous sample mean table at step n-1'), ...
                    struct('Symbol', 'W_n', 'Description', 'Sample weight at step n'), ...
                    struct('Symbol', 'X_n', 'Description', 'Sample observation at step n') ...
                ] ...
            );
        end

        function spec = getDiagramSpec(obj)
            % GETDIAGRAMSPEC  Returns Pillar 3 diagram representation (Ports & Canvas Symbol).
            spec = struct(...
                'Inputs', struct(...
                    'sample', struct('Domain', 'sample-stream',      'DType', 'real', 'Shape', obj.Shape), ...
                    'weight', struct('Domain', 'importance-weight',  'DType', 'real', 'Shape', [1 1]), ...
                    'index',  struct('Domain', 'state-action-index', 'DType', 'int',  'Shape', [1 2])...
                ), ...
                'Outputs', struct(...
                    'mean',       struct('Domain', 'estimated-mean',     'DType', 'real', 'Shape', obj.Shape), ...
                    'variance',   struct('Domain', 'estimated-variance', 'DType', 'real', 'Shape', obj.Shape), ...
                    'std_dev',    struct('Domain', 'estimated-std',      'DType', 'real', 'Shape', obj.Shape), ...
                    'cum_weight', struct('Domain', 'weight-sum',         'DType', 'real', 'Shape', obj.Shape)...
                ), ...
                'DiagramSymbol', struct(...
                    'Style',     'SimulinkBlock', ...
                    'LabelText', 'Welford Variance Accumulator', ...
                    'Formula',   'M_{2,n} = M_{2,n-1} + W_n (X_n - \mu_{n-1})(X_n - \mu_n)', ...
                    'IconPath',  '' ...
                )...
            );
        end
    end

    methods (Access = protected)
        function updateFullArray(obj, sample, weight)
            obj.WeightTable = obj.WeightTable + weight;
            valid_mask = obj.WeightTable > 0;
            if any(valid_mask, 'all')
                d1 = sample - obj.ValueTable(valid_mask);
                step_size = weight ./ obj.WeightTable(valid_mask);
                obj.ValueTable(valid_mask) = obj.ValueTable(valid_mask) + step_size .* d1;
                d2 = sample - obj.ValueTable(valid_mask);
                obj.M2Table(valid_mask) = obj.M2Table(valid_mask) + weight .* d1 .* d2;
            end
        end

        function updateIndexed(obj, sample, weight, idx)
            obj.WeightTable(idx) = obj.WeightTable(idx) + weight;
            c_val = obj.WeightTable(idx);
            if c_val > 0
                d1 = sample - obj.ValueTable(idx);
                obj.ValueTable(idx) = obj.ValueTable(idx) + (weight / c_val) * d1;
                d2 = sample - obj.ValueTable(idx);
                obj.M2Table(idx) = obj.M2Table(idx) + weight * d1 * d2;
            end
        end

        function updateSubscript2D(obj, sample, weight, s_idx, a_idx)
            obj.WeightTable(s_idx, a_idx) = obj.WeightTable(s_idx, a_idx) + weight;
            c_val = obj.WeightTable(s_idx, a_idx);
            if c_val > 0
                d1 = sample - obj.ValueTable(s_idx, a_idx);
                obj.ValueTable(s_idx, a_idx) = obj.ValueTable(s_idx, a_idx) + (weight / c_val) * d1;
                d2 = sample - obj.ValueTable(s_idx, a_idx);
                obj.M2Table(s_idx, a_idx) = obj.M2Table(s_idx, a_idx) + weight * d1 * d2;
            end
        end
    end
end
