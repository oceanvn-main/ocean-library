classdef ExponentialMovingAverage < OceanEngine.Accumulators.AbstractAccumulator
    % EXPONENTIALMOVINGAVERAGE  Online constant step-size exponential moving average accumulator.
    %
    %   Implements the constant step-size exponential moving average update rule (Sutton & Barto Sec 2.5):
    %     V_n = V_{n-1} + alpha * (X_n - V_{n-1})
    %
    %   Useful for non-stationary environments where recent rewards/samples are weighted higher.
    %
    %   Package: OceanEngine.Accumulators
    %   Reference: Sutton & Barto (2018) Reinforcement Learning, Section 2.5
    %
    %   See also: OceanEngine.Accumulators.AbstractAccumulator
    %             OceanEngine.Accumulators.WeightedMeanAccumulator

    properties (Access = private)
        Alpha double   % Constant step-size parameter in (0, 1]
    end

    methods
        function obj = ExponentialMovingAverage(shape, opts)
            % EXPONENTIALMOVINGAVERAGE Constructor.
            %
            % Inputs:
            %   shape          - Array dimensions [rows, cols] or scalar N (default [1 1])
            %   opts.Alpha     - Step size parameter in (0, 1] (default 0.1)
            %   opts.InitValue - Initial value for value table (default 0.0)
            arguments
                shape             double = [1 1]
                opts.Alpha        (1,1) double = 0.1
                opts.InitValue    (1,1) double = 0.0
            end
            if opts.Alpha <= 0 || opts.Alpha > 1
                error('ExponentialMovingAverage:InvalidAlpha', 'Alpha must be in range (0, 1].');
            end
            obj@OceanEngine.Accumulators.AbstractAccumulator(shape, 'InitValue', opts.InitValue);
            obj.Alpha = opts.Alpha;
            obj.reset();
        end

        function setAlpha(obj, alpha)
            % SETALPHA  Updates step size parameter alpha.
            arguments
                obj
                alpha (1,1) double
            end
            if alpha <= 0 || alpha > 1
                error('ExponentialMovingAverage:InvalidAlpha', 'Alpha must be in range (0, 1].');
            end
            obj.Alpha = alpha;
        end

        function a = getAlpha(obj)
            % GETALPHA  Returns step size parameter alpha.
            a = obj.Alpha;
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Returns pedagogical reflection metadata.
            meta = struct(...
                'Name', 'Exponential Moving Average', ...
                'SymbolicLatex', 'V_n = V_{n-1} + \alpha (X_n - V_{n-1})', ...
                'Formula', 'V_n = V_{n-1} + \alpha (X_n - V_{n-1})', ...
                'TextbookRef', 'Sutton & Barto (2018) Section 2.5', ...
                'Alpha', obj.Alpha, ...
                'Shape', obj.Shape, ...
                'Description', 'Constant step-size tracking accumulator for non-stationary processes.', ...
                'Symbols', [ ...
                    struct('Symbol', 'V_n', 'Description', 'Estimated value/mean table at step n'), ...
                    struct('Symbol', 'V_{n-1}', 'Description', 'Previous estimated value table at step n-1'), ...
                    struct('Symbol', '\alpha', 'Description', 'Constant step-size parameter in (0, 1]'), ...
                    struct('Symbol', 'X_n', 'Description', 'Sample return/observation at step n') ...
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
                    'value',      struct('Domain', 'estimated-mean', 'DType', 'real', 'Shape', obj.Shape), ...
                    'step_count', struct('Domain', 'step-count',     'DType', 'int',  'Shape', obj.Shape)...
                ), ...
                'DiagramSymbol', struct(...
                    'Style',     'SimulinkBlock', ...
                    'LabelText', 'Exponential Moving Average', ...
                    'Formula',   'V_n = V_{n-1} + \alpha (X_n - V_{n-1})', ...
                    'IconPath',  '' ...
                )...
            );
        end
    end

    methods (Access = protected)
        function updateFullArray(obj, sample, weight)
            obj.WeightTable = obj.WeightTable + 1;
            obj.ValueTable = obj.ValueTable + obj.Alpha * (sample - obj.ValueTable);
        end

        function updateIndexed(obj, sample, weight, idx)
            obj.WeightTable(idx) = obj.WeightTable(idx) + 1;
            obj.ValueTable(idx) = obj.ValueTable(idx) + obj.Alpha * (sample - obj.ValueTable(idx));
        end

        function updateSubscript2D(obj, sample, weight, s_idx, a_idx)
            obj.WeightTable(s_idx, a_idx) = obj.WeightTable(s_idx, a_idx) + 1;
            obj.ValueTable(s_idx, a_idx) = obj.ValueTable(s_idx, a_idx) + ...
                obj.Alpha * (sample - obj.ValueTable(s_idx, a_idx));
        end
    end
end
