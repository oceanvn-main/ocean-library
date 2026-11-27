classdef (Abstract) AbstractAccumulator < OceanEngine.Accumulators.IAccumulator
    % ABSTRACTACCUMULATOR  Abstract base class for online statistics accumulators.
    %
    %   Provides common matrix allocation, subscript indexing parsing, table
    %   getters/setters, and reset logic for all online accumulators using the
    %   Template Method Design Pattern.
    %
    %   Package: OceanEngine.Accumulators
    %   See also: OceanEngine.Accumulators.IAccumulator
    %             OceanEngine.Accumulators.WeightedMeanAccumulator
    %             OceanEngine.Accumulators.ExponentialMovingAverage
    %             OceanEngine.Accumulators.WelfordVarianceAccumulator

    properties (Access = protected)
        ValueTable   double   % Estimated value / mean table
        WeightTable  double   % Cumulative weight sum C (or counts N)
        Shape        double   % Table dimensions [rows, cols]
        InitialValue double   % Default initial value for reset
    end

    methods
        function obj = AbstractAccumulator(shape, opts)
            % ABSTRACTACCUMULATOR Constructor.
            arguments
                shape             double = [1 1]
                opts.InitValue    (1,1) double = 0.0
            end
            if isscalar(shape)
                obj.Shape = [shape 1];
            else
                obj.Shape = shape;
            end
            obj.InitialValue = opts.InitValue;
        end

        function update(obj, sample, weight, varargin)
            % UPDATE  Template method for online incremental update.
            %
            % Syntax:
            %   acc.update(sample)                 % weight = 1.0, full array
            %   acc.update(sample, weight)         % custom weight, full array
            %   acc.update(sample, weight, s_idx)  % 1D vector index s_idx
            %   acc.update(sample, weight, s, a)   % 2D matrix index (s, a)
            arguments
                obj
                sample (1,1) double
                weight (1,1) double = 1.0
            end
            arguments (Repeating)
                varargin
            end

            if isempty(varargin)
                % Full array / scalar update
                obj.updateFullArray(sample, weight);
            elseif numel(varargin) == 1
                % 1D Index update
                idx = varargin{1};
                obj.updateIndexed(sample, weight, idx);
            elseif numel(varargin) == 2
                % 2D Matrix subscript update (s, a)
                s_idx = varargin{1};
                a_idx = varargin{2};
                obj.updateSubscript2D(sample, weight, s_idx, a_idx);
            else
                error('AbstractAccumulator:InvalidIndexing', 'Supported indexing up to 2D.');
            end
        end

        function val = getValue(obj)
            % GETVALUE  Returns current value table estimate.
            val = obj.ValueTable;
        end

        function w = getWeight(obj)
            % GETWEIGHT  Returns current cumulative weight table C or counts N.
            w = obj.WeightTable;
        end

        function setValueTable(obj, val)
            % SETVALUETABLE  Direct setter for value table.
            arguments
                obj
                val double
            end
            obj.ValueTable = val;
        end

        function setWeightTable(obj, w)
            % SETWEIGHTTABLE  Direct setter for weight table.
            arguments
                obj
                w double
            end
            obj.WeightTable = w;
        end

        function reset(obj)
            % RESET  Resets tables to zero weights and initial values.
            obj.ValueTable  = repmat(obj.InitialValue, obj.Shape);
            obj.WeightTable = zeros(obj.Shape);
        end
    end

    methods (Access = protected, Abstract)
        updateFullArray(obj, sample, weight)
        updateIndexed(obj, sample, weight, idx)
        updateSubscript2D(obj, sample, weight, s_idx, a_idx)
    end
end
