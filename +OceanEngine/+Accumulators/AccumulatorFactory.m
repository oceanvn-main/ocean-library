classdef AccumulatorFactory
    % ACCUMULATORFACTORY  Static factory for online statistics accumulators.
    %
    %   Instantiates online accumulator strategies by name:
    %     acc = OceanEngine.Accumulators.AccumulatorFactory.create('Weighted', [N_states 1]);
    %     acc = OceanEngine.Accumulators.AccumulatorFactory.create('EMA', [10 2], 'Alpha', 0.05);
    %     acc = OceanEngine.Accumulators.AccumulatorFactory.create('Welford', [100 1]);
    %
    %   Package: OceanEngine.Accumulators
    %   See also: OceanEngine.Accumulators.WeightedMeanAccumulator
    %             OceanEngine.Accumulators.ExponentialMovingAverage
    %             OceanEngine.Accumulators.WelfordVarianceAccumulator

    methods (Static)
        function acc = create(strategy, shape, opts)
            % CREATE  Factory method instantiating requested accumulator.
            %
            % Inputs:
            %   strategy       - (string) 'Weighted', 'SampleAverage', 'EMA', or 'Welford'
            %   shape          - Array dimensions [rows, cols] (default [1 1])
            %   opts.InitValue - Initial value for value table (default 0.0)
            %   opts.Alpha     - Step size for EMA (default 0.1)
            arguments
                strategy          (1,1) string
                shape             double = [1 1]
                opts.InitValue    (1,1) double = 0.0
                opts.Alpha        (1,1) double = 0.1
            end

            switch lower(strategy)
                case {"weighted", "weightedmean", "sampleaverage", "unweighted", "is"}
                    acc = OceanEngine.Accumulators.WeightedMeanAccumulator(shape, 'InitValue', opts.InitValue);

                case {"ema", "exponentialmovingaverage", "constantalpha"}
                    acc = OceanEngine.Accumulators.ExponentialMovingAverage(shape, 'Alpha', opts.Alpha, 'InitValue', opts.InitValue);

                case {"welford", "welfordvariance", "runningvariance", "onlinevariance"}
                    acc = OceanEngine.Accumulators.WelfordVarianceAccumulator(shape);

                otherwise
                    error('AccumulatorFactory:UnknownStrategy', ...
                        'Unknown strategy "%s". Supported: "Weighted", "EMA", "Welford".', strategy);
            end
        end
        function strats = listStrategies()
            % LISTSTRATEGIES  Returns list of available accumulator strategies.
            strats = ["Weighted", "EMA", "Welford"];
        end

        function meta = getMetadata(strategy)
            % GETMETADATA  Returns Pillar 1 Symbolic metadata for a strategy.
            arguments
                strategy (1,1) string
            end
            temp_acc = OceanEngine.Accumulators.AccumulatorFactory.create(strategy, [1 1]);
            meta = temp_acc.getMetadata();
        end

        function spec = getDiagramSpec(strategy, shape, opts)
            % GETDIAGRAMSPEC  Returns Pillar 3 Diagram specification for a strategy.
            arguments
                strategy (1,1) string
                shape    double = [1 1]
                opts.Alpha (1,1) double = 0.1
                opts.InitValue (1,1) double = 0.0
            end
            temp_acc = OceanEngine.Accumulators.AccumulatorFactory.create(strategy, shape, ...
                'Alpha', opts.Alpha, 'InitValue', opts.InitValue);
            spec = temp_acc.getDiagramSpec();
        end

        function proto = getPrototypeSpec(strategy, shape, opts)
            % GETPROTOTYPESPEC  Returns unified 3-Pillar prototype spec for canvas registry.
            arguments
                strategy (1,1) string
                shape    double = [1 1]
                opts.Alpha (1,1) double = 0.1
                opts.InitValue (1,1) double = 0.0
            end
            temp_acc = OceanEngine.Accumulators.AccumulatorFactory.create(strategy, shape, ...
                'Alpha', opts.Alpha, 'InitValue', opts.InitValue);
            proto = struct(...
                'Strategy', strategy, ...
                'Symbolic', temp_acc.getMetadata(), ...
                'Diagram', temp_acc.getDiagramSpec(), ...
                'FactoryHandle', @(s, h) OceanEngine.Accumulators.AccumulatorFactory.create(s, h)...
            );
        end
    end
end
