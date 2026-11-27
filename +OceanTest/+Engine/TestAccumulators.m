classdef TestAccumulators < matlab.unittest.TestCase
    % TESTACCUMULATORS  Unit tests for online statistics accumulators.
    %
    % Verifies numeric correctness of:
    %   1. WeightedMeanAccumulator (sample average & weighted IS)
    %   2. ExponentialMovingAverage
    %   3. WelfordVarianceAccumulator (online mean, variance, std)
    %   4. AccumulatorFactory
    %
    % Package: OceanTest.Engine
    % See also: OceanEngine.Accumulators.WeightedMeanAccumulator

    methods (Test)
        function testWeightedMeanScalar(testCase)
            acc = OceanEngine.Accumulators.WeightedMeanAccumulator([1 1]);
            samples = [10 20 30 40 50];
            weights = [1 1 1 1 1];

            for k = 1:numel(samples)
                acc.update(samples(k), weights(k));
            end

            testCase.verifyEqual(acc.getValue(), 30, 'AbsTol', 1e-12);
            testCase.verifyEqual(acc.getWeight(), 5, 'AbsTol', 1e-12);
        end

        function testWeightedImportanceSampling(testCase)
            acc = OceanEngine.Accumulators.WeightedMeanAccumulator([1 1]);
            % Weighted average of returns [10, 20] with weights [0.5, 1.5]
            % Expected: (10*0.5 + 20*1.5) / (0.5 + 1.5) = (5 + 30) / 2.0 = 17.5
            acc.update(10, 0.5);
            acc.update(20, 1.5);

            testCase.verifyEqual(acc.getValue(), 17.5, 'AbsTol', 1e-12);
            testCase.verifyEqual(acc.getWeight(), 2.0, 'AbsTol', 1e-12);
        end

        function testSubscriptMatrixIndexing(testCase)
            acc = OceanEngine.Accumulators.WeightedMeanAccumulator([5 2]);
            % Update state 3, action 2 with sample 100, weight 2.0
            acc.update(100, 2.0, 3, 2);
            acc.update(200, 2.0, 3, 2);

            val = acc.getValue();
            w   = acc.getWeight();

            testCase.verifyEqual(val(3, 2), 150, 'AbsTol', 1e-12);
            testCase.verifyEqual(w(3, 2), 4.0, 'AbsTol', 1e-12);
            testCase.verifyEqual(val(1, 1), 0.0);
        end

        function testExponentialMovingAverage(testCase)
            acc = OceanEngine.Accumulators.ExponentialMovingAverage([1 1], 'Alpha', 0.5);
            % Update 0 -> 10: 0 + 0.5*(10 - 0) = 5
            acc.update(10);
            testCase.verifyEqual(acc.getValue(), 5.0, 'AbsTol', 1e-12);

            % Update 5 -> 20: 5 + 0.5*(20 - 5) = 12.5
            acc.update(20);
            testCase.verifyEqual(acc.getValue(), 12.5, 'AbsTol', 1e-12);
        end

        function testWelfordVariance(testCase)
            acc = OceanEngine.Accumulators.WelfordVarianceAccumulator([1 1]);
            data = [2, 4, 4, 4, 5, 5, 7, 9];

            for k = 1:numel(data)
                acc.update(data(k));
            end

            expected_mean = mean(data);
            expected_var  = var(data, 1); % population variance

            testCase.verifyEqual(acc.getValue(), expected_mean, 'AbsTol', 1e-12);
            testCase.verifyEqual(acc.getVariance('Type', 'population'), expected_var, 'AbsTol', 1e-12);
            testCase.verifyEqual(acc.getStd('Type', 'population'), std(data, 1), 'AbsTol', 1e-12);
        end

        function testAccumulatorFactory(testCase)
            acc1 = OceanEngine.Accumulators.AccumulatorFactory.create('Weighted', [3 3]);
            testCase.verifyClass(acc1, 'OceanEngine.Accumulators.WeightedMeanAccumulator');

            acc2 = OceanEngine.Accumulators.AccumulatorFactory.create('EMA', [2 2], 'Alpha', 0.2);
            testCase.verifyClass(acc2, 'OceanEngine.Accumulators.ExponentialMovingAverage');

            acc3 = OceanEngine.Accumulators.AccumulatorFactory.create('Welford', [5 1]);
            testCase.verifyClass(acc3, 'OceanEngine.Accumulators.WelfordVarianceAccumulator');
        end

        function testDiagramSpec(testCase)
            acc = OceanEngine.Accumulators.WeightedMeanAccumulator([10 2]);
            spec = acc.getDiagramSpec();
            testCase.verifyEqual(spec.DiagramSymbol.Style, 'SimulinkBlock');
            testCase.verifyEqual(spec.DiagramSymbol.LabelText, 'Weighted Mean Accumulator');
            testCase.verifyTrue(isfield(spec, 'Inputs'));
            testCase.verifyTrue(isfield(spec, 'Outputs'));
        end

        function testFactoryThreePillarIntrospection(testCase)
            strats = OceanEngine.Accumulators.AccumulatorFactory.listStrategies();
            testCase.verifyTrue(ismember('EMA', strats));

            meta = OceanEngine.Accumulators.AccumulatorFactory.getMetadata('EMA');
            testCase.verifyEqual(meta.Name, 'Exponential Moving Average');

            spec = OceanEngine.Accumulators.AccumulatorFactory.getDiagramSpec('Weighted', [5 1]);
            testCase.verifyEqual(spec.Outputs.value.Shape, [5 1]);

            proto = OceanEngine.Accumulators.AccumulatorFactory.getPrototypeSpec('Welford', [10 1]);
            testCase.verifyEqual(proto.Strategy, "Welford");
            testCase.verifyTrue(isfield(proto, 'Symbolic'));
            testCase.verifyTrue(isfield(proto, 'Diagram'));
        end
    end
end
