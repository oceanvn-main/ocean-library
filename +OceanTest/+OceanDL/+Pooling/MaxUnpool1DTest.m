classdef MaxUnpool1DTest < OceanTest.BaseTestCase
    % MAXUNPOOL1DTEST  Unit tests for OceanDL.Pooling.MaxUnpool1D,
    %   always paired against OceanDL.Pooling.MaxPool1D.

    properties
        X6
    end

    methods (TestClassSetup)
        function buildInput(testCase)
            testCase.X6 = (1:6)';
        end
    end

    methods (Test)
        function testUnpoolReconstructsWinningPixelsAtOriginalLocations(testCase)
            import OceanDL.Pooling.MaxPool1D
            import OceanDL.Pooling.MaxUnpool1D

            pool = MaxPool1D(2);
            Y = pool.forward(testCase.X6);
            sw = pool.getSwitches();

            unpool = MaxUnpool1D();
            Xhat = unpool.forward(Y, sw, pool.getInputSize());

            expected = [0; 2; 0; 4; 0; 6];
            testCase.verifyEqual(Xhat, expected, 'AbsTol', 1e-12);
        end

        function testBackwardGathersAtSwitches(testCase)
            import OceanDL.Pooling.MaxPool1D
            import OceanDL.Pooling.MaxUnpool1D

            pool = MaxPool1D(2);
            Y = pool.forward(testCase.X6);
            sw = pool.getSwitches();

            unpool = MaxUnpool1D();
            unpool.forward(Y, sw, pool.getInputSize());

            dXhat = zeros(6, 1);
            dXhat(2) = 100; dXhat(4) = 200; dXhat(6) = 300;
            dY = unpool.backward(dXhat);
            testCase.verifyEqual(dY, [100; 200; 300], 'AbsTol', 1e-12);
        end

        function testCollidingSwitchesAreSummed(testCase)
            import OceanDL.Pooling.MaxPool1D
            import OceanDL.Pooling.MaxUnpool1D

            X = [1; 5; 2];
            pool = MaxPool1D(2, 'Stride', 1);
            Y = pool.forward(X);          % Y = [5;5]
            sw = pool.getSwitches();      % both point at position 2

            unpool = MaxUnpool1D();
            Xhat = unpool.forward(Y, sw, pool.getInputSize());

            expected = [0; 10; 0];        % 5 + 5 summed
            testCase.verifyEqual(Xhat, expected, 'AbsTol', 1e-12);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.MaxUnpool1D
            unpool = MaxUnpool1D();
            testCase.verifyError(@() unpool.backward(ones(3, 1)), ...
                'OceanDL:Pooling:MaxUnpool1D:NoForwardCache');
        end
    end
end
