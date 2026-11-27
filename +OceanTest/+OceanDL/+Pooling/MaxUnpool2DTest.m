classdef MaxUnpool2DTest < OceanTest.BaseTestCase
    % MAXUNPOOL2DTEST  Unit tests for OceanDL.Pooling.MaxUnpool2D,
    %   always paired against OceanDL.Pooling.MaxPool2D.

    properties
        X4x4
    end

    methods (TestClassSetup)
        function buildInput(testCase)
            testCase.X4x4 = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
        end
    end

    methods (Test)
        function testUnpoolReconstructsWinningPixelsAtOriginalLocations(testCase)
            % Chaining MaxPool2D -> MaxUnpool2D must reproduce the ORIGINAL
            % input's values at exactly the 4 argmax locations, and zero
            % everywhere else (SegNet-style sparse reconstruction).
            import OceanDL.Pooling.MaxPool2D
            import OceanDL.Pooling.MaxUnpool2D

            pool = MaxPool2D([2 2]);
            Y = pool.forward(testCase.X4x4);
            sw = pool.getSwitches();

            unpool = MaxUnpool2D();
            Xhat = unpool.forward(Y, sw, pool.getInputSize());

            expected = zeros(4, 4);
            expected(2, 2) = 6;
            expected(2, 4) = 8;
            expected(4, 2) = 14;
            expected(4, 4) = 16;
            testCase.verifyEqual(Xhat, expected, 'AbsTol', 1e-12);
        end

        function testBackwardGathersAtSwitches(testCase)
            import OceanDL.Pooling.MaxPool2D
            import OceanDL.Pooling.MaxUnpool2D

            pool = MaxPool2D([2 2]);
            Y = pool.forward(testCase.X4x4);
            sw = pool.getSwitches();

            unpool = MaxUnpool2D();
            unpool.forward(Y, sw, pool.getInputSize());

            dXhat = zeros(4, 4);
            dXhat(2, 2) = 100;
            dXhat(2, 4) = 200;
            dXhat(4, 2) = 300;
            dXhat(4, 4) = 400;

            dY = unpool.backward(dXhat);
            testCase.verifyEqual(dY, [100 200; 300 400], 'AbsTol', 1e-12);
        end

        function testCollidingSwitchesAreSummed(testCase)
            % 3x3 input, overlapping PoolSize=[2 2] Stride=[1 1]: all 4
            % pooled outputs share the SAME argmax location (the center
            % pixel), so scattering them back must SUM, not overwrite.
            import OceanDL.Pooling.MaxPool2D
            import OceanDL.Pooling.MaxUnpool2D

            X = [1 2 3; 4 9 5; 6 7 8];
            pool = MaxPool2D([2 2], 'Stride', [1 1]);
            Y = pool.forward(X);              % Y = 9*ones(2,2)
            sw = pool.getSwitches();          % all four point at (2,2)

            unpool = MaxUnpool2D();
            Xhat = unpool.forward(Y, sw, pool.getInputSize());

            expected = zeros(3, 3);
            expected(2, 2) = 9 * 4;   % 4 colliding contributions summed
            testCase.verifyEqual(Xhat, expected, 'AbsTol', 1e-12);
        end

        function testOutputSizeAcceptsHWOnly(testCase)
            import OceanDL.Pooling.MaxPool2D
            import OceanDL.Pooling.MaxUnpool2D

            pool = MaxPool2D([2 2]);
            Y = pool.forward(testCase.X4x4);
            sw = pool.getSwitches();

            unpool = MaxUnpool2D();
            Xhat = unpool.forward(Y, sw, [4 4]);   % just [H W], C/N inferred from Y

            % size(Xhat, [1 2 3 4]) explicitly queries all 4 dims -- unlike
            % size(Xhat), it does NOT drop trailing singleton dimensions.
            testCase.verifyEqual(size(Xhat, [1 2 3 4]), [4 4 1 1]);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.MaxUnpool2D
            unpool = MaxUnpool2D();
            testCase.verifyError(@() unpool.backward(ones(2, 2)), ...
                'OceanDL:Pooling:MaxUnpool2D:NoForwardCache');
        end
    end
end
