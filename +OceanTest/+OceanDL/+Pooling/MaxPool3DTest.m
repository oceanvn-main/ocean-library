classdef MaxPool3DTest < OceanTest.BaseTestCase
    % MAXPOOL3DTEST  Unit tests for OceanDL.Pooling.MaxPool3D.

    methods (Test)
        function testHandWorkedForwardAndSwitches(testCase)
            % Single 2x2x2 window covering the whole cube: X = reshape(1:8,2,2,2)
            % puts the max (8) at (2,2,2), linear index 8.
            import OceanDL.Pooling.MaxPool3D
            X = reshape(1:8, 2, 2, 2);
            pool = MaxPool3D([2 2 2]);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, 8, 'AbsTol', 1e-12);
            testCase.verifyEqual(pool.getSwitches(), 8);
        end

        function testBackwardRoutesToArgmaxOnly(testCase)
            import OceanDL.Pooling.MaxPool3D
            X = reshape(1:8, 2, 2, 2);
            pool = MaxPool3D([2 2 2]);
            pool.forward(X);
            dX = pool.backward(1);

            expected = zeros(2, 2, 2);
            expected(2, 2, 2) = 1;
            testCase.verifyEqual(dX, expected, 'AbsTol', 1e-12);
        end

        function testBackwardAccumulatesOverOverlappingWindows(testCase)
            % 3x3x3 cube, PoolSize=[2 2 2], Stride=[1 1 1] (overlapping):
            % the center voxel is inside ALL 8 pooling windows.
            import OceanDL.Pooling.MaxPool3D
            X = ones(3, 3, 3);
            X(2, 2, 2) = 9;
            pool = MaxPool3D([2 2 2], 'Stride', [1 1 1]);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, 9 * ones(2, 2, 2), 'AbsTol', 1e-12);

            dX = pool.backward(ones(2, 2, 2));
            expected = zeros(3, 3, 3);
            expected(2, 2, 2) = 8;   % all 8 overlapping windows pick (2,2,2)
            testCase.verifyEqual(dX, expected, 'AbsTol', 1e-12);
        end

        function testBatchAndChannelsAreIndependent(testCase)
            import OceanDL.Pooling.MaxPool3D
            X1 = reshape(1:8, 2, 2, 2);
            X2 = X1 * 10;
            X = cat(5, cat(4, X1, X2), cat(4, X2, X1));   % (2x2x2x2x2)

            pool = MaxPool3D([2 2 2]);
            Y = pool.forward(X);

            testCase.verifyEqual(size(Y), [1 1 1 2 2]);
            testCase.verifyEqual(Y(1, 1, 1, 1, 1), 8, 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(1, 1, 1, 2, 1), 80, 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(1, 1, 1, 1, 2), 80, 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(1, 1, 1, 2, 2), 8, 'AbsTol', 1e-12);
        end

        function testPaddingUsesNegativeInfinitySoItNeverWins(testCase)
            import OceanDL.Pooling.MaxPool3D
            X = -5 * ones(2, 2, 2);
            pool = MaxPool3D([2 2 2], 'Padding', [1 1 1]);
            Y = pool.forward(X);
            testCase.verifyTrue(all(Y(:) <= -5 + 1e-12), ...
                'Padding must never win the max for an all-negative input.');
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.MaxPool3D
            pool = MaxPool3D([2 2 2]);
            testCase.verifyError(@() pool.backward(1), ...
                'OceanDL:Pooling:MaxPool3D:NoForwardCache');
        end

        function testPaddingTooLargeErrors(testCase)
            import OceanDL.Pooling.MaxPool3D
            testCase.verifyError(@() MaxPool3D([2 2 2], 'Padding', [2 2 2]), ...
                'OceanDL:Pooling:MaxPool3D:PaddingTooLarge');
        end
    end
end
