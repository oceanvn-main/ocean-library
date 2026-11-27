classdef AvgPool2DTest < OceanTest.BaseTestCase
    % AVGPOOL2DTEST  Unit tests for OceanDL.Pooling.AvgPool2D.
    %
    % HAND-WORKED EXAMPLE (4x4, PoolSize=[2 2], Stride=[2 2], no padding):
    %   X = [ 1  2  3  4
    %         5  6  7  8
    %         9 10 11 12
    %        13 14 15 16 ]
    %   mean[1 2;5 6]=3.5   mean[3 4;7 8]=5.5
    %   mean[9 10;13 14]=11.5   mean[11 12;15 16]=13.5
    %   Y = [3.5 5.5; 11.5 13.5]

    properties
        X4x4
    end

    methods (TestClassSetup)
        function buildInput(testCase)
            testCase.X4x4 = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
        end
    end

    methods (Test)
        function testHandWorkedForward(testCase)
            import OceanDL.Pooling.AvgPool2D
            pool = AvgPool2D([2 2]);
            Y = pool.forward(testCase.X4x4);
            testCase.verifyEqual(Y, [3.5 5.5; 11.5 13.5], 'AbsTol', 1e-12);
        end

        function testBackwardDistributesGradientUniformly(testCase)
            % Non-overlapping windows (Stride == PoolSize): every input
            % pixel gets exactly 1/poolArea of its window's upstream grad.
            import OceanDL.Pooling.AvgPool2D
            pool = AvgPool2D([2 2]);
            pool.forward(testCase.X4x4);
            dX = pool.backward(ones(2, 2));
            testCase.verifyEqual(dX, 0.25 * ones(4, 4), 'AbsTol', 1e-12);
        end

        function testBackwardAccumulatesOverOverlappingWindows(testCase)
            % 3x3 input, PoolSize=[2 2], Stride=[1 1]: the center pixel is
            % in all 4 windows, so its gradient sums 4 contributions of 1/4.
            import OceanDL.Pooling.AvgPool2D
            X = ones(3, 3);
            pool = AvgPool2D([2 2], 'Stride', [1 1]);
            pool.forward(X);
            dX = pool.backward(ones(2, 2));
            testCase.verifyEqual(dX(2, 2), 1.0, 'AbsTol', 1e-12);   % 4 * (1/4)
            testCase.verifyEqual(dX(1, 1), 0.25, 'AbsTol', 1e-12);  % only 1 window
        end

        function testCountIncludePadTrueDividesByFullPoolArea(testCase)
            % 2x2 input, PoolSize=[2 2], Stride=[2 2], Padding=[1 1]:
            % each window contains exactly one real pixel + 3 zero-pad
            % cells; CountIncludePad=true still divides by the full area (4).
            import OceanDL.Pooling.AvgPool2D
            X = [1 2; 3 4];
            pool = AvgPool2D([2 2], 'Stride', [2 2], 'Padding', [1 1], 'CountIncludePad', true);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, [0.25 0.5; 0.75 1.0], 'AbsTol', 1e-12);
        end

        function testCountIncludePadFalseDividesByRealCellsOnly(testCase)
            % Same setup, CountIncludePad=false: each window has exactly
            % ONE real cell (the rest is padding), so the average recovers
            % the original pixel value exactly.
            import OceanDL.Pooling.AvgPool2D
            X = [1 2; 3 4];
            pool = AvgPool2D([2 2], 'Stride', [2 2], 'Padding', [1 1], 'CountIncludePad', false);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, [1 2; 3 4], 'AbsTol', 1e-12);
        end

        function testBatchAndChannelsAreIndependent(testCase)
            import OceanDL.Pooling.AvgPool2D
            X1 = testCase.X4x4;
            X2 = testCase.X4x4 * 10;
            X = cat(4, X1, X2);   % (4x4x1x2)

            pool = AvgPool2D([2 2]);
            Y = pool.forward(X);

            testCase.verifyEqual(size(Y), [2 2 1 2]);
            testCase.verifyEqual(Y(:, :, 1, 1), [3.5 5.5; 11.5 13.5], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, :, 1, 2), [35 55; 115 135], 'AbsTol', 1e-12);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.AvgPool2D
            pool = AvgPool2D([2 2]);
            testCase.verifyError(@() pool.backward(ones(2, 2)), ...
                'OceanDL:Pooling:AvgPool2D:NoForwardCache');
        end
    end
end
