classdef MaxPool2DTest < OceanTest.BaseTestCase
    % MAXPOOL2DTEST  Unit tests for OceanDL.Pooling.MaxPool2D.
    %
    % HAND-WORKED EXAMPLE (4x4, PoolSize=[2 2], Stride=[2 2], no padding):
    %   X = [ 1  2  3  4
    %         5  6  7  8
    %         9 10 11 12
    %        13 14 15 16 ]
    %   Window (rows1-2,cols1-2) = [1 2;5 6]   -> max 6 at (2,2)
    %   Window (rows1-2,cols3-4) = [3 4;7 8]   -> max 8 at (2,4)
    %   Window (rows3-4,cols1-2) = [9 10;13 14]-> max 14 at (4,2)
    %   Window (rows3-4,cols3-4) = [11 12;15 16]-> max 16 at (4,4)
    %   Y = [6 8; 14 16]
    %   Switches (linear idx into 4x4) = [sub2ind([4,4],2,2) sub2ind([4,4],2,4)
    %                                     sub2ind([4,4],4,2) sub2ind([4,4],4,4)]
    %                                  = [6 14; 8 16]

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
            import OceanDL.Pooling.MaxPool2D
            pool = MaxPool2D([2 2]);
            Y = pool.forward(testCase.X4x4);
            testCase.verifyEqual(Y, [6 8; 14 16], 'AbsTol', 1e-12);
        end

        function testHandWorkedSwitches(testCase)
            import OceanDL.Pooling.MaxPool2D
            pool = MaxPool2D([2 2]);
            pool.forward(testCase.X4x4);
            sw = pool.getSwitches();
            expected = [sub2ind([4 4], 2, 2), sub2ind([4 4], 2, 4); ...
                        sub2ind([4 4], 4, 2), sub2ind([4 4], 4, 4)];
            testCase.verifyEqual(sw, expected);
        end

        function testStrideDefaultsToPoolSize(testCase)
            import OceanDL.Pooling.MaxPool2D
            pool = MaxPool2D([2 2]);
            testCase.verifyEqual(pool.Stride, [2 2]);
        end

        function testBackwardRoutesToArgmaxOnly(testCase)
            import OceanDL.Pooling.MaxPool2D
            pool = MaxPool2D([2 2]);
            pool.forward(testCase.X4x4);
            dY = ones(2, 2);
            dX = pool.backward(dY);

            expected = zeros(4, 4);
            expected(2, 2) = 1;
            expected(2, 4) = 1;
            expected(4, 2) = 1;
            expected(4, 4) = 1;
            testCase.verifyEqual(dX, expected, 'AbsTol', 1e-12);
        end

        function testBackwardAccumulatesOverOverlappingWindows(testCase)
            % 3x3 input, PoolSize=[2 2], Stride=[1 1] (overlapping): the
            % center pixel (9) wins ALL FOUR pooling windows, so its
            % gradient must be the SUM of all four upstream gradients, not
            % just the last one written.
            import OceanDL.Pooling.MaxPool2D
            X = [1 2 3; 4 9 5; 6 7 8];
            pool = MaxPool2D([2 2], 'Stride', [1 1]);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, 9 * ones(2, 2), 'AbsTol', 1e-12);

            dY = ones(2, 2);
            dX = pool.backward(dY);
            expected = zeros(3, 3);
            expected(2, 2) = 4;   % 4 overlapping windows all pick (2,2)
            testCase.verifyEqual(dX, expected, 'AbsTol', 1e-12);
        end

        function testBatchAndChannelsAreIndependent(testCase)
            import OceanDL.Pooling.MaxPool2D
            X1 = testCase.X4x4;
            X2 = testCase.X4x4 * 10;   % distinguishable second channel/sample
            X = cat(4, cat(3, X1, X2), cat(3, X2, X1));  % (4x4x2x2)

            pool = MaxPool2D([2 2]);
            Y = pool.forward(X);

            testCase.verifyEqual(size(Y), [2 2 2 2]);
            testCase.verifyEqual(Y(:, :, 1, 1), [6 8; 14 16], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, :, 2, 1), [60 80; 140 160], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, :, 1, 2), [60 80; 140 160], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, :, 2, 2), [6 8; 14 16], 'AbsTol', 1e-12);
        end

        function testPaddingUsesNegativeInfinitySoItNeverWins(testCase)
            % An all-negative input must never have a padded (zero) cell
            % win the max -- that would silently bias the output toward 0.
            import OceanDL.Pooling.MaxPool2D
            X = -5 * ones(2, 2);
            pool = MaxPool2D([2 2], 'Padding', [1 1]);
            Y = pool.forward(X);
            testCase.verifyTrue(all(Y(:) <= -5 + 1e-12), ...
                'Padding must never win the max for an all-negative input.');
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.MaxPool2D
            pool = MaxPool2D([2 2]);
            testCase.verifyError(@() pool.backward(ones(2, 2)), ...
                'OceanDL:Pooling:MaxPool2D:NoForwardCache');
        end

        function testPaddingTooLargeErrors(testCase)
            import OceanDL.Pooling.MaxPool2D
            testCase.verifyError(@() MaxPool2D([2 2], 'Padding', [1 1] * 2), ...
                'OceanDL:Pooling:MaxPool2D:PaddingTooLarge');
        end
    end
end
