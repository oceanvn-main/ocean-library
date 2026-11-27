classdef MaxPool1DTest < OceanTest.BaseTestCase
    % MAXPOOL1DTEST  Unit tests for OceanDL.Pooling.MaxPool1D.
    %
    % HAND-WORKED EXAMPLE (L=6, PoolSize=2, Stride=2, no padding):
    %   X = [1;2;3;4;5;6]
    %   window[1,2]=max 2 @pos2   window[3,4]=max 4 @pos4   window[5,6]=max 6 @pos6
    %   Y = [2;4;6],  Switches = [2;4;6]

    properties
        X6
    end

    methods (TestClassSetup)
        function buildInput(testCase)
            testCase.X6 = (1:6)';
        end
    end

    methods (Test)
        function testHandWorkedForward(testCase)
            import OceanDL.Pooling.MaxPool1D
            pool = MaxPool1D(2);
            Y = pool.forward(testCase.X6);
            testCase.verifyEqual(Y, [2; 4; 6], 'AbsTol', 1e-12);
        end

        function testHandWorkedSwitches(testCase)
            import OceanDL.Pooling.MaxPool1D
            pool = MaxPool1D(2);
            pool.forward(testCase.X6);
            testCase.verifyEqual(pool.getSwitches(), [2; 4; 6]);
        end

        function testBackwardRoutesToArgmaxOnly(testCase)
            import OceanDL.Pooling.MaxPool1D
            pool = MaxPool1D(2);
            pool.forward(testCase.X6);
            dX = pool.backward(ones(3, 1));
            testCase.verifyEqual(dX, [0;1;0;1;0;1], 'AbsTol', 1e-12);
        end

        function testBackwardAccumulatesOverOverlappingWindows(testCase)
            % L=3, PoolSize=2, Stride=1 (overlapping): position 2 (value 5)
            % wins BOTH windows, so its gradient sums both contributions.
            import OceanDL.Pooling.MaxPool1D
            X = [1; 5; 2];
            pool = MaxPool1D(2, 'Stride', 1);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, [5; 5], 'AbsTol', 1e-12);

            dX = pool.backward(ones(2, 1));
            expected = [0; 2; 0];
            testCase.verifyEqual(dX, expected, 'AbsTol', 1e-12);
        end

        function testBatchAndChannelsAreIndependent(testCase)
            import OceanDL.Pooling.MaxPool1D
            X1 = testCase.X6;
            X2 = testCase.X6 * 10;
            X = cat(3, cat(2, X1, X2), cat(2, X2, X1));  % (6x2x2)

            pool = MaxPool1D(2);
            Y = pool.forward(X);

            testCase.verifyEqual(size(Y), [3 2 2]);
            testCase.verifyEqual(Y(:, 1, 1), [2; 4; 6], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, 2, 1), [20; 40; 60], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, 1, 2), [20; 40; 60], 'AbsTol', 1e-12);
            testCase.verifyEqual(Y(:, 2, 2), [2; 4; 6], 'AbsTol', 1e-12);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.MaxPool1D
            pool = MaxPool1D(2);
            testCase.verifyError(@() pool.backward(ones(3, 1)), ...
                'OceanDL:Pooling:MaxPool1D:NoForwardCache');
        end
    end
end
