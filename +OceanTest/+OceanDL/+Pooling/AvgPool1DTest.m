classdef AvgPool1DTest < OceanTest.BaseTestCase
    % AVGPOOL1DTEST  Unit tests for OceanDL.Pooling.AvgPool1D.

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
            import OceanDL.Pooling.AvgPool1D
            pool = AvgPool1D(2);
            Y = pool.forward(testCase.X6);
            testCase.verifyEqual(Y, [1.5; 3.5; 5.5], 'AbsTol', 1e-12);
        end

        function testBackwardDistributesGradientUniformly(testCase)
            import OceanDL.Pooling.AvgPool1D
            pool = AvgPool1D(2);
            pool.forward(testCase.X6);
            dX = pool.backward(ones(3, 1));
            testCase.verifyEqual(dX, 0.5 * ones(6, 1), 'AbsTol', 1e-12);
        end

        function testBackwardAccumulatesOverOverlappingWindows(testCase)
            import OceanDL.Pooling.AvgPool1D
            X = ones(3, 1);
            pool = AvgPool1D(2, 'Stride', 1);
            pool.forward(X);
            dX = pool.backward(ones(2, 1));
            testCase.verifyEqual(dX(2), 1.0, 'AbsTol', 1e-12);   % 2 windows * 1/2
            testCase.verifyEqual(dX(1), 0.5, 'AbsTol', 1e-12);   % only 1 window
        end

        function testCountIncludePadTrueDividesByFullPoolArea(testCase)
            import OceanDL.Pooling.AvgPool1D
            X = [1; 2];
            pool = AvgPool1D(2, 'Stride', 2, 'Padding', 1, 'CountIncludePad', true);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, [0.5; 1.0], 'AbsTol', 1e-12);
        end

        function testCountIncludePadFalseDividesByRealCellsOnly(testCase)
            import OceanDL.Pooling.AvgPool1D
            X = [1; 2];
            pool = AvgPool1D(2, 'Stride', 2, 'Padding', 1, 'CountIncludePad', false);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, [1; 2], 'AbsTol', 1e-12);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.AvgPool1D
            pool = AvgPool1D(2);
            testCase.verifyError(@() pool.backward(ones(3, 1)), ...
                'OceanDL:Pooling:AvgPool1D:NoForwardCache');
        end
    end
end
