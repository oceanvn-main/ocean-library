classdef AvgPool3DTest < OceanTest.BaseTestCase
    % AVGPOOL3DTEST  Unit tests for OceanDL.Pooling.AvgPool3D.

    methods (Test)
        function testHandWorkedForward(testCase)
            % Single 2x2x2 window covering the whole cube: mean(1:8) = 4.5.
            import OceanDL.Pooling.AvgPool3D
            X = reshape(1:8, 2, 2, 2);
            pool = AvgPool3D([2 2 2]);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, 4.5, 'AbsTol', 1e-12);
        end

        function testBackwardDistributesGradientUniformly(testCase)
            import OceanDL.Pooling.AvgPool3D
            X = reshape(1:8, 2, 2, 2);
            pool = AvgPool3D([2 2 2]);
            pool.forward(X);
            dX = pool.backward(1);
            testCase.verifyEqual(dX, (1/8) * ones(2, 2, 2), 'AbsTol', 1e-12);
        end

        function testBackwardAccumulatesOverOverlappingWindows(testCase)
            import OceanDL.Pooling.AvgPool3D
            X = ones(3, 3, 3);
            pool = AvgPool3D([2 2 2], 'Stride', [1 1 1]);
            pool.forward(X);
            dX = pool.backward(ones(2, 2, 2));
            testCase.verifyEqual(dX(2, 2, 2), 1.0, 'AbsTol', 1e-12);    % 8 windows * 1/8
            testCase.verifyEqual(dX(1, 1, 1), 0.125, 'AbsTol', 1e-12);  % only 1 window
        end

        function testCountIncludePadTrueDividesByFullPoolVolume(testCase)
            import OceanDL.Pooling.AvgPool3D
            X = ones(1, 1, 1);
            pool = AvgPool3D([2 2 2], 'Stride', [2 2 2], 'Padding', [1 1 1], 'CountIncludePad', true);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, 1/8, 'AbsTol', 1e-12);
        end

        function testCountIncludePadFalseDividesByRealCellsOnly(testCase)
            import OceanDL.Pooling.AvgPool3D
            X = ones(1, 1, 1);
            pool = AvgPool3D([2 2 2], 'Stride', [2 2 2], 'Padding', [1 1 1], 'CountIncludePad', false);
            Y = pool.forward(X);
            testCase.verifyEqual(Y, 1, 'AbsTol', 1e-12);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.AvgPool3D
            pool = AvgPool3D([2 2 2]);
            testCase.verifyError(@() pool.backward(1), ...
                'OceanDL:Pooling:AvgPool3D:NoForwardCache');
        end
    end
end
