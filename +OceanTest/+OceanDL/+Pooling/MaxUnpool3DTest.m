classdef MaxUnpool3DTest < OceanTest.BaseTestCase
    % MAXUNPOOL3DTEST  Unit tests for OceanDL.Pooling.MaxUnpool3D,
    %   always paired against OceanDL.Pooling.MaxPool3D.

    methods (Test)
        function testUnpoolReconstructsWinningVoxelAtOriginalLocation(testCase)
            import OceanDL.Pooling.MaxPool3D
            import OceanDL.Pooling.MaxUnpool3D

            X = reshape(1:8, 2, 2, 2);
            pool = MaxPool3D([2 2 2]);
            Y = pool.forward(X);
            sw = pool.getSwitches();

            unpool = MaxUnpool3D();
            Xhat = unpool.forward(Y, sw, pool.getInputSize());

            expected = zeros(2, 2, 2);
            expected(2, 2, 2) = 8;
            testCase.verifyEqual(Xhat, expected, 'AbsTol', 1e-12);
        end

        function testBackwardGathersAtSwitches(testCase)
            import OceanDL.Pooling.MaxPool3D
            import OceanDL.Pooling.MaxUnpool3D

            X = reshape(1:8, 2, 2, 2);
            pool = MaxPool3D([2 2 2]);
            Y = pool.forward(X);
            sw = pool.getSwitches();

            unpool = MaxUnpool3D();
            unpool.forward(Y, sw, pool.getInputSize());

            dXhat = zeros(2, 2, 2);
            dXhat(2, 2, 2) = 500;
            dY = unpool.backward(dXhat);
            testCase.verifyEqual(dY, 500, 'AbsTol', 1e-12);
        end

        function testCollidingSwitchesAreSummed(testCase)
            % 3x3x3 cube, overlapping PoolSize=[2 2 2] Stride=[1 1 1]: all 8
            % pooled outputs share the SAME argmax voxel (the center), so
            % scattering them back must SUM, not overwrite.
            import OceanDL.Pooling.MaxPool3D
            import OceanDL.Pooling.MaxUnpool3D

            X = ones(3, 3, 3);
            X(2, 2, 2) = 9;
            pool = MaxPool3D([2 2 2], 'Stride', [1 1 1]);
            Y = pool.forward(X);              % Y = 9*ones(2,2,2)
            sw = pool.getSwitches();          % all eight point at (2,2,2)

            unpool = MaxUnpool3D();
            Xhat = unpool.forward(Y, sw, pool.getInputSize());

            expected = zeros(3, 3, 3);
            expected(2, 2, 2) = 9 * 8;         % 8 colliding contributions summed
            testCase.verifyEqual(Xhat, expected, 'AbsTol', 1e-12);
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Pooling.MaxUnpool3D
            unpool = MaxUnpool3D();
            testCase.verifyError(@() unpool.backward(1), ...
                'OceanDL:Pooling:MaxUnpool3D:NoForwardCache');
        end
    end
end
