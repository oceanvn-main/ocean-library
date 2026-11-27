classdef FlattenTest < OceanTest.BaseTestCase
    % FLATTENTEST  Unit tests for OceanDL.Conv.Flatten.

    methods (Test)
        function testForwardReshapesColumnMajor(testCase)
            import OceanDL.Conv.Flatten
            X = reshape(1:24, 2, 3, 4, 1);   % H=2,W=3,C=4,N=1
            flat = Flatten();
            A = flat.forward(X);
            testCase.verifySize(A, [24 1]);
            testCase.verifyEqual(A, reshape(X, 24, 1));
        end

        function testBackwardIsExactInverseReshape(testCase)
            import OceanDL.Conv.Flatten
            X = randn(2, 3, 4, 5);
            flat = Flatten();
            flat.forward(X);
            dA = randn(24, 5);
            dX = flat.backward(dA);
            testCase.verifyEqual(dX, reshape(dA, 2, 3, 4, 5));
        end

        function testBackwardBeforeForwardErrors(testCase)
            import OceanDL.Conv.Flatten
            flat = Flatten();
            testCase.verifyError(@() flat.backward(ones(6, 1)), ...
                'OceanDL:Conv:Flatten:NoForwardCache');
        end
    end
end
