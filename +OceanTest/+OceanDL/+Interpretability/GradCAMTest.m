classdef GradCAMTest < OceanTest.BaseTestCase
    properties
        Spec
    end

    methods (TestClassSetup)
        function buildSpec(testCase)
            import OceanDL.Conv.CNNTopologySpec
            testCase.Spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 2, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.conv2d([3 3], 3, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(4), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 1, 'LayerNames', ...
                {'conv1', 'act1', 'pool1', 'conv2', 'act2', 'flat', 'dense1', 'smax'});
        end
    end

    methods (Test)
        function testMatchesManualFormulaFromValidatedPrimitives(testCase)
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.GradCAM

            rng(21);
            X = randn(6, 6, 1, 1);
            eng = CNNEngine(testCase.Spec);
            Y = eng.forward(X);
            targetClass = 3;
            A = eng.getActivations('conv2');

            seed = zeros(size(Y));
            seed(targetClass, :) = 1;
            dAmanual = eng.gradient('smax', 'conv2', seed);
            alphaManual = mean(dAmanual, [1 2]);
            camManual = max(sum(alphaManual .* A, 3), 0);
            camManual = reshape(camManual, size(camManual, 1), size(camManual, 2));

            r = GradCAM.compute(eng, X, targetClass, 'conv2', 'OutputLayer', 'smax');
            testCase.verifyEqual(r.Alpha, alphaManual);
            testCase.verifyEqual(r.CAM, camManual, 'AbsTol', 1e-12);
            testCase.verifySize(r.CAMNormalized, size(camManual));
            testCase.verifyTrue(all(r.CAMNormalized(:) >= 0 & r.CAMNormalized(:) <= 1));
        end
    end
end
