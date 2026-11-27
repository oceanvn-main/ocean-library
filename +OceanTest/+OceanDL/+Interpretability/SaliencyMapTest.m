classdef SaliencyMapTest < OceanTest.BaseTestCase
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
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(3), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'act1', 'pool1', 'flat', 'dense1', 'smax'});
        end
    end

    methods (Test)
        function testMatchesManualTapeGradient(testCase)
            % SaliencyMap must be exactly the vanilla-gradient formula: a
            % thin wrapper over engine.gradient() with a one-hot seed --
            % engine.gradient()'s correctness is already gradchecked in
            % CNNEngineTest, so this checks the wiring/formula, not the
            % underlying autograd math again.
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.SaliencyMap

            rng(11);
            X = randn(6, 6, 1, 1);
            eng = CNNEngine(testCase.Spec);
            Y = eng.forward(X);
            targetClass = 2;

            seed = zeros(size(Y));
            seed(targetClass, :) = 1;
            dXmanual = eng.gradient('smax', 'input', seed);
            saliencyManual = max(abs(dXmanual), [], 3);

            r = SaliencyMap.compute(eng, X, targetClass, 'OutputLayer', 'smax');
            testCase.verifyEqual(r.Gradient, dXmanual);
            testCase.verifySize(r.Saliency, [6 6 1]);
            testCase.verifyEqual(reshape(r.Saliency, 6, 6), saliencyManual, 'AbsTol', 1e-12);
        end
    end
end
