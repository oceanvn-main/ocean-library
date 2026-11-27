classdef DeconvVisualizerTest < OceanTest.BaseTestCase
    methods (Test)
        function testIdentityConvReconstructsSinglePixel(testCase)
            % A 1x1, weight=1, no-bias conv (+ReLU) is a pure passthrough:
            % deconv-reconstructing from the argmax activation must give
            % back a zero array with exactly that one pixel's value, at
            % the same location -- no spatial mixing, no attenuation.
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.DeconvVisualizer

            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([1 1], 1, 'HasBias', false), ...
                CNNTopologySpec.relu() ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'act1'});
            eng = CNNEngine(spec, struct('conv1', struct('Weight', 1)));

            X = reshape(1:16, 4, 4, 1, 1);   % max value 16 at (4,4)
            r = DeconvVisualizer.reconstruct(eng, X, 'act1', 1);

            testCase.verifyEqual(r.SeedRow, 4);
            testCase.verifyEqual(r.SeedCol, 4);
            expected = zeros(4, 4, 1, 1);
            expected(4, 4, 1, 1) = 16;
            testCase.verifyEqual(r.Reconstruction, expected, 'AbsTol', 1e-10);
        end

        function testExplicitSpatialIndexOverridesArgmax(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.DeconvVisualizer

            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([1 1], 1, 'HasBias', false), ...
                CNNTopologySpec.relu() ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'act1'});
            eng = CNNEngine(spec, struct('conv1', struct('Weight', 1)));

            X = reshape(1:16, 4, 4, 1, 1);
            r = DeconvVisualizer.reconstruct(eng, X, 'act1', 1, 'SpatialIndex', [1 1]);
            testCase.verifyEqual(r.SeedRow, 1);
            testCase.verifyEqual(r.SeedCol, 1);
            expected = zeros(4, 4, 1, 1);
            expected(1, 1, 1, 1) = 1;   % X(1,1) = 1
            testCase.verifyEqual(r.Reconstruction, expected, 'AbsTol', 1e-10);
        end

        function testSigmoidInPathErrors(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.DeconvVisualizer

            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([1 1], 1, 'HasBias', false), ...
                CNNTopologySpec.sigmoid() ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'act1'});
            eng = CNNEngine(spec, struct('conv1', struct('Weight', 1)));

            testCase.verifyError(@() DeconvVisualizer.reconstruct(eng, reshape(1:16, 4, 4, 1, 1), 'act1', 1), ...
                'OceanDL:Interpretability:DeconvVisualizer:UnsupportedActivation');
        end
    end
end
