classdef ActivationMapExtractorTest < OceanTest.BaseTestCase
    methods (Test)
        function testChannelToImageMinMaxNormalizes(testCase)
            import OceanDL.Interpretability.ActivationMapExtractor
            maps = zeros(2, 2, 2, 1);
            maps(:, :, 1, 1) = [0 2; 4 6];   % min=0, max=6
            img = ActivationMapExtractor.channelToImage(maps, 1, 1);
            testCase.verifyEqual(img, [0 1/3; 2/3 1], 'AbsTol', 1e-12);
        end

        function testExtractWiresThroughEngineForward(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.ActivationMapExtractor

            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 2, 'Padding', [1 1]), ...
                CNNTopologySpec.relu() ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'act1'});
            eng = CNNEngine(spec);

            maps = ActivationMapExtractor.extract(eng, randn(6, 6, 1, 1), 'act1');
            testCase.verifySize(maps, [6 6 2 1]);
            testCase.verifyTrue(all(maps(:) >= 0));   % post-ReLU
        end
    end
end
