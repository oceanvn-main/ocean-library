classdef CNNTopologySpecTest < OceanTest.BaseTestCase
    % CNNTOPOLOGYSPECTEST  Unit tests for OceanDL.Conv.CNNTopologySpec.

    methods (Test)
        function testBasicConstructionAndAutoNames(testCase)
            import OceanDL.Conv.CNNTopologySpec
            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 4, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(3), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 2);

            testCase.verifyEqual(numel(spec.LayerSpecs), 6);
            testCase.verifyEqual(spec.InputChannels, 2);
            testCase.verifyEqual(spec.LayerNames, ...
                {'L1_conv2d', 'L2_relu', 'L3_maxpool2d', 'L4_flatten', 'L5_dense', 'L6_softmax'});
        end

        function testMaxPoolStrideDefaultsToPoolSize(testCase)
            import OceanDL.Conv.CNNTopologySpec
            s = CNNTopologySpec.maxpool2d([2 2]);
            testCase.verifyEqual(s.Stride, [2 2]);
        end

        function testCustomLayerNames(testCase)
            import OceanDL.Conv.CNNTopologySpec
            spec = CNNTopologySpec.make({CNNTopologySpec.conv2d(3, 4), CNNTopologySpec.relu()}, ...
                'LayerNames', {'conv_a', 'act_a'});
            testCase.verifyEqual(spec.LayerNames, {'conv_a', 'act_a'});
        end

        function testMustStartWithConv2d(testCase)
            import OceanDL.Conv.CNNTopologySpec
            testCase.verifyError(@() CNNTopologySpec.make({CNNTopologySpec.relu()}), ...
                'OceanDL:Conv:CNNTopologySpec:MustStartWithConv');
        end

        function testEmptyLayerListErrors(testCase)
            import OceanDL.Conv.CNNTopologySpec
            testCase.verifyError(@() CNNTopologySpec.make({}), ...
                'OceanDL:Conv:CNNTopologySpec:Empty');
        end

        function testInvalidLayerStructErrors(testCase)
            import OceanDL.Conv.CNNTopologySpec
            testCase.verifyError(@() CNNTopologySpec.make({struct('Type', 'bogus')}), ...
                'OceanDL:Conv:CNNTopologySpec:InvalidLayer');
        end

        function testLayerNamesSizeMismatchErrors(testCase)
            import OceanDL.Conv.CNNTopologySpec
            testCase.verifyError(@() CNNTopologySpec.make({CNNTopologySpec.conv2d(3, 4)}, ...
                'LayerNames', {'a', 'b'}), ...
                'OceanDL:Conv:CNNTopologySpec:SizeMismatch');
        end
    end
end
