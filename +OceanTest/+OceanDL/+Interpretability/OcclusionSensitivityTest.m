classdef OcclusionSensitivityTest < OceanTest.BaseTestCase
    % Hand-constructed network where the target-class score is EXACTLY one
    % raw pixel value (an identity 1x1 conv + a one-hot dense row, no
    % nonlinearity/softmax), so the occlusion score-drop heatmap is
    % analytically predictable: occluding the one pixel the score depends
    % on drops the score by exactly its value; occluding any other single
    % pixel drops it by exactly zero.

    methods (Test)
        function testHeatmapIsolatesTheSinglePixelEachClassDependsOn(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.OcclusionSensitivity

            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([1 1], 1, 'HasBias', false), ...
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(2, 'HasBias', false) ...
            }, 'InputChannels', 1, 'LayerNames', {'conv1', 'flat', 'dense1'});

            Wdense = zeros(2, 16);
            Wdense(1, 1) = 1;    % class 1 score == flattened index 1  == X(1,1)
            Wdense(2, 11) = 1;   % class 2 score == flattened index 11 == X(3,3)
            weights = struct('conv1', struct('Weight', 1), 'dense1', struct('Weight', Wdense));

            eng = CNNEngine(spec, weights);
            X = reshape(1:16, 4, 4, 1, 1);   % X(1,1)=1, X(3,3)=11 (column-major fill)

            r1 = OcclusionSensitivity.run(eng, X, 1, 'PatchSize', [1 1], 'Stride', [1 1]);
            testCase.verifyEqual(r1.BaseScore, 1, 'AbsTol', 1e-12);
            expected1 = zeros(4, 4); expected1(1, 1) = 1;
            testCase.verifyEqual(r1.Heatmap, expected1, 'AbsTol', 1e-12);

            r2 = OcclusionSensitivity.run(eng, X, 2, 'PatchSize', [1 1], 'Stride', [1 1]);
            testCase.verifyEqual(r2.BaseScore, 11, 'AbsTol', 1e-12);
            expected2 = zeros(4, 4); expected2(3, 3) = 11;
            testCase.verifyEqual(r2.Heatmap, expected2, 'AbsTol', 1e-12);
        end

        function testBatchedInputErrors(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.OcclusionSensitivity

            spec = CNNTopologySpec.make({CNNTopologySpec.conv2d(3, 2, 'Padding', [1 1])}, 'InputChannels', 1);
            eng = CNNEngine(spec);
            testCase.verifyError(@() OcclusionSensitivity.run(eng, zeros(6, 6, 1, 2), 1), ...
                'OceanDL:Interpretability:OcclusionSensitivity:SingleImageOnly');
        end
    end
end
