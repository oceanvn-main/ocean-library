classdef TopActivatingPatchesTest < OceanTest.BaseTestCase
    % A 3x3 box filter (all-ones kernel) over two 8x8 images: image 1 has a
    % 3x3 bright 10-valued block at rows 3-5, cols 3-5 (rest zero), image 2
    % is all zero. The window that exactly overlaps the bright block (top-
    % left corner (3,3), stride 1, no padding) sums to 9*10=90 -- the
    % unique, analytically predictable global maximum.

    methods (Test)
        function testTopPatchIsTheExactBrightBlockWindow(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.TopActivatingPatches

            spec = CNNTopologySpec.make({CNNTopologySpec.conv2d([3 3], 1, 'HasBias', false)}, ...
                'InputChannels', 1, 'LayerNames', {'conv1'});
            eng = CNNEngine(spec, struct('conv1', struct('Weight', ones(3, 3, 1, 1))));

            img1 = zeros(8, 8, 1, 1);
            img1(3:5, 3:5, 1, 1) = 10;
            img2 = zeros(8, 8, 1, 1);
            images = cat(4, img1, img2);

            r = TopActivatingPatches.find(eng, images, 'conv1', 1, 'TopK', 3);
            testCase.verifyEqual(r.Values(1), 90, 'AbsTol', 1e-10);
            testCase.verifyEqual(r.ImageIndex(1), 1);
            testCase.verifyEqual(r.OutputRow(1), 3);
            testCase.verifyEqual(r.OutputCol(1), 3);
            testCase.verifyEqual(numel(r.Patches), 3);
        end

        function testBadChannelErrors(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine
            import OceanDL.Interpretability.TopActivatingPatches

            spec = CNNTopologySpec.make({CNNTopologySpec.conv2d(3, 2, 'Padding', [1 1])}, ...
                'InputChannels', 1, 'LayerNames', {'conv1'});
            eng = CNNEngine(spec);
            testCase.verifyError(@() TopActivatingPatches.find(eng, randn(6, 6, 1, 1), 'conv1', 5), ...
                'OceanDL:Interpretability:TopActivatingPatches:BadChannel');
        end
    end
end
