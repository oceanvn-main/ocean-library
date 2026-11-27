classdef WeightVisualizerTest < OceanTest.BaseTestCase
    methods (Test)
        function testExtractTilesReturnsOnePerOutputFilter(testCase)
            import OceanDL.Conv.Conv2D
            import OceanDL.Interpretability.WeightVisualizer
            W = cat(4, ones(3, 3, 1), 2 * ones(3, 3, 1));
            conv = Conv2D([3 3], 2, 'InChannels', 1, 'Weight', W);
            tiles = WeightVisualizer.extractTiles(conv);
            testCase.verifyEqual(numel(tiles), 2);
            testCase.verifyEqual(tiles{1}, ones(3, 3, 1));
            testCase.verifyEqual(tiles{2}, 2 * ones(3, 3, 1));
        end

        function testDisplayImageMinMaxNormalizes(testCase)
            import OceanDL.Interpretability.WeightVisualizer
            tile = reshape([1 2; 3 4], 2, 2, 1);   % min=1, max=4
            img = WeightVisualizer.toDisplayImage(tile);
            testCase.verifyEqual(img, reshape([0 1/3; 2/3 1], 2, 2, 1), 'AbsTol', 1e-12);
        end

        function testConstantTileNormalizesToZero(testCase)
            import OceanDL.Interpretability.WeightVisualizer
            img = WeightVisualizer.toDisplayImage(5 * ones(2, 2, 1));
            testCase.verifyEqual(img, zeros(2, 2, 1));
        end

        function testThreeChannelTileKeptAsRGB(testCase)
            import OceanDL.Interpretability.WeightVisualizer
            tile = cat(3, zeros(2, 2), 0.5 * ones(2, 2), ones(2, 2));
            img = WeightVisualizer.toDisplayImage(tile);
            testCase.verifySize(img, [2 2 3]);
        end

        function testMultiChannelTileAveraged(testCase)
            % Cin=4 (not 1 or 3) -> averaged across channels. Each channel
            % is itself spatially constant, so the average is spatially
            % constant too (1.5 everywhere) -> min==max -> normalizes to 0.
            import OceanDL.Interpretability.WeightVisualizer
            tile = cat(3, zeros(2, 2), ones(2, 2), 2 * ones(2, 2), 3 * ones(2, 2));
            img = WeightVisualizer.toDisplayImage(tile);
            testCase.verifySize(img, [2 2]);
            testCase.verifyEqual(img, zeros(2, 2), 'AbsTol', 1e-12);
        end
    end
end
