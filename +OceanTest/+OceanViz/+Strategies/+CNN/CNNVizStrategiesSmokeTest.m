classdef CNNVizStrategiesSmokeTest < OceanTest.BaseTestCase
    % CNNVIZSTRATEGIESSMOKETEST  Renders all 8 CNN interpretability
    %   strategies against real output from the Interpretability math
    %   modules (run on a tiny toy CNNEngine), on an invisible figure.
    %   Visual/rendering smoke test only -- no pixel-level assertions,
    %   just "does it run and produce a titled axes" (per the project
    %   plan's package-4 verification bar; the math is already gradchecked
    %   in +OceanTest/+OceanDL/+Interpretability).

    properties
        Engine
        X
        Fig
    end

    methods (TestClassSetup)
        function buildEngineAndImage(testCase)
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine

            spec = CNNTopologySpec.make({ ...
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

            rng(41);
            testCase.Engine = CNNEngine(spec);
            testCase.X = rand(12, 12, 1, 1);
            testCase.Engine.forward(testCase.X);
        end
    end

    methods (TestMethodSetup)
        function openFig(testCase)
            testCase.Fig = figure('Visible', 'off');
        end
    end

    methods (TestMethodTeardown)
        function closeFig(testCase)
            close(testCase.Fig);
        end
    end

    methods (Test)
        function testFilterGrid(testCase)
            import OceanDL.Interpretability.WeightVisualizer
            import OceanViz.Strategies.CNN.FilterGridStrategy
            conv = testCase.Engine.getLayerObject('conv1');
            data = struct('Tiles', {WeightVisualizer.extractTiles(conv)});
            ax = axes(testCase.Fig);
            FilterGridStrategy().plot(ax, data, struct());
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testActivationMap(testCase)
            import OceanDL.Interpretability.ActivationMapExtractor
            import OceanViz.Strategies.CNN.ActivationMapStrategy
            maps = ActivationMapExtractor.extract(testCase.Engine, testCase.X, 'conv1');
            img = ActivationMapExtractor.channelToImage(maps, 1, 1);
            ax = axes(testCase.Fig);
            ActivationMapStrategy().plot(ax, img, struct('Title', 'conv1/ch1'));
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testPatchGrid(testCase)
            import OceanDL.Interpretability.TopActivatingPatches
            import OceanViz.Strategies.CNN.PatchGridStrategy
            images = cat(4, testCase.X, rand(12, 12, 1, 1));
            r = TopActivatingPatches.find(testCase.Engine, images, 'conv1', 1, 'TopK', 4);
            ax = axes(testCase.Fig);
            PatchGridStrategy().plot(ax, r, struct());
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testDeconv(testCase)
            import OceanDL.Interpretability.DeconvVisualizer
            import OceanViz.Strategies.CNN.DeconvStrategy
            r = DeconvVisualizer.reconstruct(testCase.Engine, testCase.X, 'act1', 1);
            ax = axes(testCase.Fig);
            DeconvStrategy().plot(ax, r, struct());
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testOcclusionHeatmap(testCase)
            import OceanDL.Interpretability.OcclusionSensitivity
            import OceanViz.Strategies.CNN.OcclusionHeatmapStrategy
            r = OcclusionSensitivity.run(testCase.Engine, testCase.X, 1, 'PatchSize', [4 4], 'Stride', [4 4]);
            ax = axes(testCase.Fig);
            OcclusionHeatmapStrategy().plot(ax, r, struct('Image', testCase.X));
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testSaliencyOverlay(testCase)
            import OceanDL.Interpretability.SaliencyMap
            import OceanViz.Strategies.CNN.SaliencyOverlayStrategy
            r = SaliencyMap.compute(testCase.Engine, testCase.X, 1, 'OutputLayer', 'dense1');
            ax = axes(testCase.Fig);
            SaliencyOverlayStrategy().plot(ax, r, struct('Image', testCase.X));
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testGradCAMOverlay(testCase)
            import OceanDL.Interpretability.GradCAM
            import OceanViz.Strategies.CNN.GradCAMOverlayStrategy
            r = GradCAM.compute(testCase.Engine, testCase.X, 1, 'conv2', 'OutputLayer', 'dense1');
            ax = axes(testCase.Fig);
            GradCAMOverlayStrategy().plot(ax, r, struct('Image', testCase.X));
            testCase.verifyNotEmpty(ax.Title.String);
        end

        function testAdversarialCompare(testCase)
            import OceanDL.Interpretability.AdversarialPerturbation
            import OceanViz.Strategies.CNN.AdversarialCompareStrategy
            r = AdversarialPerturbation.fgsm(testCase.Engine, testCase.X, 1, 0.05);
            ax = axes(testCase.Fig);
            AdversarialCompareStrategy().plot(ax, r, struct('Image', testCase.X));
            testCase.verifyNotEmpty(ax.Title.String);
        end
    end
end
