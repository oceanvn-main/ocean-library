classdef GenerativeBenchmark < matlab.perftest.TestCase
    % GENERATIVEBENCHMARK  Wall-clock cost of ONE training step (forward +
    %   backward, FP32, CPU unless a GPU is used) for the generative models,
    %   on the shapes used in the comparison experiments. These numbers were
    %   the basis of the run-time estimates (e.g. USPS VAE 0.49 s vs 2-layer
    %   HVAE 0.62 s at batch 32).
    %
    %   Measure (several samples, statistics):
    %     r = runperf('OceanTest.OceanDL.Generative.GenerativeBenchmark');
    %     sampleSummary(r)
    %   Under runtests the cases run once each as smoke tests (tag "Benchmark";
    %   exclude with runtests(..., 'ExcludeTag', 'Benchmark')).

    properties (TestParameter)
        Model = struct( ...
            'VAE_usps_32_64_64',      {{'hvae', [32 64 64], [0 1 0], 16, 32}}, ...
            'HVAE2_usps_32_64_64',    {{'hvae', [32 64 64], [0 1 1], 16, 32}}, ...
            'HVAE10_64_64_64',        {{'hvae', [64 64 64], [4 4 2], 16, 32}}, ...
            'HVAE10_bottleneck',      {{'bottleneck', [64 64 64], [4 4 2], 16, 32}}, ...
            'FlowUNet_usps_32_64',    {{'flow', [32 64], [], 16, 32}})
    end

    methods (TestClassSetup)
        function addLibrary(testCase)
            root = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
        end
    end

    methods (Test, TestTags = {'Benchmark'})
        function trainingStep(testCase, Model)
            [kind, widths, lat, H, B] = Model{:};
            X = rand(H, H, 1, B, 'single');
            switch kind
                case {'hvae', 'bottleneck'}
                    bt = "res"; if kind == "bottleneck", bt = "bottleneck"; end
                    net = OceanDL.Generative.HierarchicalVAE('Widths', widths, 'LatentsPerLevel', lat, ...
                        'LatentChannels', 4, 'BlockType', bt, 'Groups', 8, 'Seed', 1).to('DataType', "single");
                    loss = @() net.forward(X);
                case 'flow'
                    unet = OceanDL.Vision.UNet2D('InChannels', 1, 'Widths', widths, 'Groups', 8, 'Seed', 1);
                    net = OceanDL.Generative.Diffusion.DiffusionModel(unet, ...
                        OceanDL.Generative.Diffusion.Schedules.RectifiedFlow(), 'Target', "velocity", 'Seed', 2);
                    net.to('DataType', "single");
                    loss = @() net.forward(2 * X - 1);
            end
            net.zeroGrad(); L = loss(); L.backward();                  % warm-up (JIT, allocation)
            while testCase.keepMeasuring
                net.zeroGrad();
                L = loss();
                L.backward();
            end
            testCase.verifyTrue(isfinite(double(L.Data)));
        end
    end
end
