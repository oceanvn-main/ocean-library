classdef DeviceTest < OceanTest.BaseTestCase
    % DEVICETEST  Module.to / OceanDL.Device: precision and placement follow
    %   the parameters through forward, backward, the optimizer and
    %   stateDict round trips, for every model family; single-precision
    %   outputs agree with double to single-precision accuracy.

    methods (Test)
        function testDeviceCastLeavesLabelsAndMasksAlone(testCase)
            D = OceanDL.Device;
            testCase.verifyClass(D.cast(randn(2), 'DataType', "single"), 'single');
            testCase.verifyClass(D.cast(int32([1 2]), 'DataType', "single"), 'int32');
            testCase.verifyClass(D.cast(true(2), 'DataType', "single"), 'logical');
            testCase.verifyEqual(D.name(randn(2)), "cpu");
            testCase.verifyEqual(D.dataType(single(1)), "single");
            testCase.verifyError(@() D.cast(OceanDL.Autograd.Tensor(1), 'DataType', "single"), 'OceanDL:Device:Tensor');
            if ~D.gpuAvailable()
                testCase.verifyError(@() D.cast(1, 'Device', "gpu"), 'OceanDL:Device:NoGPU');
            end
        end

        function testEveryModelFamilyRunsInSingle(testCase)
            cases = OceanTest.OceanDL.nn.DeviceTest.models();
            for k = 1:size(cases, 1)
                [name, make, X, lossOf] = cases{k, :};
                ref = make(); net = make();
                net.loadStateDict(ref.stateDict());
                net.to('DataType', "single");
                testCase.verifyEqual(net.dataType(), "single", name);
                L64 = lossOf(ref, X);
                L32 = lossOf(net, single(X));
                testCase.verifyClass(L32.Data, 'single', name);
                testCase.verifyEqual(double(L32.Data), L64.Data, 'RelTol', 1e-4, name);
                L32.backward(); L64.backward();
                P32 = net.namedParameters(); P64 = ref.namedParameters();
                for i = 1:numel(P32)
                    g = P32(i).Param.Grad;
                    if isempty(g), continue; end
                    testCase.verifyClass(g, 'single', [name ' ' P32(i).Name]);
                    scale = max(abs(P64(i).Param.Grad(:))) + 1e-6;
                    testCase.verifyLessThan(max(abs(double(g(:)) - P64(i).Param.Grad(:))) / scale, 1e-2, [name ' ' P32(i).Name]);
                end
            end
        end

        function testOptimizerStateAndStateDictKeepPrecision(testCase)
            net = OceanDL.Vision.ViT('ImageSize', [8 8], 'Channels', 1, 'PatchSize', 4, 'Dim', 8, 'Depth', 1, ...
                'Heads', 2, 'NumClasses', 3, 'Seed', 1).to('DataType', "single");
            opt = OceanDL.optim.ModuleOptimizer.forModule(net, @(wd) OceanStats.Optim.AdamWOptimizer('LR', 1e-3, 'WeightDecay', wd));
            for s = 1:2
                opt.zeroGrad();
                L = OceanDL.Autograd.F.crossEntropy(net.forward(randn(8, 8, 1, 4, 'single')), [1 2 3 1]);
                L.backward(); opt.step();
            end
            P = net.parameters();
            testCase.verifyTrue(all(cellfun(@(p) isa(p.Data, 'single'), P)));
            sd = net.stateDict();
            fresh = OceanDL.Vision.ViT('ImageSize', [8 8], 'Channels', 1, 'PatchSize', 4, 'Dim', 8, 'Depth', 1, ...
                'Heads', 2, 'NumClasses', 3).to('DataType', "single");
            fresh.loadStateDict(containers.Map(keys(sd), cellfun(@double, values(sd), 'UniformOutput', false)));
            testCase.verifyEqual(fresh.dataType(), "single");                 % double state dict cast on load
            back = fresh.to('DataType', "double");
            testCase.verifyEqual(back.dataType(), "double");
        end

        function testGpuPathWhenAvailable(testCase)
            testCase.assumeTrue(OceanDL.Device.gpuAvailable(), 'no usable GPU on this machine');
            net = OceanDL.nn.Linear(4, 3, 'Seed', 1).to('DataType', "single", 'Device', "gpu");
            Y = net.forward(OceanDL.Device.cast(randn(4, 5), 'DataType', "single", 'Device', "gpu"));
            testCase.verifyEqual(OceanDL.Device.name(Y), "gpu");
            L = sum(Y .* Y, 'all'); L.backward();
            testCase.verifyEqual(OceanDL.Device.name(net.Weight.Grad), "gpu");
        end
    end

    methods (Static)
        function c = models()
            F = OceanDL.Autograd.F;
            ce = @(net, X, y) F.crossEntropy(net.forward(X), y);
            s = RandStream('mt19937ar', 'Seed', 2);
            img = randn(s, 16, 16, 1, 3);
            seq = randn(s, 3, 2, 5);
            c = {
                "ViT",  @() OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, 'PatchSize', 4, 'Dim', 16, ...
                            'Depth', 2, 'Heads', 2, 'NumClasses', 4, 'Seed', 3), img, @(n, X) ce(n, X, [1 2 3])
                "Swin", @() OceanDL.Vision.Swin('Channels', 1, 'PatchSize', 2, 'Dim', 8, 'Depths', [2 1], ...
                            'Heads', [2 2], 'WindowSize', 2, 'NumClasses', 4, 'Seed', 4), img, @(n, X) ce(n, X, [1 2 3])
                "LSTM", @() OceanDL.Recurrent.RecurrentLayer(OceanDL.Recurrent.LSTMCell(3, 4, 'Seed', 5)), seq, ...
                            @(n, X) sum(n.forward(X) .^ 2, 'all')
                "MAE",  @() OceanDL.Vision.MaskedAutoencoder(OceanDL.Vision.ViT('ImageSize', [16 16], 'Channels', 1, ...
                            'PatchSize', 4, 'Dim', 16, 'Depth', 1, 'Heads', 2, 'Positions', "sincos", 'Pooling', "mean", ...
                            'Seed', 6), 'DecoderDim', 8, 'DecoderDepth', 1, 'DecoderHeads', 2, 'Seed', 7), img, ...
                            @(n, X) n.forward(X) };
        end
    end
end
