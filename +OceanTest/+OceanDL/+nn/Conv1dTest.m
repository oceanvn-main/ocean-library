classdef Conv1dTest < OceanTest.BaseTestCase
    % CONV1DTEST  OceanDL Conv1d (Function + nn module) validated against
    %   torch.nn.functional.conv1d (forward AND gradients; dense, depthwise with
    %   channel multiplier, depthwise without bias, general groups), finite
    %   differences (GradCheck), the §3 output-length formula, padding
    %   invariance of valid frames, single precision and a speed budget.
    %   Fixtures: fixtures/gen_torch_conv1d_reference.py (torch 2.6).

    properties (Constant)
        Cases = ["dense", "dw_mult", "dw_plain", "grouped"]
    end

    methods (Test)
        function testMatchesTorchForwardAndGradients(testCase)
            import OceanDL.Layout.DimFormat
            for name = testCase.Cases
                fx = testCase.fixture(name);
                x = OceanDL.Autograd.Tensor(fx.x, 'RequiresGrad', true);
                w = OceanDL.Autograd.Tensor(fx.w, 'RequiresGrad', true);
                args = {x, w};
                if isfield(fx, 'b')
                    b = OceanDL.Autograd.Tensor(fx.b, 'RequiresGrad', true);
                    args{3} = b;
                end
                y = OceanDL.Autograd.F.conv1d(args{:}, 'Stride', fx.cfg(1), 'Padding', fx.cfg(2), ...
                    'Dilation', fx.cfg(3), 'Groups', fx.cfg(4));
                testCase.verifyEqual(y.Data, fx.y, 'AbsTol', 1e-12, char(name + " forward"));
                L = sum(y .* fx.R, 'all');
                L.backward();
                testCase.verifyEqual(x.Grad, fx.gx, 'AbsTol', 1e-12, char(name + " dX"));
                testCase.verifyEqual(w.Grad, fx.gw, 'AbsTol', 1e-12, char(name + " dW"));
                if isfield(fx, 'b')
                    testCase.verifyEqual(b.Grad, fx.gb, 'AbsTol', 1e-12, char(name + " db"));
                end
            end
        end

        function testModuleUsesTorchWeightLayout(testCase)
            % Load the torch weights into the module through loadStateDict:
            % same (Cout x Cin/G x K) order, no permutation.
            fx = testCase.fixture("dw_mult");
            [Cin, ~, ~] = size(fx.x, 1:3);
            [Cout, ~, K] = size(fx.w, 1:3);
            conv = OceanDL.nn.Conv1d(Cin, Cout, K, 'Stride', fx.cfg(1), 'Padding', fx.cfg(2), ...
                'Dilation', fx.cfg(3), 'Groups', fx.cfg(4), 'Seed', 1);
            sd = conv.stateDict();
            sd('Weight') = fx.w;  sd('Bias') = fx.b;
            conv.loadStateDict(sd);
            y = conv.forward(fx.x);
            testCase.verifyEqual(y.Data, fx.y, 'AbsTol', 1e-12);
        end

        function testPyTorchInitialisationBounds(testCase)
            conv = OceanDL.nn.Conv1d(6, 9, 4, 'Groups', 3, 'Seed', 7);
            bound = 1 / sqrt(2 * 4);                                      % fanIn = (Cin/G) K
            testCase.verifySize(conv.Weight.Data, [9 2 4]);
            testCase.verifySize(conv.Bias.Data, [9 1]);
            testCase.verifyLessThanOrEqual(max(abs([conv.Weight.Data(:); conv.Bias.Data(:)])), bound);
            testCase.verifyGreaterThan(max(abs(conv.Weight.Data(:))), 0.8 * bound);   % actually spans the range
            again = OceanDL.nn.Conv1d(6, 9, 4, 'Groups', 3, 'Seed', 7);
            testCase.verifyEqual(again.Weight.Data, conv.Weight.Data);    % seeded = reproducible
            noBias = OceanDL.nn.Conv1d(6, 9, 4, 'Bias', false, 'Seed', 7);
            testCase.verifyEqual(numel(noBias.parameters()), 1);
        end

        function testGradcheckDenseAndDepthwise(testCase)
            s = RandStream('mt19937ar', 'Seed', 3);
            f = @(x, w, b) OceanDL.Autograd.F.conv1d(x, w, b, 'Stride', 2, 'Padding', 2, 'Dilation', 2);
            r = OceanDL.Autograd.GradCheck.check(f, {randn(s, 3, 2, 9), randn(s, 2, 3, 3), randn(s, 2, 1)});
            testCase.verifyTrue(r.Passed, sprintf('dense: max err %.3g', r.MaxAbsError));
            g = @(x, w) OceanDL.Autograd.F.conv1d(x, w, [], 'Padding', 1, 'Groups', 3);
            r = OceanDL.Autograd.GradCheck.check(g, {randn(s, 3, 2, 6), randn(s, 6, 1, 3)});
            testCase.verifyTrue(r.Passed, sprintf('depthwise: max err %.3g', r.MaxAbsError));
        end

        function testOutputLengthMatchesActualOutputs(testCase)
            for k = [1 3 5]
                for st = [1 2 3]
                    for p = [0 1 2]
                        for d = [1 2]
                            conv = OceanDL.nn.Conv1d(2, 3, k, 'Stride', st, 'Padding', p, 'Dilation', d, 'Seed', 1);
                            for T = 1:12
                                expected = floor((T + 2*p - d*(k-1) - 1) / st) + 1;     % contract §3
                                if expected < 1
                                    testCase.verifyError(@() conv.forward(randn(2, 1, T)), ...
                                        'OceanDL:Autograd:Conv1d:InputTooShort');
                                    testCase.verifyEqual(conv.outputLength(T), 0);
                                else
                                    y = conv.forward(randn(2, 1, T));
                                    testCase.verifyEqual(size(y.Data, 3), expected);
                                    testCase.verifyEqual(conv.outputLength(T), expected);
                                end
                            end
                        end
                    end
                end
            end
            conv = OceanDL.nn.Conv1d(2, 3, 3, 'Padding', 1);
            testCase.verifyEqual(conv.outputLength([0 4 7]), [0 4 7]);    % empty stays empty
        end

        function testValidFramesIgnoreBatchPadding(testCase)
            % Example n inside a zero-padded batch: its first outputLength(T_n)
            % output frames equal those of the example convolved alone.
            s = RandStream('mt19937ar', 'Seed', 4);
            conv = OceanDL.nn.Conv1d(4, 5, 5, 'Stride', 2, 'Padding', 2, 'Seed', 2);
            lens = [17 9 12];
            X = zeros(4, 3, max(lens));
            for n = 1:3, X(:, n, 1:lens(n)) = randn(s, 4, 1, lens(n)); end
            Y = conv.forward(X).Data;
            outLens = conv.outputLength(lens);
            for n = 1:3
                alone = conv.forward(X(:, n, 1:lens(n))).Data;
                testCase.verifySize(alone, [5 1 outLens(n)]);
                testCase.verifyEqual(Y(:, n, 1:outLens(n)), alone, 'AbsTol', 1e-13);
            end
        end

        function testSinglePrecision(testCase)
            fx = testCase.fixture("dense");
            x = OceanDL.Autograd.Tensor(single(fx.x), 'RequiresGrad', true);
            y = OceanDL.Autograd.F.conv1d(x, single(fx.w), single(fx.b), 'Stride', fx.cfg(1), ...
                'Padding', fx.cfg(2), 'Dilation', fx.cfg(3));
            testCase.verifyClass(y.Data, 'single');
            testCase.verifyEqual(double(y.Data), fx.y, 'AbsTol', 1e-5);
            L = sum(y .* single(fx.R), 'all');
            L.backward();
            testCase.verifyClass(x.Grad, 'single');
            testCase.verifyEqual(double(x.Grad), fx.gx, 'AbsTol', 1e-5);
            conv = OceanDL.nn.Conv1d(3, 2, 3, 'DataType', "single", 'Seed', 1);
            testCase.verifyClass(conv.forward(single(randn(3, 2, 6))).Data, 'single');
        end

        function testErrors(testCase)
            testCase.verifyError(@() OceanDL.nn.Conv1d(4, 6, 3, 'Groups', 4), 'OceanDL:nn:Conv1d:BadGroups');
            conv = OceanDL.nn.Conv1d(4, 6, 3);
            testCase.verifyError(@() conv.forward(randn(3, 2, 8)), 'OceanDL:nn:Conv1d:ChannelMismatch');
            testCase.verifyError(@() OceanDL.Autograd.F.conv1d(randn(4, 2, 8), randn(6, 3, 3)), ...
                'OceanDL:Autograd:Conv1d:ChannelMismatch');
        end

        function testSpeechSizedBatchIsFast(testCase)
            % 80 log-mel channels x 16 utterances x 200 frames, k = 5: forward + backward.
            s = RandStream('mt19937ar', 'Seed', 5);
            conv = OceanDL.nn.Conv1d(80, 80, 5, 'Padding', 2, 'Seed', 1);
            dw = OceanDL.nn.Conv1d(80, 80, 5, 'Padding', 2, 'Groups', 80, 'Seed', 1);
            X = OceanDL.Autograd.Tensor(randn(s, 80, 16, 200), 'RequiresGrad', true);
            for m = {conv, dw}
                L = sum(m{1}.forward(X), 'all'); L.backward();          % warm-up
                tic;
                for r = 1:3
                    L = sum(m{1}.forward(X), 'all'); L.backward();
                end
                t = toc / 3;
                testCase.log(1, sprintf("Conv1d groups=%d 80x16x200 k=5 fwd+bwd: %.1f ms", m{1}.Groups, 1e3 * t));
                testCase.verifyLessThan(t, 2, 'speech-sized Conv1d step should be well under 2 s');
            end
        end
    end

    methods (Access = private)
        function fx = fixture(~, name)
            import OceanDL.Layout.DimFormat
            dir = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            rd = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(dir, "conv1d_" + name + "_" + n + ".npy")));
            fx.x = DimFormat.toCanonical(rd("x_NCT"), "BCT");
            fx.w = rd("w");
            fx.y = DimFormat.toCanonical(rd("y_NCT"), "BCT");
            fx.R = DimFormat.toCanonical(rd("R_NCT"), "BCT");
            fx.gx = DimFormat.toCanonical(rd("gx_NCT"), "BCT");
            fx.gw = rd("gw");
            fx.cfg = rd("cfg");
            if isfile(fullfile(dir, "conv1d_" + name + "_b.npy"))
                fx.b = reshape(rd("b"), [], 1);
                fx.gb = reshape(rd("gb"), [], 1);
            end
        end
    end
end
