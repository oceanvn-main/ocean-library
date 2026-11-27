classdef EncodersTest < OceanTest.BaseTestCase
    % ENCODERSTEST  ConvEncoder / RecurrentEncoder: length propagation, padding invariance,
    %   gradients (finite differences), config round trip.

    methods (Test)
        function testConvOutputLengthClosedForm(testCase)
            enc = OceanNLP.Speech.Models.Encoders.ConvEncoder(6, 'Channels', [5 4 3], ...
                'KernelSizes', [5 3 4], 'Strides', [2 1 3]);
            lens = 0:40;
            expect = lens;
            for l = 1:3
                k = enc.KernelSizes(l); s = enc.Strides(l); p = floor(k / 2);
                nz = expect > 0;
                expect(nz) = floor((expect(nz) + 2 * p - (k - 1) - 1) / s) + 1;   % contract sec. 3
                expect = max(expect, 0);
            end
            testCase.verifyEqual(enc.outputLength(lens), expect);
            X = randn(6, 1, 23);
            [H, l2] = enc.forward(X, 23);
            testCase.verifyEqual(size(H, 3), enc.outputLength(23));
            testCase.verifyEqual(l2, enc.outputLength(23));
            testCase.verifyEqual(enc.OutputDim, 3);
        end

        function testConvPaddingInvariance(testCase)
            enc = OceanNLP.Speech.Models.Encoders.ConvEncoder(4, 'Channels', [6 5], ...
                'KernelSizes', [5 3], 'Strides', [2 1], 'Seed', 3);
            testCase.verifyPaddingInvariant(enc, 4, [17 9 4 1]);
        end

        function testRecurrentPaddingInvariance(testCase)
            for cellType = ["gru", "lstm"]
                enc = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(4, 'HiddenSize', 3, ...
                    'NumLayers', 2, 'Cell', cellType, 'Bidirectional', true, 'Seed', 7);
                testCase.verifyEqual(enc.OutputDim, 6);
                testCase.verifyPaddingInvariant(enc, 4, [11 6 2]);
            end
            uni = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(4, 'HiddenSize', 3, ...
                'NumLayers', 1, 'Bidirectional', false);
            testCase.verifyEqual(uni.OutputDim, 3);
            testCase.verifyPaddingInvariant(uni, 4, [8 3]);
        end

        function testBidirectionalDirectionsDiffer(testCase)
            % the two directions must not share an initialisation (distinct seeds)
            enc = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(4, 'HiddenSize', 3, 'NumLayers', 1);
            L = enc.Stack.Layers{1};
            testCase.verifyNotEqual(L.ForwardLayer.Cell.Wx.Data, L.BackwardLayer.Cell.Wx.Data);
        end

        function testConvInputGradient(testCase)
            enc = OceanNLP.Speech.Models.Encoders.ConvEncoder(3, 'Channels', [4 3], ...
                'KernelSizes', [3 3], 'Strides', [2 1], 'Seed', 1);
            lens = [7 4];
            r = OceanDL.Autograd.GradCheck.check(@(x) enc.forward(x, lens), {randn(3, 2, 7)});
            testCase.verifyTrue(r.Passed, sprintf('max abs err %.3g', r.MaxAbsError));
            % padded input frames receive exactly zero gradient
            g = r.Analytic{1};
            testCase.verifyEqual(nnz(g(:, 2, 5:7)), 0);
        end

        function testRecurrentInputGradient(testCase)
            enc = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(3, 'HiddenSize', 2, 'NumLayers', 2, 'Seed', 4);
            lens = [5 3];
            r = OceanDL.Autograd.GradCheck.check(@(x) enc.forward(x, lens), {randn(3, 2, 5)});
            testCase.verifyTrue(r.Passed, sprintf('max abs err %.3g', r.MaxAbsError));
            testCase.verifyEqual(nnz(r.Analytic{1}(:, 2, 4:5)), 0);
        end

        function testParameterGradients(testCase)
            % autograd parameter gradients vs central differences (every parameter tensor)
            encs = {OceanNLP.Speech.Models.Encoders.ConvEncoder(3, 'Channels', [4 3], 'KernelSizes', [3 2], 'Strides', [2 1], 'Seed', 2), ...
                    OceanNLP.Speech.Models.Encoders.RecurrentEncoder(3, 'HiddenSize', 2, 'NumLayers', 1, 'Seed', 5), ...
                    OceanNLP.Speech.Models.Encoders.RecurrentEncoder(3, 'HiddenSize', 2, 'NumLayers', 1, 'Cell', "lstm", 'Seed', 6)};
            X = randn(3, 2, 6); lens = [6 4];
            for e = 1:numel(encs)
                err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError( ...
                    encs{e}, @(m) m.forward(X, lens), 3);
                testCase.verifyLessThan(err, 1e-6, class(encs{e}));
            end
        end

        function testConfigRoundTrip(testCase)
            encs = {OceanNLP.Speech.Models.Encoders.ConvEncoder(5, 'Channels', [4 6], 'KernelSizes', [3 5], ...
                        'Strides', [1 2], 'Activation', "relu", 'LayerNorm', false, 'Dropout', 0.2, 'Seed', 9), ...
                    OceanNLP.Speech.Models.Encoders.RecurrentEncoder(5, 'HiddenSize', 4, 'NumLayers', 2, ...
                        'Cell', "lstm", 'Bidirectional', false, 'Dropout', 0.1, 'Seed', 11)};
            X = randn(5, 2, 9); lens = [9 6];
            for e = 1:numel(encs)
                cfg = encs{e}.config();
                e2 = OceanNLP.Speech.Models.RecognizerFactory.encoderFromConfig(cfg);
                testCase.verifyClass(e2, class(encs{e}));
                testCase.verifyEqual(e2.config(), cfg);
                testCase.verifyEqual(e2.stateDict().keys(), encs{e}.stateDict().keys());
                % same seeds -> identical initial weights; identical outputs in eval mode
                encs{e}.eval(); e2.eval();
                testCase.verifyEqual(e2.forward(X, lens).Data, encs{e}.forward(X, lens).Data);
            end
        end

        function testErrors(testCase)
            testCase.verifyError(@() OceanNLP.Speech.Models.Encoders.ConvEncoder(3, 'Channels', [4 4], 'KernelSizes', 3, 'Strides', [1 1]), ...
                'OceanNLP:Speech:Models:ConvEncoder:LayerSpec');
            enc = OceanNLP.Speech.Models.Encoders.ConvEncoder(3);
            testCase.verifyError(@() enc.forward(randn(4, 1, 5), 5), 'OceanNLP:Speech:Models:ConvEncoder:InputDim');
            testCase.verifyError(@() enc.forward(randn(3, 2, 5), [5 6]), 'OceanNLP:Speech:Models:ConvEncoder:Lengths');
            rec = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(3);
            testCase.verifyError(@() rec.forward(randn(2, 1, 5), 5), 'OceanNLP:Speech:Models:RecurrentEncoder:InputDim');
            testCase.verifyError(@() OceanNLP.Speech.Models.RecognizerFactory.encoderFromConfig(struct('Type', "Nope")), ...
                'OceanNLP:Speech:Models:RecognizerFactory:UnknownType');
        end
    end

    methods
        function verifyPaddingInvariant(testCase, enc, F, lens)
            % Valid outputs of a padded batch == outputs of each utterance run alone.
            enc.eval();
            N = numel(lens); T = max(lens);
            X = zeros(F, N, T);
            for n = 1:N
                X(:, n, 1:lens(n)) = randn(F, 1, lens(n));
            end
            [H, L] = enc.forward(X, lens);
            H = H.Data;
            testCase.verifyEqual(L, enc.outputLength(lens));
            for n = 1:N
                [h1, l1] = enc.forward(X(:, n, 1:lens(n)), lens(n));
                testCase.verifyEqual(L(n), l1);
                testCase.verifyEqual(H(:, n, 1:l1), h1.Data, 'AbsTol', 1e-12);
                testCase.verifyEqual(nnz(H(:, n, l1+1:end)), 0, 'padded output frames must be zero');
            end
            % garbage in the padded input region must not change anything
            Xj = X;
            for n = 1:N
                Xj(:, n, lens(n)+1:T) = 1e3 * randn(F, 1, T - lens(n));
            end
            Hj = enc.forward(Xj, lens).Data;
            testCase.verifyEqual(Hj, H, 'AbsTol', 1e-12);
        end
    end

    methods (Static)
        function err = paramGradError(model, fn, seed)
            % Max |autograd - central difference| over every parameter entry of MODEL for
            % the projected scalar sum(fn(model) .* R).
            model.eval();
            Y = fn(model);
            R = randn(RandStream('mt19937ar', 'Seed', seed), size(Y.Data));
            model.zeroGrad();
            L = sum(fn(model) .* R, 'all');
            L.backward();
            P = model.parameters();
            h = 1e-6; err = 0;
            guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            for k = 1:numel(P)
                g = P{k}.Grad;
                if isempty(g), g = zeros(size(P{k}.Data)); end
                for i = 1:numel(P{k}.Data)
                    v = P{k}.Data(i);
                    P{k}.Data(i) = v + h; lp = sum(fn(model).Data .* R, 'all');
                    P{k}.Data(i) = v - h; lm = sum(fn(model).Data .* R, 'all');
                    P{k}.Data(i) = v;
                    err = max(err, abs((lp - lm) / (2 * h) - g(i)) / max(1, abs(g(i))));
                end
            end
        end
    end
end
