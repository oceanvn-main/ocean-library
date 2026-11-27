classdef HuBERTModelTest < OceanTest.BaseTestCase
    % HUBERTMODELTEST  Span masking statistics/contiguity/padding, masked-only loss (target
    %   perturbation), loss vs an independent cross-entropy, gradients (finite differences),
    %   padding invariance, target-frame mapping, encoder contract inside CTCRecognizer.

    methods (Test)
        function testMaskRateMatchesClosedForm(testCase)
            p = 0.13; l = 5;
            rs = RandStream('mt19937ar', 'Seed', 5);
            for Tn = [7 23 40]
                R = 4000;
                M = OceanNLP.Speech.Models.Pretrain.HuBERTModel.spanMask(repmat(Tn, 1, R), Tn, p, l, rs, ...
                    'MaxSpanFraction', 0.5, 'MinSpans', 0);
                frac = mean(M, 2);
                e = OceanNLP.Speech.Models.Pretrain.HuBERTModel.expectedMaskFraction(Tn, p, l, 0.5);
                se = std(frac) / sqrt(R);
                testCase.verifyLessThan(abs(mean(frac) - e), 4 * se + 1e-12, ...
                    sprintf('T=%d: empirical %.4f vs closed form %.4f', Tn, mean(frac), e));
                % independent brute-force expectation by exhaustive start-pattern probability:
                % P(frame t masked) = 1 - (1-p)^{#starts that cover t}
                ln = min(l, max(1, floor(0.5 * Tn)));
                cover = zeros(1, Tn);
                for s = 1:Tn
                    cover(s:min(s + ln - 1, Tn)) = cover(s:min(s + ln - 1, Tn)) + 1;
                end
                testCase.verifyEqual(e, mean(1 - (1 - p) .^ cover), 'AbsTol', 1e-14);
            end
        end

        function testSpansContiguousAndNoPadding(testCase)
            rs = RandStream('mt19937ar', 'Seed', 1);
            lens = [30 12 3 0 1 25];
            T = 34;
            for rep = 1:200
                M = OceanNLP.Speech.Models.Pretrain.HuBERTModel.spanMask(lens, T, 0.13, 5, rs, 'MaxSpanFraction', 0.5, 'MinSpans', 1);
                for n = 1:numel(lens)
                    testCase.assertFalse(any(M(n, lens(n)+1:end)), 'padded frame masked');
                    if lens(n) > 0
                        testCase.assertTrue(any(M(n, :)), 'MinSpans = 1 must mask something');
                    end
                    ln = min(5, max(1, floor(0.5 * lens(n))));
                    d = diff([false, M(n, 1:lens(n)), false]);
                    starts = find(d == 1); stops = find(d == -1) - 1;
                    % every maximal run is a union of whole spans: length >= ln unless clipped at the end
                    runLen = stops - starts + 1;
                    testCase.assertTrue(all(runLen >= ln | stops == lens(n)), 'span shorter than its length');
                end
            end
        end

        function testLossOnlyOnMaskedValidFrames(testCase)
            m = testCase.tinyModel('Alpha', 1);
            m.eval();
            [X, lens] = testCase.batch([14 9 5], 4);
            lo = m.outputLength(lens); T = max(lo);
            rs = RandStream('mt19937ar', 'Seed', 2);
            M = m.sampleMask(lo, T, rs);
            Y = randi(rs, m.NumUnits, numel(lens), T);
            L0 = m.pretrainLoss(X, lens, Y, M).Data;
            valid = (1:T) <= lo(:);
            Yp = Y;
            Yp(~M) = mod(Y(~M), m.NumUnits) + 1;          % change every unmasked AND padded target
            testCase.verifyEqual(m.pretrainLoss(X, lens, Yp, M).Data, L0);
            Ym = Y;
            Ym(M) = mod(Y(M), m.NumUnits) + 1;            % change masked targets -> loss changes
            testCase.verifyNotEqual(m.pretrainLoss(X, lens, Ym, M).Data, L0);
            % independent recomputation: -mean over masked valid frames of log softmax
            out = m.forwardMasked(X, lens, M);
            Z = out.Logits.Data;
            ll = Z - max(Z, [], 1); ll = ll - log(sum(exp(ll), 1));
            acc = 0; cnt = 0;
            for n = 1:numel(lens)
                for t = 1:lo(n)
                    if M(n, t)
                        acc = acc - ll(Y(n, t), n, t); cnt = cnt + 1;
                    end
                end
            end
            testCase.verifyEqual(L0, acc / cnt, 'AbsTol', 1e-12);
            testCase.verifyLessThanOrEqual(max(abs(Z), [], 'all'), 1 / m.Temperature + 1e-9);   % cosine / tau
            testCase.verifyFalse(any(M & ~valid, 'all'));
            % with alpha < 1 the unmasked targets matter
            m2 = testCase.tinyModel('Alpha', 0.5); m2.eval();
            testCase.verifyNotEqual(m2.pretrainLoss(X, lens, Yp, M).Data, m2.pretrainLoss(X, lens, Y, M).Data);
            % a mask over padding is refused
            Mbad = M; Mbad(3, end) = true;
            testCase.verifyError(@() m.pretrainLoss(X, lens, Y, Mbad), 'OceanNLP:Speech:Models:HuBERTModel:MaskPadding');
        end

        function testGradCheckInputAndParameters(testCase)
            m = testCase.tinyModel('Alpha', 0.7);
            [X, lens] = testCase.batch([8 5], 4);
            lo = m.outputLength(lens); T = max(lo);
            rs = RandStream('mt19937ar', 'Seed', 4);
            M = false(2, T); M(1, 2:3) = true; M(2, 1) = true;
            Y = randi(rs, m.NumUnits, 2, T);
            m.eval();
            r = OceanDL.Autograd.GradCheck.check(@(x) m.pretrainLoss(x, lens, Y, M), {X});
            testCase.verifyTrue(r.Passed, sprintf('input grad max abs err %.3g', r.MaxAbsError));
            testCase.verifyEqual(nnz(r.Analytic{1}(:, 2, lens(2)+1:end)), 0, 'padded inputs get zero gradient');
            err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError(m, @(mm) mm.pretrainLoss(X, lens, Y, M), 1);
            testCase.verifyLessThan(err, 1e-6);
            % the mask embedding receives gradient only through masked frames
            m.zeroGrad();
            L = m.pretrainLoss(X, lens, Y, false(size(M)) | M);
            L.backward();
            testCase.verifyGreaterThan(norm(m.MaskEmbedding.Grad), 0);
            m.zeroGrad();
            H = m.forward(X, lens); s = sum(H .* H, 'all'); s.backward();
            g = m.MaskEmbedding.Grad;
            testCase.verifyTrue(isempty(g) || nnz(g) == 0, 'unmasked forward must not use the mask embedding');
        end

        function testPaddingInvariance(testCase)
            m = testCase.tinyModel();
            m.eval();
            lens = [13 7 4];
            [X, ~] = testCase.batch(lens, 4);
            [H, lo] = m.forward(X, lens);
            H = H.Data;
            testCase.verifyEqual(lo, m.outputLength(lens));
            T = size(H, 3);
            rs = RandStream('mt19937ar', 'Seed', 8);
            M = m.sampleMask(lo, T, rs);
            Zb = m.forwardMasked(X, lens, M).Logits.Data;
            for n = 1:3
                [h1, l1] = m.forward(X(:, n, 1:lens(n)), lens(n));
                testCase.verifyEqual(H(:, n, 1:l1), h1.Data, 'AbsTol', 1e-10);
                testCase.verifyEqual(nnz(H(:, n, l1+1:end)), 0);
                z1 = m.forwardMasked(X(:, n, 1:lens(n)), lens(n), M(n, 1:l1)).Logits.Data;
                testCase.verifyEqual(Zb(:, n, 1:l1), z1, 'AbsTol', 1e-10);
            end
            Xj = X;
            for n = 1:3, Xj(:, n, lens(n)+1:end) = 1e3; end
            testCase.verifyEqual(m.forward(Xj, lens).Data, H, 'AbsTol', 1e-10);
            % an empty utterance in the batch stays finite and zero
            [He, le] = m.forward(cat(2, X, zeros(4, 1, size(X, 3))), [lens 0]);
            testCase.verifyEqual(le(end), 0);
            testCase.verifyTrue(all(isfinite(He.Data), 'all'));
            testCase.verifyEqual(He.Data(:, 1:3, :), H, 'AbsTol', 1e-10);
        end

        function testTargetFrameIndex(testCase)
            m = OceanNLP.Speech.Models.Pretrain.HuBERTModel(4, 'ConvChannels', [4 4], 'ConvKernels', [5 3], ...
                'ConvStrides', [2 1], 'Dim', 8, 'Depth', 1, 'Heads', 2, 'FFNDim', 8, 'NumUnits', 3, 'ProjDim', 4);
            % brute force: the receptive-field centre of output t through conv (k, s, p = floor(k/2))
            for Tin = [1 2 9 10 37]
                To = m.outputLength(Tin);
                c = 1:To;
                for l = 2:-1:1
                    k = m.ConvKernels(l); s = m.ConvStrides(l); p = floor(k / 2);
                    c = (c - 1) * s - p + 1 + (k - 1) / 2;   % centre of the window [(t-1)s - p + 1, (t-1)s - p + k]
                end
                testCase.verifyEqual(m.targetFrameIndex(To, Tin), min(c, Tin));
                testCase.verifyLessThanOrEqual(max([m.targetFrameIndex(To, Tin), 0]), Tin);
            end
        end

        function testEncoderContractInCTCRecognizer(testCase)
            OceanNLP.Speech.Models.RecognizerFactory.register("encoder", "HuBERTModel", ...
                @OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig);
            vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(["one two", "three"], 'Unit', "char");
            feats = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 4);
            h = testCase.tinyModel();
            cfgBefore = h.config();
            testCase.verifyEqual(OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig(cfgBefore).config(), cfgBefore);
            h.dropPretrainHead();
            testCase.verifyFalse(h.config().PretrainHead);
            ctc = OceanNLP.Speech.Models.CTCRecognizer(feats, vocab, {h}, 'Seed', 2);
            [X, lens] = testCase.batch([20 15], 4);
            b = struct('X', X, 'Lengths', lens, 'Mask', (1:20) <= lens(:), 'Targets', {{vocab.encode("one"), vocab.encode("two")}});
            L = ctc.loss(b);
            testCase.verifyTrue(isfinite(L.Data));
            ctc2 = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(ctc.config());
            testCase.verifyEqual(ctc2.config(), ctc.config());
            ctc2.loadStateDict(ctc.stateDict());
            ctc.eval(); ctc2.eval();
            testCase.verifyEqual(ctc2.forward(b).Logits.Data, ctc.forward(b).Logits.Data);
            % pretrained weights transfer by name into the head-less model
            src = testCase.tinyModel(); src.dropPretrainHead();
            testCase.verifyEqual(sort(src.stateDict().keys()), sort(h.stateDict().keys()));
        end

        function testHiddenStates(testCase)
            m = testCase.tinyModel();
            [X, lens] = testCase.batch([9 6], 4);
            Z0 = m.hiddenStates(X, lens, 0);
            Z2 = m.hiddenStates(X, lens, 2);
            testCase.verifyEqual(size(Z2), [m.Dim, 2, max(m.outputLength(lens))]);
            testCase.verifyNotEqual(Z0, Z2);
            testCase.verifyTrue(m.Training, 'training flag restored');
            testCase.verifyError(@() m.hiddenStates(X, lens, 3), 'OceanNLP:Speech:Models:HuBERTModel:Layer');
        end
    end

    methods (Static)
        function m = tinyModel(varargin)
            m = OceanNLP.Speech.Models.Pretrain.HuBERTModel(4, 'ConvChannels', [5 4], 'ConvKernels', [3 3], ...
                'ConvStrides', [2 1], 'Dim', 6, 'Depth', 2, 'Heads', 2, 'FFNDim', 8, 'Dropout', 0, ...
                'NumUnits', 5, 'ProjDim', 3, 'Seed', 3, varargin{:});
        end

        function [X, lens] = batch(lens, F)
            rs = RandStream('mt19937ar', 'Seed', sum(lens));
            X = zeros(F, numel(lens), max(lens));
            for n = 1:numel(lens)
                X(:, n, 1:lens(n)) = randn(rs, F, 1, lens(n));
            end
        end
    end
end
