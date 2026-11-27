classdef EncoderDecoderRecognizerTest < OceanTest.BaseTestCase
    % ENCODERDECODERRECOGNIZERTEST  Teacher forcing layout (Bos/Eos/Pad), causal decoder
    %   (future target tokens never change earlier logits), padding invariance of logits and
    %   of each loss term (masked CE token-weighted, CTC per-utterance mean), label smoothing
    %   and joint-CTC weighting against independent computations, full-model parameter
    %   gradients, transcribe input forms + Transcript honesty, step function == teacher
    %   forcing, returned cross-attention, config/factory and checkpoint save -> load ->
    %   identical transcripts.

    properties
        Vocab
        Features
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(["one two", "three", "zero"], 'Unit', "char");
            testCase.Features = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 10);
            OceanNLP.Speech.Models.RecognizerFactory.register("encoder", "ConformerEncoder", ...
                @OceanNLP.Speech.Models.Encoders.ConformerEncoder.fromConfig);
            OceanNLP.Speech.Models.RecognizerFactory.register("recognizer", "EncoderDecoderRecognizer", ...
                @OceanNLP.Speech.Models.EncoderDecoderRecognizer.fromConfig);
        end
    end

    methods (Test)
        function testTeacherForcingLayoutAndCausality(testCase)
            m = testCase.makeModel();
            m.eval();
            v = testCase.Vocab;
            b = testCase.makeBatch([20 13], ["one two", "zero"]);
            out = m.forward(b);
            K = v.numClasses();
            y1 = v.encode("one two"); y2 = v.encode("zero");
            U = numel(y1) + 1;
            testCase.verifySize(out.Logits, [K 2 U]);
            testCase.verifyEqual(out.Targets(1, :), [y1 v.Eos]);
            testCase.verifyEqual(out.Targets(2, :), [y2 v.Eos v.Pad * ones(1, U - numel(y2) - 1)]);
            testCase.verifyEqual(out.LogitLengths, [U numel(y2) + 1]);
            testCase.verifyEqual(out.Lengths, m.outputLength([20 13]));
            % causality: replace the later target tokens -> earlier logits identical
            for cut = [1 3 5]
                b2 = b; b2.Targets{1} = [y1(1:cut), repmat(v.encode("e"), 1, numel(y1) - cut)];
                o2 = m.forward(b2);
                testCase.verifyEqual(o2.Logits.Data(:, 1, 1:cut + 1), out.Logits.Data(:, 1, 1:cut + 1), 'AbsTol', 1e-12);
                testCase.verifyGreaterThan(max(abs(o2.Logits.Data(:, 1, cut + 2:end) - out.Logits.Data(:, 1, cut + 2:end)), [], 'all'), 1e-6);
            end
            testCase.verifyError(@() m.forward(rmfield(b, 'Targets')), 'OceanNLP:Speech:Models:EncoderDecoderRecognizer:NoTargets');
            bb = b; bb.Targets{2} = [v.Eos 6];
            testCase.verifyError(@() m.loss(bb), 'OceanNLP:Speech:Models:EncoderDecoderRecognizer:SpecialInTarget');
        end

        function testPaddingInvarianceAndLossTerms(testCase)
            m = testCase.makeModel('CTCWeight', 0.3, 'LabelSmoothing', 0.1, 'Dropout', 0.2);
            m.eval();
            lens = [60 44 30];
            texts = ["one two", "three", "zero"];
            [b, singles] = testCase.makeBatch(lens, texts);
            out = m.forward(b);
            s = m.losses(b);
            K = testCase.Vocab.numClasses();
            nllSum = 0; nTok = 0; ctc = zeros(1, 3);
            for n = 1:3
                o1 = m.forward(singles{n});
                u = o1.LogitLengths;
                testCase.verifyEqual(out.Logits.Data(:, n, 1:u), o1.Logits.Data, 'AbsTol', 1e-11);
                testCase.verifyEqual(out.CTCLogits.Data(:, n, 1:o1.Lengths), o1.CTCLogits.Data, 'AbsTol', 1e-11);
                % independent smoothed CE: (1-eps) NLL + eps * mean_k(-log p_k), summed over tokens
                Z = reshape(o1.Logits.Data, K, u);
                lp = Z - log(sum(exp(Z), 1));
                tg = o1.Targets(1, :);
                nll = -lp(sub2ind([K u], tg, 1:u));
                nllSum = nllSum + sum(0.9 * nll + 0.1 * (-mean(lp, 1)));
                nTok = nTok + u;
                s1 = m.losses(singles{n});
                ctc(n) = s1.CTC.Data;
            end
            testCase.verifyEqual(s.Attention.Data, nllSum / nTok, 'AbsTol', 1e-11);   % token-weighted mean
            testCase.verifyEqual(s.CTC.Data, mean(ctc), 'AbsTol', 1e-11);            % per-utterance mean
            testCase.verifyEqual(s.Total.Data, 0.7 * s.Attention.Data + 0.3 * s.CTC.Data, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.loss(b).Data, s.Total.Data);
            bj = b;                                                  % garbage in padded input frames
            for n = 1:3, bj.X(:, n, lens(n)+1:end) = 1e3 * randn(10, 1, max(lens) - lens(n)); end
            testCase.verifyEqual(m.loss(bj).Data, s.Total.Data, 'AbsTol', 1e-11);
        end

        function testCTCInfeasibleUtterance(testCase)
            % 9 frames -> 3 encoder frames cannot emit "three" (5 labels): with CTCZeroInfinity
            % (default) its CTC term is 0, the other utterance's CTC term is halved by the mean;
            % with CTCZeroInfinity = false OceanDL's error surfaces.
            m = testCase.makeModel('CTCWeight', 0.5);
            m.eval();
            [b, singles] = testCase.makeBatch([9 60], ["three", "one"]);
            s = m.losses(b);
            s2 = m.losses(singles{2});
            testCase.verifyEqual(s.CTC.Data, s2.CTC.Data / 2, 'AbsTol', 1e-12);
            testCase.verifyTrue(isfinite(s.Total.Data));
            cfg = m.config(); cfg.CTCZeroInfinity = false;
            m2 = OceanNLP.Speech.Models.EncoderDecoderRecognizer.fromConfig(cfg);
            testCase.verifyError(@() m2.loss(b), 'OceanDL:Autograd:CTCLoss:Infeasible');
        end

        function testFullModelParameterGradients(testCase)
            enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(10, 'Dim', 4, 'NumHeads', 2, 'NumBlocks', 1, ...
                'FFMult', 1, 'ConvKernel', 3, 'Seed', 1);
            m = OceanNLP.Speech.Models.EncoderDecoderRecognizer(testCase.Features, testCase.Vocab, {enc}, ...
                'DecoderLayers', 1, 'DecoderHeads', 2, 'DecoderFF', 4, 'Dropout', 0, 'LabelSmoothing', 0.1, ...
                'CTCWeight', 0.4, 'Seed', 2);
            b = testCase.makeBatch([12 9], ["one", "two"]);
            err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError(m, @(mm) mm.loss(b), 1);
            testCase.verifyLessThan(err, 1e-6);
        end

        function testTranscribeFormsStepFunctionAndAttention(testCase)
            m = testCase.makeModel('ModelId', "ed-test");
            rs = RandStream('mt19937ar', 'Seed', 5);
            w = {OceanNLP.Speech.Core.Waveform(0.1 * randn(rs, 3000, 1), 8000), ...
                 OceanNLP.Speech.Core.Waveform(0.1 * randn(rs, 1700, 1), 8000)};
            fsq = cellfun(@(x) testCase.Features.forward(x), w, 'UniformOutput', false);
            batch = OceanNLP.Speech.Models.CTCRecognizer.padSequences(fsq);
            for dec = ["greedy", "beam"]
                a = m.transcribe(w, 'Decoder', dec, 'BeamWidth', 3, 'MaxLength', 8);
                b = m.transcribe(fsq, 'Decoder', dec, 'BeamWidth', 3, 'MaxLength', 8);
                c = m.transcribe(batch, 'Decoder', dec, 'BeamWidth', 3, 'MaxLength', 8);
                d = m.transcribe(w{2}, 'Decoder', dec, 'BeamWidth', 3, 'MaxLength', 8);
                testCase.verifyEqual(b.texts(), a.texts()); testCase.verifyEqual(c.texts(), a.texts());
                testCase.verifyEqual([c.Score], [a.Score], 'AbsTol', 1e-10);
                testCase.verifyEqual(d.Text, a(2).Text);
                testCase.verifyEqual(d.Score, a(2).Score, 'AbsTol', 1e-10);
                testCase.verifyEqual(a(1).Decoder, "attention-" + dec);
                testCase.verifyEqual(a(1).ModelId, "ed-test");
                testCase.verifyEmpty(a(1).Segments);
            end
            testCase.verifyTrue(m.Training, 'transcribe must restore the training flag');
            % greedy score == teacher-forced log-prob of the hypothesis (step fn == forward)
            m.eval();
            [g, info] = m.transcribe(batch, 'Decoder', "greedy", 'MaxLength', 8, 'ReturnAttention', true, 'LengthPenalty', 0);
            for n = 1:2
                bn = struct('X', batch.X(:, n, 1:batch.Lengths(n)), 'Lengths', batch.Lengths(n), 'Targets', {{g(n).TokenIds}});
                m.StoreAttention = true;
                o = m.forward(bn);
                m.StoreAttention = false;
                K = testCase.Vocab.numClasses(); u = o.LogitLengths;
                Z = reshape(o.Logits.Data, K, u); lp = Z - log(sum(exp(Z), 1));
                tg = o.Targets(1, :);
                if g(n).Candidates(1).HitMaxLength
                    testCase.verifyEqual(numel(tg), 8);
                end
                testCase.verifyEqual(g(n).Score, sum(lp(sub2ind([K u], tg, 1:u))), 'AbsTol', 1e-10);
                A = o.CrossAttention{end};
                testCase.verifyEqual(info(n).CrossAttention, A(:, 1:o.Lengths, :, 1), 'AbsTol', 1e-12);
                testCase.verifyEqual(sum(info(n).CrossAttention, 2), ones(u, 1, 2), 'AbsTol', 1e-12);
            end
            other = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 12);
            testCase.verifyError(@() m.transcribe(other.forward(w{1})), 'OceanNLP:Speech:Models:EncoderDecoderRecognizer:InputDim');
            testCase.verifyError(@() m.transcribe("text"), 'OceanNLP:Speech:Models:EncoderDecoderRecognizer:InputType');
        end

        function testIncrementalCacheMatchesFullRecomputation(testCase)
            % Every cached step (KV cache, beam reordering / pruning of cache rows) equals the
            % uncached full-prefix recomputation to 1e-12, and transcripts, scores and N-best
            % lists are identical, for greedy and beam.
            m = testCase.makeModel();
            m.eval();
            b = testCase.makeBatch([31 18 25], ["one two", "three", "zero"]);
            o = m.forward(b);
            Mem = o.EncoderOutput.Data; lens = o.Lengths;
            v = testCase.Vocab;
            for cfg = {{'Method', "greedy"}, {'BeamWidth', 4, 'NBest', 3}, {'BeamWidth', 2, 'LengthPenalty', 0}}
                d = OceanNLP.Speech.Decoding.AttentionBeamDecoder(v, cfg{1}{:}, 'MaxLength', 9);
                step = @(p, src, st, parent) checkedStep(m, Mem, lens, p, src, st, parent);
                ref = d.search(@(p, src) m.nextLogProbs(p, Mem(:, src, :), lens(src)), lens);
                got = d.search(step, lens, 'Incremental', true, 'State', m.initDecodeState(Mem, lens));
                testCase.verifyEqual({got.Labels}, {ref.Labels});
                testCase.verifyEqual([got.Score], [ref.Score], 'AbsTol', 1e-12);
                for n = 1:numel(ref)
                    testCase.verifyEqual({got(n).Candidates.TokenIds}, {ref(n).Candidates.TokenIds});
                    testCase.verifyEqual([got(n).Candidates.LogProb], [ref(n).Candidates.LogProb], 'AbsTol', 1e-12);
                end
            end
            for dec = ["greedy", "beam"]
                a = m.transcribe(b, 'Decoder', dec, 'BeamWidth', 4, 'MaxLength', 9, 'UseCache', false);
                c = m.transcribe(b, 'Decoder', dec, 'BeamWidth', 4, 'MaxLength', 9, 'UseCache', true);
                testCase.verifyEqual(c.texts(), a.texts());
                testCase.verifyEqual([c.Score], [a.Score], 'AbsTol', 1e-12);
            end
            testCase.verifyError(@() m.nextLogProbsCached([v.Bos 6], 1, m.initDecodeState(Mem, lens), 1), ...
                'OceanNLP:Speech:Models:EncoderDecoderRecognizer:CacheLength');
        end

        function testCTCRescoring(testCase)
            % N-best rescoring: joint score = (1-l) logP_att + l logP_ctc, re-ranked; l = 0 keeps
            % the attention result; logP_ctc equals the CTC loss of the candidate.
            m = testCase.makeModel('CTCWeight', 0.3);
            m.eval();
            b = testCase.makeBatch([31 18], ["one two", "three"]);
            a = m.transcribe(b, 'Decoder', "beam", 'BeamWidth', 3, 'MaxLength', 6);
            r = m.transcribe(b, 'Decoder', "beam", 'BeamWidth', 3, 'MaxLength', 6, 'CTCRescoreWeight', 0.4);
            o = m.forward(setfield(b, 'Targets', {[6], [6]})); %#ok<SFLD>
            for n = 1:2
                c = r(n).Candidates;
                testCase.verifyEqual(r(n).Decoder, "attention-beam+ctc-rescore");
                testCase.verifyEqual(sort([c.AttentionLogProb]), sort([a(n).Candidates.LogProb]), 'AbsTol', 1e-12);
                testCase.verifyTrue(issorted([c.Score], 'descend'));
                h = c(1);
                if isfinite(h.CTCLogProb)
                    Lc = OceanDL.Autograd.F.ctcLoss(o.CTCLogits.Data(:, n, 1:o.Lengths(n)), {h.TokenIds}, ...
                        'InputLengths', o.Lengths(n), 'Blank', testCase.Vocab.Blank, 'Reduction', "sum");
                    testCase.verifyEqual(h.CTCLogProb, -Lc.Data, 'AbsTol', 1e-10);
                    testCase.verifyEqual(h.Score, (0.6 * h.AttentionLogProb + 0.4 * h.CTCLogProb) / h.Length, 'AbsTol', 1e-12);
                end
            end
            testCase.verifyError(@() m.transcribe(b, 'Decoder', "greedy", 'CTCRescoreWeight', 0.3), ...
                'OceanNLP:Speech:Models:EncoderDecoderRecognizer:CTCRescore');
        end

        function testConfigFactoryAndCheckpointRoundTrip(testCase)
            m = testCase.makeModel('CTCWeight', 0.2, 'ModelId', "rt");
            cfg = m.config();
            testCase.verifyEqual(cfg.Type, "EncoderDecoderRecognizer");
            m2 = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg);
            testCase.verifyEqual(m2.config(), cfg);
            m2.loadStateDict(m.stateDict());
            b = testCase.makeBatch([15 9], ["one", "two"]);
            m.eval(); m2.eval();
            testCase.verifyEqual(m2.forward(b).Logits.Data, m.forward(b).Logits.Data);
            file = fullfile(tempdir, "ocean_encdec_ck_test.mat");
            c = onCleanup(@() delete(file));
            OceanNLP.Speech.Training.SpeechCheckpoint.save(file, m);
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(file);
            for dec = ["greedy", "beam"]
                a = m.transcribe(b, 'Decoder', dec, 'BeamWidth', 4, 'MaxLength', 6);
                r = ck.Model.transcribe(b, 'Decoder', dec, 'BeamWidth', 4, 'MaxLength', 6);
                testCase.verifyEqual(r.texts(), a.texts());
                testCase.verifyEqual([r.Score], [a.Score]);
            end
            cfg.NumClasses = cfg.NumClasses + 1;
            testCase.verifyError(@() OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg), ...
                'OceanNLP:Speech:Models:EncoderDecoderRecognizer:ConfigNumClasses');
        end
    end

    methods
        function m = makeModel(testCase, opts)
            arguments
                testCase
                opts.CTCWeight (1,1) double = 0
                opts.LabelSmoothing (1,1) double = 0
                opts.Dropout (1,1) double = 0
                opts.ModelId (1,1) string = ""
            end
            enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(10, 'Dim', 8, 'NumHeads', 2, 'NumBlocks', 1, ...
                'ConvKernel', 3, 'Dropout', opts.Dropout, 'Seed', 1);
            m = OceanNLP.Speech.Models.EncoderDecoderRecognizer(testCase.Features, testCase.Vocab, {enc}, ...
                'DecoderLayers', 2, 'DecoderHeads', 2, 'Dropout', opts.Dropout, 'LabelSmoothing', opts.LabelSmoothing, ...
                'CTCWeight', opts.CTCWeight, 'Seed', 3, 'ModelId', opts.ModelId);
        end

        function [b, singles] = makeBatch(testCase, lens, texts)
            rs = RandStream('mt19937ar', 'Seed', sum(lens));
            N = numel(lens); T = max(lens); F = 10;
            X = zeros(F, N, T);
            singles = cell(1, N);
            tg = testCase.Vocab.encode(texts);
            if ~iscell(tg), tg = {tg}; end
            for n = 1:N
                x = randn(rs, F, 1, lens(n)) * 3 - 5;
                X(:, n, 1:lens(n)) = x;
                singles{n} = struct('X', x, 'Lengths', lens(n), 'Targets', {tg(n)});
            end
            b = struct('X', X, 'Lengths', lens, 'Targets', {tg});
        end
    end
end

function [lp, st] = checkedStep(m, Mem, lens, p, src, st, parent)
% cached step, verified against the uncached full-prefix recomputation at every step
[lp, st] = m.nextLogProbsCached(p, src, st, parent);
ref = m.nextLogProbs(p, Mem(:, src, :), lens(src));
if max(abs(lp - ref), [], 'all') > 1e-12
    error('test:CacheMismatch', 'cached step differs from recomputation by %g', max(abs(lp - ref), [], 'all'));
end
end
