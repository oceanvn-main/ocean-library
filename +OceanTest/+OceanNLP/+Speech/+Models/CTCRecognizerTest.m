classdef CTCRecognizerTest < OceanTest.BaseTestCase
    % CTCRECOGNIZERTEST  Padding never affects logits/loss, parameter gradients, transcribe
    %   input forms and Transcript honesty, exact beam scores, compatibility checks,
    %   config / factory round trips (recogniser and feature transforms).

    properties
        Vocab
        Features
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(["one two", "three", "zero"], 'Unit', "char");
            testCase.Features = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 10);
        end
    end

    methods (Test)
        function testPaddedBatchEqualsSingleUtterances(testCase)
            m = testCase.makeModel(0.2);
            m.eval();                                   % dropout off (identity)
            lens = [31 18 9];
            [b, singles] = testCase.makeBatch(lens, ["one two", "three", "zero"]);
            out = m.forward(b);
            testCase.verifyEqual(out.Lengths, m.outputLength(lens));
            Lb = m.loss(b).Data;
            Ls = zeros(1, 3);
            for n = 1:3
                o1 = m.forward(singles{n});
                testCase.verifyEqual(out.Logits.Data(:, n, 1:o1.Lengths), o1.Logits.Data, 'AbsTol', 1e-12);
                testCase.verifyEqual(nnz(out.Logits.Data(:, n, o1.Lengths+1:end)), 0);
                Ls(n) = m.loss(singles{n}).Data;
            end
            % OceanDL CTCLoss "mean": mean over the batch of per-utterance (loss / target length)
            testCase.verifyEqual(Lb, mean(Ls), 'AbsTol', 1e-12);
            % garbage in padded input frames changes nothing
            bj = b;
            for n = 1:3
                bj.X(:, n, lens(n)+1:end) = 1e3 * randn(size(bj.X, 1), 1, size(bj.X, 3) - lens(n));
            end
            testCase.verifyEqual(m.forward(bj).Logits.Data, out.Logits.Data, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.loss(bj).Data, Lb, 'AbsTol', 1e-12);
        end

        function testLossParameterGradients(testCase)
            m = testCase.makeModel(0, 'Tiny', true);
            b = testCase.makeBatch([12 7], ["one", "two"]);
            err = OceanTest.OceanNLP.Speech.Models.EncodersTest.paramGradError(m, @(mm) mm.loss(b), 1);
            testCase.verifyLessThan(err, 1e-6);
        end

        function testTranscribeInputFormsAgree(testCase)
            m = testCase.makeModel(0, 'ModelId', "ctc-test");
            rs = RandStream('mt19937ar', 'Seed', 5);
            w = {OceanNLP.Speech.Core.Waveform(0.1 * randn(rs, 3000, 1), 8000), ...
                 OceanNLP.Speech.Core.Waveform(0.1 * randn(rs, 1700, 1), 8000)};
            fsq = cellfun(@(x) testCase.Features.forward(x), w, 'UniformOutput', false);
            batch = OceanNLP.Speech.Models.CTCRecognizer.padSequences(fsq);
            for dec = ["greedy", "beam"]
                a = m.transcribe(w, 'Decoder', dec, 'BeamWidth', 5);
                b = m.transcribe(fsq, 'Decoder', dec, 'BeamWidth', 5);
                c = m.transcribe(batch, 'Decoder', dec, 'BeamWidth', 5);
                d = m.transcribe(w{2}, 'Decoder', dec, 'BeamWidth', 5);
                testCase.verifySize(a, [1 2]);
                testCase.verifyEqual(b.texts(), a.texts()); testCase.verifyEqual(c.texts(), a.texts());
                testCase.verifyEqual([b.Score], [a.Score]); testCase.verifyEqual([c.Score], [a.Score]);
                testCase.verifyEqual(d.Text, a(2).Text);
                testCase.verifyEqual(d.Score, a(2).Score, 'AbsTol', 1e-10);
                testCase.verifyEqual(a(1).Decoder, "ctc-" + dec);
                testCase.verifyEqual(a(1).ModelId, "ctc-test");
                testCase.verifyEqual(a(1).DecoderOptions.Method, dec);
                testCase.verifyEmpty(a(1).Segments);          % no invented timestamps
                testCase.verifyEqual(a(1).Language, "");
            end
            testCase.verifyTrue(m.Training, 'transcribe must restore the training flag');
        end

        function testGreedyAndBeamAgainstIndependentComputation(testCase)
            m = testCase.makeModel(0);
            b = testCase.makeBatch(4, "on");          % T' = 2 frames: every prefix fits in the beam
            m.eval();
            o = m.forward(b);
            K = testCase.Vocab.numClasses();
            Z = reshape(o.Logits.Data(:, 1, 1:o.Lengths), K, o.Lengths);
            lp = Z - log(sum(exp(Z), 1));
            g = m.transcribe(b, 'Decoder', "greedy");
            ids = OceanDL.Recurrent.CTCDecoder.greedy(Z, 1);
            testCase.verifyEqual(g.TokenIds, ids{1});
            testCase.verifyEqual(g.Text, testCase.Vocab.decode(ids{1}));
            % beam scores are exact log p(labelling | x) without fusion (CTC forward algorithm)
            bt = m.transcribe(b, 'Decoder', "beam", 'BeamWidth', 2000, 'NBest', 4);
            for h = 1:numel(bt.Candidates)
                nll = OceanDL.Autograd.Functions.CTCLoss.single(lp, bt.Candidates(h).TokenIds, 1);
                testCase.verifyEqual(bt.Candidates(h).Score, -nll, 'AbsTol', 1e-9);
            end
            testCase.verifyGreaterThanOrEqual(bt.Score, -OceanDL.Autograd.Functions.CTCLoss.single(lp, g.TokenIds, 1) - 1e-9);
        end

        function testCompatibilityChecks(testCase)
            m = testCase.makeModel(0);
            other = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 12);
            w = OceanNLP.Speech.Core.Waveform(0.1 * randn(2000, 1), 8000);
            testCase.verifyError(@() m.transcribe(other.forward(w)), 'OceanNLP:Speech:Models:CTCRecognizer:InputDim');
            sameDim = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 10, 'FMin', 100);
            testCase.verifyError(@() m.transcribe(sameDim.forward(w)), 'OceanNLP:Speech:Models:CTCRecognizer:FeatureMismatch');
            testCase.verifyError(@() m.transcribe("text"), 'OceanNLP:Speech:Models:CTCRecognizer:InputType');
            cfg = m.config(); cfg.NumClasses = cfg.NumClasses + 1;
            testCase.verifyError(@() OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg), ...
                'OceanNLP:Speech:Models:CTCRecognizer:ConfigNumClasses');
            bad = OceanNLP.Speech.Data.SpeechVocabulary.fit("ab", 'Unit', "char");
            d = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(bad);
            testCase.verifyError(@() d.decode(randn(testCase.Vocab.numClasses(), 1, 4), 4), ...
                'OceanNLP:Speech:Data:SpeechVocabulary:Incompatible');
            enc = {OceanNLP.Speech.Models.Encoders.ConvEncoder(11)};
            testCase.verifyError(@() OceanNLP.Speech.Models.CTCRecognizer(testCase.Features, testCase.Vocab, enc), ...
                'OceanNLP:Speech:Models:CTCRecognizer:DimMismatch');
            testCase.verifyError(@() m.loss(struct('X', zeros(10, 1, 3), 'Lengths', 3)), ...
                'OceanNLP:Speech:Models:CTCRecognizer:NoTargets');
        end

        function testConfigRoundTripThroughFactory(testCase)
            m = testCase.makeModel(0.1, 'ModelId', "rt");
            cfg = m.config();
            testCase.verifyEqual(cfg.Type, "CTCRecognizer");
            m2 = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg);
            testCase.verifyEqual(m2.config(), cfg);
            m2.loadStateDict(m.stateDict());
            b = testCase.makeBatch([9 5], ["one", "two"]);
            m.eval(); m2.eval();
            testCase.verifyEqual(m2.forward(b).Logits.Data, m.forward(b).Logits.Data);
            testCase.verifyError(@() OceanNLP.Speech.Models.RecognizerFactory.fromConfig(struct('Type', "NoSuchModel")), ...
                'OceanNLP:Speech:Models:RecognizerFactory:UnknownType');
            % runtime registration (how later builders / experiments extend the factory)
            OceanNLP.Speech.Models.RecognizerFactory.register("recognizer", "CTCAliasForTest", ...
                @(c) OceanNLP.Speech.Models.CTCRecognizer.fromConfig(setfield(c, 'Type', "CTCRecognizer"))); %#ok<SFLD>
            cfg.Type = "CTCAliasForTest";
            m3 = OceanNLP.Speech.Models.RecognizerFactory.fromConfig(cfg);
            testCase.verifyClass(m3, 'OceanNLP.Speech.Models.CTCRecognizer');
            testCase.verifyTrue(ismember("CTCAliasForTest", OceanNLP.Speech.Models.RecognizerFactory.types("recognizer")));
        end

        function testFeatureConfigRoundTrip(testCase)
            w = OceanNLP.Speech.Core.Waveform(0.1 * randn(RandStream('mt19937ar', 'Seed', 9), 4000, 1), 8000);
            ts = {OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 24, 'HopLength', 100, 'FMin', 20, 'MelNorm', "slaney"), ...
                  OceanNLP.Speech.Features.MFCC(8000, 'NumCoeffs', 13, 'Lifter', 22, 'NumMels', 26), ...
                  OceanNLP.Speech.Features.FeaturePipeline({OceanNLP.Speech.Features.MFCC(8000), ...
                        OceanNLP.Speech.Features.DeltaFeatures('Width', 2, 'Order', 2)})};
            for k = 1:numel(ts)
                f2 = OceanNLP.Speech.Models.RecognizerFactory.featuresFromConfig(ts{k}.config());
                testCase.verifyClass(f2, class(ts{k}));
                testCase.verifyEqual(f2.config(), ts{k}.config());
                testCase.verifyEqual(f2.forward(w).Values, ts{k}.forward(w).Values);
            end
        end
    end

    methods
        function m = makeModel(testCase, dropout, opts)
            arguments
                testCase
                dropout (1,1) double = 0
                opts.Tiny (1,1) logical = false
                opts.ModelId (1,1) string = ""
            end
            if opts.Tiny
                enc = {OceanNLP.Speech.Models.Encoders.ConvEncoder(10, 'Channels', 3, 'KernelSizes', 3, 'Strides', 2, 'Seed', 1), ...
                       OceanNLP.Speech.Models.Encoders.RecurrentEncoder(3, 'HiddenSize', 2, 'NumLayers', 1, 'Seed', 2)};
            else
                enc = {OceanNLP.Speech.Models.Encoders.ConvEncoder(10, 'Channels', [8 6], 'KernelSizes', [5 3], ...
                           'Strides', [2 1], 'Dropout', dropout, 'Seed', 1), ...
                       OceanNLP.Speech.Models.Encoders.RecurrentEncoder(6, 'HiddenSize', 5, 'NumLayers', 2, ...
                           'Dropout', dropout, 'Seed', 2)};
            end
            m = OceanNLP.Speech.Models.CTCRecognizer(testCase.Features, testCase.Vocab, enc, ...
                'Dropout', dropout, 'Seed', 3, 'ModelId', opts.ModelId);
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
