classdef SpeechProcessAdapterTest < OceanTest.BaseTestCase
    % SPEECHPROCESSADAPTERTEST  The adapter replays the real trained CTC recogniser
    %   (Results/Speech/checkpoints/ctc_fsdd_best.mat) on real held-out FSDD audio
    %   (speaker theo) and must reproduce transcribe() exactly, without touching RNG.

    properties
        Model
        EncDec
        Waves cell
    end

    methods (TestClassSetup)
        function loadRealModelAndAudio(tc)
            file = fullfile(tc.LibraryRoot, 'Results', 'Speech', 'checkpoints', 'ctc_fsdd_best.mat');
            tc.assumeTrue(isfile(file), 'Trained checkpoint not present.');
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(string(file));
            tc.Model = ck.Model;
            edFile = fullfile(tc.LibraryRoot, 'Results', 'Speech', 'checkpoints', 'encdec_fsdd_best.mat');
            if isfile(edFile)
                tc.EncDec = OceanNLP.Speech.Training.SpeechCheckpoint.load(string(edFile)).Model;
            end
            root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            names = ["7_theo_3.wav", "0_theo_12.wav", "4_theo_40.wav"];
            tc.Waves = arrayfun(@(n) OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/" + n, 'Root', root), ...
                names, 'UniformOutput', false);
        end
    end

    methods (Test)
        function transcriptEqualsTranscribe(tc)
            settings = {{'Decoder', "greedy"}, {'Decoder', "beam", 'BeamWidth', 8}, ...
                {'Decoder', "beam", 'BeamWidth', 3, 'NBest', 2}};
            for s = 1:numel(settings)
                for i = 1:numel(tc.Waves)
                    w = tc.Waves{i};
                    expected = tc.Model.transcribe(w, settings{s}{:});
                    [tr, ~] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(tc.Model, w, settings{s}{:});
                    tc.verifyEqual(tr.Text, expected.Text);
                    tc.verifyEqual(tr.TokenIds, expected.TokenIds);
                    tc.verifyEqual(tr.Score, expected.Score);
                    tc.verifyEqual(tr.Candidates, expected.Candidates);
                    tc.verifyEqual(tr.Decoder, expected.Decoder);
                    tc.verifyEqual(tr.DecoderOptions, expected.DecoderOptions);
                    tc.verifyEqual(tr.ModelId, expected.ModelId);
                end
            end
        end

        function stageLogitsEqualModelForward(tc)
            m = tc.Model;
            w = tc.Waves{1};
            [~, r] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w);
            logits = tc.artifact(r, "Logits");
            wasTraining = m.Training;
            m.eval(); restore = onCleanup(@() m.train(wasTraining)); %#ok<NASGU>
            batch = OceanNLP.Speech.Models.CTCRecognizer.padSequences({m.Features.forward(w)});
            out = m.forward(batch);
            Z = double(OceanDL.Autograd.Tensor.raw(out.Logits));
            tc.verifyEqual(logits.Values, reshape(Z(:, 1, 1:out.Lengths), size(Z, 1), []));   % bit-exact
            H = double(OceanDL.Autograd.Tensor.raw(out.EncoderOutput));
            enc = tc.artifact(r, "Encoder 2 (RecurrentEncoder)");
            tc.verifyEqual(enc.Values, reshape(H(:, 1, :), size(H, 1), []));
        end

        function frontEndStagesAreTheRealIntermediates(tc)
            m = tc.Model;
            w = tc.Waves{2};
            [~, r] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w);
            feats = m.Features.forward(w);
            tc.verifyEqual(tc.artifact(r, "Log-mel").Values, feats.Values);
            tc.verifyEqual(tc.artifact(r, "Model features").Values, feats.Values);
            % independent route: FrameGrid.powerSpectrum, then W P and the log
            P = m.Features.Grid.powerSpectrum(w.Samples);
            spec = tc.artifact(r, "Power spectrum");
            tc.verifyEqual(spec.Values, P);
            bank = tc.artifact(r, "Mel filter bank");
            tc.verifyEqual(log(max(bank.Weights * P, m.Features.LogFloor)), feats.Values, 'AbsTol', 1e-12);
            frames = tc.artifact(r, "Frames + window");
            tc.verifyEqual(size(frames.Frames), [m.Features.Grid.WindowLength, feats.numFrames()]);
            % normalised input: zero mean / unit variance per bin over the frames
            z = tc.artifact(r, "Normalised input").Values;
            tc.verifyEqual(mean(z, 2), zeros(size(z, 1), 1), 'AbsTol', 1e-10);
            % posteriors: softmax of the logits, best path = argmax
            post = tc.artifact(r, "Posteriors");
            lg = tc.artifact(r, "Logits");
            tc.verifyEqual(sum(post.Posteriors, 1), ones(1, size(lg.Values, 2)), 'AbsTol', 1e-12);
            [~, best] = max(lg.Values, [], 1);
            tc.verifyEqual(post.BestPath, best);
            tc.verifyEqual(post.Tokens, m.Vocabulary.Tokens);
            % lineage: log-mel consumes power spectrum + filter bank
            steps = r.listSteps();
            k = find([steps.Name] == "Log-mel");
            tc.verifyEqual(numel(steps(k).InputIds), 2);
        end

        function inspectionDoesNotChangeOutputsOrRng(tc)
            m = tc.Model;
            w = tc.Waves{3};
            opts = {'Decoder', "beam", 'BeamWidth', 4};
            rng(1234, 'twister');
            s0 = rng;
            absent = m.transcribe(w, opts{:});
            sAbsent = rng;
            rng(s0);
            [enabled, rEn] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, opts{:}, ...
                'Trace', OceanNLP.Speech.Inspect.SpeechTrace());
            sEnabled = rng;
            rng(s0);
            [disabled, rDis] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, opts{:}, ...
                'Trace', OceanNLP.Speech.Inspect.SpeechTrace('Enabled', false));
            sDisabled = rng;
            tc.verifyEqual(sAbsent, s0);
            tc.verifyEqual(sEnabled, s0);
            tc.verifyEqual(sDisabled, s0);
            for tr = [enabled, disabled]
                tc.verifyEqual(tr.Text, absent.Text);
                tc.verifyEqual(tr.Score, absent.Score);
                tc.verifyEqual(tr.Candidates, absent.Candidates);
            end
            tc.verifyGreaterThan(numel(rEn.listSteps()), 10);
            tc.verifyEmpty(rDis.listSteps());
        end

        function modelModesAreRestored(tc)
            m = tc.Model;
            m.train(true);
            OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, tc.Waves{1});
            tc.verifyTrue(m.Training);
            tc.verifyTrue(OceanDL.Autograd.GradMode.isEnabled());
            m.eval();
            OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, tc.Waves{1});
            tc.verifyFalse(m.Training);
            m.train(true);
        end

        function budgetsOmitLargeArtifactsButKeepMetadata(tc)
            m = tc.Model;
            w = tc.Waves{1};
            t = OceanNLP.Speech.Inspect.SpeechTrace('MaxArtifactBytes', 20e3, 'MaxTotalBytes', 60e3);
            [tr, r] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, 'Trace', t);
            tc.verifyEqual(tr.Text, m.transcribe(w).Text);
            arts = arrayfun(@(s) r.getArtifact(s.OutputIds(1)), r.listSteps());
            omitted = arts([arts.Status] == "omitted");
            tc.verifyNotEmpty(omitted);
            for a = omitted
                tc.verifyEmpty(a.Value);
                tc.verifyGreaterThan(a.ByteCount, 0);
                tc.verifyEqual(a.Kind, "struct");
                tc.verifySubstring(a.Summary, "omitted");
            end
            kept = arts([arts.Status] == "recorded");
            tc.verifyLessThanOrEqual(sum([kept.ByteCount]), 60e3);
            tc.verifyTrue(all([kept.ByteCount] <= 20e3));
            tc.verifyEqual(t.BytesUsed, sum([kept.ByteCount]));
        end

        function logitFrameTimesFollowReceptiveFieldCentres(tc)
            m = tc.Model;
            w = tc.Waves{1};
            [~, r] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w);
            lg = tc.artifact(r, "Logits");
            ft = m.Features.forward(w).FrameTimes;
            hop = m.Features.Grid.HopLength / m.Features.SampleRate;
            % brute force: compose each Conv1d's input-window centre explicitly
            c = 1:size(lg.Values, 2);
            convs = m.Encoders{1}.Convs;
            for l = numel(convs):-1:1
                k = convs{l}.KernelSize; s = convs{l}.Stride; p = convs{l}.Padding;
                first = (c - 1) * s - p + 1;           % first input frame read (1-based)
                c = (first + (first + k - 1)) / 2;     % centre of [first, first + k - 1]
            end
            tc.verifyEqual(lg.FrameTimes, ft(1) + (c - 1) * hop, 'AbsTol', 1e-12);
        end

        function rejectsBadTrace(tc)
            tc.verifyError(@() OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(tc.Model, tc.Waves{1}, ...
                'Trace', struct()), 'OceanNLP:Speech:Inspect:SpeechProcessAdapter:TraceType');
        end

        % ---- encoder-decoder branch (Results/Speech/checkpoints/encdec_fsdd_best.mat) ----

        function encDecTranscriptEqualsTranscribe(tc)
            m = tc.encDec();
            settings = {{'Decoder', "greedy"}, {'Decoder', "greedy", 'UseCache', false}, ...
                {'Decoder', "beam", 'BeamWidth', 4}, {'Decoder', "beam", 'BeamWidth', 3, 'NBest', 2}, ...
                {'Decoder', "beam", 'BeamWidth', 3, 'CTCRescoreWeight', 0.3}, ...
                {'Decoder', "beam", 'BeamWidth', 3, 'LengthPenalty', 0, 'MaxLengthRatio', 2}};
            for s = 1:numel(settings)
                for i = 1:2
                    w = tc.Waves{i};
                    expected = m.transcribe(w, settings{s}{:});
                    [tr, ~] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, settings{s}{:});
                    tc.verifyEqual(tr.Text, expected.Text);
                    tc.verifyEqual(tr.TokenIds, expected.TokenIds);
                    tc.verifyEqual(tr.Score, expected.Score);
                    tc.verifyEqual(tr.Candidates, expected.Candidates);
                    tc.verifyEqual(tr.Decoder, expected.Decoder);
                    tc.verifyEqual(tr.DecoderOptions, expected.DecoderOptions);
                    tc.verifyEqual(tr.ModelId, expected.ModelId);
                end
            end
        end

        function encDecStagesEqualModelForward(tc)
            m = tc.encDec();
            w = tc.Waves{1};
            [tr, r] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, 'Decoder', "beam", 'BeamWidth', 4);
            wasTraining = m.Training;
            m.eval(); restore = onCleanup(@() m.train(wasTraining)); %#ok<NASGU>
            batch = OceanNLP.Speech.Models.CTCRecognizer.padSequences({m.Features.forward(w)});
            batch.Targets = {tr.TokenIds};
            out = m.forward(batch);
            H = double(OceanDL.Autograd.Tensor.raw(out.EncoderOutput));
            enc = tc.artifact(r, "Encoder 1 (ConformerEncoder)");
            tc.verifyEqual(enc.Values, reshape(H(:, 1, :), size(H, 1), []));          % bit-exact
            % teacher-forced posteriors: columns are distributions; the chosen path's
            % log-probability is the attention log-prob the search reported
            post = tc.artifact(r, "Decoder posteriors");
            U = numel(tr.TokenIds) + 1;
            tc.verifySize(post.Posteriors, [m.Vocabulary.numClasses(), U]);
            tc.verifyEqual(sum(post.Posteriors, 1), ones(1, U), 'AbsTol', 1e-12);
            tc.verifyEqual(post.TokenIds, [tr.TokenIds, m.Vocabulary.Eos]);
            tc.verifyEqual(post.LogProb, tr.Candidates(1).LogProb, 'AbsTol', 1e-9);
            Z = double(OceanDL.Autograd.Tensor.raw(out.Logits));
            z = reshape(Z(:, 1, :), size(Z, 1), []);
            tc.verifyEqual(log(post.Posteriors), z - log(sum(exp(z), 1)), 'AbsTol', 1e-10);
            % cross-attention: rows are distributions over the T' valid encoder frames
            att = tc.artifact(r, "Cross-attention");
            Tp = out.Lengths(1);
            tc.verifySize(att.Values, [U, Tp]);
            tc.verifyEqual(sum(att.Values, 2), ones(U, 1), 'AbsTol', 1e-10);
            tc.verifyEqual(size(att.PerHead, 3), m.DecoderHeads);
            tc.verifyNumElements(att.FrameTimes, Tp);
            % search steps: first call extends Bos; every column is a log-distribution
            steps = tc.artifact(r, "Decoder search steps");
            tc.verifyGreaterThanOrEqual(numel(steps.Steps), U);
            tc.verifyEqual(steps.Steps(1).Prefixes, m.Vocabulary.Bos);
            for k = 1:numel(steps.Steps)
                lp = steps.Steps(k).LogProbs;
                mx = max(lp, [], 1);
                tc.verifyEqual(mx + log(sum(exp(lp - mx), 1)), zeros(1, size(lp, 2)), 'AbsTol', 1e-10);
            end
            % CTC head posteriors (the model has a joint-CTC head)
            ctc = tc.artifact(r, "CTC head posteriors");
            C = double(OceanDL.Autograd.Tensor.raw(out.CTCLogits));
            c = reshape(C(:, 1, 1:Tp), size(C, 1), Tp);
            tc.verifyEqual(ctc.Posteriors, exp(c - max(c, [], 1)) ./ sum(exp(c - max(c, [], 1)), 1), 'AbsTol', 1e-12);
        end

        function encDecInspectionDoesNotChangeOutputsOrRng(tc)
            m = tc.encDec();
            w = tc.Waves{3};
            opts = {'Decoder', "beam", 'BeamWidth', 3};
            rng(2468, 'twister');
            s0 = rng;
            absent = m.transcribe(w, opts{:});
            [enabled, rEn] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, opts{:});
            [disabled, rDis] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w, opts{:}, ...
                'Trace', OceanNLP.Speech.Inspect.SpeechTrace('Enabled', false));
            tc.verifyEqual(rng, s0);
            for tr = [enabled, disabled]
                tc.verifyEqual(tr.Text, absent.Text);
                tc.verifyEqual(tr.Candidates, absent.Candidates);
            end
            tc.verifyEmpty(rDis.listSteps());
            tc.verifyGreaterThan(numel(rEn.listSteps()), 10);
            tc.verifyFalse(m.StoreAttention);
            m.train(true);
            OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, w);
            tc.verifyTrue(m.Training);
            tc.verifyTrue(OceanDL.Autograd.GradMode.isEnabled());
            m.eval();
        end

        function encDecShortInputEqualsTranscribe(tc)
            m = tc.encDec();
            short = OceanNLP.Speech.Core.Waveform(tc.Waves{1}.Samples(1:50), 8000);
            [tr, ~] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(m, short);
            tc.verifyEqual(tr.Text, m.transcribe(short).Text);
            [trc, ~] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(tc.Model, short);
            tc.verifyEqual(trc.Text, tc.Model.transcribe(short).Text);
        end

        function optionsMustApplyToTheModel(tc)
            tc.verifyError(@() OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(tc.encDec(), tc.Waves{1}, ...
                'LMWeight', 0.5), 'OceanNLP:Speech:Inspect:SpeechProcessAdapter:OptionNotApplicable');
            tc.verifyError(@() OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(tc.Model, tc.Waves{1}, ...
                'LengthPenalty', 0.5), 'OceanNLP:Speech:Inspect:SpeechProcessAdapter:OptionNotApplicable');
        end
    end

    methods
        function m = encDec(tc)
            tc.assumeNotEmpty(tc.EncDec, 'Encoder-decoder checkpoint not present.');
            m = tc.EncDec;
        end

        function v = artifact(tc, reader, name)
            steps = reader.listSteps();
            k = find([steps.Name] == name, 1);
            tc.assertNotEmpty(k, "No step named " + name);
            a = reader.getArtifact(steps(k).OutputIds(1));
            tc.assertEqual(a.Status, "recorded");
            v = a.Value;
        end
    end
end
