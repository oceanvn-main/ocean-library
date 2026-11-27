classdef RecognitionPipelineTest < OceanTest.BaseTestCase
    % RECOGNITIONPIPELINETEST  The pipeline with the trained FSDD checkpoints on real audio:
    %   its transcript equals recognizer.transcribe on the same conditioned audio (CTC and
    %   encoder-decoder, trace on and off), long input is segmented with detector times,
    %   evaluation equals an independent WER/CER computation, RNG untouched.

    properties
        Ctc
        EncDec
        Root string
        Theo cell
    end

    methods (TestClassSetup)
        function loadModels(tc)
            dir0 = fullfile(tc.LibraryRoot, 'Results', 'Speech', 'checkpoints');
            tc.assumeTrue(isfile(fullfile(dir0, 'ctc_fsdd_best.mat')) && isfile(fullfile(dir0, 'encdec_fsdd_best.mat')), ...
                'Trained checkpoints not present.');
            tc.Ctc = OceanNLP.Speech.Training.SpeechCheckpoint.load(string(fullfile(dir0, 'ctc_fsdd_best.mat'))).Model;
            tc.EncDec = OceanNLP.Speech.Training.SpeechCheckpoint.load(string(fullfile(dir0, 'encdec_fsdd_best.mat'))).Model;
            tc.Root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            names = ["7_theo_3.wav", "0_theo_12.wav", "4_theo_40.wav"];
            tc.Theo = arrayfun(@(n) OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/" + n, 'Root', tc.Root), ...
                names, 'UniformOutput', false);
        end
    end

    methods (Test)
        function ctcEqualsTranscribeOnConditionedAudio(tc)
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'Segmentation', "never");
            settings = {{'Decoder', "greedy"}, {'Decoder', "beam", 'BeamWidth', 8}, ...
                {'Decoder', "beam", 'BeamWidth', 3, 'NBest', 2}};
            for s = 1:numel(settings)
                for i = 1:numel(tc.Theo)
                    for trace = [false, true]
                        r = p.run(tc.Theo{i}, 'Trace', trace, settings{s}{:});
                        tc.verifySameTranscript(r.Transcript, tc.Ctc.transcribe(r.Conditioned, settings{s}{:}));
                        tc.verifyEqual(r.numPieces(), 1);
                        tc.verifyEqual(numel(r.Readers), double(trace));
                    end
                end
            end
        end

        function encoderDecoderEqualsTranscribeOnConditionedAudio(tc)
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.EncDec, 'Segmentation', "never");
            settings = {{'Decoder', "greedy"}, {'Decoder', "beam", 'BeamWidth', 3}, ...
                {'Decoder', "beam", 'BeamWidth', 3, 'CTCRescoreWeight', 0.3}};
            for s = 1:numel(settings)
                for i = 1:2
                    for trace = [false, true]
                        r = p.run(tc.Theo{i}, 'Trace', trace, settings{s}{:});
                        tc.verifySameTranscript(r.Transcript, tc.EncDec.transcribe(r.Conditioned, settings{s}{:}));
                    end
                end
            end
        end

        function conditioningIsHarmlessOnCleanFsdd(tc)
            % FSDD is already 8 kHz mono: conditioning = DC removal + gain only, recorded in History.
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'Segmentation', "never");
            r = p.run(tc.Theo{1});
            x = tc.Theo{1}.Samples;
            g = 10 ^ (r.Report.GainDB / 20);
            tc.verifyEqual(r.Conditioned.Samples, (x - mean(x)) * g, 'AbsTol', 1e-12);
            tc.verifyEqual(r.Conditioned.History, [tc.Theo{1}.History, ...
                sprintf("remove DC offset %.3g", mean(x)), r.Conditioned.History(end)]);
            tc.verifyEqual(r.Transcript.Segments, struct('Text', r.Transcript.Text, 'StartTime', 0, ...
                'EndTime', r.Conditioned.duration()));
        end

        function traceDoesNotChangeOutputsOrRng(tc)
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.EncDec);
            rng(4321, 'twister'); s0 = rng;
            a = p.run(tc.Theo{3}, 'Decoder', "beam", 'BeamWidth', 3);
            sA = rng;
            b = p.run(tc.Theo{3}, 'Trace', true, 'Decoder', "beam", 'BeamWidth', 3);
            sB = rng;
            tc.verifyEqual(sA, s0);
            tc.verifyEqual(sB, s0);
            tc.verifySameTranscript(a.Transcript, b.Transcript);
            tc.verifyEqual(a.Segments, b.Segments);
            tc.verifyGreaterThan(numel(b.reader(1).listSteps()), 10);
            tc.verifyError(@() a.reader(1), 'OceanNLP:Speech:Inference:PipelineResult:NoTrace');
        end

        function longComposedInputIsSegmentedWithDetectorTimes(tc)
            % 12 real theo recordings concatenated (the UtteranceComposer convention):
            % longer than MaxSegment, so "auto" segments it at detected pauses.
            names = ["3_theo_1", "8_theo_2", "1_theo_5", "6_theo_9", "2_theo_11", "9_theo_14", "5_theo_20", "0_theo_31", "7_theo_33", "4_theo_36", "2_theo_38", "6_theo_42"];
            refs = strings(1, numel(names));
            x = [];
            for k = 1:numel(names)
                w = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/" + names(k) + ".wav", 'Root', tc.Root);
                x = [x; w.Samples]; %#ok<AGROW>
                refs(k) = OceanNLP.Speech.Data.SpeechManifest.DigitWords(str2double(extractBefore(names(k), "_")) + 1);   % dataset label
            end
            reference = strjoin(refs, " ");
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc);
            r = p.run(struct('Samples', x, 'SampleRate', 8000, 'Source', "composed theo"), 'Reference', reference);
            tc.assumeGreaterThan(r.Conditioned.duration(), p.Detector.MaxSegment);
            tc.verifyTrue(r.Segmented);
            P = r.numPieces();
            tc.verifyEqual(P, numel(r.Segments));
            tc.verifyGreaterThan(P, 1);
            % piece transcripts are the recogniser's own output on the detector's pieces
            for k = 1:P
                tc.verifySameTranscript(r.SegmentTranscripts(k), tc.Ctc.transcribe(r.SegmentWaveforms(k)));
            end
            texts = arrayfun(@(t) t.Text, r.SegmentTranscripts);
            tc.verifyEqual(r.Transcript.Text, strjoin(texts(strlength(texts) > 0), " "));
            tc.verifyEqual([r.Transcript.Segments.StartTime], [r.Segments.Start]);
            tc.verifyEqual([r.Transcript.Segments.EndTime], [r.Segments.End]);
            tc.verifyEqual(r.Transcript.Score, NaN);
            % evaluation equals an independent WER/CER computation
            n = tc.Ctc.Vocabulary.Normalizer;
            wer = OceanNLP.Speech.Evaluation.WordErrorRate.compute(reference, r.Transcript.Text, 'Normalizer', n);
            cer = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(reference, r.Transcript.Text, 'Normalizer', n);
            tc.verifyEqual(r.Evaluation.WER.Rate, wer.Rate);
            tc.verifyEqual(r.Evaluation.CER.Rate, cer.Rate);
            whole = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'Segmentation', "never").run( ...
                struct('Samples', x, 'SampleRate', 8000), 'Reference', reference);
            tc.log(1, sprintf('composed 12 words (%.2f s): %d pieces, segmented WER %.3f "%s"; unsegmented WER %.3f "%s"', ...
                r.Conditioned.duration(), P, r.Evaluation.WER.Rate, r.Transcript.Text, ...
                whole.Evaluation.WER.Rate, whole.Transcript.Text));
        end

        function long44kRecordingRunsWithTimings(tc)
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Counting-16-44p1-mono-15secs.wav');
            tc.assumeTrue(isfile(f));
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc);
            r = p.run(string(f), 'Decoder', "beam", 'BeamWidth', 4);
            tc.verifyEqual(r.Conditioned.SampleRate, 8000);
            tc.verifyEqual(r.Report.InputRate, 44100);
            tc.verifyTrue(r.Segmented);
            tc.verifyEqual(r.numPieces(), numel(r.Segments));
            tm = r.Timings;
            for fn = ["Input", "Conditioning", "Detection", "Segmentation", "Recognition", "Evaluation", "Total"]
                tc.verifyGreaterThanOrEqual(tm.(fn), 0);
            end
            tc.verifyEqual(tm.Recognition, sum(tm.PerSegment), 'AbsTol', 1e-12);
            tc.verifyGreaterThanOrEqual(tm.Total, tm.Conditioning + tm.Detection + tm.Recognition);
            tc.verifyEmpty(r.Evaluation);
            tc.verifyEqual(r.DecodeOptions, struct('Decoder', "beam", 'BeamWidth', 4));
            tc.log(1, sprintf('Counting 15.5 s @44.1 kHz: "%s" (%d pieces), total %.2f s (condition %.2f, detect %.3f, recognise %.2f)', ...
                r.Transcript.Text, r.numPieces(), tm.Total, tm.Conditioning, tm.Detection, tm.Recognition));
        end

        function silentInputGivesEmptyTranscriptAndWarning(tc)
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc);
            r = p.run(struct('Samples', zeros(8000, 1), 'SampleRate', 8000));
            tc.verifyEqual(r.Transcript.Text, "");
            tc.verifyEqual(r.numPieces(), 0);
            tc.verifyTrue(any(contains(r.Warnings, "No speech detected")));
            tc.verifyTrue(any(contains(r.Warnings, "Silent input")));
        end

        function quietRoomNoiseIsNotSpeechAfterNormalisation(tc)
            % critic round 2, N1: level normalisation (up to +30 dB) used to lift quiet room
            % noise above the detector's absolute floor, so a silent microphone recording was
            % "recognised". The floor now applies to the recorded level (before the gain).
            s = RandStream('mt19937ar', 'Seed', 4);
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc);
            for level = [1e-4, 3e-4]                         % about -80 and -70 dBFS RMS
                r = p.run(struct('Samples', level * randn(s, 8000, 1), 'SampleRate', 8000));
                tc.verifyGreaterThan(r.Report.GainDB, 0, 'normalisation did apply gain');
                tc.verifyEqual(r.numPieces(), 0, sprintf('noise at %g', level));
                tc.verifyEqual(r.Transcript.Text, "");
                tc.verifyTrue(any(contains(r.Warnings, "No speech detected")));
            end
            % real speech is still found with the same settings
            r = p.run(tc.Theo{1});
            tc.verifyGreaterThan(r.numPieces(), 0);
        end

        function stereoIntegerInputIsConditioned(tc)
            x = tc.Theo{2}.Samples;
            xi = int16(round([x, x] * 32767));
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'Segmentation', "never");
            r = p.run(struct('Samples', xi, 'SampleRate', 8000));
            tc.verifyEqual(r.Report.Channels, 2);
            tc.verifyEqual(r.Conditioned.numChannels(), 1);
            tc.verifySameTranscript(r.Transcript, tc.Ctc.transcribe(r.Conditioned));
            tc.verifyTrue(any(contains(r.Input.History, "int16 PCM scaled")));
        end

        function detectorOffAndOptionErrors(tc)
            p = OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'UseDetector', false);
            r = p.run(tc.Theo{1});
            tc.verifyEmpty(r.Segments);
            tc.verifyEmpty(r.DetectorInfo);
            tc.verifySameTranscript(r.Transcript, tc.Ctc.transcribe(r.Conditioned));
            tc.verifyError(@() OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'UseDetector', false, ...
                'Segmentation', "always"), 'OceanNLP:Speech:Inference:RecognitionPipeline:NeedsDetector');
            tc.verifyError(@() OceanNLP.Speech.Inference.RecognitionPipeline(tc.Ctc, 'Conditioner', ...
                OceanNLP.Speech.Inference.AudioConditioner('TargetRate', 16000)), ...
                'OceanNLP:Speech:Inference:RecognitionPipeline:RateMismatch');
            tc.verifyError(@() p.run(tc.Theo{1}, 'Trace'), 'OceanNLP:Speech:Inference:RecognitionPipeline:NameValue');
        end
    end

    methods
        function verifySameTranscript(tc, a, b)
            tc.verifyEqual(a.Text, b.Text);
            tc.verifyEqual(a.TokenIds, b.TokenIds);
            tc.verifyEqual(a.Score, b.Score);
            tc.verifyEqual(a.Candidates, b.Candidates);
            tc.verifyEqual(a.Decoder, b.Decoder);
            tc.verifyEqual(a.DecoderOptions, b.DecoderOptions);
            tc.verifyEqual(a.ModelId, b.ModelId);
        end
    end
end
