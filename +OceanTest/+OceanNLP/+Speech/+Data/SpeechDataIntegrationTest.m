classdef SpeechDataIntegrationTest < OceanTest.BaseTestCase
    % SPEECHDATAINTEGRATIONTEST  Data pipeline with builder B1's real components
    %   (Features.LogMel + Core.FrameGrid, Signal.ResampleSignal) on FSDD.
    %   The other Data tests use a test double and stay independent of B1.

    properties
        Composed
        Vocab
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.assumeEqual(exist('OceanNLP.Speech.Features.LogMel', 'class'), 8, 'LogMel (B1) not available.');
            s = OceanTest.OceanNLP.Speech.Data.fsddTestManifest().canonicalFSDDSplit();
            testCase.Composed = OceanNLP.Speech.Data.UtteranceComposer.compose(s.bySplit("test"), 'Count', 9, 'Seed', 21);
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(s.bySplit("train").Entries.Transcript);
        end
    end

    methods (Test)
        function testLogMelBatchesAtNativeRate(testCase)
            lm = OceanNLP.Speech.Features.LogMel(8000);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Composed, lm, 'Vocabulary', testCase.Vocab);
            L = ds.lengths();                                  % via LogMel.Grid.numFrames, no audio
            for i = 1:ds.length()
                f = ds.item(i).Features;
                testCase.verifyEqual(L(i), f.numFrames());
                testCase.verifyEqual(f.Values, lm.forward(OceanNLP.Speech.Core.Waveform(ds.item(i).Waveform.Samples, 8000)).Values);
            end
            b = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 4, 'SortByLength', true, 'Seed', 2);
            B = b.batch(1);
            testCase.verifyEqual(size(B.X, 1), lm.outputDim(1));
            testCase.verifyEqual(B.Lengths, L(B.Indices));
        end

        function testResampledPathPredictsLengths(testCase)
            testCase.assumeEqual(exist('OceanMath.Signal.ResampleSignal', 'class'), 8, 'ResampleSignal (B1) not available.');
            lm = OceanNLP.Speech.Features.LogMel(16000);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Composed, lm, 'SampleRate', 16000);
            L = ds.lengths();                                  % S' = ceil(S p / q), no audio
            for i = 1:3
                it = ds.item(i);
                S = round(testCase.Composed.Entries.Duration(i) * 8000);
                testCase.verifyEqual(it.Waveform.numSamples(), ceil(S * 2));
                testCase.verifyEqual(it.Features.numFrames(), L(i));
                testCase.verifyEqual(it.Waveform.History(end), "resampled 8000 -> 16000 Hz (OceanMath.Signal.ResampleSignal)");
            end
            % LogMel refuses a rate mismatch rather than resampling silently
            bad = OceanNLP.Speech.Data.AudioDataset(testCase.Composed, lm);
            testCase.verifyError(@() bad.item(1), 'OceanNLP:Speech:Features:LogMel:SampleRateMismatch');
        end
    end
end
