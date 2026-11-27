classdef AudioDatasetTest < OceanTest.BaseTestCase
    % AUDIODATASETTEST  Lazy items on real FSDD audio, lengths, cache, resampling guard.

    properties
        Test        % 24 test-speaker utterances
        Vocab
    end

    methods (TestClassSetup)
        function build(testCase)
            s = OceanTest.OceanNLP.Speech.Data.fsddTestManifest().canonicalFSDDSplit();
            te = s.bySplit("test");
            testCase.Test = te.subset(round(linspace(1, te.height(), 24)));
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(s.bySplit("train").Entries.Transcript);
        end
    end

    methods (Test)
        function testItemContentsAgainstDirectComputation(testCase)
            fe = OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(8000);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe, 'Vocabulary', testCase.Vocab);
            testCase.verifyEqual(ds.numel(), 24);
            testCase.verifyEqual(ds.length(), 24);
            for i = [1 7 24]
                it = ds.item(i);
                x = audioread(testCase.Test.resolve(i));
                testCase.verifyEqual(it.Waveform.Samples, x);
                testCase.verifyEqual(it.Waveform.Source, testCase.Test.Entries.AudioPath(i));
                % frame 1 log energy, straight from the samples
                testCase.verifyEqual(it.Features.Values(1, 1), log(max(sum(x(1:200).^2), 1e-10)), 'AbsTol', 1e-12);
                testCase.verifyEqual(it.Features.numFrames(), 1 + floor((numel(x) - 200) / 80));
                testCase.verifyEqual(it.Text, testCase.Test.Entries.Transcript(i));
                testCase.verifyEqual(it.TokenIds, testCase.Vocab.encode(it.Text));
                testCase.verifyEqual(it.Speaker, "theo");
            end
        end

        function testLengthsWithoutAudioEqualLoadedLengths(testCase)
            counting = OceanTest.OceanNLP.Speech.Data.FrameEnergyCountingTestTransform(8000);
            plain = OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(8000);
            a = OceanNLP.Speech.Data.AudioDataset(testCase.Test, counting, 'Cache', false);
            b = OceanNLP.Speech.Data.AudioDataset(testCase.Test, plain);
            La = a.lengths(); Lb = b.lengths();
            testCase.verifyEqual(La, Lb);
            testCase.verifyEqual(La(5), b.item(5).Features.numFrames());
            % composed entries: durations summed in the manifest still give exact sample counts
            c = OceanNLP.Speech.Data.UtteranceComposer.compose(testCase.Test, 'Count', 6, 'Seed', 4);
            ca = OceanNLP.Speech.Data.AudioDataset(c, counting);
            cb = OceanNLP.Speech.Data.AudioDataset(c, plain);
            testCase.verifyEqual(ca.lengths(), cb.lengths());
        end

        function testCacheReturnsSameObjects(testCase)
            fe = OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(8000);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe);
            a = ds.item(3); b = ds.item(3);
            testCase.verifyTrue(a.Features == b.Features);
            nc = OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe, 'Cache', false);
            c = nc.item(3); d = nc.item(3);
            testCase.verifyFalse(c.Features == d.Features);
            testCase.verifyEqual(c.Features.Values, d.Features.Values);
        end

        function testResamplingGuard(testCase)
            fe16 = OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(16000);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe16, 'SampleRate', 16000);
            if exist('OceanMath.Signal.ResampleSignal', 'class') ~= 8
                testCase.verifyError(@() ds.item(1), 'OceanNLP:Speech:Data:AudioDataset:ResamplerUnavailable');
            else
                it = ds.item(1);                         % integration path (B1's resampler present)
                x = audioread(testCase.Test.resolve(1));
                testCase.verifyEqual(it.Waveform.SampleRate, 16000);
                testCase.verifyEqual(it.Waveform.numSamples(), ceil(numel(x) * 2));
                testCase.verifyTrue(any(contains(it.Waveform.History, "resampled 8000 -> 16000")));
            end
        end

        function testNormalizerMismatchAndAugmentType(testCase)
            fe = OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(8000);
            other = OceanNLP.Speech.Data.TranscriptNormalizer('Lowercase', false);
            testCase.verifyError(@() OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe, ...
                'Vocabulary', testCase.Vocab, 'Normalizer', other), ...
                'OceanNLP:Speech:Data:AudioDataset:NormalizerMismatch');
            testCase.verifyError(@() OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe, 'Augment', 3), ...
                'OceanNLP:Speech:Data:AudioDataset:Augment');
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe);
            testCase.verifyError(@() ds.item(25), 'OceanNLP:Speech:Data:AudioDataset:Index');
        end

        function testAugmentedItemIsReproduciblePerEpoch(testCase)
            fe = OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(8000);
            aug = OceanNLP.Speech.Data.AudioAugment('Seed', 5, 'Gain', [-6 6]);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Test, fe, 'Augment', aug);
            ds.setEpoch(2); a = ds.item(1);
            ds.setEpoch(2); b = ds.item(1);
            testCase.verifyEqual(a.Waveform.Samples, b.Waveform.Samples);
            x = audioread(testCase.Test.resolve(1));
            ratio = a.Waveform.Samples(x ~= 0) ./ x(x ~= 0);
            testCase.verifyLessThan(max(abs(ratio - ratio(1))), 1e-12);
            testCase.verifyTrue(abs(20 * log10(ratio(1))) <= 6);
        end
    end
end
