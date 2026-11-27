classdef UtteranceComposerTest < OceanTest.BaseTestCase
    % UTTERANCECOMPOSERTEST  Composed utterances: same speaker/split, provenance, determinism, audio.

    properties
        Split
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.Split = OceanTest.OceanNLP.Speech.Data.fsddTestManifest().canonicalFSDDSplit();
        end
    end

    methods (Test)
        function testProvenanceSpeakerSplitAndTranscript(testCase)
            src = testCase.Split;
            c = OceanNLP.Speech.Data.UtteranceComposer.compose(src, 'Count', 80, 'Seed', 3);
            testCase.verifyEqual(c.height(), 80);
            S = src.Entries;
            for u = 1:c.height()
                e = c.Entries(u, :);
                ids = split(extractAfter(e.Id, "_"), "+");
                [found, loc] = ismember(ids, S.Id);
                testCase.verifyTrue(all(found));
                rows = S(loc, :);
                testCase.verifyTrue(numel(ids) >= 2 && numel(ids) <= 5);
                testCase.verifyEqual(numel(unique(ids)), numel(ids));           % no reuse within an utterance
                testCase.verifyEqual(unique(rows.Speaker), e.Speaker);
                testCase.verifyEqual(unique(rows.Split), e.Split);
                testCase.verifyEqual(e.Transcript, join(rows.Transcript, " "));
                testCase.verifyEqual(e.AudioPath, join(rows.AudioPath, "+"));
                testCase.verifyEqual(e.Duration, sum(rows.Duration), 'AbsTol', 1e-12);
            end
        end

        function testDeterministicPerSeed(testCase)
            a = OceanNLP.Speech.Data.UtteranceComposer.compose(testCase.Split, 'Count', 30, 'Seed', 9);
            b = OceanNLP.Speech.Data.UtteranceComposer.compose(testCase.Split, 'Count', 30, 'Seed', 9);
            c = OceanNLP.Speech.Data.UtteranceComposer.compose(testCase.Split, 'Count', 30, 'Seed', 10);
            testCase.verifyEqual(a.Entries, b.Entries);
            testCase.verifyNotEqual(a.Entries.Id, c.Entries.Id);
            g = rng; OceanNLP.Speech.Data.UtteranceComposer.compose(testCase.Split, 'Count', 5);
            testCase.verifyEqual(rng, g);                                    % global RNG untouched
        end

        function testComposedAudioIsConcatenationOfComponents(testCase)
            c = OceanNLP.Speech.Data.UtteranceComposer.compose(testCase.Split, 'Count', 4, 'Seed', 1);
            ds = OceanNLP.Speech.Data.AudioDataset(c, OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(8000));
            for u = 1:c.height()
                files = c.resolve(u);
                x = [];
                for f = files
                    x = [x; audioread(f)]; %#ok<AGROW>
                end
                it = ds.item(u);
                testCase.verifyEqual(it.Waveform.Samples, x);
                testCase.verifyEqual(it.Waveform.duration(), c.Entries.Duration(u), 'AbsTol', 1e-12);
                testCase.verifyEqual(it.Text, c.Entries.Transcript(u));
            end
        end

        function testRejectsTinyGroupsAndBadRange(testCase)
            one = testCase.Split.subset(1);
            testCase.verifyError(@() OceanNLP.Speech.Data.UtteranceComposer.compose(one, 'Count', 1), ...
                'OceanNLP:Speech:Data:UtteranceComposer:GroupTooSmall');
            testCase.verifyError(@() OceanNLP.Speech.Data.UtteranceComposer(testCase.Split, 'MinWords', 4, 'MaxWords', 3), ...
                'OceanNLP:Speech:Data:UtteranceComposer:Range');
        end
    end
end
