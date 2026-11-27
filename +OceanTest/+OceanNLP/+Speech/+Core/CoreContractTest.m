classdef CoreContractTest < OceanTest.BaseTestCase
    % CORECONTRACTTEST  Phase 0 value classes: Waveform, FeatureSequence, Transcript.

    methods (Test)
        function testWaveformReadsRealFileWithRootRelativeSource(testCase)
            root = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD');
            w = OceanNLP.Speech.Core.Waveform.read("recordings/0_jackson_0.wav", 'Root', root);
            info = audioinfo(fullfile(root, 'recordings', '0_jackson_0.wav'));
            testCase.verifyEqual(w.SampleRate, info.SampleRate);
            testCase.verifyEqual(w.numSamples(), info.TotalSamples);
            testCase.verifyEqual(w.duration(), info.TotalSamples / info.SampleRate, 'AbsTol', 1e-12);
            testCase.verifyEqual(w.Source, "recordings/0_jackson_0.wav");
        end

        function testWaveformRejectsAmbiguousAndInvalidInput(testCase)
            W = @(varargin) OceanNLP.Speech.Core.Waveform(varargin{:});
            testCase.verifyError(@() W(randn(1, 100), 8000), 'OceanNLP:Speech:Core:Waveform:Orientation');
            testCase.verifyError(@() W([], 8000), 'OceanNLP:Speech:Core:Waveform:Empty');
            testCase.verifyError(@() W([1; NaN], 8000), 'OceanNLP:Speech:Core:Waveform:Nonfinite');
            testCase.verifyError(@() W([1; 2], 0), 'OceanNLP:Speech:Core:Waveform:SampleRate');
            testCase.verifyError(@() W([1; 2], Inf), 'OceanNLP:Speech:Core:Waveform:SampleRate');
        end

        function testToMonoRecordsHistory(testCase)
            x = [(1:4)', (5:8)'];
            w = OceanNLP.Speech.Core.Waveform(x, 16000);
            m = w.toMono();
            testCase.verifyEqual(m.Samples, mean(x, 2));
            testCase.verifyEqual(m.History, "mono mean of 2 channels");
            testCase.verifyEqual(w.toMono("first").Samples, x(:, 1));
        end

        function testWaveformRejectsTransposedAndAcceptsIntegers(testCase)
            W = @(varargin) OceanNLP.Speech.Core.Waveform(varargin{:});
            testCase.verifyError(@() W(randn(2, 100), 8000), 'OceanNLP:Speech:Core:Waveform:Orientation');
            testCase.verifyError(@() W([1; 2], 8000, 'SourceOffset', -1), 'OceanNLP:Speech:Core:Waveform:SourceOffset');
            w = W(int16([1; -2; 3]), 8000);                % integers cannot be non-finite
            testCase.verifyClass(w.Samples, 'double');
            testCase.verifyEqual(w.Samples, [1; -2; 3]);
            testCase.verifyEqual(W(5, 8000).numSamples(), 1);   % a single mono sample is unambiguous
        end

        function testSegmentTracksOffsetAndHistory(testCase)
            fs = 1000;
            x = [(1:10)', -(1:10)'];
            w = OceanNLP.Speech.Core.Waveform(x, fs, 'SourceOffset', 2);
            s = w.segment([3 6]);
            testCase.verifyEqual(s.Samples, x(3:6, :));
            testCase.verifyEqual(s.SourceOffset, 2 + 2 / fs, 'AbsTol', 1e-15);
            testCase.verifyEqual(s.History, "segment samples 3-6 (0.0020-0.0060 s)");
            t = w.segment([0.002 0.006], 'Units', "seconds");     % [t0, t1) -> samples 3..6
            testCase.verifyEqual(t.Samples, s.Samples);
            testCase.verifyEqual(s.times('Reference', "source"), 2 + (2:5)' / fs, 'AbsTol', 1e-15);
            testCase.verifySameHandle(w.segment([1 10]), w);          % full range: no copy
            testCase.verifyError(@() w.segment([0 3]), 'OceanNLP:Speech:Core:Waveform:Range');
            testCase.verifyError(@() w.segment([4 11]), 'OceanNLP:Speech:Core:Waveform:Range');
            testCase.verifyError(@() w.segment([5 4]), 'OceanNLP:Speech:Core:Waveform:Range');
            testCase.verifyError(@() w.segment([1.5 4]), 'OceanNLP:Speech:Core:Waveform:Range');
            d = s.derive(s.Samples(2:end, :), fs, "trim", 'SourceOffset', s.SourceOffset + 1 / fs);
            testCase.verifyEqual(d.SourceOffset, s.SourceOffset + 1 / fs);
            testCase.verifyEqual(s.derive(2 * s.Samples, fs, "gain").SourceOffset, s.SourceOffset);
        end

        function testReadRangeEqualsSegmentOfFullRead(testCase)
            root = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD');
            file = "recordings/0_jackson_0.wav";
            full = OceanNLP.Speech.Core.Waveform.read(file, 'Root', root);
            part = OceanNLP.Speech.Core.Waveform.read(file, 'Root', root, 'Range', [101 900]);
            testCase.verifyEqual(part.Samples, full.Samples(101:900, :));
            testCase.verifyEqual(part.SourceOffset, 100 / full.SampleRate);
            testCase.verifyEqual(part.Source, file);
            sec = OceanNLP.Speech.Core.Waveform.read(file, 'Root', root, 'Range', [0.05 0.1], 'Units', "seconds");
            testCase.verifyEqual(sec.Samples, full.segment([0.05 0.1], 'Units', "seconds").Samples);
            testCase.verifyError(@() OceanNLP.Speech.Core.Waveform.read(file, 'Root', root, ...
                'Range', [1 full.numSamples() + 1]), 'OceanNLP:Speech:Core:Waveform:Range');
        end

        function testReadSourceIsPathAware(testCase)
            root = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD');
            absFile = fullfile(root, 'recordings', '0_jackson_0.wav');
            w = OceanNLP.Speech.Core.Waveform.read(string(absFile), 'Root', root + string(filesep));
            testCase.verifyEqual(w.Source, "recordings/0_jackson_0.wav");
            if ispc
                w = OceanNLP.Speech.Core.Waveform.read(string(upper(absFile)), 'Root', lower(root));
                testCase.verifyEqual(lower(w.Source), "recordings/0_jackson_0.wav");
            end
            % 'record' is a string prefix of 'recordings' but not a parent folder.
            testCase.verifyWarning(@() OceanNLP.Speech.Core.Waveform.read(string(absFile), ...
                'Root', fullfile(root, 'record')), 'OceanNLP:Speech:Core:Waveform:OutsideRoot');
        end

        function testConcatChecksAndRecords(testCase)
            a = OceanNLP.Speech.Core.Waveform((1:3)', 8000, 'Source', "a.wav");
            b = OceanNLP.Speech.Core.Waveform((4:5)', 8000, 'Source', "b.wav").derive((4:5)', 8000, "gain 0 dB");
            c = OceanNLP.Speech.Core.Waveform.concat(a, b);
            testCase.verifyEqual(c.Samples, (1:5)');
            testCase.verifyEqual(c.Source, "a.wav+b.wav");
            testCase.verifyEqual(c.History, "concatenated 2 waveforms (component histories: [] + [gain 0 dB])");
            testCase.verifyEqual(OceanNLP.Speech.Core.Waveform.concat(a, a).History, "concatenated 2 waveforms");
            testCase.verifyError(@() OceanNLP.Speech.Core.Waveform.concat(a, ...
                OceanNLP.Speech.Core.Waveform((1:3)', 16000)), 'OceanNLP:Speech:Core:Waveform:ConcatMismatch');
            testCase.verifyError(@() OceanNLP.Speech.Core.Waveform.concat(a, 1), 'OceanNLP:Speech:Core:Waveform:Concat');
        end

        function testFeatureSequenceShapesAndValidation(testCase)
            V = reshape(1:12, 3, 4);
            f = OceanNLP.Speech.Core.FeatureSequence(V, 'FrameTimes', (0:3) * 0.01, 'FrequencyHz', [100; 200; 300]);
            testCase.verifyEqual(size(f.toCBT()), [3 1 4]);
            testCase.verifyEqual(squeeze(f.toCBT()), V);
            testCase.verifyError(@() OceanNLP.Speech.Core.FeatureSequence(V, 'FrameTimes', 1:3), ...
                'OceanNLP:Speech:Core:FeatureSequence:FrameTimes');
            testCase.verifyError(@() OceanNLP.Speech.Core.FeatureSequence([1 Inf], 'FrameTimes', [0 1]), ...
                'OceanNLP:Speech:Core:FeatureSequence:Nonfinite');
            empty = OceanNLP.Speech.Core.FeatureSequence(zeros(3, 0));     % shorter than one frame
            testCase.verifyEqual(empty.numFrames(), 0);
        end

        function testTranscriptDefaultsAreHonest(testCase)
            tr = OceanNLP.Speech.Core.Transcript("one two");
            testCase.verifyEmpty(tr.Segments);
            testCase.verifyEqual(tr.Language, "");
            testCase.verifyTrue(isnan(tr.Score));
            testCase.verifyEqual(tr.RawText, "one two");
            both = [tr, OceanNLP.Speech.Core.Transcript("three")];
            testCase.verifyEqual(both.texts(), ["one two", "three"]);
        end
    end
end
