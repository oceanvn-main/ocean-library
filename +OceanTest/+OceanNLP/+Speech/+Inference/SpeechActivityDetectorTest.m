classdef SpeechActivityDetectorTest < OceanTest.BaseTestCase
    % SPEECHACTIVITYDETECTORTEST  The Ocean energy endpointer on real speech, validated against
    %   MATLAB Audio Toolbox detectSpeech (used here only, never in the library).
    %
    %   AGREEMENT PROTOCOL: both detectors run on the same conditioned 8 kHz waveform
    %   (AudioConditioner defaults). The Ocean detector uses settings matched to
    %   detectSpeech's defaults (no padding, MergeDistance 5 hops = 50 ms -> MinSilence
    %   0.05 s, MinSpeech 0, no MaxSegment), Sensitivity 0.5. Agreement = fraction of a
    %   10 ms time grid on which both say speech or both say non-speech.
    %   MEASURED (R2026a, 2026-10-07): Counting 15.5 s 0.952, SpeechDFT 5 s 0.942,
    %   Rainbow 114 s 0.859, 40 FSDD files (RandStream mt19937ar seed 7) mean 0.861,
    %   min 0.667 (0.30 s file "7_yweweler_17": a few 10 ms grid points decide it).
    %   The asserted minima below are these measurements rounded down by ~0.02-0.03 so
    %   that an unrelated numeric change does not flip the test, while a real
    %   regression of the endpointer (several points of agreement) does.

    properties (Constant)
        MinAgreeCounting = 0.93
        MinAgreeSpeechDFT = 0.92
        MinAgreeRainbow = 0.83
        MinAgreeFsddMean = 0.83
    end

    properties
        Cond
        Root string
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.Cond = OceanNLP.Speech.Inference.AudioConditioner('TargetRate', 8000);
            tc.Root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
        end
    end

    methods (Test)
        function agreesWithDetectSpeechOnToolboxSamples(tc)
            tc.assumeAudioToolbox();
            names = ["Counting-16-44p1-mono-15secs.wav", "SpeechDFT-16-8-mono-5secs.wav", "Rainbow-16-8-mono-114secs.wav"];
            mins = [tc.MinAgreeCounting, tc.MinAgreeSpeechDFT, tc.MinAgreeRainbow];
            for k = 1:numel(names)
                f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', names(k));
                tc.assumeTrue(isfile(f), "sample missing: " + names(k));
                [agree, bd, nRef, nOurs] = tc.compareWithDetectSpeech(string(f));
                tc.log(1, sprintf('%s: agreement %.3f, %d detectSpeech vs %d Ocean segments, median |boundary diff| %.3f s', ...
                    names(k), agree, nRef, nOurs, bd));
                tc.verifyGreaterThanOrEqual(agree, mins(k), names(k));
            end
        end

        function agreesWithDetectSpeechOnFsdd(tc)
            tc.assumeAudioToolbox();
            d = dir(fullfile(tc.Root, 'FSDD', 'recordings', '*.wav'));
            tc.assumeNotEmpty(d, 'FSDD not present.');
            rs = RandStream('mt19937ar', 'Seed', 7);
            pick = d(randperm(rs, numel(d), 40));
            agree = zeros(1, numel(pick));
            for k = 1:numel(pick)
                agree(k) = tc.compareWithDetectSpeech(string(fullfile(pick(k).folder, pick(k).name)));
            end
            tc.log(1, sprintf('FSDD (40 files): mean agreement %.3f, median %.3f, min %.3f', mean(agree), median(agree), min(agree)));
            tc.verifyGreaterThanOrEqual(mean(agree), tc.MinAgreeFsddMean);
        end

        function countingSegmentsMatchDetectSpeechOneToOne(tc)
            % Default (padded) settings: each detectSpeech region on the 15 s counting
            % recording lies in its own Ocean segment (measured: 10 vs 10).
            tc.assumeAudioToolbox();
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Counting-16-44p1-mono-15secs.wav');
            tc.assumeTrue(isfile(f));
            w = tc.Cond.forward(OceanNLP.Speech.Core.Waveform.read(string(f)));
            idx = detectSpeech(w.Samples, w.SampleRate);
            segs = OceanNLP.Speech.Inference.SpeechActivityDetector().detect(w);
            tc.verifyEqual(numel(segs), size(idx, 1));
            owner = zeros(1, size(idx, 1));
            for k = 1:size(idx, 1)
                mid = mean(idx(k, :) - [1 0]) / w.SampleRate;
                j = find([segs.Start] <= mid & [segs.End] >= mid, 1);
                tc.verifyNotEmpty(j);
                if ~isempty(j), owner(k) = j; end
            end
            tc.verifyEqual(numel(unique(owner)), size(idx, 1));
        end

        function silenceAroundRealSpeech(tc)
            w0 = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/4_theo_40.wav", 'Root', tc.Root);
            fs = w0.SampleRate;
            x = [zeros(fs / 2, 1); w0.Samples; zeros(fs / 2, 1)];
            d = OceanNLP.Speech.Inference.SpeechActivityDetector();
            [segs, info] = d.detect(OceanNLP.Speech.Core.Waveform(x, fs));
            tc.verifyNumElements(segs, 1);
            tc.verifyGreaterThanOrEqual(segs(1).Start, 0.5 - d.Pad - 1e-9);
            tc.verifyLessThanOrEqual(segs(1).End, 0.5 + w0.duration() + d.Pad + d.FrameSeconds);   % a frame may reach past the last speech sample
            tc.verifyEqual(numel(info.Energy), numel(info.FrameTimes));
            tc.verifyEqual(info.IsSpeech, info.FrameTimes >= segs(1).Start & info.FrameTimes <= segs(1).End);
        end

        function energyMatchesDirectFraming(tc)
            % cumulative-sum energy == explicit FrameSignal framing (independent route)
            w = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/0_theo_12.wav", 'Root', tc.Root);
            d = OceanNLP.Speech.Inference.SpeechActivityDetector();
            [~, info] = d.detect(w);
            L = round(d.FrameSeconds * w.SampleRate); H = round(d.HopSeconds * w.SampleRate);
            frames = OceanMath.Signal.FrameSignal.frame(w.Samples, L, H);
            tc.verifyEqual(info.Energy, 10 * log10(mean(frames .^ 2, 1) + 1e-12), 'AbsTol', 1e-9);
        end

        function silentAndVeryShortInputs(tc)
            d = OceanNLP.Speech.Inference.SpeechActivityDetector();
            [segs, info] = d.detect(OceanNLP.Speech.Core.Waveform(zeros(16000, 1), 8000));
            tc.verifyEmpty(segs);
            tc.verifySubstring(info.Note, "No speech");
            w0 = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/4_theo_40.wav", 'Root', tc.Root);
            [segs, info] = d.detect(OceanNLP.Speech.Core.Waveform(w0.Samples(1:100), 8000));   % < one frame
            tc.verifyNumElements(info.Energy, 1);
            tc.verifyLessThanOrEqual(numel(segs), 1);
            tc.verifyError(@() d.trim(w0, struct('Start', {}, 'End', {})), ...
                'OceanNLP:Speech:Inference:SpeechActivityDetector:NoSpeech');
        end

        function maxSegmentSplitsAtLowestEnergyFrame(tc)
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Counting-16-44p1-mono-15secs.wav');
            tc.assumeTrue(isfile(f));
            w = tc.Cond.forward(OceanNLP.Speech.Core.Waveform.read(string(f)));
            % bridge every pause -> one long segment, then split it
            whole = OceanNLP.Speech.Inference.SpeechActivityDetector('MinSilence', 5, 'MaxSegment', Inf).detect(w);
            tc.assumeNumElements(whole, 1);
            d = OceanNLP.Speech.Inference.SpeechActivityDetector('MinSilence', 5, 'MaxSegment', 8);
            [segs, info] = d.detect(w);
            tc.verifyTrue(all([segs.End] - [segs.Start] <= 8 + 1e-12));
            % brute force: the first split is the argmin of E over the allowed window
            s = whole.Start; e = whole.End; m = min(d.MinSpeech, (e - s) / 4);
            win = find(info.FrameTimes >= s + m & info.FrameTimes <= e - m);
            [~, k] = min(info.Energy(win));
            tc.verifyTrue(any(abs(info.SplitTimes - info.FrameTimes(win(k))) < 1e-12));
            tc.verifyEqual(segs(1).Start, s);
            tc.verifyEqual(segs(end).End, e);
            tc.verifyEqual([segs(2:end).Start], [segs(1:end-1).End]);    % contiguous pieces
        end

        function longRecordingSegmentsAreCapped(tc)
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Rainbow-16-8-mono-114secs.wav');
            tc.assumeTrue(isfile(f));
            w = tc.Cond.forward(OceanNLP.Speech.Core.Waveform.read(string(f)));
            d = OceanNLP.Speech.Inference.SpeechActivityDetector('MaxSegment', 3);
            t = tic;
            segs = d.detect(w);
            tc.log(1, sprintf('Rainbow 114 s: %d segments in %.3f s', numel(segs), toc(t)));
            tc.verifyTrue(all([segs.End] - [segs.Start] <= 3 + 1e-12));
            tc.verifyTrue(all(diff([segs.Start]) > 0));
            parts = d.split(w, segs);
            tc.verifyNumElements(parts, numel(segs));
            durs = arrayfun(@(p) p.duration(), parts);
            tc.verifyEqual(durs, [segs.End] - [segs.Start], 'AbsTol', 2 / w.SampleRate);
            tc.verifyTrue(contains(parts(1).History(end), "segment 1/" + numel(segs)));
        end

        function trimKeepsFirstToLastSpeech(tc)
            w0 = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/4_theo_40.wav", 'Root', tc.Root);
            x = [zeros(4000, 1); w0.Samples; zeros(4000, 1)];
            w = OceanNLP.Speech.Core.Waveform(x, 8000);
            d = OceanNLP.Speech.Inference.SpeechActivityDetector();
            segs = d.detect(w);
            wt = d.trim(w, segs);
            n1 = round(segs(1).Start * 8000) + 1; n2 = round(segs(end).End * 8000);
            tc.verifyEqual(wt.Samples, x(n1:n2));
            tc.verifySubstring(wt.History(end), "trim to speech");
        end

        function configRoundTrip(tc)
            d = OceanNLP.Speech.Inference.SpeechActivityDetector('Sensitivity', 0.8, 'MinSpeech', 0.1, ...
                'MinSilence', 0.3, 'Pad', 0.05, 'MaxSegment', 5);
            tc.verifyEqual(OceanNLP.Speech.Inference.SpeechActivityDetector.fromConfig(d.config()).config(), d.config());
            tc.verifyError(@() OceanNLP.Speech.Inference.SpeechActivityDetector('MaxSegment', 0.2, 'MinSpeech', 0.15), ...
                'OceanNLP:Speech:Inference:SpeechActivityDetector:MaxSegment');
        end
    end

    methods
        function assumeAudioToolbox(tc)
            tc.assumeTrue(~isempty(which('detectSpeech')), ...
                'Audio Toolbox detectSpeech not available.');
        end

        function [agree, medBoundary, nRef, nOurs] = compareWithDetectSpeech(tc, file)
            w = tc.Cond.forward(OceanNLP.Speech.Core.Waveform.read(file));
            fs = w.SampleRate;
            idx = detectSpeech(w.Samples, fs);
            d = OceanNLP.Speech.Inference.SpeechActivityDetector('Sensitivity', 0.5, 'MaxSegment', Inf, ...
                'Pad', 0, 'MinSilence', 0.05, 'MinSpeech', 0);
            segs = d.detect(w);
            g = (0:0.01:w.duration())';
            ref = false(size(g)); ours = false(size(g));
            for k = 1:size(idx, 1), ref = ref | (g >= (idx(k, 1) - 1) / fs & g <= idx(k, 2) / fs); end
            for k = 1:numel(segs), ours = ours | (g >= segs(k).Start & g <= segs(k).End); end
            agree = mean(ref == ours);
            nRef = size(idx, 1);
            nOurs = numel(segs);
            rb = [(idx(:, 1) - 1) / fs; idx(:, 2) / fs];
            ob = [[segs.Start]'; [segs.End]'];
            medBoundary = NaN;
            if ~isempty(rb) && ~isempty(ob)
                medBoundary = median(min(abs(rb - ob'), [], 2));
            end
        end
    end
end
