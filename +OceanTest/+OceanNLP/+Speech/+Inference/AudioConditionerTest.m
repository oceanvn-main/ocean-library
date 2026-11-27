classdef AudioConditionerTest < OceanTest.BaseTestCase
    % AUDIOCONDITIONERTEST  Real audio (FSDD, Audio Toolbox speech samples) through the
    %   conditioner; every expected value is recomputed independently from the samples.

    properties
        Fsdd            % 7_theo_3.wav (8 kHz mono)
        FsddFile string
        Counting        % Counting-16-44p1-mono-15secs.wav (44.1 kHz)
    end

    methods (TestClassSetup)
        function loadAudio(tc)
            root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            tc.FsddFile = string(fullfile(root, 'FSDD', 'recordings', '7_theo_3.wav'));
            tc.assumeTrue(isfile(tc.FsddFile), 'FSDD not present.');
            tc.Fsdd = OceanNLP.Speech.Core.Waveform.read(tc.FsddFile);
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Counting-16-44p1-mono-15secs.wav');
            if isfile(f)
                tc.Counting = OceanNLP.Speech.Core.Waveform.read(string(f));
            end
        end
    end

    methods (Test)
        function stereo44kIsMixedResampledAndRecorded(tc)
            tc.assumeNotEmpty(tc.Counting, 'Audio Toolbox sample not present.');
            x = tc.Counting.Samples;
            w = OceanNLP.Speech.Core.Waveform([x, 0.5 * x], 44100, 'Source', "counting-stereo");
            c = OceanNLP.Speech.Inference.AudioConditioner('TargetRate', 8000, 'Normalize', false);
            [y, rep] = c.forward(w);
            % independent route: the same steps by hand
            xm = mean([x, 0.5 * x], 2);
            xm = xm - mean(xm);
            expected = OceanMath.Signal.ResampleSignal.resample(xm, 44100, 8000);
            tc.verifyEqual(y.Samples, expected, 'AbsTol', 1e-12);
            tc.verifyEqual(y.SampleRate, 8000);
            tc.verifyEqual(rep.Channels, 2);
            tc.verifyEqual(rep.InputRate, 44100);
            tc.verifyEqual(rep.OutputRate, 8000);
            tc.verifyEqual(rep.DurationIn, size(x, 1) / 44100, 'AbsTol', 1e-12);
            tc.verifyEqual(rep.DurationOut, numel(expected) / 8000, 'AbsTol', 1e-12);
            tc.verifyEqual(rep.RMSdBFS, 20 * log10(sqrt(mean(expected .^ 2))), 'AbsTol', 1e-9);
            tc.verifyTrue(any(contains(y.History, "mono mean of 2 channels")));
            tc.verifyTrue(any(contains(y.History, "resample 44100 Hz -> 8000 Hz")));
            tc.verifyTrue(any(contains(y.History, "remove DC")));
            tc.verifyEqual(numel(rep.Steps), 3);
            tc.verifyEqual(y.Source, "counting-stereo");
        end

        function dcOffsetIsMeasuredAndRemoved(tc)
            x = tc.Fsdd.Samples + 0.1;
            [y, rep] = OceanNLP.Speech.Inference.AudioConditioner('Normalize', false).forward( ...
                OceanNLP.Speech.Core.Waveform(x, 8000));
            tc.verifyEqual(rep.DCOffset, mean(x), 'AbsTol', 1e-15);
            tc.verifyEqual(mean(y.Samples), 0, 'AbsTol', 1e-15);
            tc.verifyEqual(y.Samples, x - mean(x), 'AbsTol', 1e-15);
        end

        function clippingIsReportedAndGainNeverClips(tc)
            x = min(max(tc.Fsdd.Samples * 40, -1), 1);          % hard-clipped real speech
            c = OceanNLP.Speech.Inference.AudioConditioner();
            [y, rep] = c.forward(OceanNLP.Speech.Core.Waveform(x, 8000));
            tc.verifyEqual(rep.ClippedFraction, nnz(abs(x) >= 0.999) / numel(x));
            tc.verifyGreaterThan(rep.ClippedFraction, 0.001);
            tc.verifyTrue(any(startsWith(rep.Warnings, "Clipped:")));
            tc.verifyLessThan(max(abs(y.Samples)), 1);
        end

        function normalisationHitsTargetRms(tc)
            [y, rep] = OceanNLP.Speech.Inference.AudioConditioner('TargetRMSdBFS', -20).forward(tc.Fsdd);
            tc.verifyEqual(20 * log10(sqrt(mean(y.Samples .^ 2))), -20, 'AbsTol', 1e-9);
            tc.verifyEqual(rep.OutputRMSdBFS, -20, 'AbsTol', 1e-9);
            tc.verifyEqual(rep.GainDB, -20 - rep.RMSdBFS, 'AbsTol', 1e-9);
            tc.verifyTrue(any(startsWith(y.History, "gain ")));
        end

        function veryQuietInputIsCappedAndWarned(tc)
            w = OceanNLP.Speech.Core.Waveform(tc.Fsdd.Samples * 1e-4, 8000);
            [y, rep] = OceanNLP.Speech.Inference.AudioConditioner('MaxGainDB', 30).forward(w);
            tc.verifyEqual(rep.GainDB, 30);
            tc.verifyTrue(any(startsWith(rep.Warnings, "Very quiet")));
            tc.verifyTrue(any(startsWith(rep.Warnings, "Gain limited to +30 dB")));
            tc.verifyTrue(all(isfinite(y.Samples)));
        end

        function silentInputDoesNotBlowUp(tc)
            w = OceanNLP.Speech.Core.Waveform(zeros(8000, 2), 8000);
            [y, rep] = OceanNLP.Speech.Inference.AudioConditioner().forward(w);
            tc.verifyEqual(rep.RMSdBFS, -Inf);
            tc.verifyEqual(rep.PeakDBFS, -Inf);
            tc.verifyEqual(rep.GainDB, 0);
            tc.verifyEqual(y.Samples, zeros(8000, 1));
            tc.verifyTrue(any(startsWith(rep.Warnings, "Silent input")));
        end

        function veryShortInputIsWarnedNotPadded(tc)
            w = OceanNLP.Speech.Core.Waveform(tc.Fsdd.Samples(1:400), 8000);    % 50 ms
            [y, rep] = OceanNLP.Speech.Inference.AudioConditioner('MinDuration', 0.1).forward(w);
            tc.verifyEqual(y.numSamples(), 400);
            tc.verifyTrue(any(startsWith(rep.Warnings, "Very short input")));
        end

        function integerPcmIsScaledLikeAudioread(tc)
            xi = audioread(tc.FsddFile, 'native');
            tc.assumeClass(xi, 'int16');
            w = OceanNLP.Speech.Inference.AudioConditioner.toWaveform(struct('Samples', xi, 'SampleRate', 8000));
            tc.verifyEqual(w.Samples, tc.Fsdd.Samples);              % audioread 'double' route
            tc.verifyTrue(any(contains(w.History, "int16 PCM scaled")));
            u = OceanNLP.Speech.Inference.AudioConditioner.toWaveform(struct('Samples', uint8([0; 128; 255]), 'SampleRate', 8000));
            tc.verifyEqual(u.Samples, [-1; 0; 127 / 128]);
            % an unscaled integer-valued double waveform is reported, not silently accepted
            [~, rep] = OceanNLP.Speech.Inference.AudioConditioner().forward(OceanNLP.Speech.Core.Waveform(double(xi), 8000));
            tc.verifyTrue(any(contains(rep.Warnings, "unscaled integer PCM")));
        end

        function inputRoutesAgree(tc)
            a = OceanNLP.Speech.Inference.AudioConditioner.toWaveform(tc.FsddFile);
            b = OceanNLP.Speech.Inference.AudioConditioner.toWaveform(struct('Samples', tc.Fsdd.Samples, 'SampleRate', 8000));
            tc.verifyEqual(a.Samples, b.Samples);
            tc.verifyError(@() OceanNLP.Speech.Inference.AudioConditioner.toWaveform(42), ...
                'OceanNLP:Speech:Inference:AudioConditioner:InputType');
        end

        function longInputIsMemorySane(tc)
            f = fullfile(matlabroot, 'toolbox', 'audio', 'samples', 'Rainbow-16-8-mono-114secs.wav');
            tc.assumeTrue(isfile(f), 'Audio Toolbox sample not present.');
            w = OceanNLP.Speech.Core.Waveform.read(string(f));
            t = tic;
            [y, rep] = OceanNLP.Speech.Inference.AudioConditioner('TargetRate', 16000).forward(w);
            el = toc(t);
            tc.log(1, sprintf('Rainbow 114 s, 8 kHz -> 16 kHz conditioned in %.2f s', el));
            tc.verifyEqual(y.numSamples(), 2 * w.numSamples());
            tc.verifyEqual(rep.DurationOut, rep.DurationIn, 'AbsTol', 1e-9);
        end

        function configRoundTripAndNoRng(tc)
            c = OceanNLP.Speech.Inference.AudioConditioner('TargetRate', 16000, 'MixDown', "first", ...
                'TargetRMSdBFS', -25, 'MaxGainDB', 12, 'ClipThreshold', 0.99, 'MinDuration', 0.2);
            c2 = OceanNLP.Speech.Inference.AudioConditioner.fromConfig(c.config());
            tc.verifyEqual(c2.config(), c.config());
            rng(77, 'twister'); s0 = rng;
            [y1, r1] = c.forward(tc.Fsdd);
            [y2, r2] = c2.forward(tc.Fsdd);
            tc.verifyEqual(rng, s0);
            tc.verifyEqual(y1.Samples, y2.Samples);
            tc.verifyEqual(r1, r2);
        end
    end
end
