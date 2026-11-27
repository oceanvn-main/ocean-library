classdef AudioAugmentTest < OceanTest.BaseTestCase
    % AUDIOAUGMENTTEST  Disabled = identity with no draws; gain/noise/SpecAugment semantics.

    properties
        W           % a real FSDD waveform
    end

    methods (TestClassSetup)
        function load(testCase)
            root = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech');
            testCase.W = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/7_nicolas_3.wav", 'Root', root);
        end
    end

    methods (Test)
        function testDisabledIsIdentityAndDrawsNothing(testCase)
            a = OceanNLP.Speech.Data.AudioAugment('Seed', 3);
            s0 = a.streamState();
            g = rng;
            testCase.verifyFalse(a.isEnabled());
            testCase.verifyTrue(a.forward(testCase.W) == testCase.W);
            f = OceanNLP.Speech.Core.FeatureSequence(ones(3, 4), 'FrameTimes', 0:3);
            testCase.verifyTrue(a.forward(f) == f);
            testCase.verifyEqual(a.streamState(), s0);
            testCase.verifyEqual(rng, g);
            onlyWave = OceanNLP.Speech.Data.AudioAugment('Gain', [0 1]);
            s1 = onlyWave.streamState();
            testCase.verifyTrue(onlyWave.forwardFeatures(f) == f);          % feature part disabled
            testCase.verifyEqual(onlyWave.streamState(), s1);
        end

        function testGainIsUniformScaleInRange(testCase)
            a = OceanNLP.Speech.Data.AudioAugment('Seed', 1, 'Gain', [-6 6]);
            y = a.forward(testCase.W);
            x = testCase.W.Samples;
            r = y.Samples(x ~= 0) ./ x(x ~= 0);
            testCase.verifyLessThan(max(abs(r - r(1))), 1e-12);
            gdb = 20 * log10(r(1));
            testCase.verifyTrue(gdb >= -6 && gdb <= 6);
            testCase.verifyTrue(contains(y.History(end), sprintf("%+.6g dB", gdb)));
            b = OceanNLP.Speech.Data.AudioAugment('Seed', 1, 'Gain', [-6 6]);
            testCase.verifyEqual(b.forward(testCase.W).Samples, y.Samples);
        end

        function testNoiseHitsRequestedSNR(testCase)
            a = OceanNLP.Speech.Data.AudioAugment('Seed', 2, 'NoiseSNR', [20 20]);
            y = a.forward(testCase.W);
            x = testCase.W.Samples;
            snr = 10 * log10(mean(x.^2) / mean((y.Samples - x).^2));
            % realized noise power of S i.i.d. N(0,s^2) draws: relative sd sqrt(2/S) (~0.1 dB here)
            tol = 10 * log10(1 + 5 * sqrt(2 / numel(x)));
            testCase.verifyEqual(snr, 20, 'AbsTol', tol);
        end

        function testSpecAugmentMasks(testCase)
            V = reshape(1:400, 10, 40);
            f = OceanNLP.Speech.Core.FeatureSequence(V, 'FrameTimes', (0:39) / 100);
            a = OceanNLP.Speech.Data.AudioAugment('Seed', 4, 'FreqMasks', 2, 'MaxFreqMask', 3, ...
                'TimeMasks', 2, 'MaxTimeMask', 6);
            y = a.forward(f);
            info = y.Config.SpecAugment;
            fill = mean(V, 'all');
            masked = false(size(V));
            for k = 1:size(info.FreqMaskRows, 1)
                rr = info.FreqMaskRows(k, :);
                testCase.verifyLessThanOrEqual(rr(2) - rr(1) + 1, 3);
                masked(rr(1):rr(2), :) = true;
            end
            for k = 1:size(info.TimeMaskFrames, 1)
                tt = info.TimeMaskFrames(k, :);
                testCase.verifyLessThanOrEqual(tt(2) - tt(1) + 1, 6);
                masked(:, tt(1):tt(2)) = true;
            end
            testCase.verifyEqual(y.Values(masked), repmat(fill, nnz(masked), 1));
            testCase.verifyEqual(y.Values(~masked), V(~masked));
            a.setEpoch(0);
            testCase.verifyEqual(a.forward(f).Values, y.Values);
            testCase.verifyError(@() a.forward(3), 'OceanNLP:Speech:Data:AudioAugment:InputType');
            testCase.verifyError(@() OceanNLP.Speech.Data.AudioAugment('Gain', [3 1]), ...
                'OceanNLP:Speech:Data:AudioAugment:Range');
        end
    end
end
