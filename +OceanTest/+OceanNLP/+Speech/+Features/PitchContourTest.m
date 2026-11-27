classdef PitchContourTest < OceanTest.BaseTestCase
    % PITCHCONTOURTEST  YIN pitch on closed-form periodic signals (truth known analytically),
    %   silence, noise, grid agreement and input policy. Comparison against MATLAB
    %   `pitch` / librosa.pyin on real speech is in validate_features_external.m.

    methods (Test)
        function testPureTonesAtKnownF0(testCase)
            for fs = [8000 16000]
                for f0 = [87.3 100 150.5 220 333.3]
                    x = 0.5 * sin(2 * pi * f0 * (0:fs - 1)' / fs + 0.3);
                    p = OceanNLP.Speech.Features.PitchContour(fs).forward(OceanNLP.Speech.Core.Waveform(x, fs));
                    est = p.Values(1, :);
                    testCase.verifyTrue(all(est > 0), sprintf('fs=%d f0=%g all voiced', fs, f0));
                    testCase.verifyLessThan(max(abs(est - f0)) / f0, 0.01, sprintf('fs=%d f0=%g', fs, f0));
                    testCase.verifyGreaterThan(min(p.Values(2, :)), 0.9);
                end
            end
        end

        function testHarmonicComplexWithWeakFundamental(testCase)
            % sum of harmonics 1..8 of 120 Hz with 1/k amplitudes, fundamental at 10 %: period is still 1/120 s
            fs = 8000;
            n = (0:fs - 1)';
            f0 = 120;
            x = 0.1 * sin(2 * pi * f0 * n / fs);
            for k = 2:8
                x = x + (1 / k) * sin(2 * pi * k * f0 * n / fs + k);
            end
            p = OceanNLP.Speech.Features.PitchContour(fs).forward(OceanNLP.Speech.Core.Waveform(x, fs));
            testCase.verifyLessThan(max(abs(p.Values(1, :) - f0)) / f0, 0.01);
        end

        function testSilenceAndNoise(testCase)
            fs = 8000;
            pt = OceanNLP.Speech.Features.PitchContour(fs);
            s = pt.forward(OceanNLP.Speech.Core.Waveform(zeros(fs, 1), fs));
            testCase.verifyEqual(s.Values, zeros(2, s.numFrames()));
            rs = RandStream('mt19937ar', 'Seed', 7);
            nz = pt.forward(OceanNLP.Speech.Core.Waveform(randn(rs, fs, 1), fs));
            testCase.verifyTrue(all(isfinite(nz.Values), 'all'));
            testCase.verifyLessThan(mean(nz.Values(1, :) > 0), 0.2);   % white noise is (mostly) unvoiced
            testCase.verifyTrue(all(nz.Values(2, :) >= 0 & nz.Values(2, :) <= 1));
        end

        function testGridAgreementAndShortInput(testCase)
            fs = 8000;
            w = OceanNLP.Speech.Core.Waveform.read("recordings/5_yweweler_12.wav", ...
                'Root', fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD'));
            for center = [false true]
                pt = OceanNLP.Speech.Features.PitchContour(fs, 'Center', center);
                p = pt.forward(w);
                g = OceanNLP.Speech.Core.FrameGrid(fs, 'Center', center);
                testCase.verifyEqual(p.FrameTimes, g.frameTimes(w.numSamples()));
                testCase.verifySize(p.Values, [2, g.numFrames(w.numSamples())]);
                v = p.Values(1, p.Values(1, :) > 0);
                testCase.verifyTrue(all(v >= 50 & v <= 400));
                lm = OceanNLP.Speech.Features.LogMel(fs, 'Center', center).forward(w);
                testCase.verifyEqual(p.FrameTimes, lm.FrameTimes);
            end
            short = OceanNLP.Speech.Features.PitchContour(fs).forward(OceanNLP.Speech.Core.Waveform(randn(100, 1), fs));
            testCase.verifySize(short.Values, [2 0]);
        end

        function testValidation(testCase)
            testCase.verifyError(@() OceanNLP.Speech.Features.PitchContour(8000, 'MinF0', 300, 'MaxF0', 200), ...
                'OceanNLP:Speech:Features:PitchContour:InvalidRange');
            testCase.verifyError(@() OceanNLP.Speech.Features.PitchContour(16000).forward( ...
                OceanNLP.Speech.Core.Waveform(randn(1000, 1), 8000)), ...
                'OceanNLP:Speech:Features:PitchContour:SampleRateMismatch');
            testCase.verifyEqual(OceanNLP.Speech.Features.PitchContour(8000).outputDim(), 2);
        end
    end
end
