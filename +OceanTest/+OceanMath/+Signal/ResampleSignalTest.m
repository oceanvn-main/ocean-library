classdef ResampleSignalTest < OceanTest.BaseTestCase
    % RESAMPLESIGNALTEST  OceanMath.Signal.ResampleSignal vs scipy.signal.resample_poly
    %   driven by an independently designed scipy.signal.firwin Kaiser filter, on
    %   real audio (FSDD 8 kHz and the MATLAB Counting 44.1 kHz sample).

    methods (Test)
        function testFilterDesignMatchesFirwin(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            for c = fx.meta.res_cases'
                [h, p, q] = OceanMath.Signal.ResampleSignal.design(c.fs_in, c.fs_out, ...
                    'HalfWidth', c.half_width, 'KaiserBeta', c.beta);
                testCase.verifyEqual([p q], [c.up c.down], c.name);
                testCase.verifyEqual(h, fx.col([c.name '_h']), 'AbsTol', 1e-14, c.name);
            end
        end

        function testOutputMatchesResamplePoly(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            for c = fx.meta.res_cases'
                if c.fs_in == 44100
                    x = fx.col('res_in_44k');
                else
                    x = fx.col('res_in_8k');
                end
                y = OceanMath.Signal.ResampleSignal.resample(x, c.fs_in, c.fs_out, ...
                    'HalfWidth', c.half_width, 'KaiserBeta', c.beta);
                ref = fx.col(c.name);
                testCase.verifyEqual(numel(y), ceil(numel(x) * c.up / c.down), c.name);
                testCase.verifyEqual(y, ref, 'AbsTol', 1e-12, c.name);
            end
        end

        function testRealAudioInputsAreTheFiles(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            x = audioread(fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD', 'recordings', '3_theo_10.wav'), 'double');
            testCase.verifyEqual(x, fx.col('res_in_8k'));
        end

        function testPassbandToneSurvivesRoundTrip(testCase)
            % a 440 Hz tone (closed form) is in the passband of 8k->16k->8k
            fs = 8000;
            n = (0:7999)';
            x = sin(2 * pi * 440 * n / fs);
            y = OceanMath.Signal.ResampleSignal.resample(x, fs, 16000);
            n2 = (0:numel(y) - 1)';
            ref = sin(2 * pi * 440 * n2 / 16000);
            mid = 400:numel(y) - 400;                               % away from the zero-padded edges
            testCase.verifyLessThan(max(abs(y(mid) - ref(mid))), 1e-3);
            z = OceanMath.Signal.ResampleSignal.resample(y, 16000, fs);
            testCase.verifySize(z, size(x));
            testCase.verifyLessThan(max(abs(z(200:end-200) - x(200:end-200))), 1e-3);
        end

        function testAliasingToneIsRemoved(testCase)
            % 3.5 kHz at 16 kHz is above the 2 kHz output Nyquist of 16k->4k: must be attenuated
            fs = 16000;
            x = sin(2 * pi * 3500 * (0:15999)' / fs);
            y = OceanMath.Signal.ResampleSignal.resample(x, fs, 4000);
            testCase.verifyLessThan(sqrt(mean(y(200:end-200) .^ 2)), 1e-3 * sqrt(mean(x .^ 2)));
        end

        function testMultichannelEqualRatesAndEmpty(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            x = fx.col('res_in_8k');
            X = [x, -2 * x];
            Y = OceanMath.Signal.ResampleSignal.resample(X, 8000, 16000);
            y = OceanMath.Signal.ResampleSignal.resample(x, 8000, 16000);
            testCase.verifyEqual(Y, [y, -2 * y], 'AbsTol', 1e-15);
            testCase.verifyEqual(OceanMath.Signal.ResampleSignal.resample(X, 8000, 8000), X);
            testCase.verifySize(OceanMath.Signal.ResampleSignal.resample(zeros(0, 2), 8000, 16000), [0 2]);
        end
    end
end
