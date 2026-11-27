classdef FrameSignalTest < OceanTest.BaseTestCase
    % FRAMESIGNALTEST  OceanMath.Signal.FrameSignal + SpectralAnalyzer.framePowerSpectrum.
    %   References: numpy pad/librosa.util.frame, scipy get_window, numpy rfft
    %   (fixtures/make_speech_signal_fixtures.py), brute-force frame counting.

    methods (Test)
        function testInputFixtureIsTheRealRecording(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            wavFile = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD', 'recordings', '0_jackson_0.wav');
            testCase.assumeTrue(isfile(wavFile), 'FSDD recording not present; dataset is not distributed with the library.');
            x = audioread(wavFile, 'double');
            testCase.verifyEqual(x(1:numel(fx.col('frame_input'))), fx.col('frame_input'));
        end

        function testFramesMatchNumpyPadAndLibrosaFrame(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            x = fx.col('frame_input');
            for LH = [200 80; 7 3]'
                L = LH(1); H = LH(2);
                [fr, s0] = OceanMath.Signal.FrameSignal.frame(x, L, H);
                testCase.verifyEqual(fr, fx.raw(sprintf('frames_nocenter_L%d_H%d', L, H)));
                testCase.verifyEqual(s0, (0:size(fr, 2) - 1) * H);
                for mode = ["constant", "reflect"]
                    [fr, s0] = OceanMath.Signal.FrameSignal.frame(x, L, H, 'Center', true, 'PadMode', mode);
                    testCase.verifyEqual(fr, fx.raw(sprintf('frames_center_%s_L%d_H%d', mode, L, H)), ...
                        sprintf('%s L=%d', mode, L));
                    testCase.verifyEqual(s0, (0:size(fr, 2) - 1) * H - floor(L / 2));
                end
            end
        end

        function testNumFramesAgreesWithBruteForce(testCase)
            % count every start position whose L samples fit in the padded signal
            for S = [0 1 5 6 7 8 199 200 201 1000]
                for L = [1 6 7 200]
                    for H = [1 3 80]
                        for c = [false true]
                            P = c * floor(L / 2);
                            brute = 0;
                            st = 0;
                            while st + L <= S + 2 * P
                                brute = brute + 1;
                                st = st + H;
                            end
                            testCase.verifyEqual(OceanMath.Signal.FrameSignal.numFrames(S, L, H, c), brute, ...
                                sprintf('S=%d L=%d H=%d c=%d', S, L, H, c));
                        end
                    end
                end
            end
        end

        function testEdgeCases(testCase)
            [fr, s0] = OceanMath.Signal.FrameSignal.frame(zeros(0, 1), 200, 80);
            testCase.verifyEqual(size(fr), [200 0]);
            testCase.verifyEmpty(s0);
            [fr, ~] = OceanMath.Signal.FrameSignal.frame((1:150)', 200, 80);          % S < L, no centring
            testCase.verifySize(fr, [200 0]);
            [fr, s0] = OceanMath.Signal.FrameSignal.frame((1:150)', 200, 80, 'Center', true);  % zeros pad
            testCase.verifyEqual(size(fr, 2), OceanMath.Signal.FrameSignal.numFrames(150, 200, 80, true));
            testCase.verifyEqual(fr(101:200, 1), (1:100)');                             % s_1 = -100
            testCase.verifyEqual(s0(1), -100);
            testCase.verifyError(@() OceanMath.Signal.FrameSignal.frame((1:100)', 200, 80, 'Center', true, ...
                'PadMode', "reflect"), 'OceanMath:Signal:FrameSignal:TooShortForReflect');
        end

        function testWindowsMatchScipy(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            for L = [200 7]
                for periodic = [true false]
                    tag = sprintf('L%d_%s', L, string(ternary(periodic, "periodic", "symmetric")));
                    for type = ["hann", "hamming", "povey"]
                        w = OceanMath.Signal.FrameSignal.window(type, L, 'Periodic', periodic);
                        testCase.verifyEqual(w, fx.col(sprintf('win_%s_%s', type, tag)), 'AbsTol', 1e-14, ...
                            sprintf('%s %s', type, tag));
                    end
                end
            end
            testCase.verifyEqual(OceanMath.Signal.FrameSignal.window("rectangular", 5), ones(5, 1));
            testCase.verifyEqual(OceanMath.Signal.FrameSignal.window("hann", 1), 1);
        end

        function testFramePowerSpectrumMatchesNumpyRfft(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            x = fx.col('frame_input');
            fr = OceanMath.Signal.FrameSignal.frame(x, 200, 80) .* OceanMath.Signal.FrameSignal.window("hann", 200);
            for nfft = [256 255 200]                                     % even, odd, = L
                P = OceanMath.Signal.SpectralAnalyzer.framePowerSpectrum(fr, nfft);
                ref = fx.raw(sprintf('pow_nfft%d', nfft));
                testCase.verifySize(P, [floor(nfft / 2) + 1, size(fr, 2)]);
                testCase.verifyEqual(P, ref, 'AbsTol', 1e-12 * max(ref(:)), sprintf('nfft=%d', nfft));
            end
            testCase.verifySize(OceanMath.Signal.SpectralAnalyzer.framePowerSpectrum(zeros(200, 0), 255), [128 0]);
            testCase.verifyError(@() OceanMath.Signal.SpectralAnalyzer.framePowerSpectrum(fr, 128), ...
                'OceanMath:Signal:SpectralAnalyzer:NFFTTooSmall');
        end

        function testPowerSpectrumParsevalOnRealFrame(testCase)
            % sum_k |X_k|^2 (two-sided) = N sum_n x_n^2 ; rebuild the two-sided sum from the one-sided bins
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            x = fx.col('frame_input');
            fr = OceanMath.Signal.FrameSignal.frame(x, 200, 80);
            for nfft = [256 255]
                P = OceanMath.Signal.SpectralAnalyzer.framePowerSpectrum(fr, nfft);
                if mod(nfft, 2) == 0
                    two = P(1, :) + 2 * sum(P(2:end-1, :), 1) + P(end, :);
                else
                    two = P(1, :) + 2 * sum(P(2:end, :), 1);
                end
                testCase.verifyEqual(two, nfft * sum(fr .^ 2, 1), 'RelTol', 1e-12);
            end
        end
    end

    methods (Static)
        function fx = fixtures()
            dir_ = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            raw = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(dir_, [char(n) '.npy'])));
            fx = struct('raw', raw, 'col', @(n) reshape(raw(n), [], 1), ...
                'meta', jsondecode(fileread(fullfile(dir_, 'speech_signal_fixtures.json'))));
        end
    end
end

function out = ternary(c, a, b)
if c
    out = a;
else
    out = b;
end
end
