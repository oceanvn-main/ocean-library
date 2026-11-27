classdef FrameGridTest < OceanTest.BaseTestCase
    % FRAMEGRIDTEST  OceanNLP.Speech.Core.FrameGrid time conventions (contract sec. 3).

    methods (Test)
        function testDefaultsFollowTheContract(testCase)
            for fs = [8000 16000 44100]
                g = OceanNLP.Speech.Core.FrameGrid(fs);
                testCase.verifyEqual(g.WindowLength, round(0.025 * fs));
                testCase.verifyEqual(g.HopLength, round(0.010 * fs));
                testCase.verifyEqual(g.NFFT, 2 ^ nextpow2(round(0.025 * fs)));
                testCase.verifyGreaterThanOrEqual(g.NFFT, g.WindowLength);
            end
            testCase.verifyError(@() OceanNLP.Speech.Core.FrameGrid(8000, 'WindowLength', 300, 'NFFT', 256), ...
                'OceanNLP:Speech:Core:FrameGrid:NFFTTooSmall');
        end

        function testStartsTimesIntervalsAgreeWithFrames(testCase)
            % frame t's samples (from the padded signal) must be x(sampleInterval(t)) inside [1,S]
            x = (1:1001)';                                       % sample index as value: locates samples exactly
            for center = [false true]
                for L = [200 7]
                    g = OceanNLP.Speech.Core.FrameGrid(8000, 'WindowLength', L, 'HopLength', 3, 'Center', center);
                    [fr, s0] = g.frame(x);
                    T = g.numFrames(numel(x));
                    testCase.verifySize(fr, [L T]);
                    testCase.verifyEqual(g.frameStarts(numel(x)), s0);
                    iv = g.sampleInterval((1:T)', numel(x));
                    for t = [1, ceil(T / 2), T]
                        idx = iv(t, 1):iv(t, 2);
                        inside = idx >= 1 & idx <= numel(x);
                        testCase.verifyEqual(fr(inside, t), x(idx(inside)));
                        testCase.verifyEqual(fr(~inside, t), zeros(nnz(~inside), 1));   % constant padding
                    end
                    % centre time = mean of the interval's sample times (0-based / fs)
                    testCase.verifyEqual(g.frameTimes(numel(x)), (mean(iv, 2)' - 1) / 8000, 'AbsTol', 1e-15);
                    ivc = g.sampleInterval((1:T)', numel(x), 'Clip', true);
                    testCase.verifyGreaterThanOrEqual(ivc(:, 1), 1);
                    testCase.verifyLessThanOrEqual(ivc(:, 2), numel(x));
                end
            end
            g = OceanNLP.Speech.Core.FrameGrid(8000);
            testCase.verifyError(@() g.sampleInterval(100, 1000), 'OceanNLP:Speech:Core:FrameGrid:FrameIndex');
        end

        function testBinFrequenciesOddAndEvenNFFT(testCase)
            for n = [255 256]
                g = OceanNLP.Speech.Core.FrameGrid(8000, 'NFFT', n);
                f = g.binFrequencies();
                testCase.verifySize(f, [floor(n / 2) + 1, 1]);
                testCase.verifyEqual(f(2) - f(1), 8000 / n, 'AbsTol', 1e-12);
                testCase.verifyEqual(f(end), floor(n / 2) * 8000 / n, 'AbsTol', 1e-9);
            end
        end

        function testConfigRoundTrip(testCase)
            g = OceanNLP.Speech.Core.FrameGrid(16000, 'WindowLength', 400, 'HopLength', 160, 'Window', "hamming", ...
                'Center', true, 'PadMode', "reflect");
            c = g.config();
            h = OceanNLP.Speech.Core.FrameGrid.fromOptions(c.SampleRate, rmfield(c, ["Type", "SampleRate"]));
            testCase.verifyEqual(h.config(), c);
        end
    end
end
