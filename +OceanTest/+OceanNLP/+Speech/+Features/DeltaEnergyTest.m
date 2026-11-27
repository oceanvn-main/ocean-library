classdef DeltaEnergyTest < OceanTest.BaseTestCase
    % DELTAENERGYTEST  DeltaFeatures vs librosa.feature.delta(mode='nearest'); EnergyContour vs numpy.

    methods (Test)
        function testDeltasMatchLibrosaNearest(testCase)
            fx = OceanTest.OceanNLP.Speech.Features.LogMelMFCCTest.fixtures();
            c = fx.meta.cases(1);
            mfcc = testCase.caseAMfcc(fx);
            for N = 1:3
                out = OceanNLP.Speech.Features.DeltaFeatures('Width', N, 'Order', 2).forward(mfcc);
                F = mfcc.numBins();
                testCase.verifySize(out.Values, [3 * F, mfcc.numFrames()]);
                testCase.verifyEqual(out.Values(1:F, :), mfcc.Values);
                testCase.verifyEqual(out.Values(F+1:2*F, :), fx.raw(sprintf('A_delta_N%d', N)), 'AbsTol', 1e-7, sprintf('N=%d', N));
                testCase.verifyEqual(out.Values(2*F+1:end, :), fx.raw(sprintf('A_ddelta_N%d', N)), 'AbsTol', 1e-7, sprintf('N=%d', N));
                testCase.verifyEqual(out.FrameTimes, mfcc.FrameTimes);
            end
            testCase.verifyEqual(c.name, 'A');
        end

        function testDeltaClosedFormOnLinearRamp(testCase)
            % interior delta of c_t = a t is exactly a (sum n*2n / (2 sum n^2) = 1)
            f = OceanNLP.Speech.Core.FeatureSequence(3 * (1:20), 'FrameTimes', (0:19) / 100);
            out = OceanNLP.Speech.Features.DeltaFeatures('Width', 2, 'Order', 1).forward(f);
            testCase.verifyEqual(out.Values(2, 3:18), 3 * ones(1, 16), 'AbsTol', 1e-12);
        end

        function testDeltaEdgeCasesAndContract(testCase)
            d = OceanNLP.Speech.Features.DeltaFeatures();
            testCase.verifyEqual(d.outputDim(13), 39);
            empty = OceanNLP.Speech.Core.FeatureSequence(zeros(13, 0), 'FrameTimes', zeros(1, 0));
            testCase.verifySize(d.forward(empty).Values, [39 0]);
            one = OceanNLP.Speech.Core.FeatureSequence((1:13)', 'FrameTimes', 0);
            testCase.verifyEqual(d.forward(one).Values(14:end), zeros(26, 1));
            id = OceanNLP.Speech.Features.DeltaFeatures('Order', 0).forward(one);
            testCase.verifyEqual(id.Values, one.Values);
            testCase.verifyError(@() d.forward(OceanNLP.Speech.Core.Waveform(randn(100, 1), 8000)), ...
                'OceanNLP:Speech:Features:DeltaFeatures:InputType');
            fm = OceanNLP.Speech.Core.FeatureSequence(ones(2, 3), 'FrameTimes', 1:3, 'FrequencyHz', [100; 200]);
            testCase.verifyEqual(d.forward(fm).FrequencyHz, [100; 200; 100; 200; 100; 200]);
        end

        function testEnergyMatchesNumpy(testCase)
            fx = OceanTest.OceanNLP.Speech.Features.LogMelMFCCTest.fixtures();
            w = OceanNLP.Speech.Core.Waveform(fx.col('A_input'), 8000);
            e = OceanNLP.Speech.Features.EnergyContour(8000, 'WindowLength', 200, 'HopLength', 80).forward(w);
            testCase.verifyEqual(e.Values, reshape(fx.raw('A_energy'), 1, []), 'AbsTol', 1e-10);
            testCase.verifyEqual(e.AxisNames, ["energy", "time"]);
            g = OceanNLP.Speech.Core.FrameGrid(8000, 'WindowLength', 200, 'HopLength', 80);
            testCase.verifyEqual(e.FrameTimes, g.frameTimes(w.numSamples()));
        end

        function testEnergyEdgeCases(testCase)
            et = OceanNLP.Speech.Features.EnergyContour(8000);
            s = et.forward(OceanNLP.Speech.Core.Waveform(zeros(800, 1), 8000));
            testCase.verifyEqual(s.Values, log(1e-10) * ones(1, s.numFrames()));
            testCase.verifySize(et.forward(OceanNLP.Speech.Core.Waveform(ones(10, 1), 8000)).Values, [1 0]);
            testCase.verifyError(@() et.forward(OceanNLP.Speech.Core.Waveform(ones(800, 1), 16000)), ...
                'OceanNLP:Speech:Features:EnergyContour:SampleRateMismatch');
            % windowed energy = sum (w x)^2, closed form for a constant signal
            ew = OceanNLP.Speech.Features.EnergyContour(8000, 'ApplyWindow', true) ...
                .forward(OceanNLP.Speech.Core.Waveform(ones(800, 1), 8000));
            win = OceanMath.Signal.FrameSignal.window("hann", 200);
            testCase.verifyEqual(ew.Values, log(sum(win .^ 2)) * ones(1, ew.numFrames()), 'AbsTol', 1e-12);
        end
    end

    methods
        function mf = caseAMfcc(~, fx)
            c = fx.meta.cases(1);
            w = OceanNLP.Speech.Core.Waveform(fx.col('A_input'), c.sr);
            mf = OceanNLP.Speech.Features.MFCC(c.sr, 'WindowLength', c.L, 'HopLength', c.H, 'NFFT', c.n_fft, ...
                'NumMels', c.n_mels, 'FMax', c.fmax, 'NumCoeffs', 13).forward(w);
        end
    end
end
