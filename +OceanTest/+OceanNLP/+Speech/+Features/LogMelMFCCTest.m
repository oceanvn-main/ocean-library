classdef LogMelMFCCTest < OceanTest.BaseTestCase
    % LOGMELMFCCTEST  LogMel and MFCC vs librosa on real speech (fixtures/make_feature_fixtures.py)
    %   plus the contract edge cases (silence, stereo, rate mismatch, short input).

    methods (Test)
        function testLogMelAndMFCCMatchLibrosa(testCase)
            fx = OceanTest.OceanNLP.Speech.Features.LogMelMFCCTest.fixtures();
            for c = fx.meta.cases'
                w = OceanNLP.Speech.Core.Waveform(fx.col([c.name '_input']), c.sr);
                scale = "slaney";
                if c.htk
                    scale = "htk";
                end
                args = {'WindowLength', c.L, 'HopLength', c.H, 'NFFT', c.n_fft, 'Window', string(c.window), ...
                    'Center', c.center, 'PadMode', string(c.pad_mode), 'NumMels', c.n_mels, 'FMin', c.fmin, ...
                    'FMax', c.fmax, 'MelScale', scale, 'MelNorm', string(c.norm), 'Power', c.power, ...
                    'LogFloor', fx.meta.log_floor};
                lm = OceanNLP.Speech.Features.LogMel(c.sr, args{:}).forward(w);
                ref = fx.raw([c.name '_logmel']);
                testCase.verifySize(lm.Values, size(ref), c.name);
                testCase.verifyEqual(lm.Values, ref, 'AbsTol', 1e-8, c.name);      % log domain: ~1e-8 relative
                mf = OceanNLP.Speech.Features.MFCC(c.sr, args{:}, 'NumCoeffs', 13).forward(w);
                testCase.verifyEqual(mf.Values, fx.raw([c.name '_mfcc13']), 'AbsTol', 1e-7, c.name);
                mfl = OceanNLP.Speech.Features.MFCC(c.sr, args{:}, 'NumCoeffs', 13, 'Lifter', 22).forward(w);
                testCase.verifyEqual(mfl.Values, fx.raw([c.name '_mfcc13_lift22']), 'AbsTol', 1e-6, c.name);
                % time axis = the FrameGrid's
                g = OceanNLP.Speech.Core.FrameGrid(c.sr, 'WindowLength', c.L, 'HopLength', c.H, 'NFFT', c.n_fft, ...
                    'Center', c.center, 'PadMode', string(c.pad_mode));
                testCase.verifyEqual(lm.FrameTimes, g.frameTimes(w.numSamples()), c.name);
                testCase.verifyEqual(mf.FrameTimes, lm.FrameTimes, c.name);
            end
        end

        function testInputsAreTheRealFiles(testCase)
            fx = OceanTest.OceanNLP.Speech.Features.LogMelMFCCTest.fixtures();
            x = audioread(fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD', 'recordings', '3_theo_10.wav'), 'double');
            testCase.verifyEqual(x, fx.col('A_input'));
        end

        function testAxesUnitsAndConfig(testCase)
            lmT = OceanNLP.Speech.Features.LogMel(8000);
            w = testCase.fsddWave();
            lm = lmT.forward(w);
            testCase.verifyEqual(lm.AxisNames, ["mel", "time"]);
            testCase.verifyEqual(lm.Units, "log power");
            testCase.verifyEqual(lm.FrequencyHz, lmT.CentersHz);
            testCase.verifyEqual(lm.Source, w.Source);
            testCase.verifyEqual(lm.Config.Type, "LogMel");
            testCase.verifyEqual(lmT.outputDim(), 40);
            mf = OceanNLP.Speech.Features.MFCC(8000);
            testCase.verifyEqual(mf.outputDim(), 13);
            testCase.verifyEmpty(mf.forward(w).FrequencyHz);
            % intermediates are the stages actually used by forward
            st = lmT.intermediates(w);
            testCase.verifyEqual(log(max(lmT.Filters * st.PowerSpectrum, 1e-10)), lm.Values);
        end

        function testSilenceIsFiniteAtTheFloor(testCase)
            w = OceanNLP.Speech.Core.Waveform(zeros(8000, 1), 8000);
            lm = OceanNLP.Speech.Features.LogMel(8000).forward(w);
            testCase.verifyEqual(lm.Values, log(1e-10) * ones(size(lm.Values)));
            mf = OceanNLP.Speech.Features.MFCC(8000).forward(w);
            testCase.verifyTrue(all(isfinite(mf.Values), 'all'));
        end

        function testStereoIsMixedAndRecorded(testCase)
            x = testCase.fsddWave().Samples;
            w2 = OceanNLP.Speech.Core.Waveform([x, 0.5 * x], 8000);
            lm = OceanNLP.Speech.Features.LogMel(8000).forward(w2);
            ref = OceanNLP.Speech.Features.LogMel(8000).forward(OceanNLP.Speech.Core.Waveform(0.75 * x, 8000));
            testCase.verifyEqual(lm.Values, ref.Values, 'AbsTol', 1e-12);
            testCase.verifyEqual(lm.Config.InputHistory, "mono mean of 2 channels");
        end

        function testRejectsRateMismatchAndWrongInput(testCase)
            w = testCase.fsddWave();
            testCase.verifyError(@() OceanNLP.Speech.Features.LogMel(16000).forward(w), ...
                'OceanNLP:Speech:Features:LogMel:SampleRateMismatch');
            testCase.verifyError(@() OceanNLP.Speech.Features.MFCC(16000).forward(w), ...
                'OceanNLP:Speech:Features:LogMel:SampleRateMismatch');
            testCase.verifyError(@() OceanNLP.Speech.Features.LogMel(8000).forward(randn(100, 1)), ...
                'OceanNLP:Speech:Features:LogMel:InputType');
            testCase.verifyError(@() OceanNLP.Speech.Features.MFCC(8000, 'NumCoeffs', 41), ...
                'OceanNLP:Speech:Features:MFCC:TooManyCoeffs');
        end

        function testShortSignals(testCase)
            w = OceanNLP.Speech.Core.Waveform(randn(150, 1), 8000);       % shorter than one 200-sample frame
            lm = OceanNLP.Speech.Features.LogMel(8000).forward(w);
            testCase.verifySize(lm.Values, [40 0]);
            testCase.verifySize(lm.FrameTimes, [1 0]);
            lmc = OceanNLP.Speech.Features.LogMel(8000, 'Center', true).forward(w);
            testCase.verifyEqual(lmc.numFrames(), OceanMath.Signal.FrameSignal.numFrames(150, 200, 80, true));
            testCase.verifyError(@() OceanNLP.Speech.Features.LogMel(8000, 'Center', true, 'PadMode', "reflect") ...
                .forward(OceanNLP.Speech.Core.Waveform(randn(90, 1), 8000)), ...
                'OceanMath:Signal:FrameSignal:TooShortForReflect');
        end
    end

    methods
        function w = fsddWave(testCase)
            w = OceanNLP.Speech.Core.Waveform.read("recordings/0_jackson_0.wav", ...
                'Root', fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD'));
        end
    end

    methods (Static)
        function fx = fixtures()
            dir_ = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            raw = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(dir_, [char(n) '.npy'])));
            fx = struct('raw', raw, 'col', @(n) reshape(raw(n), [], 1), ...
                'meta', jsondecode(fileread(fullfile(dir_, 'feature_fixtures.json'))));
        end
    end
end
