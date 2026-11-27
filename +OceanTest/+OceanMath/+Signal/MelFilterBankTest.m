classdef MelFilterBankTest < OceanTest.BaseTestCase
    % MELFILTERBANKTEST  OceanMath.Signal.MelFilterBank vs librosa.filters.mel / hz_to_mel.

    methods (Test)
        function testMatchesLibrosaFilterBanks(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            cfgs = fx.meta.mel_cfgs;
            for i = 1:numel(cfgs)
                c = cfgs(i);
                scale = "slaney";
                if c.htk
                    scale = "htk";
                end
                [W, centres] = OceanMath.Signal.MelFilterBank.build(c.n_mels, c.n_fft, c.sr, ...
                    'FMin', c.fmin, 'FMax', c.fmax, 'Scale', scale, 'Norm', string(c.norm));
                ref = fx.raw(c.name);
                testCase.verifySize(W, [c.n_mels, floor(c.n_fft / 2) + 1], c.name);
                testCase.verifyEqual(W, ref, 'AbsTol', 1e-12 * max(ref(:)), c.name);
                testCase.verifyEqual(centres, fx.col([c.name '_centres']), 'RelTol', 1e-12, c.name);
                testCase.verifyGreaterThanOrEqual(W, 0, c.name);
                testCase.verifyTrue(all(diff(centres) > 0), c.name);
            end
        end

        function testHzMelConversionsMatchLibrosaAndInvert(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            f = fx.col('hz_grid');
            for scale = ["htk", "slaney"]
                m = OceanMath.Signal.MelFilterBank.hz2mel(f, scale);
                testCase.verifyEqual(m, fx.col("hz2mel_" + scale), 'AbsTol', 1e-10, scale);
                testCase.verifyEqual(OceanMath.Signal.MelFilterBank.mel2hz(m, scale), f, 'AbsTol', 1e-8, scale);
            end
        end

        function testInvalidRangeAndEmptyFilterWarning(testCase)
            testCase.verifyError(@() OceanMath.Signal.MelFilterBank.build(40, 256, 8000, 'FMax', 5000), ...
                'OceanMath:Signal:MelFilterBank:InvalidRange');
            testCase.verifyError(@() OceanMath.Signal.MelFilterBank.build(40, 256, 8000, 'FMin', 3000, 'FMax', 2000), ...
                'OceanMath:Signal:MelFilterBank:InvalidRange');
            % 128 filters on a 64-point FFT cannot all contain a bin
            testCase.verifyWarning(@() OceanMath.Signal.MelFilterBank.build(128, 64, 8000), ...
                'OceanMath:Signal:MelFilterBank:EmptyFilters');
        end
    end
end
