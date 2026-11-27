classdef CepstralTransformTest < OceanTest.BaseTestCase
    % CEPSTRALTRANSFORMTEST  DCT-II vs scipy.fft.dct, lifter vs librosa.feature.mfcc(lifter=22).

    methods (Test)
        function testDctMatchesScipy(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            X = fx.raw('dct_input');
            testCase.verifyEqual(OceanMath.Signal.CepstralTransform.dct(X, 13), fx.raw('dct_ortho_13'), 'AbsTol', 1e-12);
            testCase.verifyEqual(OceanMath.Signal.CepstralTransform.dct(X, 13, 'Norm', "none"), fx.raw('dct_none_13'), 'AbsTol', 1e-12);
            testCase.verifyEqual(OceanMath.Signal.CepstralTransform.dct(X, 40), fx.raw('dct_ortho_40'), 'AbsTol', 1e-12);
        end

        function testOrthoBasisIsOrthonormal(testCase)
            D = OceanMath.Signal.CepstralTransform.dctMatrix(40, 40, "ortho");
            testCase.verifyEqual(D * D', eye(40), 'AbsTol', 1e-12);
        end

        function testLifterMatchesLibrosa(testCase)
            fx = OceanTest.OceanMath.Signal.FrameSignalTest.fixtures();
            C = OceanMath.Signal.CepstralTransform.lifter(OceanMath.Signal.CepstralTransform.dct(fx.raw('dct_input'), 13), 22);
            testCase.verifyEqual(C, fx.raw('mfcc_lifter22'), 'AbsTol', 1e-12);
            C0 = OceanMath.Signal.CepstralTransform.dct(fx.raw('dct_input'), 13);
            testCase.verifyEqual(OceanMath.Signal.CepstralTransform.lifter(C0, 0), C0);
        end

        function testTooManyCoefficients(testCase)
            testCase.verifyError(@() OceanMath.Signal.CepstralTransform.dct(ones(5, 2), 6), ...
                'OceanMath:Signal:CepstralTransform:TooManyCoeffs');
        end
    end
end
