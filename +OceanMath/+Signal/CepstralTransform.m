classdef CepstralTransform
    % CEPSTRALTRANSFORM  DCT-II along dimension 1 and sinusoidal cepstral liftering.
    %
    %   C = OceanMath.Signal.CepstralTransform.dct(X, numCoeffs)                 % 'Norm', "ortho"
    %   C = OceanMath.Signal.CepstralTransform.dct(X, numCoeffs, 'Norm', "none")
    %   C = OceanMath.Signal.CepstralTransform.lifter(C, Lift)
    %   D = OceanMath.Signal.CepstralTransform.dctMatrix(numCoeffs, N, norm)
    %
    %   DCT-II (Ahmed, Natarajan & Rao 1974), X is N x T, k = 0..numCoeffs-1:
    %     "none"   C(k, t) = 2 sum_{n=0}^{N-1} X(n, t) cos(pi k (2n + 1) / (2N))
    %     "ortho"  C(k, t) = s_k sqrt(2/N) sum_{n=0}^{N-1} X(n, t) cos(pi k (2n + 1) / (2N)),
    %              s_0 = 1/sqrt(2), s_k = 1 otherwise (orthonormal basis)
    %   These are scipy.fft.dct(type=2, norm=None | 'ortho', axis=0)[:numCoeffs].
    %   For MFCCs X is the log mel spectrum (Davis & Mermelstein 1980).
    %
    %   LIFTER (sinusoidal, Juang, Rabiner & Wilpon 1987; HTK "CEPLIFTER"):
    %     C'(k, t) = (1 + (Lift/2) sin(pi k / Lift)) C(k, t),   k = 1..K  (1-based row index)
    %   The row index starts at 1 for the FIRST stored coefficient (the c0 row),
    %   which is librosa.feature.mfcc's convention (HTK applies the same weight
    %   to c1..cK with c0 kept separately). Lift = 0 is the identity.

    methods (Static)
        function C = dct(X, numCoeffs, opts)
            arguments
                X (:,:) double {mustBeReal}
                numCoeffs (1,1) double {mustBeInteger, mustBePositive}
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["ortho", "none"])} = "ortho"
            end
            N = size(X, 1);
            if numCoeffs > N
                error('OceanMath:Signal:CepstralTransform:TooManyCoeffs', ...
                    'Requested %d coefficients from %d input rows.', numCoeffs, N);
            end
            D = OceanMath.Signal.CepstralTransform.dctMatrix(numCoeffs, N, opts.Norm);
            C = D * X;
        end

        function D = dctMatrix(numCoeffs, N, norm)
            % DCTMATRIX  numCoeffs x N DCT-II basis (see class doc for the scaling).
            arguments
                numCoeffs (1,1) double {mustBeInteger, mustBePositive}
                N (1,1) double {mustBeInteger, mustBePositive}
                norm (1,1) string {mustBeMember(norm, ["ortho", "none"])} = "ortho"
            end
            k = (0:numCoeffs-1)';
            n = 0:N-1;
            D = cos(pi * k .* (2 * n + 1) / (2 * N));
            if norm == "ortho"
                D = sqrt(2 / N) * D;
                D(1, :) = D(1, :) / sqrt(2);
            else
                D = 2 * D;
            end
        end

        function C = lifter(C, Lift)
            arguments
                C (:,:) double {mustBeReal}
                Lift (1,1) double {mustBeNonnegative, mustBeFinite}
            end
            if Lift == 0
                return;
            end
            k = (1:size(C, 1))';
            C = C .* (1 + (Lift / 2) * sin(pi * k / Lift));
        end
    end
end
