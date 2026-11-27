classdef ResampleSignal
    % RESAMPLESIGNAL  Rational-factor polyphase resampler with a Kaiser-windowed sinc low-pass.
    %
    %   y = OceanMath.Signal.ResampleSignal.resample(x, fsIn, fsOut)
    %   y = OceanMath.Signal.ResampleSignal.resample(x, fsIn, fsOut, 'HalfWidth', 16, 'KaiserBeta', 8.6)
    %   [h, p, q] = OceanMath.Signal.ResampleSignal.design(fsIn, fsOut, ...)   % the FIR actually used
    %
    %   THEORY (Oppenheim & Schafer 3rd ed. sec. 4.6.3, change of rate by a
    %   non-integer factor; Crochiere & Rabiner 1983 polyphase form):
    %
    %     p / q = fsOut / fsIn in lowest terms (gcd for integer rates, rat otherwise)
    %     1. upsample by p (insert p-1 zeros)          -> rate p fsIn
    %     2. low-pass h at min(fsIn, fsOut)/2          (cutoff pi / r, r = max(p, q))
    %     3. keep every q-th sample                    -> rate fsOut
    %
    %   FILTER (odd length 2K + 1, K = HalfWidth * r, m = -K..K):
    %     g[m] = sinc(m / r) * I0(beta sqrt(1 - (m/K)^2)) / I0(beta),  sinc(u) = sin(pi u)/(pi u)
    %     h[m] = p * g[m] / sum_m g[m]           (unit DC gain per phase -> scaled by p)
    %   HalfWidth = number of sinc zero crossings kept on each side at the low rate.
    %   This is exactly scipy.signal.firwin(2K+1, 1/r, window=('kaiser', beta)) * p.
    %
    %   POLYPHASE OUTPUT (zero-phase, group delay K compensated):
    %     y[m] = sum_n x[n] h[m q - n p],   m = 0 .. ceil(S p / q) - 1,  |m q - n p| <= K
    %   Only the taps that meet a non-zero (original) sample are evaluated, i.e.
    %   the polyphase decomposition; samples outside [0, S-1] are zero
    %   (scipy.signal.resample_poly padtype='constant' convention).
    %
    %   WHY NOT PolyphaseEngine: that class implements integer-M DECIMATION
    %   (filter-then-downsample with a user FIR, output length floor-based, no
    %   interpolation by p, no delay compensation). Rational p/q resampling needs
    %   the interpolating phase selection (m q - n p) mod p, so it is implemented
    %   here directly; PolyphaseEngine is left unchanged.
    %
    %   Input x is S x C (columns = channels); each column is resampled
    %   independently. Equal rates return x unchanged. Empty x returns 0 x C.

    methods (Static)
        function y = resample(x, fsIn, fsOut, opts)
            arguments
                x (:,:) double {mustBeReal}
                fsIn (1,1) double {mustBePositive, mustBeFinite}
                fsOut (1,1) double {mustBePositive, mustBeFinite}
                opts.HalfWidth (1,1) double {mustBeInteger, mustBePositive} = 16
                opts.KaiserBeta (1,1) double {mustBeNonnegative, mustBeFinite} = 8.6
            end
            [h, p, q] = OceanMath.Signal.ResampleSignal.design(fsIn, fsOut, ...
                'HalfWidth', opts.HalfWidth, 'KaiserBeta', opts.KaiserBeta);
            [S, C] = size(x);
            if p == q
                y = x;
                return;
            end
            M = ceil(S * p / q);
            y = zeros(M, C);
            if S == 0
                return;
            end
            K = (numel(h) - 1) / 2;
            J = floor(2 * K / p) + 1;                 % max taps meeting real samples
            chunk = max(1, floor(4e6 / J));
            for m0 = 0:chunk:M-1
                m = (m0:min(M-1, m0 + chunk - 1))';
                u = m * q;                            % position on the upsampled grid
                n = ceil((u - K) / p) + (0:J-1);      % candidate input samples (0-based)
                d = u - n * p;                        % filter offset, h index d + K + 1
                valid = abs(d) <= K & n >= 0 & n <= S - 1;
                hv = zeros(size(d));
                hv(valid) = h(d(valid) + K + 1);
                nIdx = min(max(n, 0), S - 1) + 1;
                for c = 1:C
                    xc = x(:, c);
                    y(m + 1, c) = sum(hv .* reshape(xc(nIdx), size(nIdx)), 2);
                end
            end
        end

        function [h, p, q] = design(fsIn, fsOut, opts)
            % DESIGN  The (2K+1)-tap interpolation filter (column) and the reduced ratio p/q.
            arguments
                fsIn (1,1) double {mustBePositive, mustBeFinite}
                fsOut (1,1) double {mustBePositive, mustBeFinite}
                opts.HalfWidth (1,1) double {mustBeInteger, mustBePositive} = 16
                opts.KaiserBeta (1,1) double {mustBeNonnegative, mustBeFinite} = 8.6
            end
            [p, q] = OceanMath.Signal.ResampleSignal.ratio(fsIn, fsOut);
            r = max(p, q);
            K = opts.HalfWidth * r;
            m = (-K:K)';
            u = m / r;
            g = ones(size(u));
            nz = u ~= 0;
            g(nz) = sin(pi * u(nz)) ./ (pi * u(nz));
            w = besseli(0, opts.KaiserBeta * sqrt(max(0, 1 - (m / K) .^ 2))) / besseli(0, opts.KaiserBeta);
            g = g .* w;
            h = p * g / sum(g);
        end

        function [p, q] = ratio(fsIn, fsOut)
            % RATIO  fsOut/fsIn = p/q in lowest terms; errors when not representable.
            arguments
                fsIn (1,1) double {mustBePositive, mustBeFinite}
                fsOut (1,1) double {mustBePositive, mustBeFinite}
            end
            if fsIn == round(fsIn) && fsOut == round(fsOut)
                g = gcd(fsIn, fsOut);
                p = fsOut / g;
                q = fsIn / g;
            else
                [p, q] = rat(fsOut / fsIn, 1e-12 * fsOut / fsIn);
                if abs(p / q - fsOut / fsIn) > 1e-12 * fsOut / fsIn
                    error('OceanMath:Signal:ResampleSignal:Irrational', ...
                        'Rate ratio %.17g/%.17g has no exact small rational form.', fsOut, fsIn);
                end
            end
        end
    end
end
