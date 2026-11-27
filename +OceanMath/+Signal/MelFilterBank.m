classdef MelFilterBank
    % MELFILTERBANK  Triangular mel-spaced filter bank and Hz <-> mel conversions.
    %
    %   [W, centersHz] = OceanMath.Signal.MelFilterBank.build(M, nfft, fs)
    %   [W, centersHz] = OceanMath.Signal.MelFilterBank.build(M, nfft, fs, 'FMin', 0, 'FMax', fs/2, ...
    %                       'Scale', "htk", 'Norm', "none")
    %   m = OceanMath.Signal.MelFilterBank.hz2mel(f, "htk");   f = ...mel2hz(m, "htk")
    %
    %   MEL SCALES:
    %     "htk"     m = 2595 log10(1 + f / 700)                     (O'Shaughnessy 1987; HTK book)
    %     "slaney"  m = f / (200/3)                     for f <  1000 Hz
    %               m = 15 + ln(f / 1000) / (ln(6.4) / 27)  for f >= 1000 Hz
    %               (Slaney 1998 Auditory Toolbox: linear below 1 kHz, log above)
    %
    %   FILTERS (Davis & Mermelstein 1980 triangular bank; identical construction
    %   to librosa.filters.mel):
    %     mel points  mu_i = mel(FMin) + i (mel(FMax) - mel(FMin)) / (M + 1),  i = 0..M+1
    %     edges       g_i  = mel2hz(mu_i)    (Hz; g_{i+1} is the centre of filter i)
    %     bins        f_k  = k fs / nfft,    k = 0..floor(nfft/2)
    %     W(i, k) = max(0, min( (f_k - g_{i-1}) / (g_i - g_{i-1}),
    %                           (g_{i+1} - f_k) / (g_{i+1} - g_i) ))   for filter i = 1..M
    %   Norm "slaney" multiplies row i by 2 / (g_{i+1} - g_{i-1}) (constant area);
    %   "none" keeps unit peak height (peak reached only when a bin sits on g_i).
    %
    %   W is M x (floor(nfft/2)+1), nonnegative; centersHz (M x 1) strictly
    %   increasing. A filter with no bin inside its support (too many mels for
    %   nfft) produces an all-zero row and warning
    %   'OceanMath:Signal:MelFilterBank:EmptyFilters' (librosa warns likewise).

    methods (Static)
        function [W, centersHz] = build(numFilters, nfft, fs, opts)
            arguments
                numFilters (1,1) double {mustBeInteger, mustBePositive}
                nfft (1,1) double {mustBeInteger, mustBePositive}
                fs (1,1) double {mustBePositive, mustBeFinite}
                opts.FMin (1,1) double {mustBeNonnegative, mustBeFinite} = 0
                opts.FMax (1,1) double {mustBePositive, mustBeFinite} = fs / 2
                opts.Scale (1,1) string {mustBeMember(opts.Scale, ["htk", "slaney"])} = "htk"
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["none", "slaney"])} = "none"
            end
            if opts.FMax <= opts.FMin || opts.FMax > fs / 2
                error('OceanMath:Signal:MelFilterBank:InvalidRange', ...
                    'Need 0 <= FMin < FMax <= fs/2 (got FMin=%g, FMax=%g, fs/2=%g).', opts.FMin, opts.FMax, fs / 2);
            end
            mf = OceanMath.Signal.MelFilterBank;
            melPts = linspace(mf.hz2mel(opts.FMin, opts.Scale), mf.hz2mel(opts.FMax, opts.Scale), numFilters + 2);
            g = mf.mel2hz(melPts, opts.Scale);              % 1 x (M+2) edges in Hz
            fk = (0:floor(nfft / 2)) * fs / nfft;           % 1 x K bin frequencies
            gd = diff(g);                                   % 1 x (M+1)
            ramps = g(:) - fk;                              % (M+2) x K : g_i - f_k
            lower = -ramps(1:numFilters, :) ./ gd(1:numFilters)';
            upper = ramps(3:numFilters + 2, :) ./ gd(2:numFilters + 1)';
            W = max(0, min(lower, upper));
            if opts.Norm == "slaney"
                W = W .* (2 ./ (g(3:numFilters + 2) - g(1:numFilters)))';
            end
            centersHz = g(2:numFilters + 1)';
            if any(all(W == 0, 2))
                warning('OceanMath:Signal:MelFilterBank:EmptyFilters', ...
                    '%d of %d mel filters contain no FFT bin (increase nfft or reduce the number of filters).', ...
                    nnz(all(W == 0, 2)), numFilters);
            end
        end

        function m = hz2mel(f, scale)
            % HZ2MEL  Frequency (Hz, any shape) to mel on the chosen scale.
            arguments
                f double {mustBeReal}
                scale (1,1) string {mustBeMember(scale, ["htk", "slaney"])} = "htk"
            end
            if scale == "htk"
                m = 2595 * log10(1 + f / 700);
            else
                fsp = 200 / 3;
                minLogHz = 1000;
                minLogMel = minLogHz / fsp;
                logStep = log(6.4) / 27;
                m = f / fsp;
                hi = f >= minLogHz;
                m(hi) = minLogMel + log(f(hi) / minLogHz) / logStep;
            end
        end

        function f = mel2hz(m, scale)
            % MEL2HZ  Inverse of hz2mel on the same scale.
            arguments
                m double {mustBeReal}
                scale (1,1) string {mustBeMember(scale, ["htk", "slaney"])} = "htk"
            end
            if scale == "htk"
                f = 700 * (10 .^ (m / 2595) - 1);
            else
                fsp = 200 / 3;
                minLogHz = 1000;
                minLogMel = minLogHz / fsp;
                logStep = log(6.4) / 27;
                f = fsp * m;
                hi = m >= minLogMel;
                f(hi) = minLogHz * exp(logStep * (m(hi) - minLogMel));
            end
        end
    end
end
