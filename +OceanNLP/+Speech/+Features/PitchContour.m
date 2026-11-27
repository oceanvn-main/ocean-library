classdef PitchContour < OceanNLP.Speech.Core.IFeatureTransform
    % PITCHCONTOUR  Fundamental frequency (F0) and voicing strength per frame (YIN estimator).
    %
    %   t  = OceanNLP.Speech.Features.PitchContour(fs)
    %   t  = OceanNLP.Speech.Features.PitchContour(fs, <FrameGrid options>, 'MinF0', 50, ...
    %          'MaxF0', 400, 'VoicingThreshold', 0.25)
    %   fs = t.forward(waveform)     % 2 x T: row 1 F0 in Hz (0 = unvoiced), row 2 voicing in [0,1]
    %
    %   ESTIMATOR: YIN (de Cheveigne & Kawahara 2002, J. Acoust. Soc. Am. 111(4)),
    %   steps 2-5, on the same frame grid (same T and centre times) as LogMel:
    %     lags          tau_min = max(2, floor(fs / MaxF0)),  tau_max = ceil(fs / MinF0)
    %     segment       W + tau_max samples centred on frame t's centre, W = WindowLength:
    %                   z[j] = x[a_t + j], a_t = s_t - floor(tau_max/2), except that a_t is
    %                   slid into [0, S - W - tau_max] near the signal ends (so edge frames
    %                   analyse real samples, off-centre by < tau_max/2, instead of zeros);
    %                   zero beyond the signal only when S < W + tau_max
    %     difference    d(tau) = sum_{j=0}^{W-1} (z[j] - z[j + tau])^2
    %                          = e_0 + e_tau - 2 r(tau)   (FFT cross-correlation, sliding energies)
    %     CMNDF         d'(0) = 1,  d'(tau) = d(tau) / ((1/tau) sum_{j=1}^{tau} d(j))
    %                   (d' := 1 where the running sum is 0, e.g. digital silence)
    %     threshold     tau* = first tau in [tau_min, tau_max] with d'(tau) < VoicingThreshold,
    %                   then followed down to its local minimum; if none, tau* = argmin d'
    %                   and the frame is unvoiced
    %     refinement    parabolic interpolation of d' at tau*: tau^ = tau* + delta,
    %                   delta = (d'(tau*-1) - d'(tau*+1)) / (2 (d'(tau*-1) - 2 d'(tau*) + d'(tau*+1)))
    %     output        F0 = fs / tau^ (clamped to [MinF0, MaxF0]) when voiced, else 0;
    %                   voicing strength = min(1, max(0, 1 - d'(tau*)))
    %   VoicingThreshold default 0.25 (the YIN paper uses 0.1): chosen on the FSDD
    %   train speakers as the best voicing F1 vs librosa.pyin among thresholds whose
    %   gross pitch error stays <= 10 %; at 0.1 only ~50 % of pyin-voiced frames were
    %   voiced. Evidence: Experiments/Scripts/Speech/validate_pitch_threshold.m ->
    %   Results/Speech/pitch_threshold_validation.md.
    %   Not implemented: YIN step 6 (best local estimate) and any smoothing /
    %   Viterbi tracking across frames (pYIN) -- each frame is independent.
    %   Same input policy as LogMel. Grid PadMode does not apply to the lag
    %   extension (see 'segment').

    properties (SetAccess = immutable)
        SampleRate (1,1) double
        Grid OceanNLP.Speech.Core.FrameGrid
        MinF0 (1,1) double
        MaxF0 (1,1) double
        VoicingThreshold (1,1) double
    end

    methods
        function obj = PitchContour(sampleRate, opts)
            arguments
                sampleRate (1,1) double {mustBePositive, mustBeFinite}
                opts.WindowLength double {mustBeScalarOrEmpty} = []
                opts.HopLength double {mustBeScalarOrEmpty} = []
                opts.NFFT double {mustBeScalarOrEmpty} = []
                opts.Window (1,1) string = "hann"
                opts.Periodic (1,1) logical = true
                opts.Center (1,1) logical = false
                opts.PadMode (1,1) string = "constant"
                opts.MinF0 (1,1) double {mustBePositive, mustBeFinite} = 50
                opts.MaxF0 (1,1) double {mustBePositive, mustBeFinite} = 400
                opts.VoicingThreshold (1,1) double {mustBePositive, mustBeLessThanOrEqual(opts.VoicingThreshold, 1)} = 0.25
            end
            if opts.MinF0 >= opts.MaxF0 || opts.MaxF0 > sampleRate / 4
                error('OceanNLP:Speech:Features:PitchContour:InvalidRange', ...
                    'Need MinF0 < MaxF0 <= fs/4 (lag >= 4 samples); got [%g, %g] at %g Hz.', ...
                    opts.MinF0, opts.MaxF0, sampleRate);
            end
            obj.SampleRate = sampleRate;
            obj.Grid = OceanNLP.Speech.Core.FrameGrid.fromOptions(sampleRate, ...
                rmfield(opts, ["MinF0", "MaxF0", "VoicingThreshold"]));
            obj.MinF0 = opts.MinF0;
            obj.MaxF0 = opts.MaxF0;
            obj.VoicingThreshold = opts.VoicingThreshold;
        end

        function fs = forward(obj, input)
            arguments
                obj
                input
            end
            [x, w] = prepareWaveform(input, obj.SampleRate, "PitchContour");
            S = numel(x);
            starts = obj.Grid.frameStarts(S);
            [f0, strength] = obj.estimate(x, starts);
            cfg = obj.config();
            cfg.InputHistory = w.History;
            fs = OceanNLP.Speech.Core.FeatureSequence([f0; strength], ...
                'FrameTimes', obj.Grid.frameTimes(S), 'AxisNames', ["pitch", "time"], ...
                'Units', "row 1: F0 Hz (0 = unvoiced); row 2: voicing strength [0,1]", ...
                'Config', cfg, 'Source', w.Source);
        end

        function [lagMin, lagMax] = lagRange(obj)
            % LAGRANGE  Integer lag search range [tau_min, tau_max] in samples.
            lagMin = max(2, floor(obj.SampleRate / obj.MaxF0));
            lagMax = ceil(obj.SampleRate / obj.MinF0);
        end

        function cfg = config(obj)
            [lo, hi] = obj.lagRange();
            cfg = struct('Type', "PitchContour", 'Method', "YIN (CMNDF + absolute threshold + parabolic)", ...
                'SampleRate', obj.SampleRate, 'Grid', obj.Grid.config(), 'MinF0', obj.MinF0, ...
                'MaxF0', obj.MaxF0, 'VoicingThreshold', obj.VoicingThreshold, 'LagRange', [lo, hi]);
        end

        function d = outputDim(obj, inputDim)
            arguments
                obj %#ok<INUSA>
                inputDim double = [] %#ok<INUSA> fixed output size
            end
            d = 2;
        end
    end

    methods (Access = private)
        function [f0, strength] = estimate(obj, x, starts)
            T = numel(starts);
            f0 = zeros(1, T);
            strength = zeros(1, T);
            if T == 0
                return;
            end
            [lagMin, lagMax] = obj.lagRange();
            W = obj.Grid.WindowLength;
            n = W + lagMax;
            S = numel(x);
            % segment matrix (n x T), zero outside the signal
            a0 = starts - floor(lagMax / 2);                 % 0-based segment starts (centred)
            a0 = min(max(a0, 0), max(S - n, 0));             % slid inside the signal near its ends
            idx = (0:n-1)' + a0;
            inside = idx >= 0 & idx <= S - 1;
            Z = zeros(n, T);
            Z(inside) = x(idx(inside) + 1);
            % difference function via cross-correlation and sliding energies
            A = Z(1:W, :);
            nf = 2 ^ nextpow2(W + n);
            R = real(ifft(conj(fft(A, nf)) .* fft(Z, nf)));
            r = R(1:lagMax + 1, :);                          % r(tau+1) = sum_j z[j] z[j+tau]
            cs = [zeros(1, T); cumsum(Z .^ 2, 1)];
            tau = (0:lagMax)';
            e0 = cs(W + 1, :);
            et = cs(tau + W + 1, :) - cs(tau + 1, :);
            d = max(e0 + et - 2 * r, 0);
            % cumulative mean normalised difference
            run = cumsum(d(2:end, :), 1);                    % sum_{j=1}^{tau} d(j)
            dn = ones(lagMax + 1, T);
            ok = run > 0;
            tt = repmat(tau(2:end), 1, T);
            tmp = ones(lagMax, T);
            tmp(ok) = d([false(1, T); ok]) .* tt(ok) ./ run(ok);
            dn(2:end, :) = tmp;
            thr = obj.VoicingThreshold;
            for t = 1:T
                c = dn(:, t);                                % c(tau+1)
                if e0(t) == 0 && all(et(:, t) == 0)
                    continue;                                % digital silence: unvoiced, strength 0
                end
                band = c(lagMin + 1:lagMax + 1);
                k = find(band < thr, 1, 'first');
                voiced = ~isempty(k);
                if voiced
                    L = lagMin + k - 1;
                    while L < lagMax && c(L + 2) < c(L + 1)
                        L = L + 1;
                    end
                else
                    [~, k] = min(band);
                    L = lagMin + k - 1;
                end
                strength(t) = min(1, max(0, 1 - c(L + 1)));
                if ~voiced
                    continue;
                end
                delta = 0;
                if L > 1 && L < lagMax
                    den = c(L) - 2 * c(L + 1) + c(L + 2);
                    if den > 0
                        delta = (c(L) - c(L + 2)) / (2 * den);
                        delta = max(-1, min(1, delta));
                    end
                end
                f0(t) = min(obj.MaxF0, max(obj.MinF0, obj.SampleRate / (L + delta)));
            end
        end
    end

    methods (Static)
        function f = fromConfig(cfg)
            % FROMCONFIG  Inverse of config().
            arguments
                cfg (1,1) struct
            end
            g = gridArgs(cfg.Grid);
            f = OceanNLP.Speech.Features.PitchContour(cfg.SampleRate, g{:}, ...
                'MinF0', cfg.MinF0, 'MaxF0', cfg.MaxF0, 'VoicingThreshold', cfg.VoicingThreshold);
        end
    end
end
