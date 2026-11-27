classdef FrameSignal
    % FRAMESIGNAL  Static framing and analysis-window utilities (short-time analysis).
    %
    %   [frames, starts0] = OceanMath.Signal.FrameSignal.frame(x, L, H)
    %   [frames, starts0] = OceanMath.Signal.FrameSignal.frame(x, L, H, 'Center', true, 'PadMode', "reflect")
    %   T = OceanMath.Signal.FrameSignal.numFrames(S, L, H, center)
    %   w = OceanMath.Signal.FrameSignal.window("hann", L, 'Periodic', true)
    %
    %   FRAME GRID (Phase 0 contract sec. 3; Rabiner & Schafer, "Theory and
    %   Applications of Digital Speech Processing", ch. 6 short-time analysis):
    %
    %     P   = Center ? floor(L/2) : 0          (pad on both sides)
    %     T   = (S + 2P >= L) ? 1 + floor((S + 2P - L) / H) : 0
    %     s_t = (t - 1) H - P                    (0-based source sample of frame t)
    %     frame t = xpad[(t-1)H + 1 .. (t-1)H + L]  (1-based into the padded signal)
    %
    %   PAD MODES:
    %     "constant"  zeros outside [0, S-1]
    %     "reflect"   mirror WITHOUT repeating the edge sample (numpy 'reflect'):
    %                 x[-k] = x[k], x[S-1+k] = x[S-1-k], k = 1..P; requires S > P.
    %
    %   WINDOWS (closed form, n = 0..L-1, D = L for periodic, L-1 for symmetric):
    %     hann         w[n] = 0.5 - 0.5 cos(2 pi n / D)
    %     hamming      w[n] = 0.54 - 0.46 cos(2 pi n / D)
    %     rectangular  w[n] = 1
    %     povey        w[n] = (0.5 - 0.5 cos(2 pi n / D))^0.85   (Kaldi; Kaldi itself
    %                  uses the symmetric D = L-1 -> pass 'Periodic', false to match)
    %   A periodic window is the first L points of the symmetric window of length
    %   L+1 (the DFT-even window used by STFT libraries: scipy get_window(fftbins=True)).
    %   L = 1 returns w = 1 for every type.
    %
    %   Pure static functions; no state; no toolbox calls.

    methods (Static)
        function [frames, starts0] = frame(x, L, H, opts)
            % FRAME  Slice a mono column signal into overlapping frames (L x T).
            %   starts0(t) = s_t, the 0-based source sample of the first frame sample.
            arguments
                x (:,1) double {mustBeReal}
                L (1,1) double {mustBeInteger, mustBePositive}
                H (1,1) double {mustBeInteger, mustBePositive}
                opts.Center (1,1) logical = false
                opts.PadMode (1,1) string {mustBeMember(opts.PadMode, ["constant", "reflect"])} = "constant"
            end
            S = numel(x);
            P = 0;
            if opts.Center
                P = floor(L / 2);
            end
            T = OceanMath.Signal.FrameSignal.numFrames(S, L, H, opts.Center);
            starts0 = (0:T-1) * H - P;
            if T == 0
                frames = zeros(L, 0);
                return;
            end
            if P > 0
                if opts.PadMode == "reflect"
                    if S <= P
                        error('OceanMath:Signal:FrameSignal:TooShortForReflect', ...
                            'Reflect padding by %d samples needs more than %d samples (got %d).', P, P, S);
                    end
                    left = x(P+1:-1:2);
                    right = x(S-1:-1:S-P);
                else
                    left = zeros(P, 1);
                    right = zeros(P, 1);
                end
                xp = [left; x; right];
            else
                xp = x;
            end
            idx = (1:L)' + (0:T-1) * H;   % L x T indices into the padded signal
            frames = reshape(xp(idx), L, T);
        end

        function T = numFrames(S, L, H, center)
            % NUMFRAMES  T = 1 + floor((S + 2P - L)/H) when S + 2P >= L, else 0.
            arguments
                S (1,1) double {mustBeInteger, mustBeNonnegative}
                L (1,1) double {mustBeInteger, mustBePositive}
                H (1,1) double {mustBeInteger, mustBePositive}
                center (1,1) logical = false
            end
            P = 0;
            if center
                P = floor(L / 2);
            end
            if S + 2 * P >= L
                T = 1 + floor((S + 2 * P - L) / H);
            else
                T = 0;
            end
        end

        function w = window(type, L, opts)
            % WINDOW  Closed-form analysis window (column, L x 1). See class doc.
            arguments
                type (1,1) string {mustBeMember(type, ["hann", "hamming", "rectangular", "povey"])}
                L (1,1) double {mustBeInteger, mustBePositive}
                opts.Periodic (1,1) logical = true
            end
            if L == 1
                w = 1;
                return;
            end
            n = (0:L-1)';
            if opts.Periodic
                D = L;
            else
                D = L - 1;
            end
            switch type
                case "hann"
                    w = 0.5 - 0.5 * cos(2 * pi * n / D);
                case "hamming"
                    w = 0.54 - 0.46 * cos(2 * pi * n / D);
                case "rectangular"
                    w = ones(L, 1);
                case "povey"
                    w = (0.5 - 0.5 * cos(2 * pi * n / D)) .^ 0.85;
            end
        end
    end
end
