classdef FrameGrid < handle
    % FRAMEGRID  The analysis frame grid shared by every frame-based speech feature.
    %
    %   g = OceanNLP.Speech.Core.FrameGrid(fs)
    %   g = OceanNLP.Speech.Core.FrameGrid(fs, 'WindowLength', L, 'HopLength', H, 'NFFT', n, ...
    %         'Window', "hann", 'Periodic', true, 'Center', false, 'PadMode', "constant")
    %
    %   Defaults: L = round(0.025 fs) (25 ms), H = round(0.010 fs) (10 ms),
    %   NFFT = 2^nextpow2(L). NFFT >= L is required.
    %
    %   TIME CONVENTIONS (Phase 0 contract sec. 3, frozen):
    %     P   = Center ? floor(L/2) : 0
    %     T   = (S + 2P >= L) ? 1 + floor((S + 2P - L)/H) : 0
    %     s_t = (t - 1) H - P                         0-based first sample of frame t
    %     c_t = (s_t + (L - 1)/2) / fs                frame centre time (s)
    %     sampleInterval(t) = [s_t + 1, s_t + L]      1-based, may reach into padding
    %     f_k = k fs / NFFT, k = 0..floor(NFFT/2)     one-sided bin frequencies
    %   Each frame is multiplied by window() (length L) and zero-padded at the
    %   end to NFFT before the DFT (powerSpectrum). Framing itself is
    %   OceanMath.Signal.FrameSignal; this class binds it to a sample rate and
    %   to the conventions above so every feature shares one time axis.
    %
    %   Edge cases: Center=false and S < L gives T = 0 (empty, not an error);
    %   Center=true with PadMode="reflect" needs S > P (error
    %   'OceanMath:Signal:FrameSignal:TooShortForReflect').

    properties (SetAccess = immutable)
        SampleRate (1,1) double
        WindowLength (1,1) double
        HopLength (1,1) double
        NFFT (1,1) double
        Window (1,1) string
        Periodic (1,1) logical
        Center (1,1) logical
        PadMode (1,1) string
    end

    methods
        function obj = FrameGrid(sampleRate, opts)
            arguments
                sampleRate (1,1) double {mustBePositive, mustBeFinite}
                opts.WindowLength double {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = []
                opts.HopLength double {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = []
                opts.NFFT double {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = []
                opts.Window (1,1) string {mustBeMember(opts.Window, ["hann", "hamming", "rectangular", "povey"])} = "hann"
                opts.Periodic (1,1) logical = true
                opts.Center (1,1) logical = false
                opts.PadMode (1,1) string {mustBeMember(opts.PadMode, ["constant", "reflect"])} = "constant"
            end
            L = opts.WindowLength;
            if isempty(L)
                L = max(1, round(0.025 * sampleRate));
            end
            H = opts.HopLength;
            if isempty(H)
                H = max(1, round(0.010 * sampleRate));
            end
            n = opts.NFFT;
            if isempty(n)
                n = 2 ^ nextpow2(L);
            end
            if n < L
                error('OceanNLP:Speech:Core:FrameGrid:NFFTTooSmall', 'NFFT (%d) must be >= WindowLength (%d).', n, L);
            end
            obj.SampleRate = sampleRate;
            obj.WindowLength = L;
            obj.HopLength = H;
            obj.NFFT = n;
            obj.Window = opts.Window;
            obj.Periodic = opts.Periodic;
            obj.Center = opts.Center;
            obj.PadMode = opts.PadMode;
        end

        function P = padding(obj)
            % PADDING  Samples padded on each side: floor(L/2) when centred, else 0.
            P = 0;
            if obj.Center
                P = floor(obj.WindowLength / 2);
            end
        end

        function T = numFrames(obj, S)
            arguments
                obj
                S (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            T = OceanMath.Signal.FrameSignal.numFrames(S, obj.WindowLength, obj.HopLength, obj.Center);
        end

        function s = frameStarts(obj, S)
            % FRAMESTARTS  1 x T 0-based source sample of each frame's first sample (s_t).
            arguments
                obj
                S (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            % Starts use zero-based signal coordinates even though MATLAB arrays
            % are one-based.  A negative start denotes samples supplied by left
            % padding, not an index that should be applied directly to x.
            s = (0:obj.numFrames(S) - 1) * obj.HopLength - obj.padding();
        end

        function c = frameTimes(obj, S)
            % FRAMETIMES  1 x T frame centre times c_t = (s_t + (L-1)/2) / fs, seconds.
            arguments
                obj
                S (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            % (L-1)/2 is the midpoint of sample positions 0..L-1.  For even L
            % this is a valid half-sample time between the two middle samples.
            c = (obj.frameStarts(S) + (obj.WindowLength - 1) / 2) / obj.SampleRate;
        end

        function iv = sampleInterval(obj, t, S, opts)
            % SAMPLEINTERVAL  numel(t) x 2 [first, last] 1-based source samples of frames t.
            %   'Clip', true clips to [1, S] (a frame lying wholly in padding then
            %   has first > last).
            arguments
                obj
                t (:,1) double {mustBeInteger, mustBePositive}
                S (1,1) double {mustBeInteger, mustBeNonnegative}
                opts.Clip (1,1) logical = false
            end
            T = obj.numFrames(S);
            if any(t > T)
                error('OceanNLP:Speech:Core:FrameGrid:FrameIndex', 'Frame index exceeds the %d frames of a %d-sample signal.', T, S);
            end
            s = (t - 1) * obj.HopLength - obj.padding();
            iv = [s + 1, s + obj.WindowLength];
            if opts.Clip
                iv = [max(iv(:, 1), 1), min(iv(:, 2), S)];
            end
        end

        function [frames, starts0] = frame(obj, x)
            % FRAME  L x T raw (unwindowed) frames of a mono column signal, and s_t.
            arguments
                obj
                x (:,1) double {mustBeReal}
            end
            [frames, starts0] = OceanMath.Signal.FrameSignal.frame(x, obj.WindowLength, obj.HopLength, ...
                'Center', obj.Center, 'PadMode', obj.PadMode);
        end

        function w = window(obj)
            % WINDOW  L x 1 analysis window.
            w = OceanMath.Signal.FrameSignal.window(obj.Window, obj.WindowLength, 'Periodic', obj.Periodic);
        end

        function P = powerSpectrum(obj, x)
            % POWERSPECTRUM  (floor(NFFT/2)+1) x T |DFT(window .* frame)|^2 of a mono column.
            arguments
                obj
                x (:,1) double {mustBeReal}
            end
            frames = obj.frame(x);
            % frame() is intentionally raw.  Apply the analysis window here so
            % callers such as pitch/energy may choose different frame treatment.
            P = OceanMath.Signal.SpectralAnalyzer.framePowerSpectrum(frames .* obj.window(), obj.NFFT);
        end

        function f = binFrequencies(obj)
            % BINFREQUENCIES  (floor(NFFT/2)+1) x 1 bin frequencies k fs / NFFT in Hz.
            f = (0:floor(obj.NFFT / 2))' * obj.SampleRate / obj.NFFT;
        end

        function cfg = config(obj)
            cfg = struct('Type', "FrameGrid", 'SampleRate', obj.SampleRate, ...
                'WindowLength', obj.WindowLength, 'HopLength', obj.HopLength, 'NFFT', obj.NFFT, ...
                'Window', obj.Window, 'Periodic', obj.Periodic, 'Center', obj.Center, 'PadMode', obj.PadMode);
        end
    end

    methods (Static)
        function g = fromOptions(sampleRate, opts)
            % FROMOPTIONS  Build a grid from a struct holding (a subset of) the grid options.
            %   Feature classes forward their grid name-value options through this.
            arguments
                sampleRate (1,1) double
                opts (1,1) struct
            end
            names = ["WindowLength", "HopLength", "NFFT", "Window", "Periodic", "Center", "PadMode"];
            args = {};
            for k = 1:numel(names)
                if isfield(opts, names(k))
                    args = [args, {names(k), opts.(names(k))}]; %#ok<AGROW>
                end
            end
            g = OceanNLP.Speech.Core.FrameGrid(sampleRate, args{:});
        end
    end
end
