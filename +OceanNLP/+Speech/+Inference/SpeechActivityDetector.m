classdef SpeechActivityDetector < handle
    % SPEECHACTIVITYDETECTOR  Energy endpointing with an adaptive noise floor, double-threshold
    %   hysteresis, minimum speech / silence durations, padding and a maximum segment length.
    %
    %   d = OceanNLP.Speech.Inference.SpeechActivityDetector('FrameSeconds', 0.025, ...
    %         'HopSeconds', 0.010, 'Sensitivity', 0.5, 'MinSpeech', 0.15, 'MinSilence', 0.25, ...
    %         'Pad', 0.10, 'MaxSegment', 3.0)
    %   [segments, info] = d.detect(w)    segments: 1 x J struct Start, End (seconds)
    %   wt    = d.trim(w, segments)       samples from the first Start to the last End
    %   parts = d.split(w, segments)      1 x J Waveform array (one per segment)
    %   cfg = d.config();  d2 = SpeechActivityDetector.fromConfig(cfg)
    %
    %   ALGORITHM (classic energy endpointing, Rabiner & Sambur 1975 / Rabiner & Schafer,
    %   "Theory and Applications of Digital Speech Processing" sec. 10.3, with an
    %   adaptive percentile threshold in place of a calibrated silence interval):
    %     1. Frames: L = round(FrameSeconds f_s), H = round(HopSeconds f_s), not centred,
    %        T = 1 + floor((S - L)/H)  (OceanMath.Signal.FrameSignal.numFrames; a signal
    %        shorter than L is one frame of length S). Multichannel input is averaged.
    %     2. Log energy (rectangular window, from a cumulative sum -> O(S) time and memory,
    %        no L x T frame matrix, so minutes of audio are cheap):
    %          E_t = 10 log10( (1/L) sum_{n in frame t} x[n]^2 + 1e-12 )   dBFS
    %     3. Adaptive levels:  N = Q_0.10(E) (noise floor),  P = Q_0.95(E) (speech level),
    %        R = P - N  (Q_p = linear-interpolation quantile of the frame energies).
    %     4. Thresholds (Sensitivity s in [0, 1], higher = more speech):
    %          a_on = 0.6 - 0.4 s,   theta_on = N + a_on R,
    %          theta_off = max(N + a_on R / 2, AbsoluteFloorDB)
    %        Double-threshold hysteresis: a speech run is a maximal run of frames with
    %        E_t >= theta_off that contains at least one frame with E_t >= theta_on.
    %        Degenerate levels: P < AbsoluteFloorDB -> no speech (silence / digital zero);
    %        R < MinRangeDB -> no pauses to adapt to, so every frame is speech (stationary
    %        signal: continuous speech or stationary noise cannot be told apart by energy).
    %     5. Durations: pauses shorter than MinSilence are bridged, then runs shorter than
    %        MinSpeech are dropped. Run [t1, t2] spans [(t1 - 1) H, (t2 - 1) H + L) samples.
    %     6. Padding: each segment grows by Pad on both sides (clipped to [0, duration]),
    %        overlapping segments merge.
    %     7. MaxSegment: a segment longer than MaxSegment is split at the centre of its
    %        lowest-energy frame inside [Start + m, End - m], m = min(MinSpeech, len / 4),
    %        recursively until every piece is <= MaxSegment (Inf disables splitting).
    %   LIMITS (honest): energy only -- weak unvoiced onsets (/s/ in "six", /f/ in "five")
    %   may fall below theta_off and are recovered only by Pad; loud non-speech (music,
    %   door slams) is detected as speech. Validated against MATLAB Audio Toolbox
    %   detectSpeech in the tests only (frame agreement reported there).
    %
    %   info: FrameTimes (frame centres, s), Energy (dB), Threshold (theta_on, dB),
    %     ThresholdOff, NoiseFloor, SpeechLevel, IsSpeech (frame centre inside a final
    %     segment), RawIsSpeech (step-4 hysteresis decision), SplitTimes, FrameLength,
    %     HopLength, SampleRate, Note.

    properties (SetAccess = immutable)
        FrameSeconds (1,1) double
        HopSeconds (1,1) double
        Sensitivity (1,1) double
        MinSpeech (1,1) double
        MinSilence (1,1) double
        Pad (1,1) double
        MaxSegment (1,1) double
        AbsoluteFloorDB (1,1) double
        MinRangeDB (1,1) double
    end

    methods
        function obj = SpeechActivityDetector(opts)
            arguments
                opts.FrameSeconds (1,1) double {mustBePositive, mustBeFinite} = 0.025
                opts.HopSeconds (1,1) double {mustBePositive, mustBeFinite} = 0.010
                opts.Sensitivity (1,1) double {mustBeGreaterThanOrEqual(opts.Sensitivity, 0), mustBeLessThanOrEqual(opts.Sensitivity, 1)} = 0.5
                opts.MinSpeech (1,1) double {mustBeNonnegative, mustBeFinite} = 0.15
                opts.MinSilence (1,1) double {mustBeNonnegative, mustBeFinite} = 0.25
                opts.Pad (1,1) double {mustBeNonnegative, mustBeFinite} = 0.10
                opts.MaxSegment (1,1) double {mustBePositive} = 3.0
                opts.AbsoluteFloorDB (1,1) double {mustBeFinite} = -60
                opts.MinRangeDB (1,1) double {mustBeNonnegative, mustBeFinite} = 6
            end
            if opts.MaxSegment <= 2 * opts.MinSpeech
                error('OceanNLP:Speech:Inference:SpeechActivityDetector:MaxSegment', ...
                    'MaxSegment (%g s) must exceed 2 x MinSpeech (%g s).', opts.MaxSegment, opts.MinSpeech);
            end
            obj.FrameSeconds = opts.FrameSeconds;
            obj.HopSeconds = opts.HopSeconds;
            obj.Sensitivity = opts.Sensitivity;
            obj.MinSpeech = opts.MinSpeech;
            obj.MinSilence = opts.MinSilence;
            obj.Pad = opts.Pad;
            obj.MaxSegment = opts.MaxSegment;
            obj.AbsoluteFloorDB = opts.AbsoluteFloorDB;
            obj.MinRangeDB = opts.MinRangeDB;
        end

        function [segments, info] = detect(obj, w, opts)
            % DETECT  Speech segments of W.  'GainDB': gain already applied to W (e.g. by
            %   AudioConditioner level normalisation). The absolute floor refers to the
            %   RECORDED level, so it is moved by the same gain: floor = AbsoluteFloorDB + GainDB.
            %   Without this, normalisation (up to +30 dB) lifts room noise above the floor
            %   and a quiet recording is "detected" as speech.
            arguments
                obj
                w (1,1) OceanNLP.Speech.Core.Waveform
                opts.GainDB (1,1) double {mustBeFinite} = 0
            end
            floorDB = obj.AbsoluteFloorDB + opts.GainDB;
            fs = w.SampleRate;
            x = w.Samples;
            if size(x, 2) > 1
                x = mean(x, 2);
            end
            S = numel(x);
            dur = S / fs;
            L = max(1, round(obj.FrameSeconds * fs));
            H = max(1, round(obj.HopSeconds * fs));
            if S < L
                L = S;
            end
            T = OceanMath.Signal.FrameSignal.numFrames(S, L, H, false);
            starts0 = (0:T-1) * H;
            c = [0; cumsum(x .^ 2)];
            E = reshape(10 * log10((c(starts0 + L + 1) - c(starts0 + 1)) / L + 1e-12), 1, []);
            clear c
            times = (starts0 + L / 2) / fs;

            N = quantileLinear(E, 0.10);
            P = quantileLinear(E, 0.95);
            R = P - N;
            aOn = 0.6 - 0.4 * obj.Sensitivity;
            thOn = N + aOn * R;
            thOff = max(N + aOn * R / 2, floorDB);
            note = "";
            if P < floorDB
                raw = false(1, T);
                note = sprintf("No speech: 95th-percentile frame energy %.1f dBFS is below the absolute floor %g dBFS", ...
                    P, floorDB);
            elseif R < obj.MinRangeDB
                raw = E >= floorDB;
                note = sprintf("Energy range %.1f dB < %g dB: no pauses to adapt to, frames above %g dBFS taken as speech", ...
                    R, obj.MinRangeDB, floorDB);
            else
                above = E >= thOff;
                raw = false(1, T);
                [r1, r2] = runsOf(above);
                for k = 1:numel(r1)
                    if any(E(r1(k):r2(k)) >= thOn)
                        raw(r1(k):r2(k)) = true;
                    end
                end
            end

            % durations (frames)
            [r1, r2] = runsOf(raw);
            st = (r1 - 1) * H / fs;
            en = ((r2 - 1) * H + L) / fs;
            [st, en] = bridgeGaps(st, en, obj.MinSilence);
            keep = (en - st) >= obj.MinSpeech;
            st = st(keep); en = en(keep);
            % padding + merge
            st = max(0, st - obj.Pad);
            en = min(dur, en + obj.Pad);
            [st, en] = bridgeGaps(st, en, 0);
            % max segment
            splits = [];
            if isfinite(obj.MaxSegment)
                ns = []; ne = [];
                for k = 1:numel(st)
                    [a, b, sp] = obj.splitLong(st(k), en(k), times, E);
                    ns = [ns, a]; ne = [ne, b]; splits = [splits, sp]; %#ok<AGROW>
                end
                st = ns; en = ne;
            end
            segments = struct('Start', num2cell(st), 'End', num2cell(en));
            if isempty(st)
                segments = struct('Start', {}, 'End', {});
            end
            isSpeech = false(1, T);
            for k = 1:numel(st)
                isSpeech = isSpeech | (times >= st(k) & times <= en(k));
            end
            info = struct('FrameTimes', times, 'Energy', E, 'Threshold', thOn, 'ThresholdOff', thOff, ...
                'NoiseFloor', N, 'SpeechLevel', P, 'IsSpeech', isSpeech, 'RawIsSpeech', raw, ...
                'SplitTimes', sort(splits), 'FrameLength', L, 'HopLength', H, 'SampleRate', fs, 'Note', note);
        end

        function wt = trim(obj, w, segments) %#ok<INUSL>
            % TRIM  Keep [segments(1).Start, segments(end).End]; pauses between segments stay.
            arguments
                obj
                w (1,1) OceanNLP.Speech.Core.Waveform
                segments struct
            end
            if isempty(segments)
                error('OceanNLP:Speech:Inference:SpeechActivityDetector:NoSpeech', 'No speech segment to trim to.');
            end
            [n1, n2] = sampleRange(min([segments.Start]), max([segments.End]), w);
            wt = w.derive(w.Samples(n1:n2, :), w.SampleRate, sprintf("trim to speech %.3f-%.3f s (SpeechActivityDetector)", ...
                (n1 - 1) / w.SampleRate, n2 / w.SampleRate));
        end

        function parts = split(obj, w, segments) %#ok<INUSL>
            % SPLIT  One Waveform per segment (History records the segment's span).
            arguments
                obj
                w (1,1) OceanNLP.Speech.Core.Waveform
                segments struct
            end
            J = numel(segments);
            parts = OceanNLP.Speech.Core.Waveform.empty(1, 0);
            for k = 1:J
                [n1, n2] = sampleRange(segments(k).Start, segments(k).End, w);
                parts(k) = w.derive(w.Samples(n1:n2, :), w.SampleRate, ...
                    sprintf("segment %d/%d: %.3f-%.3f s (SpeechActivityDetector)", k, J, ...
                    (n1 - 1) / w.SampleRate, n2 / w.SampleRate));
            end
        end

        function cfg = config(obj)
            cfg = struct('Type', "SpeechActivityDetector", 'Version', 1, 'FrameSeconds', obj.FrameSeconds, ...
                'HopSeconds', obj.HopSeconds, 'Sensitivity', obj.Sensitivity, 'MinSpeech', obj.MinSpeech, ...
                'MinSilence', obj.MinSilence, 'Pad', obj.Pad, 'MaxSegment', obj.MaxSegment, ...
                'AbsoluteFloorDB', obj.AbsoluteFloorDB, 'MinRangeDB', obj.MinRangeDB);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if ~isfield(cfg, 'Type') || string(cfg.Type) ~= "SpeechActivityDetector"
                error('OceanNLP:Speech:Inference:SpeechActivityDetector:ConfigType', 'Config is not a SpeechActivityDetector config.');
            end
            names = ["FrameSeconds", "HopSeconds", "Sensitivity", "MinSpeech", "MinSilence", "Pad", ...
                "MaxSegment", "AbsoluteFloorDB", "MinRangeDB"];
            args = {};
            for n = names
                if isfield(cfg, n)
                    args = [args, {char(n), cfg.(n)}]; %#ok<AGROW>
                end
            end
            obj = OceanNLP.Speech.Inference.SpeechActivityDetector(args{:});
        end
    end

    methods (Access = private)
        function [a, b, splits] = splitLong(obj, s, e, times, E)
            if e - s <= obj.MaxSegment
                a = s; b = e; splits = [];
                return;
            end
            m = min(obj.MinSpeech, (e - s) / 4);
            idx = find(times >= s + m & times <= e - m);
            if isempty(idx)
                c = (s + e) / 2;                     % no frame centre inside: halve
            else
                [~, k] = min(E(idx));
                c = times(idx(k));
            end
            [a1, b1, s1] = obj.splitLong(s, c, times, E);
            [a2, b2, s2] = obj.splitLong(c, e, times, E);
            a = [a1, a2]; b = [b1, b2]; splits = [s1, c, s2];
        end
    end
end

function q = quantileLinear(v, p)
% Linear-interpolation quantile (Hyndman & Fan type 7), no toolbox.
v = sort(v(:));
n = numel(v);
if n == 1
    q = v;
    return;
end
h = (n - 1) * p + 1;
lo = floor(h);
hi = min(lo + 1, n);
q = v(lo) + (h - lo) * (v(hi) - v(lo));
end

function [r1, r2] = runsOf(mask)
% Start / end indices (1-based, inclusive) of the true runs of a logical row.
d = diff([false, mask(:).', false]);
r1 = find(d == 1);
r2 = find(d == -1) - 1;
end

function [st, en] = bridgeGaps(st, en, minGap)
% Merge consecutive intervals whose gap is < minGap (or overlapping when minGap = 0).
if isempty(st)
    return;
end
os = st(1); oe = en(1);
ns = []; ne = [];
for k = 2:numel(st)
    if st(k) - oe < minGap || st(k) <= oe
        oe = max(oe, en(k));
    else
        ns(end + 1) = os; ne(end + 1) = oe; %#ok<AGROW>
        os = st(k); oe = en(k);
    end
end
st = [ns, os]; en = [ne, oe];
end

function [n1, n2] = sampleRange(t1, t2, w)
S = w.numSamples();
n1 = min(S, max(1, round(t1 * w.SampleRate) + 1));
n2 = max(n1, min(S, round(t2 * w.SampleRate)));
end
