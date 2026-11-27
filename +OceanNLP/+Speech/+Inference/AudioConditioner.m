classdef AudioConditioner < handle
    % AUDIOCONDITIONER  Real-world audio -> the mono, model-rate, level-normalised waveform a
    %   recogniser expects, with an honest report of what the input was and what was done.
    %
    %   c = OceanNLP.Speech.Inference.AudioConditioner('TargetRate', 8000, 'MixDown', "mean", ...
    %         'RemoveDC', true, 'Normalize', true, 'TargetRMSdBFS', -20, 'MaxGainDB', 30, ...
    %         'ClipThreshold', 0.999, 'MinDuration', 0.1)
    %   [w2, report] = c.forward(w)          Waveform -> conditioned Waveform
    %   cfg = c.config();   c2 = OceanNLP.Speech.Inference.AudioConditioner.fromConfig(cfg)
    %   w   = OceanNLP.Speech.Inference.AudioConditioner.toWaveform(input)
    %         input = Waveform | audio file path | struct(Samples, SampleRate[, Source])
    %
    %   STEPS (in this order; each one that changes the samples appends a note to
    %   Waveform.History and to report.Steps). x is S x C, f_s the input rate.
    %     0. Measure on the RAW input (all channels):
    %          ClippedFraction = #{|x| >= ClipThreshold} / (S C)
    %          PeakDBFS (input) is reported after mixing (step 5), see below.
    %     1. Mix down C > 1 channels: "mean" x_m[n] = (1/C) sum_c x_c[n], or "first".
    %        A mean mix whose RMS is more than 6 dB below the mean channel RMS is
    %        flagged (channels partly cancel: out-of-phase / spaced microphones).
    %     2. DC removal: DCOffset = (1/S) sum_n x_m[n];  x_m <- x_m - DCOffset.
    %     3. Resample f_s -> TargetRate with OceanMath.Signal.ResampleSignal (rational p/q
    %        polyphase Kaiser-sinc low-pass; no toolbox). Mixing first means only one
    %        channel is resampled (memory/time linear in S).
    %     4. Level (after 1-3):  RMS = sqrt((1/S') sum x^2),  RMSdBFS = 20 log10 RMS,
    %        PeakDBFS = 20 log10 max|x|  (full scale = 1; digital silence -> -Inf).
    %        These are measured AFTER resampling: the anti-aliasing low-pass can overshoot
    %        (Gibbs ringing), so PeakDBFS may exceed 0 dBFS even for an unclipped input
    %        (e.g. +1.5 dB for a 0.99-amplitude square wave). ClippedFraction (step 0) is
    %        the measure of real clipping in the recording.
    %     5. Normalise (optional) to TargetRMSdBFS:
    %          g_dB = min(TargetRMSdBFS - RMSdBFS, MaxGainDB, -PeakDBFS - 0.1)
    %        i.e. capped by MaxGainDB and by the peak headroom (never creates clipping),
    %        x <- 10^(g_dB/20) x.  Silent input (RMS = 0) is never amplified (g_dB = 0).
    %        Note: RMS is over the whole signal, so long pauses lower it and raise the gain;
    %        the headroom cap keeps the loudest sample below full scale.
    %     6. Minimum-duration guard: DurationOut < MinDuration -> a warning (the audio is
    %        NOT padded: zero-padding would add content the speaker never produced).
    %
    %   INTEGER INPUT (toWaveform with struct Samples of an integer class): signed
    %   intN samples are scaled by 1 / 2^(N-1) (int16: 1/32768), unsigned by
    %   (x - 2^(N-1)) / 2^(N-1) -- the audioread convention -- before anything else.
    %   A Waveform already holds doubles; samples beyond +-1 there are reported
    %   ("peak above full scale -- unscaled integer PCM?").
    %
    %   report: InputRate, OutputRate, Channels, DurationIn, DurationOut, PeakDBFS,
    %     RMSdBFS (both measured after steps 1-3, before gain), ClippedFraction,
    %     DCOffset, GainDB, Steps (string), Warnings (string, human-readable)
    %     + OutputPeakDBFS, OutputRMSdBFS (after gain), ChannelRMSdBFS (1 x C, raw).
    %
    %   Warnings are measured facts with the threshold that triggered them; they never
    %   change the samples. No random numbers are drawn.

    properties (SetAccess = immutable)
        TargetRate (1,1) double
        MixDown (1,1) string
        RemoveDC (1,1) logical
        Normalize (1,1) logical
        TargetRMSdBFS (1,1) double
        MaxGainDB (1,1) double
        ClipThreshold (1,1) double
        MinDuration (1,1) double
        ClipWarnFraction (1,1) double
        QuietDBFS (1,1) double
    end

    methods
        function obj = AudioConditioner(opts)
            arguments
                opts.TargetRate (1,1) double {mustBePositive, mustBeFinite} = 8000
                opts.MixDown (1,1) string {mustBeMember(opts.MixDown, ["mean", "first"])} = "mean"
                opts.RemoveDC (1,1) logical = true
                opts.Normalize (1,1) logical = true
                opts.TargetRMSdBFS (1,1) double {mustBeFinite, mustBeNonpositive} = -20
                opts.MaxGainDB (1,1) double {mustBeFinite, mustBeNonnegative} = 30
                opts.ClipThreshold (1,1) double {mustBePositive, mustBeFinite} = 0.999
                opts.MinDuration (1,1) double {mustBeNonnegative, mustBeFinite} = 0.1
                opts.ClipWarnFraction (1,1) double {mustBeNonnegative} = 1e-3
                opts.QuietDBFS (1,1) double {mustBeFinite} = -40
            end
            obj.TargetRate = opts.TargetRate;
            obj.MixDown = opts.MixDown;
            obj.RemoveDC = opts.RemoveDC;
            obj.Normalize = opts.Normalize;
            obj.TargetRMSdBFS = opts.TargetRMSdBFS;
            obj.MaxGainDB = opts.MaxGainDB;
            obj.ClipThreshold = opts.ClipThreshold;
            obj.MinDuration = opts.MinDuration;
            obj.ClipWarnFraction = opts.ClipWarnFraction;
            obj.QuietDBFS = opts.QuietDBFS;
        end

        function [w, report] = forward(obj, w)
            arguments
                obj
                w (1,1) OceanNLP.Speech.Core.Waveform
            end
            fsIn = w.SampleRate;
            C = w.numChannels();
            durIn = w.duration();
            warns = strings(1, 0);
            steps = strings(1, 0);

            % 0. raw measurements
            x0 = w.Samples;
            clipped = nnz(abs(x0) >= obj.ClipThreshold) / numel(x0);
            chanRms = sqrt(mean(x0 .^ 2, 1));
            chanDB = OceanNLP.Speech.Inference.AudioConditioner.toDB(chanRms);
            rawPeak = max(abs(x0), [], 'all');
            clear x0
            if rawPeak > 1
                warns(end + 1) = sprintf("Peak %.1f dBFS is above full scale (|x| up to %.4g > 1): unscaled integer PCM?", ...
                    20 * log10(rawPeak), rawPeak);
            end
            if clipped >= obj.ClipWarnFraction && clipped > 0
                warns(end + 1) = sprintf("Clipped: %.2f %% of samples at |x| >= %g", 100 * clipped, obj.ClipThreshold);
            end

            % 1. mix down
            if C > 1
                w = w.toMono(obj.MixDown);
                steps(end + 1) = sprintf("mix down %d channels (%s)", C, obj.MixDown);
                mixRms = sqrt(mean(w.Samples .^ 2));
                if obj.MixDown == "mean" && mixRms > 0 && ...
                        20 * log10(mixRms) < 20 * log10(mean(chanRms)) - 6
                    warns(end + 1) = sprintf("Channels partly cancel in the mean mix (mix %.1f dBFS vs channels %.1f dBFS)", ...
                        20 * log10(mixRms), 20 * log10(mean(chanRms)));
                end
            end

            % 2. DC removal
            dc = mean(w.Samples);
            if obj.RemoveDC && dc ~= 0
                w = w.derive(w.Samples - dc, w.SampleRate, sprintf("remove DC offset %.3g", dc));
                steps(end + 1) = sprintf("remove DC %.3g", dc);
            end

            % 3. resample
            if fsIn ~= obj.TargetRate
                y = OceanMath.Signal.ResampleSignal.resample(w.Samples, fsIn, obj.TargetRate);
                if isempty(y)
                    error('OceanNLP:Speech:Inference:AudioConditioner:TooShort', ...
                        'Resampling %d samples from %g Hz to %g Hz leaves no samples.', w.numSamples(), fsIn, obj.TargetRate);
                end
                w = w.derive(y, obj.TargetRate, sprintf("resample %g Hz -> %g Hz (OceanMath.Signal.ResampleSignal)", ...
                    fsIn, obj.TargetRate));
                steps(end + 1) = sprintf("resample %g -> %g Hz", fsIn, obj.TargetRate);
            end

            % 4. level
            x = w.Samples;
            rms = sqrt(mean(x .^ 2));
            peak = max(abs(x));
            rmsDB = OceanNLP.Speech.Inference.AudioConditioner.toDB(rms);
            peakDB = OceanNLP.Speech.Inference.AudioConditioner.toDB(peak);
            if rms == 0
                warns(end + 1) = "Silent input: every sample is zero (no level normalisation applied)";
            elseif rmsDB < obj.QuietDBFS
                warns(end + 1) = sprintf("Very quiet: %.1f dBFS RMS (below %g dBFS)", rmsDB, obj.QuietDBFS);
            end

            % 5. normalise
            gainDB = 0;
            if obj.Normalize && rms > 0
                want = obj.TargetRMSdBFS - rmsDB;
                gainDB = min([want, obj.MaxGainDB, -peakDB - 0.1]);
                if want > obj.MaxGainDB && gainDB == obj.MaxGainDB
                    warns(end + 1) = sprintf("Gain limited to +%g dB (MaxGainDB); output RMS %.1f dBFS instead of %g dBFS", ...
                        obj.MaxGainDB, rmsDB + gainDB, obj.TargetRMSdBFS);
                elseif gainDB < want - 1e-9 && gainDB == -peakDB - 0.1
                    warns(end + 1) = sprintf("Gain limited by peak headroom to %+.1f dB (target %g dBFS RMS not reached)", ...
                        gainDB, obj.TargetRMSdBFS);
                end
                if gainDB ~= 0
                    w = w.derive(x * 10 ^ (gainDB / 20), w.SampleRate, sprintf("gain %+.2f dB (RMS %.1f -> %.1f dBFS)", ...
                        gainDB, rmsDB, rmsDB + gainDB));
                    steps(end + 1) = sprintf("gain %+.2f dB", gainDB);
                end
            end

            % 6. duration guard
            durOut = w.duration();
            if durOut < obj.MinDuration
                warns(end + 1) = sprintf("Very short input: %.3f s (below MinDuration %g s); not padded", durOut, obj.MinDuration);
            end

            report = struct('InputRate', fsIn, 'OutputRate', w.SampleRate, 'Channels', C, ...
                'DurationIn', durIn, 'DurationOut', durOut, 'PeakDBFS', peakDB, 'RMSdBFS', rmsDB, ...
                'ClippedFraction', clipped, 'DCOffset', dc, 'GainDB', gainDB, 'Steps', steps, ...
                'Warnings', warns, ...
                'OutputPeakDBFS', peakDB + gainDB, 'OutputRMSdBFS', rmsDB + gainDB, 'ChannelRMSdBFS', chanDB);
        end

        function cfg = config(obj)
            cfg = struct('Type', "AudioConditioner", 'Version', 1, 'TargetRate', obj.TargetRate, ...
                'MixDown', obj.MixDown, 'RemoveDC', obj.RemoveDC, 'Normalize', obj.Normalize, ...
                'TargetRMSdBFS', obj.TargetRMSdBFS, 'MaxGainDB', obj.MaxGainDB, ...
                'ClipThreshold', obj.ClipThreshold, 'MinDuration', obj.MinDuration, ...
                'ClipWarnFraction', obj.ClipWarnFraction, 'QuietDBFS', obj.QuietDBFS);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if ~isfield(cfg, 'Type') || string(cfg.Type) ~= "AudioConditioner"
                error('OceanNLP:Speech:Inference:AudioConditioner:ConfigType', 'Config is not an AudioConditioner config.');
            end
            names = ["TargetRate", "MixDown", "RemoveDC", "Normalize", "TargetRMSdBFS", "MaxGainDB", ...
                "ClipThreshold", "MinDuration", "ClipWarnFraction", "QuietDBFS"];
            args = {};
            for n = names
                if isfield(cfg, n)
                    args = [args, {char(n), cfg.(n)}]; %#ok<AGROW>
                end
            end
            obj = OceanNLP.Speech.Inference.AudioConditioner(args{:});
        end

        function w = toWaveform(input)
            % TOWAVEFORM  Waveform | file path | struct(Samples, SampleRate[, Source]) -> Waveform.
            arguments
                input
            end
            if isa(input, 'OceanNLP.Speech.Core.Waveform')
                if ~isscalar(input)
                    error('OceanNLP:Speech:Inference:AudioConditioner:InputType', 'Pass one Waveform.');
                end
                w = input;
            elseif (ischar(input) || isstring(input)) && isscalar(string(input))
                w = OceanNLP.Speech.Core.Waveform.read(string(input));
            elseif isstruct(input) && isscalar(input) && all(isfield(input, {'Samples', 'SampleRate'}))
                x = input.Samples;
                note = strings(1, 0);
                if isinteger(x)
                    cls = class(x);
                    bits = 8 * numel(typecast(zeros(1, 1, cls), 'uint8'));
                    half = 2 ^ (bits - 1);
                    if startsWith(cls, 'u')
                        x = (double(x) - half) / half;
                    else
                        x = double(x) / half;
                    end
                    note = sprintf("%s PCM scaled to [-1, 1) (1/%d)", cls, half);
                elseif islogical(x) || ~isnumeric(x)
                    error('OceanNLP:Speech:Inference:AudioConditioner:InputType', 'Samples must be numeric.');
                end
                if isrow(x) && numel(x) > 1
                    x = x(:);
                    note = [note, "row vector taken as mono column"];
                end
                src = "";
                if isfield(input, 'Source'), src = string(input.Source); end
                w = OceanNLP.Speech.Core.Waveform(double(x), double(input.SampleRate), 'Source', src, 'History', note);
            else
                error('OceanNLP:Speech:Inference:AudioConditioner:InputType', ...
                    'Input must be a Waveform, an audio file path, or struct(Samples, SampleRate); got %s.', class(input));
            end
        end

        function d = toDB(a)
            % TODB  20 log10(a), -Inf for 0.
            d = 20 * log10(a);
        end
    end
end
