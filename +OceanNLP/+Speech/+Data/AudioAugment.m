classdef AudioAugment < handle
    % AUDIOAUGMENT  Seeded waveform augmentation + SpecAugment masking.
    %
    %   a = OceanNLP.Speech.Data.AudioAugment('Seed', 0, 'Gain', [-6 6], 'NoiseSNR', [10 30], ...
    %         'TimeMasks', 2, 'MaxTimeMask', 20, 'FreqMasks', 2, 'MaxFreqMask', 8)
    %   w2 = a.forwardWaveform(w)     Core.Waveform -> Core.Waveform (History records draws)
    %   f2 = a.forwardFeatures(f)     Core.FeatureSequence -> FeatureSequence (Config.SpecAugment)
    %   y  = a.forward(x)             dispatches on the input class
    %   a.setEpoch(e)                 restart the private stream at Seed + e (reproducible epochs)
    %
    %   Every option defaults to "off". A disabled component returns its input
    %   object unchanged and draws NO random numbers (the stream state is untouched).
    %   The class owns its RandStream (mt19937ar); it never touches the global RNG.
    %
    %   WAVEFORM (draws in this order, per call):
    %     gain    g ~ U[Gain(1), Gain(2)] dB,             x <- x * 10^(g/20)
    %     noise   q ~ U[NoiseSNR(1), NoiseSNR(2)] dB,     x <- x + sigma * n,  n ~ N(0, 1) i.i.d.,
    %             sigma^2 = P_x / 10^(q/10),  P_x = mean(x(:).^2)  (power after gain)
    %             so the expected SNR is q dB (a silent input gets no noise: P_x = 0).
    %
    %   SPECAUGMENT (Park et al. 2019, "SpecAugment", frequency + time masking; time
    %   warping is not implemented). On F x T values, for each of FreqMasks masks:
    %     f ~ U{0..min(MaxFreqMask, F)},  f0 ~ U{0..F-f},  rows f0+1 .. f0+f <- mask value
    %   then for each of TimeMasks masks:
    %     t ~ U{0..min(MaxTimeMask, T)},  t0 ~ U{0..T-t},  columns t0+1 .. t0+t <- mask value
    %   MaskValue "mean" = mean of the utterance's (unmasked) values, the value the
    %   paper's zero equals after per-utterance mean normalization; or a number.

    properties (SetAccess = private)
        Seed (1,1) double
        Gain (1,:) double
        NoiseSNR (1,:) double
        TimeMasks (1,1) double
        FreqMasks (1,1) double
        MaxTimeMask (1,1) double
        MaxFreqMask (1,1) double
        MaskValue
        Epoch (1,1) double = 0
    end

    properties (Access = private)
        Stream
    end

    methods
        function obj = AudioAugment(opts)
            arguments
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.Gain (1,:) double {mustBeFinite} = []
                opts.NoiseSNR (1,:) double {mustBeFinite} = []
                opts.TimeMasks (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.FreqMasks (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.MaxTimeMask (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.MaxFreqMask (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                opts.MaskValue = "mean"
            end
            for name = ["Gain", "NoiseSNR"]
                r = opts.(name);
                if ~(isempty(r) || (numel(r) == 2 && r(1) <= r(2)))
                    error('OceanNLP:Speech:Data:AudioAugment:Range', '%s must be [] or [lo hi] with lo <= hi.', name);
                end
            end
            mv = opts.MaskValue;
            if ~((isstring(mv) || ischar(mv)) && string(mv) == "mean") && ~(isnumeric(mv) && isscalar(mv) && isfinite(mv))
                error('OceanNLP:Speech:Data:AudioAugment:MaskValue', 'MaskValue must be "mean" or a finite number.');
            end
            obj.Seed = opts.Seed;
            obj.Gain = opts.Gain;
            obj.NoiseSNR = opts.NoiseSNR;
            obj.TimeMasks = opts.TimeMasks;
            obj.FreqMasks = opts.FreqMasks;
            obj.MaxTimeMask = opts.MaxTimeMask;
            obj.MaxFreqMask = opts.MaxFreqMask;
            if ischar(mv), mv = string(mv); end
            obj.MaskValue = mv;
            obj.setEpoch(0);
        end

        function setEpoch(obj, epoch)
            arguments
                obj
                epoch (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            obj.Epoch = epoch;
            obj.Stream = RandStream('mt19937ar', 'Seed', obj.Seed + epoch);
        end

        function tf = isEnabled(obj)
            tf = obj.waveformEnabled() || obj.featuresEnabled();
        end

        function tf = waveformEnabled(obj)
            tf = ~isempty(obj.Gain) || ~isempty(obj.NoiseSNR);
        end

        function tf = featuresEnabled(obj)
            tf = (obj.TimeMasks > 0 && obj.MaxTimeMask > 0) || (obj.FreqMasks > 0 && obj.MaxFreqMask > 0);
        end

        function y = forward(obj, x)
            if isa(x, 'OceanNLP.Speech.Core.Waveform')
                y = obj.forwardWaveform(x);
            elseif isa(x, 'OceanNLP.Speech.Core.FeatureSequence')
                y = obj.forwardFeatures(x);
            else
                error('OceanNLP:Speech:Data:AudioAugment:InputType', ...
                    'Input must be a Core.Waveform or Core.FeatureSequence, got %s.', class(x));
            end
        end

        function w = forwardWaveform(obj, w)
            arguments
                obj
                w (1,1) OceanNLP.Speech.Core.Waveform
            end
            if ~obj.waveformEnabled()
                return;                                   % identity, no draws
            end
            x = w.Samples;
            notes = strings(1, 0);
            if ~isempty(obj.Gain)
                g = obj.Gain(1) + (obj.Gain(2) - obj.Gain(1)) * rand(obj.Stream);
                x = x * 10^(g / 20);
                notes(end+1) = sprintf("augment gain %+.6g dB", g);
            end
            if ~isempty(obj.NoiseSNR)
                q = obj.NoiseSNR(1) + (obj.NoiseSNR(2) - obj.NoiseSNR(1)) * rand(obj.Stream);
                px = mean(x(:).^2);
                n = randn(obj.Stream, size(x));
                x = x + sqrt(px / 10^(q / 10)) * n;
                notes(end+1) = sprintf("augment white noise SNR %.6g dB (seed %d, epoch %d)", q, obj.Seed, obj.Epoch);
            end
            w = w.derive(x, w.SampleRate, join(notes, "; "));
        end

        function f = forwardFeatures(obj, f)
            arguments
                obj
                f (1,1) OceanNLP.Speech.Core.FeatureSequence
            end
            if ~obj.featuresEnabled()
                return;                                   % identity, no draws
            end
            V = f.Values;
            [F, T] = size(V);
            if isstring(obj.MaskValue)
                fill = mean(V, 'all');
                if isempty(V), fill = 0; end
            else
                fill = obj.MaskValue;
            end
            freq = zeros(0, 2); time = zeros(0, 2);
            if obj.MaxFreqMask > 0
                for k = 1:obj.FreqMasks
                    width = randi(obj.Stream, [0, min(obj.MaxFreqMask, F)]);
                    start = randi(obj.Stream, [0, F - width]);
                    V(start+1:start+width, :) = fill;
                    freq(end+1, :) = [start + 1, start + width]; %#ok<AGROW>
                end
            end
            if obj.MaxTimeMask > 0
                for k = 1:obj.TimeMasks
                    width = randi(obj.Stream, [0, min(obj.MaxTimeMask, T)]);
                    start = randi(obj.Stream, [0, T - width]);
                    V(:, start+1:start+width) = fill;
                    time(end+1, :) = [start + 1, start + width]; %#ok<AGROW>
                end
            end
            cfg = f.Config;
            cfg(1).SpecAugment = struct('FreqMaskRows', freq, 'TimeMaskFrames', time, ...
                'MaskValue', fill, 'Seed', obj.Seed, 'Epoch', obj.Epoch);
            f = f.derive(V, 'FrequencyHz', f.FrequencyHz, 'Config', cfg);
        end

        function cfg = config(obj)
            cfg = struct('Type', "AudioAugment", 'Seed', obj.Seed, 'Gain', obj.Gain, ...
                'NoiseSNR', obj.NoiseSNR, 'TimeMasks', obj.TimeMasks, 'FreqMasks', obj.FreqMasks, ...
                'MaxTimeMask', obj.MaxTimeMask, 'MaxFreqMask', obj.MaxFreqMask, ...
                'MaskValue', obj.MaskValue, 'Epoch', obj.Epoch);
        end

        function s = streamState(obj)
            % STREAMSTATE  Current generator state (for reproducibility checks).
            s = obj.Stream.State;
        end
    end
end
