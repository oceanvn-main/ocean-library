classdef FrameEnergyTestTransform < OceanNLP.Speech.Core.IFeatureTransform
    % FRAMEENERGYTESTTRANSFORM  Small, real-computing IFeatureTransform test double.
    %
    %   Lets the Data tests run without builder B1's feature package. For mono
    %   x (multichannel is averaged), rectangular frames of L = round(0.025 fs)
    %   samples with hop H = round(0.010 fs), no centering:
    %
    %     T = S >= L ? 1 + floor((S - L) / H) : 0
    %     row 1:  log(max(sum_n x_t[n]^2, 1e-10))              natural-log frame energy
    %     row 2:  (1 / (L-1)) sum_n [sign x_t[n] ~= sign x_t[n-1]]   zero-crossing rate
    %
    %   This class has NO numFrames(S) method, so AudioDataset.lengths() must load
    %   audio and count; the subclass FrameEnergyCountingTestTransform adds
    %   numFrames(S) to exercise the no-audio path.

    properties (SetAccess = immutable)
        SampleRate (1,1) double
        L (1,1) double
        H (1,1) double
    end

    methods
        function obj = FrameEnergyTestTransform(sampleRate)
            arguments
                sampleRate (1,1) double {mustBePositive}
            end
            obj.SampleRate = sampleRate;
            obj.L = round(0.025 * sampleRate);
            obj.H = round(0.010 * sampleRate);
        end

        function fs = forward(obj, w)
            if ~isa(w, 'OceanNLP.Speech.Core.Waveform')
                error('OceanTest:FrameEnergyTestTransform:InputType', 'Expects a Waveform.');
            end
            if w.SampleRate ~= obj.SampleRate
                error('OceanTest:FrameEnergyTestTransform:SampleRateMismatch', 'Rate %g ~= %g.', w.SampleRate, obj.SampleRate);
            end
            x = mean(w.Samples, 2);
            T = obj.countFrames(numel(x));
            V = zeros(2, T);
            for t = 1:T
                seg = x((t-1)*obj.H + (1:obj.L));
                V(1, t) = log(max(sum(seg.^2), 1e-10));
                s = sign(seg);
                V(2, t) = nnz(s(2:end) ~= s(1:end-1)) / (obj.L - 1);
            end
            times = ((0:T-1) * obj.H + (obj.L - 1) / 2) / obj.SampleRate;
            fs = OceanNLP.Speech.Core.FeatureSequence(V, 'FrameTimes', times, ...
                'AxisNames', ["feature", "time"], 'Units', "log energy / rate", ...
                'Config', obj.config(), 'Source', w.Source);
        end

        function cfg = config(obj)
            cfg = struct('Type', "FrameEnergyTestTransform", 'SampleRate', obj.SampleRate, ...
                'WindowLength', obj.L, 'HopLength', obj.H);
        end

        function d = outputDim(~, ~)
            d = 2;
        end
    end

    methods (Access = protected)
        function T = countFrames(obj, S)
            if S >= obj.L
                T = 1 + floor((S - obj.L) / obj.H);
            else
                T = 0;
            end
        end
    end
end
