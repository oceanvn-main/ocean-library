classdef EnergyContour < OceanNLP.Speech.Core.IFeatureTransform
    % ENERGYCONTOUR  Natural-log short-time frame energy on the FrameGrid.
    %
    %   t  = OceanNLP.Speech.Features.EnergyContour(fs)
    %   t  = OceanNLP.Speech.Features.EnergyContour(fs, <FrameGrid options>, 'Floor', 1e-10, ...
    %          'ApplyWindow', false)
    %   fs = t.forward(waveform)          % 1 x T, axes ["energy","time"], units "log energy"
    %
    %   E_t = ln( max( sum_{n=0}^{L-1} v[n]^2 x_t[n]^2, Floor ) )
    %   v = 1 (raw frame, HTK's log energy computed before windowing) by default;
    %   'ApplyWindow', true uses the grid's analysis window v = w.
    %   Silence gives ln(Floor) (finite). Same input policy as LogMel.

    properties (SetAccess = immutable)
        SampleRate (1,1) double
        Grid OceanNLP.Speech.Core.FrameGrid
        Floor (1,1) double
        ApplyWindow (1,1) logical
    end

    methods
        function obj = EnergyContour(sampleRate, opts)
            arguments
                sampleRate (1,1) double {mustBePositive, mustBeFinite}
                opts.WindowLength double {mustBeScalarOrEmpty} = []
                opts.HopLength double {mustBeScalarOrEmpty} = []
                opts.NFFT double {mustBeScalarOrEmpty} = []
                opts.Window (1,1) string = "hann"
                opts.Periodic (1,1) logical = true
                opts.Center (1,1) logical = false
                opts.PadMode (1,1) string = "constant"
                opts.Floor (1,1) double {mustBePositive} = 1e-10
                opts.ApplyWindow (1,1) logical = false
            end
            obj.SampleRate = sampleRate;
            obj.Grid = OceanNLP.Speech.Core.FrameGrid.fromOptions(sampleRate, opts);
            obj.Floor = opts.Floor;
            obj.ApplyWindow = opts.ApplyWindow;
        end

        function fs = forward(obj, input)
            arguments
                obj
                input
            end
            [x, w] = prepareWaveform(input, obj.SampleRate, "EnergyContour");
            frames = obj.Grid.frame(x);
            if obj.ApplyWindow
                frames = frames .* obj.Grid.window();
            end
            E = log(max(sum(frames .^ 2, 1), obj.Floor));
            cfg = obj.config();
            cfg.InputHistory = w.History;
            fs = OceanNLP.Speech.Core.FeatureSequence(reshape(E, 1, []), ...
                'FrameTimes', obj.Grid.frameTimes(numel(x)), 'AxisNames', ["energy", "time"], ...
                'Units', "log energy", 'Config', cfg, 'Source', w.Source);
        end

        function cfg = config(obj)
            cfg = struct('Type', "EnergyContour", 'SampleRate', obj.SampleRate, 'Grid', obj.Grid.config(), ...
                'Floor', obj.Floor, 'ApplyWindow', obj.ApplyWindow);
        end

        function d = outputDim(obj, inputDim)
            arguments
                obj %#ok<INUSA>
                inputDim double = [] %#ok<INUSA> fixed output size
            end
            d = 1;
        end
    end

    methods (Static)
        function f = fromConfig(cfg)
            % FROMCONFIG  Inverse of config().
            arguments
                cfg (1,1) struct
            end
            g = gridArgs(cfg.Grid);
            f = OceanNLP.Speech.Features.EnergyContour(cfg.SampleRate, g{:}, ...
                'Floor', cfg.Floor, 'ApplyWindow', cfg.ApplyWindow);
        end
    end
end
