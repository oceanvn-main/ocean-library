classdef LogMel < OceanNLP.Speech.Core.IFeatureTransform
    % LOGMEL  Log mel-filter-bank energies (log-mel spectrogram) of a waveform.
    %
    %   t  = OceanNLP.Speech.Features.LogMel(fs)
    %   t  = OceanNLP.Speech.Features.LogMel(fs, 'WindowLength', L, 'HopLength', H, 'NFFT', n, ...
    %          'Window', "hann", 'Center', false, 'NumMels', 40, 'FMin', 0, 'FMax', [], ...
    %          'MelScale', "htk", 'MelNorm', "none", 'Power', 2, 'LogFloor', 1e-10)
    %   fs = t.forward(waveform)        % FeatureSequence, NumMels x T, axes ["mel","time"]
    %   st = t.intermediates(waveform)  % actual frames / spectra / mel energies (inspection)
    %
    %   COMPUTATION (frame t, bin k, filter m; FrameGrid conventions):
    %     x_t[n]   = frame t of the mono signal, n = 0..L-1
    %     X_t[k]   = sum_n w[n] x_t[n] e^{-j 2 pi k n / NFFT}
    %     P_t[k]   = |X_t[k]|^Power                 (Power = 2: power, 1: magnitude)
    %     E_t[m]   = sum_k W[m, k] P_t[k]           (OceanMath.Signal.MelFilterBank)
    %     Y[m, t]  = ln(max(E_t[m], LogFloor))      (natural log)
    %   Silence gives Y = ln(LogFloor) (finite). FrequencyHz = mel centres (Hz).
    %   Equivalent to log(max(librosa.feature.melspectrogram(..., center=False), LogFloor))
    %   with the same window/nfft/htk/norm.
    %
    %   INPUT POLICY: a Waveform at exactly SampleRate (else
    %   'OceanNLP:Speech:Features:LogMel:SampleRateMismatch'); multichannel input
    %   is mixed with toMono("mean"), recorded in Config.InputHistory.
    %   FMax = [] means fs/2.

    properties (SetAccess = immutable)
        SampleRate (1,1) double
        Grid OceanNLP.Speech.Core.FrameGrid
        NumMels (1,1) double
        FMin (1,1) double
        FMax (1,1) double
        MelScale (1,1) string
        MelNorm (1,1) string
        Power (1,1) double
        LogFloor (1,1) double
        Filters (:,:) double      % NumMels x (floor(NFFT/2)+1)
        CentersHz (:,1) double    % NumMels x 1
    end

    methods
        function obj = LogMel(sampleRate, opts)
            arguments
                sampleRate (1,1) double {mustBePositive, mustBeFinite}
                opts.WindowLength double {mustBeScalarOrEmpty} = []
                opts.HopLength double {mustBeScalarOrEmpty} = []
                opts.NFFT double {mustBeScalarOrEmpty} = []
                opts.Window (1,1) string = "hann"
                opts.Periodic (1,1) logical = true
                opts.Center (1,1) logical = false
                opts.PadMode (1,1) string = "constant"
                opts.NumMels (1,1) double {mustBeInteger, mustBePositive} = 40
                opts.FMin (1,1) double {mustBeNonnegative} = 0
                opts.FMax double {mustBeScalarOrEmpty} = []
                opts.MelScale (1,1) string {mustBeMember(opts.MelScale, ["htk", "slaney"])} = "htk"
                opts.MelNorm (1,1) string {mustBeMember(opts.MelNorm, ["none", "slaney"])} = "none"
                opts.Power (1,1) double {mustBePositive} = 2
                opts.LogFloor (1,1) double {mustBePositive} = 1e-10
            end
            obj.SampleRate = sampleRate;
            obj.Grid = OceanNLP.Speech.Core.FrameGrid.fromOptions(sampleRate, opts);
            fmax = opts.FMax;
            if isempty(fmax)
                fmax = sampleRate / 2;
            end
            obj.NumMels = opts.NumMels;
            obj.FMin = opts.FMin;
            obj.FMax = fmax;
            obj.MelScale = opts.MelScale;
            obj.MelNorm = opts.MelNorm;
            obj.Power = opts.Power;
            obj.LogFloor = opts.LogFloor;
            [obj.Filters, obj.CentersHz] = OceanMath.Signal.MelFilterBank.build(opts.NumMels, ...
                obj.Grid.NFFT, sampleRate, 'FMin', opts.FMin, 'FMax', fmax, ...
                'Scale', opts.MelScale, 'Norm', opts.MelNorm);
        end

        function fs = forward(obj, input)
            arguments
                obj
                input
            end
            st = obj.intermediates(input);
            fs = OceanNLP.Speech.Core.FeatureSequence(st.LogMel, 'FrameTimes', st.FrameTimes, ...
                'FrequencyHz', obj.CentersHz, 'AxisNames', ["mel", "time"], 'Units', obj.units(), ...
                'Config', obj.recordedConfig(st.Waveform), 'Source', st.Waveform.Source);
        end

        function st = intermediates(obj, input)
            % INTERMEDIATES  Every stage of the computation on this waveform (no recomputation
            %   elsewhere): Waveform (mono), Frames (L x T raw), Window, PowerSpectrum
            %   (|X|^Power), MelEnergies, LogMel, FrameTimes, BinFrequencies.
            arguments
                obj
                input
            end
            [x, w] = prepareWaveform(input, obj.SampleRate, "LogMel");
            % Shape flow: x Sx1 -> frames LxT -> spectrum KxT -> mel MxT.
            % Columns retain the same frame identity throughout; only the row
            % coordinate changes from samples, to FFT bins, to mel bands.
            frames = obj.Grid.frame(x);
            win = obj.Grid.window();
            % Implicit expansion applies the same Lx1 window to every frame column.
            P = OceanMath.Signal.SpectralAnalyzer.framePowerSpectrum(frames .* win, obj.Grid.NFFT);
            if obj.Power ~= 2
                % framePowerSpectrum returns |X|^2.  This conversion yields
                % (|X|^2)^(Power/2) = |X|^Power without recomputing the FFT.
                P = P .^ (obj.Power / 2);
            end
            % Filters is MxK and P is KxT.  Each row of Filters is one
            % overlapping mel band, so E(m,t) is band m's energy in frame t.
            E = obj.Filters * P;
            % Flooring is part of the numerical contract: digital silence maps
            % to log(LogFloor), never -Inf, so it is safe to batch and train on.
            st = struct('Waveform', w, 'Frames', frames, 'Window', win, 'PowerSpectrum', P, ...
                'MelEnergies', E, 'LogMel', log(max(E, obj.LogFloor)), ...
                'FrameTimes', obj.Grid.frameTimes(numel(x)), 'BinFrequencies', obj.Grid.binFrequencies());
        end

        function cfg = config(obj)
            cfg = struct('Type', "LogMel", 'SampleRate', obj.SampleRate, 'Grid', obj.Grid.config(), ...
                'NumMels', obj.NumMels, 'FMin', obj.FMin, 'FMax', obj.FMax, 'MelScale', obj.MelScale, ...
                'MelNorm', obj.MelNorm, 'Power', obj.Power, 'LogFloor', obj.LogFloor);
        end

        function d = outputDim(obj, inputDim)
            arguments
                obj
                inputDim double = [] %#ok<INUSA> fixed output size
            end
            d = obj.NumMels;
        end
    end

    methods (Access = private)
        function u = units(obj)
            if obj.Power == 2
                u = "log power";
            elseif obj.Power == 1
                u = "log magnitude";
            else
                u = sprintf("log |X|^%g", obj.Power);
            end
        end

        function cfg = recordedConfig(obj, w)
            cfg = obj.config();
            cfg.InputHistory = w.History;
        end
    end

    methods (Static)
        function f = fromConfig(cfg)
            % FROMCONFIG  Inverse of config(): rebuild the transform (config equality is tested).
            arguments
                cfg (1,1) struct
            end
            g = gridArgs(cfg.Grid);
            f = OceanNLP.Speech.Features.LogMel(cfg.SampleRate, g{:}, ...
                'NumMels', cfg.NumMels, 'FMin', cfg.FMin, 'FMax', cfg.FMax, 'MelScale', cfg.MelScale, ...
                'MelNorm', cfg.MelNorm, 'Power', cfg.Power, 'LogFloor', cfg.LogFloor);
        end
    end
end
