classdef MFCC < OceanNLP.Speech.Core.IFeatureTransform
    % MFCC  Mel-frequency cepstral coefficients = DCT-II (orthonormal) of the log-mel spectrum.
    %
    %   t  = OceanNLP.Speech.Features.MFCC(fs)
    %   t  = OceanNLP.Speech.Features.MFCC(fs, 'NumCoeffs', 13, 'Lifter', 0, 'NumMels', 40, ...
    %          <every LogMel / FrameGrid option>)
    %   fs = t.forward(waveform)       % FeatureSequence, NumCoeffs x T, axes ["cepstral","time"]
    %
    %   COMPUTATION (Davis & Mermelstein 1980):
    %     Y = LogMel(waveform)                           (NumMels x T, natural log)
    %     C(k, t) = s_k sqrt(2/M) sum_{m=0}^{M-1} Y(m, t) cos(pi k (2m + 1) / (2M)),
    %               k = 0..NumCoeffs-1, s_0 = 1/sqrt(2), s_k = 1      (scipy dct norm='ortho')
    %     C'(k) = (1 + (Lift/2) sin(pi k / Lift)) C(k)  (row index k = 1..NumCoeffs; Lift = 0: none)
    %   Row 1 is c0 (proportional to the mean log mel energy; no separate log-energy
    %   replacement). Matches librosa.feature.mfcc(S=log-mel, dct_type=2, norm='ortho',
    %   lifter=Lift) when S is the natural-log mel spectrogram.
    %
    %   Same input policy as LogMel (sample-rate check, mono mixdown recorded).

    properties (SetAccess = immutable)
        NumCoeffs (1,1) double
        Lifter (1,1) double
        LogMelTransform OceanNLP.Speech.Features.LogMel
    end

    methods
        function obj = MFCC(sampleRate, opts)
            arguments
                sampleRate (1,1) double {mustBePositive, mustBeFinite}
                opts.NumCoeffs (1,1) double {mustBeInteger, mustBePositive} = 13
                opts.Lifter (1,1) double {mustBeNonnegative} = 0
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
                opts.MelScale (1,1) string = "htk"
                opts.MelNorm (1,1) string = "none"
                opts.Power (1,1) double {mustBePositive} = 2
                opts.LogFloor (1,1) double {mustBePositive} = 1e-10
            end
            if opts.NumCoeffs > opts.NumMels
                error('OceanNLP:Speech:Features:MFCC:TooManyCoeffs', ...
                    'NumCoeffs (%d) cannot exceed NumMels (%d).', opts.NumCoeffs, opts.NumMels);
            end
            obj.NumCoeffs = opts.NumCoeffs;
            obj.Lifter = opts.Lifter;
            % Forward every non-MFCC option to the owned LogMel transform.  This
            % keeps framing/mel conventions in one implementation instead of
            % duplicating a subtly different acoustic front end here.
            lm = rmfield(opts, ["NumCoeffs", "Lifter"]);
            args = namedargs2cell(lm);
            obj.LogMelTransform = OceanNLP.Speech.Features.LogMel(sampleRate, args{:});
        end

        function fs = forward(obj, input)
            arguments
                obj
                input
            end
            st = obj.LogMelTransform.intermediates(input);
            % The DCT acts down the mel rows independently for every time column:
            % NumMels x T -> NumCoeffs x T.  FrameTimes therefore do not change.
            C = OceanMath.Signal.CepstralTransform.dct(st.LogMel, obj.NumCoeffs, 'Norm', "ortho");
            % Liftering only rescales coefficient rows; Lifter=0 is the identity.
            C = OceanMath.Signal.CepstralTransform.lifter(C, obj.Lifter);
            cfg = obj.config();
            cfg.InputHistory = st.Waveform.History;
            % Cepstral rows combine every mel band and therefore have no
            % single physical FrequencyHz coordinate.
            fs = OceanNLP.Speech.Core.FeatureSequence(C, 'FrameTimes', st.FrameTimes, ...
                'AxisNames', ["cepstral", "time"], 'Units', "cepstral coefficient (log power)", ...
                'Config', cfg, 'Source', st.Waveform.Source);
        end

        function cfg = config(obj)
            cfg = struct('Type', "MFCC", 'NumCoeffs', obj.NumCoeffs, 'Lifter', obj.Lifter, ...
                'DCT', "II-ortho", 'LogMel', obj.LogMelTransform.config());
        end

        function d = outputDim(obj, inputDim)
            arguments
                obj
                inputDim double = [] %#ok<INUSA> fixed output size
            end
            d = obj.NumCoeffs;
        end
    end

    methods (Static)
        function f = fromConfig(cfg)
            % FROMCONFIG  Inverse of config().
            arguments
                cfg (1,1) struct
            end
            c = cfg.LogMel;
            g = gridArgs(c.Grid);
            f = OceanNLP.Speech.Features.MFCC(c.SampleRate, 'NumCoeffs', cfg.NumCoeffs, 'Lifter', cfg.Lifter, ...
                g{:}, 'NumMels', c.NumMels, 'FMin', c.FMin, 'FMax', c.FMax, ...
                'MelScale', c.MelScale, 'MelNorm', c.MelNorm, 'Power', c.Power, 'LogFloor', c.LogFloor);
        end
    end
end
