classdef ConvEncoder < OceanDL.nn.Module
    % CONVENCODER  Stack of 1-D convolutions over time (optional stride subsampling)
    %   for padded speech feature batches, with exact length propagation.
    %
    %   enc = OceanNLP.Speech.Models.Encoders.ConvEncoder(inputDim, 'Channels', [128 128], ...
    %           'KernelSizes', [5 3], 'Strides', [2 1], 'Activation', "gelu", ...
    %           'LayerNorm', true, 'Dropout', 0, 'Seed', 0)
    %   [H, lens] = enc.forward(X, lens)    X: F x N x T (Tensor or array), lens: 1 x N
    %   lens2 = enc.outputLength(lens);  D = enc.OutputDim;  cfg = enc.config()
    %   enc2 = ConvEncoder.fromConfig(cfg)  (same architecture; weights via loadStateDict)
    %
    %   LAYER l (l = 1..L), input Z_{l-1} (C_{l-1} x N x T_{l-1}), Z_0 = X:
    %     A_l   = Conv1d(Z_{l-1}; k_l, stride s_l, padding p_l = floor(k_l/2))
    %     B_l   = LayerNorm(A_l)  over channels, per frame          (if LayerNorm)
    %     Z_l   = M_l .* Dropout(act(B_l))                          act = GELU | ReLU
    %     M_l(n,t) = [t <= T_l(n)],  T_l(n) = floor((T_{l-1}(n) + 2p_l - k_l)/s_l) + 1
    %   (contract sec. 3 Conv1d length formula, dilation 1; T(n) = 0 stays 0).
    %
    %   PADDING NEVER LEAKS: the input is masked first, and every layer's output is
    %   re-masked, so frames past T_l(n) are exactly zero when the next convolution
    %   reads them -- the same zeros the convolution's own zero padding supplies when
    %   utterance n is run alone. A valid output frame t <= T_l(n) reads input frames
    %   up to (t-1)s_l - p_l + k_l <= T_{l-1}(n) + p_l, i.e. only valid frames or zeros.
    %   Hence the valid outputs of a padded batch equal the per-utterance outputs
    %   (tested to floating-point round-off).
    %
    %   Initialisation: Conv1d/LayerNorm PyTorch defaults; layer l draws from
    %   RandStream seeds Seed + l - 1 (deterministic, independent of the global RNG).

    properties (SetAccess = protected)
        Convs cell = {}
        Norms cell = {}
        Drop
        InputDim (1,1) double
        Channels (1,:) double
        KernelSizes (1,:) double
        Strides (1,:) double
        Activation (1,1) string
        UseLayerNorm (1,1) logical
        DropoutP (1,1) double
        Seed (1,1) double
    end

    properties (Dependent)
        OutputDim
    end

    methods
        function obj = ConvEncoder(inputDim, opts)
            arguments
                inputDim (1,1) double {mustBePositive, mustBeInteger}
                opts.Channels (1,:) double {mustBePositive, mustBeInteger} = [128 128]
                opts.KernelSizes (1,:) double {mustBePositive, mustBeInteger} = [5 3]
                opts.Strides (1,:) double {mustBePositive, mustBeInteger} = [2 1]
                opts.Activation (1,1) string {mustBeMember(opts.Activation, ["gelu", "relu"])} = "gelu"
                opts.LayerNorm (1,1) logical = true
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            L = numel(opts.Channels);
            if L == 0 || numel(opts.KernelSizes) ~= L || numel(opts.Strides) ~= L
                error('OceanNLP:Speech:Models:ConvEncoder:LayerSpec', ...
                    'Channels, KernelSizes and Strides must have the same nonzero length (got %d, %d, %d).', ...
                    L, numel(opts.KernelSizes), numel(opts.Strides));
            end
            obj.InputDim = inputDim;
            obj.Channels = opts.Channels;
            obj.KernelSizes = opts.KernelSizes;
            obj.Strides = opts.Strides;
            obj.Activation = opts.Activation;
            obj.UseLayerNorm = opts.LayerNorm;
            obj.DropoutP = opts.Dropout;
            obj.Seed = opts.Seed;
            cin = inputDim;
            obj.Convs = cell(1, L);
            obj.Norms = cell(1, 0);
            for l = 1:L
                k = opts.KernelSizes(l);
                obj.Convs{l} = OceanDL.nn.Conv1d(cin, opts.Channels(l), k, 'Stride', opts.Strides(l), ...
                    'Padding', floor(k / 2), 'Seed', opts.Seed + l - 1);
                if opts.LayerNorm
                    obj.Norms{l} = OceanDL.nn.LayerNorm(opts.Channels(l));
                end
                cin = opts.Channels(l);
            end
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
        end

        function d = get.OutputDim(obj)
            d = obj.Channels(end);
        end

        function [H, lens] = forward(obj, X, lens)
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
            end
            [F, N, T] = size(X, 1:3);
            if F ~= obj.InputDim
                error('OceanNLP:Speech:Models:ConvEncoder:InputDim', ...
                    'Expected %d input features (F x N x T), got %d.', obj.InputDim, F);
            end
            if numel(lens) ~= N || any(lens > T)
                error('OceanNLP:Speech:Models:ConvEncoder:Lengths', ...
                    'lens must hold N = %d lengths <= T = %d.', N, T);
            end
            H = OceanDL.Layout.SequenceMask.apply(OceanDL.Autograd.Tensor.wrap(X), ...
                OceanDL.Layout.SequenceMask.fromLengths(lens, T));
            for l = 1:numel(obj.Convs)
                H = obj.Convs{l}.forward(H);
                lens = obj.Convs{l}.outputLength(lens);
                if obj.UseLayerNorm
                    H = obj.Norms{l}.forward(H);
                end
                if obj.Activation == "gelu"
                    H = OceanDL.Autograd.F.gelu(H);
                else
                    H = OceanDL.Autograd.F.relu(H);
                end
                H = obj.Drop.forward(H);
                H = OceanDL.Layout.SequenceMask.apply(H, ...
                    OceanDL.Layout.SequenceMask.fromLengths(lens, size(H, 3)));
            end
        end

        function lens = outputLength(obj, lens)
            arguments
                obj
                lens double {mustBeNonnegative, mustBeInteger}
            end
            for l = 1:numel(obj.Convs)
                lens = obj.Convs{l}.outputLength(lens);
            end
        end

        function cfg = config(obj)
            cfg = struct('Type', "ConvEncoder", 'InputDim', obj.InputDim, 'Channels', obj.Channels, ...
                'KernelSizes', obj.KernelSizes, 'Strides', obj.Strides, 'Activation', obj.Activation, ...
                'LayerNorm', obj.UseLayerNorm, 'Dropout', obj.DropoutP, 'Seed', obj.Seed, ...
                'OutputDim', obj.OutputDim);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if string(cfg.Type) ~= "ConvEncoder"
                error('OceanNLP:Speech:Models:ConvEncoder:ConfigType', 'Config Type is "%s", not "ConvEncoder".', string(cfg.Type));
            end
            obj = OceanNLP.Speech.Models.Encoders.ConvEncoder(cfg.InputDim, 'Channels', cfg.Channels, ...
                'KernelSizes', cfg.KernelSizes, 'Strides', cfg.Strides, 'Activation', cfg.Activation, ...
                'LayerNorm', cfg.LayerNorm, 'Dropout', cfg.Dropout, 'Seed', cfg.Seed);
        end
    end
end
