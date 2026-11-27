classdef ConformerEncoder < OceanDL.nn.Module
    % CONFORMERENCODER  Convolutional subsampling + a stack of Conformer blocks
    %   (Gulati et al. 2020) for padded speech feature batches, with exact length propagation.
    %
    %   enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(inputDim, 'Dim', 144, 'NumHeads', 4, ...
    %           'NumBlocks', 4, 'FFMult', 4, 'ConvKernel', 15, 'SubsampleStrides', [2 2], ...
    %           'SubsampleKernel', 3, 'PosEnc', "rope", 'Dropout', 0.1, 'AttnDropout', 0, 'Seed', 0)
    %   [H, lens] = enc.forward(X, lens)     X: F x N x T (Tensor or array), lens 1 x N
    %                                        H: Dim x N x T' Tensor, lens = outputLength(lens)
    %   lens2 = enc.outputLength(lens);  D = enc.OutputDim;  cfg = enc.config()
    %   enc2  = ConformerEncoder.fromConfig(cfg)   (same architecture and seeds; weights via loadStateDict)
    %
    %   COMPUTATION, Z_0 = M_0 .* X:
    %     subsampling layer l = 1..S (stride s_l, kernel k, padding p = floor(k/2)):
    %       Z_l = M_l .* ReLU(Conv1d(Z_{l-1})),  T_l(n) = floor((T_{l-1}(n) + 2p - k)/s_l) + 1
    %       (contract sec. 3 Conv1d length formula; T(n) = 0 stays 0)
    %     H_0 = M_S .* Drop(W_in Z_S + b_in [+ PE_abs])        Linear Dim -> Dim
    %     H_b = ConformerBlock_b(H_{b-1}; key mask M_S),  b = 1..B
    %   M_l(n, t) = [t <= T_l(n)].
    %   Subsampling uses 1-D convolutions over time (feature axis = channels) instead of
    %   the paper's 2-D convolutions over (time, frequency): same 4x time reduction with
    %   fewer operations on CPU (documented deviation). Total reduction prod(s_l).
    %
    %   POSITIONS (PosEnc): "rope" (default) -- rotary embeddings inside every block's
    %   self-attention (relative, as the paper's relative positional MHSA; see
    %   OceanDL.Transformer.ConformerBlock); "abs" -- the fixed sinusoidal encoding of
    %   Vaswani et al. (OceanDL.Transformer.PositionalEncoding.sincos1d) added once after
    %   the input projection, blocks without RoPE.
    %
    %   PADDING NEVER LEAKS: masked input, re-masked after each subsampling conv (as in
    %   ConvEncoder), key masks + pre-depthwise-conv masking inside the blocks; the valid
    %   outputs of a padded batch equal the per-utterance outputs (tested).
    %   Initialisation: PyTorch defaults; sub-module seeds derived from Seed.

    properties (SetAccess = protected)
        Subsample cell = {}
        InputProj
        Blocks cell = {}
        Drop
        InputDim (1,1) double
        Dim (1,1) double
        NumHeads (1,1) double
        NumBlocks (1,1) double
        FFMult (1,1) double
        ConvKernel (1,1) double
        SubsampleStrides (1,:) double
        SubsampleKernel (1,1) double
        PosEnc (1,1) string
        DropoutP (1,1) double
        AttnDropoutP (1,1) double
        Seed (1,1) double
    end

    properties (Dependent)
        OutputDim
    end

    methods
        function obj = ConformerEncoder(inputDim, opts)
            arguments
                inputDim (1,1) double {mustBePositive, mustBeInteger}
                opts.Dim (1,1) double {mustBePositive, mustBeInteger} = 144
                opts.NumHeads (1,1) double {mustBePositive, mustBeInteger} = 4
                opts.NumBlocks (1,1) double {mustBeNonnegative, mustBeInteger} = 4
                opts.FFMult (1,1) double {mustBePositive} = 4
                opts.ConvKernel (1,1) double {mustBePositive, mustBeInteger} = 15
                opts.SubsampleStrides (1,:) double {mustBePositive, mustBeInteger} = [2 2]
                opts.SubsampleKernel (1,1) double {mustBePositive, mustBeInteger} = 3
                opts.PosEnc (1,1) string {mustBeMember(opts.PosEnc, ["rope", "abs"])} = "rope"
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.AttnDropout (1,1) double {mustBeGreaterThanOrEqual(opts.AttnDropout, 0), mustBeLessThan(opts.AttnDropout, 1)} = 0
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            if isempty(opts.SubsampleStrides)
                error('OceanNLP:Speech:Models:ConformerEncoder:Subsample', 'SubsampleStrides needs at least one layer.');
            end
            if opts.PosEnc == "abs" && mod(opts.Dim, 2)
                error('OceanNLP:Speech:Models:ConformerEncoder:OddDim', 'Sinusoidal positions need an even Dim (got %d).', opts.Dim);
            end
            if mod(opts.Dim, opts.NumHeads)
                error('OceanNLP:Speech:Models:ConformerEncoder:HeadSplit', 'Dim %d is not divisible by NumHeads %d.', opts.Dim, opts.NumHeads);
            end
            obj.InputDim = inputDim;
            obj.Dim = opts.Dim;
            obj.NumHeads = opts.NumHeads;
            obj.NumBlocks = opts.NumBlocks;
            obj.FFMult = opts.FFMult;
            obj.ConvKernel = opts.ConvKernel;
            obj.SubsampleStrides = opts.SubsampleStrides;
            obj.SubsampleKernel = opts.SubsampleKernel;
            obj.PosEnc = opts.PosEnc;
            obj.DropoutP = opts.Dropout;
            obj.AttnDropoutP = opts.AttnDropout;
            obj.Seed = opts.Seed;
            k = opts.SubsampleKernel;
            cin = inputDim;
            obj.Subsample = cell(1, numel(opts.SubsampleStrides));
            for l = 1:numel(opts.SubsampleStrides)
                obj.Subsample{l} = OceanDL.nn.Conv1d(cin, opts.Dim, k, 'Stride', opts.SubsampleStrides(l), ...
                    'Padding', floor(k / 2), 'Seed', opts.Seed + l - 1);
                cin = opts.Dim;
            end
            obj.InputProj = OceanDL.nn.Linear(opts.Dim, opts.Dim, 'Seed', opts.Seed + 50);
            blockPos = "rope";
            if opts.PosEnc == "abs", blockPos = "none"; end
            obj.Blocks = cell(1, opts.NumBlocks);
            for b = 1:opts.NumBlocks
                obj.Blocks{b} = OceanDL.Transformer.ConformerBlock(opts.Dim, opts.NumHeads, 'FFMult', opts.FFMult, ...
                    'ConvKernel', opts.ConvKernel, 'Dropout', opts.Dropout, 'AttnDropout', opts.AttnDropout, ...
                    'PosEnc', blockPos, 'Seed', opts.Seed + 100 * b);
            end
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
        end

        function d = get.OutputDim(obj)
            d = obj.Dim;
        end

        function [H, lens] = forward(obj, X, lens)
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
            end
            [F, N, T] = size(X, 1:3);
            if F ~= obj.InputDim
                error('OceanNLP:Speech:Models:ConformerEncoder:InputDim', ...
                    'Expected %d input features (F x N x T), got %d.', obj.InputDim, F);
            end
            if numel(lens) ~= N || any(lens > T)
                error('OceanNLP:Speech:Models:ConformerEncoder:Lengths', 'lens must hold N = %d lengths <= T = %d.', N, T);
            end
            SM = OceanDL.Layout.SequenceMask;
            H = SM.apply(OceanDL.Autograd.Tensor.wrap(X), SM.fromLengths(lens, T));
            for l = 1:numel(obj.Subsample)
                H = OceanDL.Autograd.F.relu(obj.Subsample{l}.forward(H));
                lens = obj.Subsample{l}.outputLength(lens);
                H = SM.apply(H, SM.fromLengths(lens, size(H, 3)));
            end
            Tp = size(H, 3);
            H = obj.InputProj.forward(H);
            if obj.PosEnc == "abs" && Tp > 0
                H = H + OceanDL.Transformer.PositionalEncoding.sincos1d(Tp, obj.Dim);
            end
            M = SM.fromLengths(lens, Tp);
            H = SM.apply(obj.Drop.forward(H), M);
            for b = 1:numel(obj.Blocks)
                H = obj.Blocks{b}.forward(H, 'KeyMask', M);
            end
        end

        function lens = outputLength(obj, lens)
            arguments
                obj
                lens double {mustBeNonnegative, mustBeInteger}
            end
            for l = 1:numel(obj.Subsample)
                lens = obj.Subsample{l}.outputLength(lens);
            end
        end

        function cfg = config(obj)
            cfg = struct('Type', "ConformerEncoder", 'InputDim', obj.InputDim, 'Dim', obj.Dim, ...
                'NumHeads', obj.NumHeads, 'NumBlocks', obj.NumBlocks, 'FFMult', obj.FFMult, ...
                'ConvKernel', obj.ConvKernel, 'SubsampleStrides', obj.SubsampleStrides, ...
                'SubsampleKernel', obj.SubsampleKernel, 'PosEnc', obj.PosEnc, 'Dropout', obj.DropoutP, ...
                'AttnDropout', obj.AttnDropoutP, 'Seed', obj.Seed, 'OutputDim', obj.OutputDim);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if string(cfg.Type) ~= "ConformerEncoder"
                error('OceanNLP:Speech:Models:ConformerEncoder:ConfigType', 'Config Type is "%s", not "ConformerEncoder".', string(cfg.Type));
            end
            obj = OceanNLP.Speech.Models.Encoders.ConformerEncoder(cfg.InputDim, 'Dim', cfg.Dim, 'NumHeads', cfg.NumHeads, ...
                'NumBlocks', cfg.NumBlocks, 'FFMult', cfg.FFMult, 'ConvKernel', cfg.ConvKernel, ...
                'SubsampleStrides', reshape(cfg.SubsampleStrides, 1, []), 'SubsampleKernel', cfg.SubsampleKernel, ...
                'PosEnc', cfg.PosEnc, 'Dropout', cfg.Dropout, 'AttnDropout', cfg.AttnDropout, 'Seed', cfg.Seed);
        end
    end
end
