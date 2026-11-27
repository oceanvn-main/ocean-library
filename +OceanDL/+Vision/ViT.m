classdef ViT < OceanDL.nn.Module
    % VIT  Vision Transformer (Dosovitskiy et al. 2021; Szeliski 2nd ed., Sec. 5.5.3).
    %
    %   image (H x W x C x N)
    %     -> PatchEmbedding: T = (H/p)(W/p) tokens of width D          (D x N x T)
    %     -> prepend a learned [CLS] token                             (D x N x (T+1))
    %     -> + position information: "learned" | "sincos" (2-D, fixed) | "none"
    %     -> TransformerEncoder (Depth blocks, Pre-LN default, final LayerNorm)
    %     -> summary: the [CLS] output ("cls") or the mean of patch tokens ("mean")
    %     -> Linear head -> logits (NumClasses x N)
    %
    %   Resolution is NOT fixed by the architecture: with "sincos" or "none"
    %   positions the same model runs on any H, W divisible by the patch size
    %   (positions are computed for the actual grid). "learned" positions are
    %   sized for ImageSize and error on other sizes.
    %
    %   vit = OceanDL.Vision.ViT('ImageSize', [48 48], 'Channels', 3, 'PatchSize', 6, ...
    %             'Dim', 96, 'Depth', 6, 'Heads', 4, 'NumClasses', 8);
    %   logits = vit.forward(X);

    properties (SetAccess = protected)
        Patch
        ClsToken                % Parameter (D x 1 x 1), [] when Pooling = "mean"
        PosEmbed                % LearnedPositionalEmbedding or []
        Encoder
        Head
        PosMode (1,1) string
        Pooling (1,1) string
        ImageSize (1,2) double
        PosDropout (1,1) double
    end

    methods
        function obj = ViT(opts)
            arguments
                opts.ImageSize (1,2) double {mustBePositive, mustBeInteger} = [48 48]
                opts.Channels (1,1) double {mustBePositive, mustBeInteger} = 3
                opts.PatchSize (1,:) double {mustBePositive, mustBeInteger} = 6
                opts.Dim (1,1) double {mustBePositive, mustBeInteger} = 96
                opts.Depth (1,1) double {mustBePositive, mustBeInteger} = 6
                opts.Heads (1,1) double {mustBePositive, mustBeInteger} = 4
                opts.MLPRatio (1,1) double {mustBePositive} = 4
                opts.NumClasses (1,1) double {mustBePositive, mustBeInteger} = 10
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["pre", "post"])} = "pre"
                opts.Positions (1,1) string {mustBeMember(opts.Positions, ["learned", "sincos", "none"])} = "learned"
                opts.Pooling (1,1) string {mustBeMember(opts.Pooling, ["cls", "mean"])} = "cls"
                opts.Dropout (1,1) double = 0
                opts.AttnDropout (1,1) double = 0
                opts.DropPath (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            seed = @(k) OceanDL.Vision.ViT.offset(opts.Seed, k);
            D = opts.Dim;
            obj.ImageSize = opts.ImageSize;
            obj.PosMode = opts.Positions;
            obj.Pooling = opts.Pooling;
            obj.PosDropout = opts.Dropout;
            obj.Patch = OceanDL.Vision.PatchEmbedding(opts.Channels, D, opts.PatchSize, 'Seed', seed(1), 'DataType', opts.DataType);
            T = prod(opts.ImageSize ./ (opts.PatchSize .* [1 1]));
            if any(mod(opts.ImageSize, opts.PatchSize .* [1 1]))
                error('OceanDL:Vision:ViT:PatchSize', 'ImageSize must be divisible by PatchSize.');
            end
            extra = 0;
            if opts.Pooling == "cls"
                obj.ClsToken = OceanDL.nn.Parameter(cast(0.02 * OceanDL.Vision.ViT.randnSeeded(seed(2), D, 1, 1), opts.DataType));
                extra = 1;
            end
            if opts.Positions == "learned"
                obj.PosEmbed = OceanDL.Transformer.LearnedPositionalEmbedding(T + extra, D, 'Seed', seed(3), 'DataType', opts.DataType);
            end
            obj.Encoder = OceanDL.Transformer.TransformerEncoder(opts.Depth, D, opts.Heads, round(opts.MLPRatio * D), ...
                'Norm', opts.Norm, 'Dropout', opts.Dropout, 'AttnDropout', opts.AttnDropout, ...
                'DropPath', opts.DropPath, 'FinalNorm', true, 'Seed', seed(4), 'DataType', opts.DataType);
            obj.Head = OceanDL.nn.Linear(D, opts.NumClasses, 'Seed', seed(5), 'DataType', opts.DataType);
        end

        function [logits, tokens] = forward(obj, X)
            [tokens, grid] = obj.embed(X);
            tokens = obj.Encoder.forward(tokens);
            if obj.Pooling == "cls"
                summary = tokens(:, :, 1);
            else
                summary = mean(tokens, 3);
            end
            [D, N] = size(summary, 1:2);
            logits = obj.Head.forward(reshape(summary, D, N));
            if nargout > 1
                tokens = struct('Tokens', tokens, 'Grid', grid, 'HasCls', obj.Pooling == "cls");
            end
        end

        function [tokens, grid] = embed(obj, X)
            % Patch tokens (+ [CLS]) with positions added -- the encoder input.
            [tokens, grid] = obj.Patch.forward(X);
            [D, N, ~] = size(tokens, 1:3);
            if obj.Pooling == "cls"
                cls = obj.ClsToken + zeros(D, N, 1, 'like', OceanDL.Autograd.Tensor.raw(tokens));   % broadcast to the batch
                tokens = cat(3, cls, tokens);
            end
            switch obj.PosMode
                case "learned"
                    tokens = obj.PosEmbed.forward(tokens);
                case "sincos"
                    P = OceanDL.Transformer.PositionalEncoding.sincos2d(grid(1), grid(2), D);
                    if obj.Pooling == "cls"
                        P = cat(3, zeros(D, 1, 1), P);                  % [CLS] carries no position
                    end
                    tokens = tokens + cast(P, 'like', OceanDL.Autograd.Tensor.raw(tokens));
            end
            tokens = OceanDL.Autograd.F.dropout(tokens, obj.PosDropout, obj.Training);
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end

        function W = randnSeeded(seed, varargin)
            if isempty(seed), W = randn(varargin{:});
            else, W = randn(RandStream('mt19937ar', 'Seed', seed), varargin{:}); end
        end
    end
end
