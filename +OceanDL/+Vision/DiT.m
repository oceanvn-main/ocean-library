classdef DiT < OceanDL.nn.Module
    % DIT  Diffusion Transformer (Peebles & Xie 2023): a ViT over latent or
    %   pixel patches as the denoiser, forward(x, t, y) -> same shape as x.
    %
    %   tokens = PatchEmbedding(x) [+ fixed 2-D sincos positions]   (D x N x T)
    %   c      = TimestepEmbedding(t) [+ ClassEmbed(y)]             (D x N)
    %   Depth x DiTBlock(tokens, c)                                (adaLN-Zero)
    %   final: Norm -> adaLN (shift, scale) -> Linear(D -> p p C_out), both
    %          ZERO-initialised -> unpatchify                        (H x W x C_out x N)
    %
    %   Modern options (SD3 / FLUX / Qwen-Image era): 'Positions' "rope"
    %   (2-D axial rotary instead of additive sincos), 'Norm' "rms",
    %   'FFN' "swiglu", 'QKNorm' "rms". Class label K + 1 (or y = []) is the
    %   NULL label for classifier-free guidance.

    properties (SetAccess = protected)
        Patch
        TimeEmbed
        ClassEmbed
        Blocks cell = {}
        FinalAdaLN
        FinalLinear
        NumClasses (1,1) double
        Positions (1,1) string
        NormKind (1,1) string
        OutChannels (1,1) double
        Heads (1,1) double
    end

    methods
        function obj = DiT(opts)
            arguments
                opts.InChannels (1,1) double = 4
                opts.OutChannels (1,1) double = 0
                opts.PatchSize (1,1) double = 2
                opts.Dim (1,1) double = 64
                opts.Depth (1,1) double = 4
                opts.Heads (1,1) double = 4
                opts.MLPRatio (1,1) double = 4
                opts.NumClasses (1,1) double = 0
                opts.Positions (1,1) string {mustBeMember(opts.Positions, ["sincos", "rope"])} = "sincos"
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["layer", "rms"])} = "layer"
                opts.FFN (1,1) string {mustBeMember(opts.FFN, ["gelu", "swiglu"])} = "gelu"
                opts.QKNorm (1,1) string {mustBeMember(opts.QKNorm, ["none", "rms", "layer"])} = "none"
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            dt = {'DataType', opts.DataType};
            D = opts.Dim;
            obj.OutChannels = opts.OutChannels; if obj.OutChannels == 0, obj.OutChannels = opts.InChannels; end
            obj.NumClasses = opts.NumClasses; obj.Positions = opts.Positions; obj.NormKind = opts.Norm; obj.Heads = opts.Heads;
            obj.Patch = OceanDL.Vision.PatchEmbedding(opts.InChannels, D, opts.PatchSize, 'Seed', s(1), dt{:});
            obj.TimeEmbed = OceanDL.nn.TimestepEmbedding(D, 'Seed', s(2), dt{:});
            if opts.NumClasses > 0
                obj.ClassEmbed = OceanDL.nn.Embedding(opts.NumClasses + 1, D, 'Seed', s(3), dt{:});
            end
            for b = 1:opts.Depth
                obj.Blocks{b} = OceanDL.Vision.DiTBlock(D, opts.Heads, 'MLPRatio', opts.MLPRatio, 'Norm', opts.Norm, ...
                    'FFN', opts.FFN, 'QKNorm', opts.QKNorm, 'Seed', s(100 * b), dt{:});
            end
            [obj.FinalAdaLN, obj.FinalLinear] = OceanDL.Vision.DiT.finalLayer(D, prod(obj.Patch.PatchSize) * obj.OutChannels, s(9000), dt);
        end

        function y = forward(obj, x, t, labels)
            if nargin < 4, labels = []; end
            [tok, grid] = obj.Patch.forward(x);
            [tok, rope] = OceanDL.Vision.DiT.addPositions(tok, grid, obj.Positions, obj.Heads);
            c = OceanDL.Vision.DiT.condition(obj.TimeEmbed, obj.ClassEmbed, obj.NumClasses, t, labels, size(x, 4));
            for b = 1:numel(obj.Blocks)
                tok = obj.Blocks{b}.forward(tok, c, rope);
            end
            y = OceanDL.Vision.DiT.finalProject(tok, c, obj.FinalAdaLN, obj.FinalLinear, obj.NormKind, ...
                obj.Patch.PatchSize, grid);
        end
    end

    methods (Static)
        function [ada, lin] = finalLayer(D, out, seed, dt)
            ada = OceanDL.nn.Linear(D, 2 * D, 'Seed', seed, dt{:});
            lin = OceanDL.nn.Linear(D, out, 'Seed', OceanDL.Vision.ConvEncoder.offset(seed, 1), dt{:});
            ada.Weight.Data(:) = 0; ada.Bias.Data(:) = 0;
            lin.Weight.Data(:) = 0; lin.Bias.Data(:) = 0;
        end

        function [tok, rope] = addPositions(tok, grid, kind, heads)
            rope = [];
            D = size(tok, 1);
            if kind == "sincos"
                P = OceanDL.Transformer.PositionalEncoding.sincos2d(grid(1), grid(2), D);
                tok = tok + cast(P, 'like', OceanDL.Autograd.Tensor.raw(tok));
            else
                rope = OceanDL.Transformer.RoPE.axial2d(grid(1), grid(2), D / heads);
            end
        end

        function c = condition(timeEmbed, classEmbed, K, t, labels, N)
            c = timeEmbed.forward(t);
            if ~isempty(classEmbed)
                if isempty(labels), labels = repmat(K + 1, N, 1); end
                c = c + classEmbed.forward(reshape(labels, N, 1));
            end
        end

        function y = finalProject(tok, c, ada, lin, normKind, p, grid)
            m = ada.forward(OceanDL.Autograd.F.silu(c));
            [sh, sc] = OceanDL.Vision.DiTBlock.chunks(m, size(tok, 1), 2);
            h = OceanDL.Vision.DiTBlock.modulate(OceanDL.Vision.DiTBlock.norm(tok, normKind), sh, sc);
            y = OceanDL.Vision.MaskedAutoencoder.unpatchify(lin.forward(h), p, grid(1), grid(2));
        end
    end
end
