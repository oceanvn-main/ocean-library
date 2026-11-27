classdef MMDiT < OceanDL.nn.Module
    % MMDIT  Multimodal Diffusion Transformer (Esser et al. 2024, SD3; FLUX;
    %   Qwen-Image), sized for class-conditional use in this library.
    %
    %   image tokens x = PatchEmbedding(image) [+ sincos]      (D x N x T)
    %   context tokens z = CtxEmbed(y) -> ContextTokens tokens (D x N x Tc)
    %     (in SD3 z is the TEXT sequence from T5 / CLIP; here a learned token
    %      sequence per class -- the same mechanism, with a label as "prompt";
    %      the NULL label K + 1 gives the unconditional context)
    %   pooled condition c = TimestepEmbedding(t) + ClassEmbed(y)   (D x N)
    %   Depth x MMDiTBlock(x, z, c);  final adaLN + Linear on x -> unpatchify.

    properties (SetAccess = protected)
        Patch
        TimeEmbed
        ClassEmbed
        CtxEmbed
        Blocks cell = {}
        FinalAdaLN
        FinalLinear
        NumClasses (1,1) double
        ContextTokens (1,1) double
        Positions (1,1) string
        NormKind (1,1) string
        OutChannels (1,1) double
        Heads (1,1) double
    end

    methods
        function obj = MMDiT(opts)
            arguments
                opts.InChannels (1,1) double = 4
                opts.OutChannels (1,1) double = 0
                opts.PatchSize (1,1) double = 2
                opts.Dim (1,1) double = 64
                opts.Depth (1,1) double = 4
                opts.Heads (1,1) double = 4
                opts.MLPRatio (1,1) double = 4
                opts.NumClasses (1,1) double = 0
                opts.ContextTokens (1,1) double {mustBePositive, mustBeInteger} = 4
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
            obj.NumClasses = opts.NumClasses; obj.ContextTokens = opts.ContextTokens;
            obj.Positions = opts.Positions; obj.NormKind = opts.Norm; obj.Heads = opts.Heads;
            obj.Patch = OceanDL.Vision.PatchEmbedding(opts.InChannels, D, opts.PatchSize, 'Seed', s(1), dt{:});
            obj.TimeEmbed = OceanDL.nn.TimestepEmbedding(D, 'Seed', s(2), dt{:});
            if opts.NumClasses > 0
                obj.ClassEmbed = OceanDL.nn.Embedding(opts.NumClasses + 1, D, 'Seed', s(3), dt{:});
            end
            obj.CtxEmbed = OceanDL.nn.Embedding(opts.NumClasses + 1, D * opts.ContextTokens, 'Seed', s(4), dt{:});
            for b = 1:opts.Depth
                obj.Blocks{b} = OceanDL.Vision.MMDiTBlock(D, opts.Heads, 'MLPRatio', opts.MLPRatio, 'Norm', opts.Norm, ...
                    'FFN', opts.FFN, 'QKNorm', opts.QKNorm, 'Seed', s(100 * b), dt{:});
            end
            [obj.FinalAdaLN, obj.FinalLinear] = OceanDL.Vision.DiT.finalLayer(D, prod(obj.Patch.PatchSize) * obj.OutChannels, s(9000), dt);
        end

        function y = forward(obj, x, t, labels)
            if nargin < 4, labels = []; end
            N = size(x, 4);
            [tok, grid] = obj.Patch.forward(x);
            [tok, rope] = OceanDL.Vision.DiT.addPositions(tok, grid, obj.Positions, obj.Heads);
            c = OceanDL.Vision.DiT.condition(obj.TimeEmbed, obj.ClassEmbed, obj.NumClasses, t, labels, N);
            ids = labels;
            if isempty(ids), ids = repmat(obj.NumClasses + 1, N, 1); end
            D = size(tok, 1); Tc = obj.ContextTokens;
            z = permute(reshape(obj.CtxEmbed.forward(reshape(ids, N, 1)), D, Tc, N), [1 3 2]);   % D x N x Tc
            for b = 1:numel(obj.Blocks)
                [tok, z] = obj.Blocks{b}.forward(tok, z, c, rope);
            end
            y = OceanDL.Vision.DiT.finalProject(tok, c, obj.FinalAdaLN, obj.FinalLinear, obj.NormKind, ...
                obj.Patch.PatchSize, grid);
        end
    end
end
