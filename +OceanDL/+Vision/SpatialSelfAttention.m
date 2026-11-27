classdef SpatialSelfAttention < OceanDL.nn.Module
    % SPATIALSELFATTENTION  Self-attention over the H*W positions of a feature
    %   map (the "mid" attention block of AutoencoderKL / DDPM):
    %
    %     y = x + MHA( tokens( GN(x) ) )     tokens: (H x W x C x N) -> (C x N x HW)
    %
    %   Reuses the shared OceanDL.Transformer.MultiHeadAttention; tokens are
    %   in column-major position order (row index fastest), as PatchEmbedding.
    %   Cost is O((HW)^2), so it is placed at the lowest resolution only.

    properties (SetAccess = protected)
        Norm
        Attention
    end

    methods
        function obj = SpatialSelfAttention(channels, opts)
            arguments
                channels (1,1) double {mustBePositive, mustBeInteger}
                opts.Heads (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Groups (1,1) double = 32
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            obj.Norm = OceanDL.nn.GroupNorm(OceanDL.nn.GroupNorm.groupsFor(channels, opts.Groups), channels, 'DataType', opts.DataType);
            obj.Attention = OceanDL.Transformer.MultiHeadAttention(channels, opts.Heads, 'Seed', opts.Seed, 'DataType', opts.DataType);
        end

        function y = forward(obj, x)
            [H, W, C, N] = size(x, 1:4);
            t = reshape(permute(obj.Norm.forward(x), [3 4 1 2]), C, N, H * W);
            a = obj.Attention.forward(t);
            y = x + permute(reshape(a, C, N, H, W), [3 4 1 2]);
        end
    end
end
