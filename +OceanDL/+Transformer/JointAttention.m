classdef JointAttention < OceanDL.nn.Module
    % JOINTATTENTION  MMDiT attention (Esser et al. 2024, SD3; FLUX double-
    %   stream blocks): two token streams with SEPARATE projection weights
    %   (and QK-norms) attend JOINTLY -- queries, keys and values of both
    %   streams are concatenated along the token axis, one softmax runs over
    %   all tokens, and the output is split back and projected per stream.
    %
    %   [yA, yB] = ja.forward(xA, xB)                 xA (D x N x Ta), xB (D x N x Tb)
    %   [yA, yB] = ja.forward(xA, xB, ropeA)          rotary positions on stream A only
    %
    %   Equivalent to one MultiHeadAttention over [xA, xB] when both streams
    %   share weights -- the MMDiT difference is modality-specific weights.

    properties (SetAccess = protected)
        A           % MultiHeadAttention (weights of stream A)
        B           % MultiHeadAttention (weights of stream B)
        AttnDropout (1,1) double
    end

    methods
        function obj = JointAttention(dim, heads, opts)
            arguments
                dim (1,1) double
                heads (1,1) double
                opts.QKNorm (1,1) string = "none"
                opts.AttnDropout (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.ConvEncoder.offset(opts.Seed, k);
            obj.A = OceanDL.Transformer.MultiHeadAttention(dim, heads, 'QKNorm', opts.QKNorm, 'Seed', s(1), 'DataType', opts.DataType);
            obj.B = OceanDL.Transformer.MultiHeadAttention(dim, heads, 'QKNorm', opts.QKNorm, 'Seed', s(10), 'DataType', opts.DataType);
            obj.AttnDropout = opts.AttnDropout;
        end

        function [yA, yB] = forward(obj, xA, xB, ropeA)
            if nargin < 4, ropeA = []; end
            split = @(x) OceanDL.Transformer.MultiHeadAttention.splitHeads(x, obj.A.NumHeads);
            [Qa, Ka] = obj.A.normalizeQK(split(obj.A.Query.forward(xA)), split(obj.A.Key.forward(xA)));
            [Qb, Kb] = obj.B.normalizeQK(split(obj.B.Query.forward(xB)), split(obj.B.Key.forward(xB)));
            if ~isempty(ropeA)
                Qa = OceanDL.Transformer.RoPE.apply(Qa, ropeA);
                Ka = OceanDL.Transformer.RoPE.apply(Ka, ropeA);
            end
            Q = cat(2, Qa, Qb); K = cat(2, Ka, Kb);
            V = cat(2, split(obj.A.Value.forward(xA)), split(obj.B.Value.forward(xB)));
            O = OceanDL.Transformer.MultiHeadAttention.headAttention(Q, K, V, 'Dropout', obj.AttnDropout, 'Training', obj.Training);
            Ta = size(xA, 3);
            yA = obj.A.Output.forward(OceanDL.Transformer.MultiHeadAttention.mergeHeads(O(:, 1:Ta, :, :)));
            yB = obj.B.Output.forward(OceanDL.Transformer.MultiHeadAttention.mergeHeads(O(:, Ta+1:end, :, :)));
        end
    end
end
