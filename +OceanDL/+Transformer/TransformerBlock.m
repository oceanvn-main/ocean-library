classdef TransformerBlock < OceanDL.nn.Module
    % TRANSFORMERBLOCK  One encoder block: multi-head self-attention + MLP,
    %   each wrapped in a residual connection and a LayerNorm.
    %
    %   Norm "pre" (default; GPT-2, ViT, Swin, MAE -- Xiong et al. 2020):
    %     x = x + DropPath(Dropout(Attn(LN1(x))))
    %     x = x + DropPath(Dropout(MLP (LN2(x))))
    %     The residual path is an untouched identity from input to output,
    %     so deep stacks train stably without learning-rate warmup.
    %   Norm "post" (original Transformer 2017, BERT):
    %     x = LN1(x + Dropout(Attn(x)))
    %     x = LN2(x + Dropout(MLP(x)))
    %     Every residual passes through a LayerNorm; typically needs warmup.
    %
    %   Semantics match PyTorch nn.TransformerEncoderLayer(norm_first = Norm=="pre",
    %   activation = Activation, batch_first=False) on "CBT" tokens.

    properties (SetAccess = protected)
        Attention
        MLP
        Norm1
        Norm2
        Norm (1,1) string = "pre"
        Dropout (1,1) double = 0
        DropPath (1,1) double = 0
    end

    methods
        function obj = TransformerBlock(dim, numHeads, mlpHidden, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                numHeads (1,1) double {mustBePositive, mustBeInteger}
                mlpHidden (1,1) double {mustBePositive, mustBeInteger} = 4 * dim
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["pre", "post"])} = "pre"
                opts.Activation (1,1) string {mustBeMember(opts.Activation, ["gelu", "relu"])} = "gelu"
                opts.Dropout (1,1) double = 0
                opts.AttnDropout (1,1) double = 0
                opts.DropPath (1,1) double = 0
                opts.Epsilon (1,1) double = 1e-5
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Transformer.TransformerBlock.offset(opts.Seed, k);
            obj.Attention = OceanDL.Transformer.MultiHeadAttention(dim, numHeads, ...
                'AttnDropout', opts.AttnDropout, 'Seed', s(0), 'DataType', opts.DataType);
            obj.MLP = OceanDL.Transformer.FeedForward(dim, mlpHidden, 'Activation', opts.Activation, ...
                'Dropout', opts.Dropout, 'Seed', s(10), 'DataType', opts.DataType);
            obj.Norm1 = OceanDL.nn.LayerNorm(dim, 'Epsilon', opts.Epsilon, 'DataType', opts.DataType);
            obj.Norm2 = OceanDL.nn.LayerNorm(dim, 'Epsilon', opts.Epsilon, 'DataType', opts.DataType);
            obj.Norm = opts.Norm;
            obj.Dropout = opts.Dropout;
            obj.DropPath = opts.DropPath;
        end

        function x = forward(obj, x, opts)
            arguments
                obj
                x
                opts.KeyMask = []
                opts.AttnMask = []
            end
            F = OceanDL.Autograd.F;
            branch = @(y) F.dropPath(F.dropout(y, obj.Dropout, obj.Training), obj.DropPath, obj.Training);
            attend = @(y) obj.Attention.forward(y, 'KeyMask', opts.KeyMask, 'AttnMask', opts.AttnMask);
            if obj.Norm == "pre"
                x = x + branch(attend(obj.Norm1.forward(x)));
                x = x + branch(obj.MLP.forward(obj.Norm2.forward(x)));
            else
                x = obj.Norm1.forward(x + F.dropout(attend(x), obj.Dropout, obj.Training));
                x = obj.Norm2.forward(x + F.dropout(obj.MLP.forward(x), obj.Dropout, obj.Training));
            end
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
