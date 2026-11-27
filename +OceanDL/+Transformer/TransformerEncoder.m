classdef TransformerEncoder < OceanDL.nn.Module
    % TRANSFORMERENCODER  A stack of Depth TransformerBlocks on "CBT" tokens,
    %   plus a final LayerNorm (on by default for Pre-LN stacks, whose
    %   residual stream is otherwise never normalised at the output).
    %   DropPath rate grows linearly with depth, 0 -> DropPath (stochastic
    %   depth schedule of DeiT / Swin).

    properties (SetAccess = protected)
        Layers cell = {}
        FinalNorm
        Depth (1,1) double
    end

    methods
        function obj = TransformerEncoder(depth, dim, numHeads, mlpHidden, opts)
            arguments
                depth (1,1) double {mustBePositive, mustBeInteger}
                dim (1,1) double {mustBePositive, mustBeInteger}
                numHeads (1,1) double {mustBePositive, mustBeInteger}
                mlpHidden (1,1) double {mustBePositive, mustBeInteger} = 4 * dim
                opts.Norm (1,1) string {mustBeMember(opts.Norm, ["pre", "post"])} = "pre"
                opts.FinalNorm = []                       % default: true for "pre", false for "post"
                opts.Activation (1,1) string = "gelu"
                opts.Dropout (1,1) double = 0
                opts.AttnDropout (1,1) double = 0
                opts.DropPath (1,1) double = 0
                opts.Epsilon (1,1) double = 1e-5
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            obj.Depth = depth;
            rates = linspace(0, opts.DropPath, depth);
            obj.Layers = cell(1, depth);
            for l = 1:depth
                seed = [];
                if ~isempty(opts.Seed), seed = opts.Seed + 100 * l; end
                obj.Layers{l} = OceanDL.Transformer.TransformerBlock(dim, numHeads, mlpHidden, ...
                    'Norm', opts.Norm, 'Activation', opts.Activation, 'Dropout', opts.Dropout, ...
                    'AttnDropout', opts.AttnDropout, 'DropPath', rates(l), 'Epsilon', opts.Epsilon, ...
                    'Seed', seed, 'DataType', opts.DataType);
            end
            finalNorm = opts.FinalNorm;
            if isempty(finalNorm), finalNorm = opts.Norm == "pre"; end
            if finalNorm
                obj.FinalNorm = OceanDL.nn.LayerNorm(dim, 'Epsilon', opts.Epsilon, 'DataType', opts.DataType);
            end
        end

        function x = forward(obj, x, opts)
            arguments
                obj
                x
                opts.KeyMask = []
                opts.AttnMask = []
            end
            for l = 1:obj.Depth
                x = obj.Layers{l}.forward(x, 'KeyMask', opts.KeyMask, 'AttnMask', opts.AttnMask);
            end
            if ~isempty(obj.FinalNorm)
                x = obj.FinalNorm.forward(x);
            end
        end

        function setStoreAttention(obj, tf)
            % Keep every layer's attention weights (for rollout / visualisation).
            for l = 1:obj.Depth
                obj.Layers{l}.Attention.StoreAttention = tf;
            end
        end
    end
end
