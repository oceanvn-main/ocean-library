classdef LearnedPositionalEmbedding < OceanDL.nn.Module
    % LEARNEDPOSITIONALEMBEDDING  One trainable vector per token position,
    %   added to the tokens (BERT, ViT): x_t <- x_t + p_t.
    %   Weight (D x 1 x T), init N(0, 0.02^2) as in ViT; broadcast over the batch.

    properties (SetAccess = protected)
        Weight
    end

    methods
        function obj = LearnedPositionalEmbedding(numPositions, dim, opts)
            arguments
                numPositions (1,1) double {mustBePositive, mustBeInteger}
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.Std (1,1) double = 0.02
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            if isempty(opts.Seed)
                W = randn(dim, 1, numPositions);
            else
                W = randn(RandStream('mt19937ar', 'Seed', opts.Seed), dim, 1, numPositions);
            end
            obj.Weight = OceanDL.nn.Parameter(cast(opts.Std * W, opts.DataType));
        end

        function y = forward(obj, x)
            T = size(x, 3);
            if T ~= size(obj.Weight, 3)
                error('OceanDL:Transformer:LearnedPositionalEmbedding:Length', ...
                    'Built for %d positions, got %d tokens.', size(obj.Weight, 3), T);
            end
            y = x + obj.Weight;
        end
    end
end
