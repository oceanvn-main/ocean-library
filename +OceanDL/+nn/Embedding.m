classdef Embedding < OceanDL.nn.Module
    % EMBEDDING  Trainable lookup table (PyTorch nn.Embedding), column-major:
    %   Weight (Dim x NumEmbeddings); ids of any shape (values 1..NumEmbeddings)
    %   -> (Dim x size(ids)). Token ids (N x T) therefore give the (D x N x T)
    %   layout used by the recurrent layers.
    %   Gradient: only the looked-up columns receive gradient, and a column
    %   used several times receives the SUM (Index op's scatter-add).
    %   Init N(0, 1), as in PyTorch.

    properties (SetAccess = protected)
        Weight
        NumEmbeddings (1,1) double
        Dim (1,1) double
    end

    methods
        function obj = Embedding(numEmbeddings, dim, opts)
            arguments
                numEmbeddings (1,1) double {mustBePositive, mustBeInteger}
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.NumEmbeddings = numEmbeddings;
            obj.Dim = dim;
            if isempty(opts.Seed)
                W = randn(dim, numEmbeddings);
            else
                W = randn(RandStream('mt19937ar', 'Seed', opts.Seed), dim, numEmbeddings);
            end
            obj.Weight = OceanDL.nn.Parameter(cast(W, opts.DataType));
        end

        function y = forward(obj, ids)
            if any(ids(:) < 1 | ids(:) > obj.NumEmbeddings | ids(:) ~= round(ids(:)))
                error('OceanDL:nn:Embedding:BadIndex', ...
                    'Ids must be integers in 1..%d.', obj.NumEmbeddings);
            end
            y = OceanDL.Autograd.F.embedding(obj.Weight, ids);
        end
    end
end
