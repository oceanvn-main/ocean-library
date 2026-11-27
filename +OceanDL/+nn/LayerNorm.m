classdef LayerNorm < OceanDL.nn.Module
    % LAYERNORM  PyTorch nn.LayerNorm over the feature axis (dim 1) of
    %   "CB" / "CBT" tensors: y = gamma .* (x - mu) ./ sqrt(var + eps) + beta.
    %   Weight (gamma) init 1, Bias (beta) init 0.  See Autograd.Functions.LayerNorm.

    properties (SetAccess = protected)
        Weight
        Bias
        Epsilon (1,1) double = 1e-5
    end

    methods
        function obj = LayerNorm(dim, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.Epsilon (1,1) double {mustBePositive} = 1e-5
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.Weight = OceanDL.nn.Parameter(ones(dim, 1, opts.DataType));
            obj.Bias = OceanDL.nn.Parameter(zeros(dim, 1, opts.DataType));
            obj.Epsilon = opts.Epsilon;
        end

        function y = forward(obj, x)
            y = OceanDL.Autograd.F.layerNorm(x, obj.Weight, obj.Bias, obj.Epsilon);
        end
    end
end
