classdef RMSNorm < OceanDL.nn.Module
    % RMSNORM  Root-mean-square normalisation over the feature axis (dim 1)
    %   (Zhang & Sennrich 2019; LLaMA, SD3 / FLUX QK-norm, Qwen-Image):
    %     y = x ./ sqrt(mean(x.^2, 1) + eps) .* weight
    %   No mean subtraction and no bias -- cheaper than LayerNorm, and in
    %   practice as good. 'Affine' false drops the weight (adaLN blocks supply
    %   their own scale). Works on any N-D Tensor whose axis 1 is features.

    properties (SetAccess = protected)
        Weight = []
        Epsilon (1,1) double
    end

    methods
        function obj = RMSNorm(dim, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.Epsilon (1,1) double {mustBePositive} = 1e-6
                opts.Affine (1,1) logical = true
                opts.DataType (1,1) string = "double"
            end
            obj.Epsilon = opts.Epsilon;
            if opts.Affine, obj.Weight = OceanDL.nn.Parameter(ones(dim, 1, opts.DataType)); end
        end

        function y = forward(obj, x)
            x = OceanDL.Autograd.Tensor.wrap(x);
            y = x ./ sqrt(mean(x .^ 2, 1) + obj.Epsilon);
            if ~isempty(obj.Weight), y = y .* obj.Weight; end
        end
    end
end
