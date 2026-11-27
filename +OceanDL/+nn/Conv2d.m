classdef Conv2d < OceanDL.nn.Module
    % CONV2D  2-D convolution module (PyTorch nn.Conv2d) on (H x W x C x N).
    %   Weight (KH x KW x Cin x Cout), Bias (Cout x 1),
    %   init U(-1/sqrt(fanIn), 1/sqrt(fanIn)), fanIn = KH*KW*Cin.
    %   Uses the shift-and-multiply Autograd.Functions.Conv2d op.

    properties (SetAccess = protected)
        Weight
        Bias
        Stride (1,2) double
        Padding (1,2) double
    end

    methods
        function obj = Conv2d(inChannels, outChannels, kernelSize, opts)
            arguments
                inChannels (1,1) double {mustBePositive, mustBeInteger}
                outChannels (1,1) double {mustBePositive, mustBeInteger}
                kernelSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride (1,:) double = 1
                opts.Padding (1,:) double = 0
                opts.Bias (1,1) logical = true
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            k = kernelSize .* [1 1];                   % scalar -> square
            obj.Stride = opts.Stride .* [1 1];
            obj.Padding = opts.Padding .* [1 1];
            u = OceanDL.nn.Linear.uniform(opts.Seed);
            bound = 1 / sqrt(k(1) * k(2) * inChannels);
            W = bound * (2 * reshape(u(prod(k) * inChannels * outChannels, 1), [k inChannels outChannels]) - 1);
            obj.Weight = OceanDL.nn.Parameter(cast(W, opts.DataType));
            if opts.Bias
                obj.Bias = OceanDL.nn.Parameter(cast(bound * (2 * u(outChannels, 1) - 1), opts.DataType));
            end
        end

        function y = forward(obj, x)
            y = OceanDL.Autograd.F.conv2d(x, obj.Weight, obj.Bias, ...
                'Stride', obj.Stride, 'Padding', obj.Padding);
        end
    end
end
