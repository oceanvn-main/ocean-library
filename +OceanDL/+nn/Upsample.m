classdef Upsample < OceanDL.nn.Module
    % UPSAMPLE  Nearest-neighbour upsampling module (PyTorch nn.Upsample,
    %   mode "nearest") on (H x W x C x N). No parameters.

    properties (SetAccess = protected)
        Scale (1,2) double
    end

    methods
        function obj = Upsample(scale)
            arguments
                scale (1,:) double {mustBePositive, mustBeInteger} = 2
            end
            obj.Scale = scale .* [1 1];
        end

        function y = forward(obj, x)
            y = OceanDL.Autograd.F.upsample(x, obj.Scale);
        end
    end
end
