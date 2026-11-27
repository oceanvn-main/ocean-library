classdef Legacy < OceanDL.nn.Module
    % LEGACY  Wrap an existing PARAMETER-FREE buffer-on-self layer
    %   (MaxPool2D/AvgPool2D/MaxPool1D/..., Conv.Flatten) as a module of the
    %   new Tensor graph -- its own forward()/backward() become the op and
    %   its VJP (see Autograd.Functions.LegacyLayer, incl. the stale-cache guard).
    %
    %   pool = OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2]));

    properties (SetAccess = protected)
        Layer
    end

    properties (SetAccess = private)
        CallCount (1,1) double = 0
    end

    methods
        function obj = Legacy(layer)
            obj.Layer = layer;
        end

        function y = forward(obj, x)
            obj.CallCount = obj.CallCount + 1;
            y = OceanDL.Autograd.Functions.LegacyLayer(obj.Layer, obj, obj.CallCount).apply(x);
        end
    end
end
