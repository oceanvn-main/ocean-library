classdef LegacyLayer < OceanDL.Autograd.Function
    % LEGACYLAYER  Adapter: an existing buffer-on-self layer (forward/backward
    %   pair, e.g. OceanDL.Pooling.MaxPool2D, Conv.Flatten)
    %   becomes a node of the Tensor graph.
    %
    %   forward  -> layer.forward(X);  VJP -> layer.backward(g)
    %
    %   SAFETY: a legacy layer keeps ONE forward cache on itself. If the same
    %   layer object ran forward again before this node's backward (e.g. it
    %   was applied twice in one graph), its cache belongs to the other call
    %   and the gradient would be silently wrong. The owning nn.Legacy module
    %   counts calls; backward() refuses to run on a stale cache.
    %   Only PARAMETER-FREE layers are supported: layers with weights have
    %   native ops (Conv2d, MatMul) whose parameters live in the graph.

    properties (SetAccess = private)
        Layer
        Owner                 % nn.Legacy module (call counter)
        CallIndex (1,1) double
    end

    methods
        function obj = LegacyLayer(layer, owner, callIndex)
            obj.Layer = layer;
            obj.Owner = owner;
            obj.CallIndex = callIndex;
        end

        function Y = forward(obj, X)
            Y = obj.Layer.forward(X);
        end

        function grads = backward(obj, g)
            if obj.Owner.CallCount ~= obj.CallIndex
                error('OceanDL:Autograd:LegacyLayer:StaleCache', ...
                    ['Legacy layer %s was run again (call %d) before backward of call %d; its forward ', ...
                     'cache is overwritten. Use a separate layer object per use.'], ...
                    class(obj.Layer), obj.Owner.CallCount, obj.CallIndex);
            end
            grads = {obj.Layer.backward(g)};
        end
    end
end
