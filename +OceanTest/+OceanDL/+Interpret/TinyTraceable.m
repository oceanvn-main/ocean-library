classdef TinyTraceable < OceanDL.nn.Module & OceanDL.Interpret.ITraceable
    % TINYTRACEABLE  Test model with closed-form answers: one 3x3 convolution
    %   (known weights, no bias) followed by SiLU.
    %     "enc.L1.conv"  = conv(x)          (linear: its gradient IS the weight pattern)
    %     "enc.L1.act"   = silu(conv(x))
    %   The weights are symmetric under a 180-degree rotation, so correlation and
    %   convolution conventions give the same answers.

    properties
        Conv
    end

    methods
        function obj = TinyTraceable(W)
            if nargin < 1, W = [1 -1 1; -1 2 -1; 1 -1 1]; end
            obj.Conv = OceanDL.nn.Conv2d(1, 1, 3, 'Padding', 1, 'Bias', false);
            obj.Conv.Weight.Data = reshape(W, 3, 3, 1, 1);
        end

        function y = forward(obj, x)
            y = OceanDL.Autograd.F.silu(obj.Conv.forward(x));
        end

        function [acts, order] = trace(obj, X, varargin)
            acts = containers.Map('KeyType', 'char', 'ValueType', 'any');
            c = obj.Conv.forward(X);
            acts('enc.L1.conv') = c;
            acts('enc.L1.act') = OceanDL.Autograd.F.silu(c);
            order = ["enc.L1.conv", "enc.L1.act"];
        end
    end
end
