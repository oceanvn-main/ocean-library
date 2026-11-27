classdef ReLUActivation < OceanDL.Activation.IActivation
    % RELUACTIVATION  Rectified Linear Unit — f(z) = max(0, z).
    %
    %   He Normal initialization recommended with this activation.
    %   Dead neuron risk: neurons that receive only negative inputs
    %   will never recover (zero gradient).
    %
    %   Usage:
    %     act = OceanDL.Activation.ReLUActivation();
    %     A   = act.forward(Z);    % A = max(0, Z)
    %     dA  = act.grad(Z);       % dA = (Z > 0)

    methods
        function A = forward(~, Z)
            A = max(0, Z);
        end

        function dA = grad(~, Z)
            dA = double(Z > 0);
        end

        function s = name(~)
            s = 'relu';
        end
    end
end
