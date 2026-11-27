classdef LinearActivation < OceanDL.Activation.IActivation
    % LINEARACTIVATION  Identity activation — f(z) = z.
    %
    %   Used for input layers and regression output layers.
    %
    %   Usage:
    %     act = OceanDL.Activation.LinearActivation();
    %     A   = act.forward(Z);   % A == Z
    %     dA  = act.grad(Z);      % dA == ones(size(Z))

    methods
        function A = forward(~, Z)
            A = Z;
        end

        function dA = grad(~, Z)
            dA = ones(size(Z));
        end

        function s = name(~)
            s = 'linear';
        end
    end
end
