classdef TanhActivation < OceanDL.Activation.IActivation
    % TANHACTIVATION  Hyperbolic tangent — f(z) = tanh(z).
    %
    %   Output range: (-1, 1). Zero-centered; generally preferred over
    %   sigmoid for hidden layers. Still susceptible to vanishing gradients
    %   at saturation. Glorot Uniform initialization recommended.
    %
    %   Usage:
    %     act = OceanDL.Activation.TanhActivation();
    %     A   = act.forward(Z);    % A = tanh(Z)
    %     dA  = act.grad(Z);       % dA = 1 - tanh(Z).^2

    methods
        function A = forward(~, Z)
            A = tanh(Z);
        end

        function dA = grad(~, Z)
            dA = 1 - tanh(Z).^2;
        end

        function s = name(~)
            s = 'tanh';
        end
    end
end
