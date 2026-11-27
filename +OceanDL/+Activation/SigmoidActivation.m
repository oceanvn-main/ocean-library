classdef SigmoidActivation < OceanDL.Activation.IActivation
    % SIGMOIDACTIVATION  Logistic sigmoid — f(z) = 1 / (1 + exp(-z)).
    %
    %   Output range: (0, 1). Commonly used for binary classification outputs.
    %   Susceptible to vanishing gradients in deep networks (saturation near 0/1).
    %   Glorot Uniform initialization recommended.
    %
    %   Usage:
    %     act = OceanDL.Activation.SigmoidActivation();
    %     A   = act.forward(Z);    % A = 1/(1+exp(-Z))
    %     dA  = act.grad(Z);       % dA = A .* (1 - A)

    methods
        function A = forward(~, Z)
            A = 1 ./ (1 + exp(-Z));
        end

        function dA = grad(~, Z)
            A  = 1 ./ (1 + exp(-Z));
            dA = A .* (1 - A);
        end

        function s = name(~)
            s = 'sigmoid';
        end
    end
end
