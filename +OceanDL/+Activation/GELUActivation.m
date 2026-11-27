classdef GELUActivation < OceanDL.Activation.IActivation
    % GELUACTIVATION  Gaussian Error Linear Unit — f(z) = z * Phi(z).
    %
    %   Phi(z) = normcdf(z) = 0.5 * (1 + erf(z / sqrt(2))).
    %   Smooth, non-monotone activation. Preferred in transformer architectures
    %   (BERT, GPT). Avoids the "dead neuron" problem of ReLU.
    %   Glorot Uniform initialization recommended.
    %
    %   Ref: Hendrycks & Gimpel, 2016 (arXiv:1606.08415)
    %
    %   Usage:
    %     act = OceanDL.Activation.GELUActivation();
    %     A   = act.forward(Z);
    %     dA  = act.grad(Z);

    methods
        function A = forward(~, Z)
            A = Z .* normcdf(Z);
        end

        function dA = grad(~, Z)
            % d/dz [z * Phi(z)] = Phi(z) + z * phi(z)
            % where phi(z) = normpdf(z) is the standard normal PDF
            dA = normcdf(Z) + Z .* normpdf(Z);
        end

        function s = name(~)
            s = 'gelu';
        end
    end
end
