classdef SwishActivation < OceanDL.Activation.IActivation
    % SWISHACTIVATION  Swish (SiLU) — f(z) = z * sigmoid(z).
    %
    %   Self-gated activation proposed by Ramachandran et al., 2017.
    %   Smooth, non-monotone. Outperforms ReLU on deep networks.
    %   Also known as SiLU (Sigmoid Linear Unit).
    %   Glorot Uniform initialization recommended.
    %
    %   Ref: Ramachandran et al., 2017 (arXiv:1710.05941)
    %
    %   Usage:
    %     act = OceanDL.Activation.SwishActivation();
    %     A   = act.forward(Z);
    %     dA  = act.grad(Z);

    methods
        function A = forward(~, Z)
            sig = 1 ./ (1 + exp(-Z));
            A   = Z .* sig;
        end

        function dA = grad(~, Z)
            % d/dz [z * sigma(z)] = sigma(z) + z * sigma(z) * (1 - sigma(z))
            %                     = sigma(z) * (1 + z * (1 - sigma(z)))
            sig = 1 ./ (1 + exp(-Z));
            dA  = sig .* (1 + Z .* (1 - sig));
        end

        function s = name(~)
            s = 'swish';
        end
    end
end
