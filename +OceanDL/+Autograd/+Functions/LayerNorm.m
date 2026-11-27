classdef LayerNorm < OceanDL.Autograd.Function
    % LAYERNORM  Layer normalisation over the FEATURE axis (dim 1) of every
    %   token independently (Ba, Kiros & Hinton 2016) -- PyTorch nn.LayerNorm
    %   semantics on the repo's features-first layout (D x N [x T]):
    %
    %     mu    = mean(x, 1)                    sigma^2 = mean((x - mu).^2, 1)    (biased)
    %     xhat  = (x - mu) ./ sqrt(sigma^2 + eps)
    %     y     = gamma .* xhat + beta          gamma, beta: (D x 1)
    %
    %   NOTE: eps is INSIDE the square root (PyTorch / TF / JAX). The legacy
    %   OceanDL.Norm.LayerNormLayer divides by (std + eps) instead; the two
    %   differ by O(eps), negligible for BERT's eps = 1e-12.
    %
    % VJP (fused -- one node instead of ~8 primitive ops), with gh = g .* gamma,
    %   s = 1 / sqrt(sigma^2 + eps):
    %     dx     = s .* (gh - mean(gh, 1) - xhat .* mean(gh .* xhat, 1))
    %     dgamma = g .* xhat      (summed over tokens by the Engine's broadcast rule)
    %     dbeta  = g
    %   The two mean() terms remove the directions the normalisation itself
    %   pins down (shifting all features, scaling all features).

    properties (SetAccess = private)
        Epsilon (1,1) double = 1e-5
    end

    methods
        function obj = LayerNorm(epsilon)
            if nargin > 0
                obj.Epsilon = epsilon;
            end
        end

        function Y = forward(obj, X, gamma, beta)
            mu = mean(X, 1);
            xc = X - mu;
            s = 1 ./ sqrt(mean(xc.^2, 1) + obj.Epsilon);
            xhat = xc .* s;
            Y = gamma .* xhat + beta;
            obj.Saved.Xhat = xhat;
            obj.Saved.S = s;
            obj.Saved.Gamma = gamma;
        end

        function grads = backward(obj, g)
            xhat = obj.Saved.Xhat;
            gh = g .* obj.Saved.Gamma;
            dX = obj.Saved.S .* (gh - mean(gh, 1) - xhat .* mean(gh .* xhat, 1));
            grads = {dX, g .* xhat, g};
        end
    end
end
