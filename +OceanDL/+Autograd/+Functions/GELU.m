classdef GELU < OceanDL.Autograd.Function
    % GELU  Gaussian Error Linear Unit (Hendrycks & Gimpel 2016), the
    %   Transformer feed-forward nonlinearity (BERT, ViT, GPT).
    %
    %   Exact (PyTorch default):  y = x Phi(x) = x (1 + erf(x / sqrt 2)) / 2
    %     dy/dx = Phi(x) + x phi(x),   phi(x) = exp(-x^2/2) / sqrt(2 pi)
    %   Approximate "tanh" (GPT-2, PyTorch approximate='tanh'):
    %     y = x (1 + tanh(u)) / 2,  u = sqrt(2/pi) (x + 0.044715 x^3)
    %     dy/dx = (1 + tanh u)/2 + x (1 - tanh^2 u) sqrt(2/pi) (1 + 3*0.044715 x^2) / 2
    %   Uses base-MATLAB erf (no Statistics Toolbox normcdf).

    properties (SetAccess = private)
        Approximate (1,1) string = "none"
    end

    methods
        function obj = GELU(approximate)
            if nargin > 0
                obj.Approximate = approximate;
            end
        end

        function Y = forward(obj, X)
            obj.Saved.X = X;
            if obj.Approximate == "tanh"
                u = sqrt(2/pi) * (X + 0.044715 * X.^3);
                obj.Saved.Th = tanh(u);
                Y = 0.5 * X .* (1 + obj.Saved.Th);
            else
                Y = 0.5 * X .* (1 + erf(X / sqrt(2)));
            end
        end

        function grads = backward(obj, g)
            X = obj.Saved.X;
            if obj.Approximate == "tanh"
                th = obj.Saved.Th;
                d = 0.5 * (1 + th) + 0.5 * X .* (1 - th.^2) .* sqrt(2/pi) .* (1 + 3 * 0.044715 * X.^2);
            else
                d = 0.5 * (1 + erf(X / sqrt(2))) + X .* exp(-X.^2 / 2) / sqrt(2*pi);
            end
            grads = {g .* d};
        end
    end
end
