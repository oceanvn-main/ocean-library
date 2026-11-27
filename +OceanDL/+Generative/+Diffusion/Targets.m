classdef Targets
    % TARGETS  What the network predicts, and exact conversions between the
    %   choices, for x = a x0 + s eps (a = alpha(t), s = sigma(t); da, ds their
    %   time derivatives). Every kind carries the same information:
    %
    %     "eps"       eps                          (DDPM, Ho 2020)
    %     "x0"        x0                           (clean image)
    %     "v"         v = a eps - s x0             (Salimans & Ho 2022)
    %     "velocity"  u = da x0 + ds eps = dx/dt   (flow matching: eps - x0)
    %
    %   Inverse maps (from the 2 x 2 linear system [x; p] = M [x0; eps]):
    %     eps:       x0 = (x - s p) / a
    %     x0:        eps = (x - a p) / s
    %     v:         x0 = (a x - s p) / (a^2 + s^2),  eps = (s x + a p) / (a^2 + s^2)
    %     velocity:  x0 = (ds x - s p) / det,  eps = (a p - da x) / det,  det = a ds - s da
    %   score = grad log p(x_t) = -E[eps | x_t] / s  (Tweedie).
    %   Works on arrays and Tensors (elementwise algebra only).

    methods (Static)
        function [x0, eps] = toX0Eps(kind, p, x, a, s, da, ds)
            switch kind
                case "eps"
                    eps = p; x0 = (x - s .* p) ./ a;
                case "x0"
                    x0 = p; eps = (x - a .* p) ./ s;
                case "v"
                    n = a .^ 2 + s .^ 2;
                    x0 = (a .* x - s .* p) ./ n; eps = (s .* x + a .* p) ./ n;
                case "velocity"
                    det = a .* ds - s .* da;
                    x0 = (ds .* x - s .* p) ./ det; eps = (a .* p - da .* x) ./ det;
                otherwise
                    error('OceanDL:Generative:Diffusion:Targets:Kind', 'Unknown target "%s".', kind);
            end
        end

        function p = fromX0Eps(kind, x0, eps, a, s, da, ds)
            switch kind
                case "eps",      p = eps;
                case "x0",       p = x0;
                case "v",        p = a .* eps - s .* x0;
                case "velocity", p = da .* x0 + ds .* eps;
                otherwise
                    error('OceanDL:Generative:Diffusion:Targets:Kind', 'Unknown target "%s".', kind);
            end
        end

        function f = x0ErrorFactor(kind, a, s, da, ds)
            % d x0_hat / d prediction: an error e in the prediction is an error
            % f .* e in the implied clean image (used by min-SNR weighting).
            switch kind
                case "eps",      f = -s ./ a;
                case "x0",       f = ones(size(a));
                case "v",        f = -s ./ (a .^ 2 + s .^ 2);
                case "velocity", f = -s ./ (a .* ds - s .* da);
                otherwise,       f = ones(size(a));
            end
        end
    end
end
