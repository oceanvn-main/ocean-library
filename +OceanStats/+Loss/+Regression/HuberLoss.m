% DEPRECATED: HuberLoss has been renamed to PseudoHuberLoss for clarity.
%   - For the smooth approximation (original behaviour): use PseudoHuberLoss
%   - For the textbook piecewise Huber (Friedman 2001 / ESL §10.14): use ClassicHuberLoss
%   This file is kept for backward compatibility; it will not be updated.
classdef HuberLoss < OceanStats.Loss.ITreeLoss
    % HUBERLOSS  Robust Huber Loss for regression.
    %
    %   FORMULA (per sample, using residual r = y - p):
    %     L(y,p)  = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
    %     dL/dp   = -r / sqrt(1 + (r/delta)^2)           [gradient]
    %     d²L/dp² =  1 / (1 + (r/delta)^2)^(3/2)        [hessian]
    %
    %   WHY PSEUDO-HUBER vs CLASSIC HUBER:
    %     Classic Huber is piecewise (L2 near 0, L1 far out) and non-smooth at the
    %     transition point. Pseudo-Huber is fully smooth and differentiable everywhere,
    %     making it ideal for gradient-based and Newton-step algorithms.
    %
    %   PARAMETER:
    %     Delta — controls the transition from L2 (near 0) to L1 (far from 0).
    %             - Small Delta → more robust (closer to MAE)
    %             - Large Delta → closer to MSE
    %             - Typical range: 0.5 to 3.0 depending on target scale.
    %
    %   WHEN TO USE:
    %     Regression with outliers when you still need a smooth, non-zero Hessian
    %     (e.g., for Newton-step boosting or second-order optimizers).
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.HuberLoss('Delta', 1.5);
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %     h    = loss.hessian(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Regression.MSELoss
    %             OceanStats.Loss.Regression.MAELoss
    %             OceanStats.Loss.ILoss

    properties
        % DELTA  Transition point between L2 and L1 regimes (default: 1.0)
        Delta double = 1.0
    end

    methods
        function obj = HuberLoss(options)
            % HUBERLOSS  Constructor with named options.
            %
            %   obj = HuberLoss('Delta', 1.5)
            arguments
                options.Delta double {mustBePositive} = 1.0
            end
            obj.Delta = options.Delta;
        end

        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            L = mean(d^2 * (sqrt(1 + (r ./ d).^2) - 1));
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            % Negative because gradient is w.r.t. Y_pred, not residual
            g = -r ./ sqrt(1 + (r ./ d).^2);
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            h = 1 ./ (1 + (r ./ d).^2).^(3/2);
        end

        function n = name(obj)
            n = sprintf("Huber (delta=%.2f)", obj.Delta);
        end

        function gamma = initPrediction(~, Y_true)
            % F_0 = mean(Y) — a reasonable starting point for smooth losses.
            gamma = mean(Y_true);
        end

        function gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
            % Approximation of Friedman's (2001) Huber leaf estimator:
            %   gamma_k ≈ median(r) + mean(sign(r - median(r)) * min(|r - median(r)|, delta))
            % where r = residuals in leaf = Y_true - F_prev.
            %
            % This avoids a full 1D line search while matching the true optimum
            % closely. It reduces to the median for pure L1 and to the mean for pure L2.
            r       = Y_true(leafMask) - F_prev(leafMask);
            r_med   = median(r);
            diff    = r - r_med;
            d       = obj.Delta;
            gamma   = r_med + mean(sign(diff) .* min(abs(diff), d));
        end
    end
end
