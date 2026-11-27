classdef MAELoss < OceanStats.Loss.ITreeLoss
    % MAELOSS  Mean Absolute Error (L1) loss for regression.
    %
    %   FORMULA:
    %     L(y, p)   = |y - p|                 [per sample]
    %     dL/dp     = sign(p - y)             [gradient; ±1, undefined at 0 → use 0]
    %     d²L/dp²   = 0                       [hessian; technically undefined at 0]
    %
    %   WHEN TO USE:
    %     Robust regression where outliers are present. L1 loss saturates at ±1
    %     gradient, so extreme errors do not dominate the update.
    %
    %   LIMITATION:
    %     Hessian is zero almost everywhere — not suitable for pure Newton-step
    %     methods (like XGBoost). Use PseudoHuberLoss for a smooth alternative,
    %     or ClassicHuberLoss for the textbook piecewise Huber (Friedman 2001).
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.MAELoss();
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Regression.PseudoHuberLoss
    %             OceanStats.Loss.Regression.ClassicHuberLoss
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            L = mean(abs(Y_true - Y_pred));
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            r = Y_pred - Y_true;
            g = sign(r);                  % +1, 0, or -1
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            % Hessian is 0 almost everywhere; return small epsilon to avoid
            % division-by-zero in Newton-step methods.
            h = zeros(size(Y_pred));
        end

        function n = name(~)
            n = "MAE";
        end

        function gamma = initPrediction(~, Y_true)
            % F_0 = median(Y) — minimizes sum of absolute errors.
            % KEY DIFFERENCE from MSE: median instead of mean.
            gamma = median(Y_true);
        end

        function gamma = computeLeafConstant(~, Y_true, F_prev, leafMask)
            % Leaf optimal = MEDIAN of residuals in this leaf.
            % This is the critical Step 2c: the leaf value is NO LONGER the mean
            % (what the surrogate squared-error tree gives). It is the MEDIAN,
            % which is robust to outliers. This is what makes L1-boosting genuinely
            % different from L2-boosting.
            residuals = Y_true(leafMask) - F_prev(leafMask);
            gamma = median(residuals);
        end
    end
end
