classdef ClassicHuberLoss < OceanStats.Loss.ITreeLoss
    % CLASSICHUBERLOSS  Classic piecewise Huber loss for robust regression.
    %
    %   This is the loss function used in Friedman (2001) MART algorithm and
    %   described in ESL §10.3 / §10.14. It is piecewise: quadratic (L2) near
    %   zero and linear (L1) beyond the threshold delta.
    %
    %   FORMULA (per sample, using residual r = y - F):
    %     L(y,F) = r^2 / 2                   if |r| <= delta   [quadratic zone]
    %     L(y,F) = delta * (|r| - delta/2)   if |r| >  delta   [linear zone]
    %
    %   GRADIENT (negative gradient = pseudo-residuals for step 2a of Alg 10.3):
    %     -dL/dF = r                          if |r| <= delta
    %     -dL/dF = delta * sign(r)            if |r| >  delta
    %
    %   HESSIAN (d²L/dF²):
    %     d²L/dF² = 1                         if |r| <= delta
    %     d²L/dF² = 0                         if |r| >  delta
    %     NOTE: The hessian is zero outside the quadratic zone. This makes the
    %     classic Huber non-smooth at delta and poorly suited to Newton-step
    %     boosting. For that, use PseudoHuberLoss instead.
    %
    %   LEAF CONSTANT (Friedman 2001, Algorithm 10.3, Step 2c):
    %     gamma_k = argmin_g sum_{i in leaf k} L(y_i, F_{m-1}(x_i) + g)
    %
    %     This has no fully closed-form solution, so Friedman (2001) gives an
    %     approximation (Eq. 11) that matches numerically:
    %       r_k    = Y_true - F_prev  (residuals for samples in leaf k)
    %       r_med  = median(r_k)
    %       gamma  = r_med + mean(sign(r_k - r_med) * min(|r_k - r_med|, delta))
    %
    %     This is exact for MAE (delta -> 0) where gamma = median(r), and
    %     becomes the mean when delta -> inf (MSE regime).
    %
    %   PARAMETER:
    %     Delta — threshold between quadratic and linear zones.
    %             - Set to the alpha-th quantile of |residuals| at each round
    %               in the original Friedman (2001) implementation (alpha=0.9).
    %             - Fixed delta is simpler and still effective; 0.5 * IQR of
    %               initial residuals is one data-driven choice.
    %
    %   WHEN TO USE:
    %     This is the textbook choice for robust gradient boosting (MART).
    %     Use PseudoHuberLoss if you need a smooth, non-zero Hessian everywhere.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.ClassicHuberLoss('Delta', 1.35);
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);   % = pseudo-residuals for MART
    %
    %   References:
    %     Friedman, J. (2001). Greedy function approximation: A gradient
    %       boosting machine. Annals of Statistics, 29(5), 1189-1232.
    %     Hastie, Tibshirani, Friedman (2009). ESL §10.3, §10.14.
    %
    %   See also: OceanStats.Loss.Regression.PseudoHuberLoss  (smooth variant)
    %             OceanStats.Loss.Regression.MSELoss
    %             OceanStats.Loss.Regression.MAELoss
    %             OceanStats.Loss.ILoss

    properties
        % DELTA  Threshold between quadratic (L2) and linear (L1) zones.
        % Default follows a common practical choice: 1.35 * sigma,
        % where sigma is the residual standard deviation. Here we expose it
        % as a free parameter and compute it from data in the experiment script.
        Delta double = 1.0
    end

    methods
        function obj = ClassicHuberLoss(options)
            % CLASSICHUBERLOSS  Constructor.
            %
            %   obj = ClassicHuberLoss('Delta', 1.35)
            arguments
                options.Delta double {mustBePositive} = 1.0
            end
            obj.Delta = options.Delta;
        end

        function L = forward(obj, Y_true, Y_pred)
            % FORWARD  Mean Huber loss over all samples.
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            % Piecewise: quadratic when |r| <= delta, linear outside
            inlier  = abs(r) <= d;
            L_inlier  = r(inlier).^2 / 2;
            L_outlier = d * (abs(r(~inlier)) - d/2);
            L = mean([L_inlier; L_outlier]);
        end

        function g = gradient(obj, Y_true, Y_pred)
            % GRADIENT  Negative gradient w.r.t. Y_pred (= pseudo-residuals).
            %
            %   In Step 2a of Algorithm 10.3, we compute:
            %     r_i = -dL/dF(y_i, F_{m-1}(x_i))
            %   which is the pseudo-residual:
            %     r_i = y_i - F              if |y_i - F| <= delta
            %     r_i = delta * sign(y_i-F)  if |y_i - F| >  delta
            %
            %   The effect: large residuals (outliers) are clipped to +/-delta,
            %   so they exert bounded influence on the next tree — this is the
            %   robustness property of Huber boosting.
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            % Negative gradient (stored positive for the GBM convention r = -grad)
            g = zeros(size(r));
            inlier     = abs(r) <= d;
            g(inlier)  = -r(inlier);               % quadratic zone: -dL/dF = r
            g(~inlier) = -d .* sign(r(~inlier));   % linear zone:   -dL/dF = delta*sign(r)
        end

        function h = hessian(obj, Y_true, Y_pred)
            % HESSIAN  d²L/dF² per sample.
            %
            %   1 in the quadratic zone (identical to MSE), 0 in the linear zone.
            %   NOTE: The zero Hessian outside delta means classic Huber is not
            %   suited to pure Newton-step methods. Use PseudoHuberLoss for that.
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            h = double(abs(r) <= d);   % 1 inside, 0 outside
        end

        function n = name(obj)
            n = sprintf("ClassicHuber (delta=%.2f)", obj.Delta);
        end

        function gamma = initPrediction(~, Y_true)
            % F_0 = mean(Y) — standard initialisation for symmetric losses.
            % Friedman (2001) also uses mean(Y) for the Huber case.
            gamma = mean(Y_true);
        end

        function gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
            % COMPUTELEAFCONSTANT  Friedman (2001) Eq. 11 leaf optimiser.
            %
            %   This is the closed-form approximation to:
            %     gamma_k = argmin_g sum_{i in leaf} L(y_i, F_{m-1}(x_i) + g)
            %
            %   Step 1: Start at the median of leaf residuals (optimal for MAE).
            %   Step 2: Correct with a bounded mean of deviations from the median.
            %           The min(., delta) clip prevents outliers from corrupting
            %           the correction — the robustness of Huber is preserved here.
            %
            %   Special cases:
            %     delta -> 0  → gamma = median(r)   [pure L1 / MAE]
            %     delta -> inf → gamma ≈ mean(r)    [pure L2 / MSE]
            r      = Y_true(leafMask) - F_prev(leafMask);
            r_med  = median(r);
            diff   = r - r_med;
            d      = obj.Delta;
            gamma  = r_med + mean(sign(diff) .* min(abs(diff), d));
        end
    end
end
