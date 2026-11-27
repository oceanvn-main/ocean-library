classdef PseudoHuberLoss < OceanStats.Loss.ITreeLoss
    % PSEUDOHUBERLOSS  Pseudo-Huber (smooth) loss for robust regression.
    %
    %   WHAT IT IS:
    %     A smooth, everywhere-differentiable approximation of the classic
    %     piecewise Huber loss. The curve is L2-shaped near zero and L1-shaped
    %     far from zero, but without the sharp kink at the transition.
    %
    %   FORMULA (per sample, using residual r = y - p):
    %     L(y,p)  = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
    %     dL/dp   = -r / sqrt(1 + (r/delta)^2)           [gradient w.r.t. p]
    %     d²L/dp² =  1 / (1 + (r/delta)^2)^(3/2)        [hessian]
    %
    %   WHY PSEUDO-HUBER vs CLASSIC HUBER (see also ClassicHuberLoss):
    %     Classic Huber is piecewise — quadratic for |r| <= delta, linear beyond.
    %     The kink at the transition makes it non-smooth, which can cause issues
    %     in Newton-step or second-order boosting.
    %     Pseudo-Huber is fully smooth everywhere, making it well-suited for
    %     gradient methods that rely on a non-zero, well-behaved Hessian.
    %
    %   PARAMETER:
    %     Delta — controls the transition from L2 (near 0) to L1 (far from 0).
    %             - Small Delta → more robust (closer to MAE)
    %             - Large Delta → closer to MSE
    %             - Typical range: 0.5 to 3.0 depending on target scale.
    %
    %   WHEN TO USE:
    %     Regression with outliers when you need a smooth Hessian (e.g., for
    %     second-order / Newton-step boosting). For the textbook MART algorithm
    %     (Friedman 2001, ESL §10.14), use ClassicHuberLoss instead.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.PseudoHuberLoss('Delta', 1.5);
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %     h    = loss.hessian(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Regression.ClassicHuberLoss   (textbook Huber)
    %             OceanStats.Loss.Regression.MSELoss
    %             OceanStats.Loss.Regression.MAELoss
    %             OceanStats.Loss.ILoss

    properties
        % DELTA  Transition point between L2 and L1 regimes (default: 1.0)
        Delta double = 1.0
    end

    methods
        function obj = PseudoHuberLoss(options)
            % PSEUDOHUBERLOSS  Constructor with named options.
            %
            %   obj = PseudoHuberLoss('Delta', 1.5)
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
            % Negative gradient w.r.t. Y_pred (pseudo-residuals for boosting)
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            g = -r ./ sqrt(1 + (r ./ d).^2);
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            d = obj.Delta;
            h = 1 ./ (1 + (r ./ d).^2).^(3/2);
        end

        function n = name(obj)
            n = sprintf("PseudoHuber (delta=%.2f)", obj.Delta);
        end

        function gamma = initPrediction(~, Y_true)
            % F_0 = mean(Y) — a reasonable starting point for smooth losses.
            gamma = mean(Y_true);
        end

        function gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
            % Approximation of Friedman's (2001) Huber leaf estimator.
            % NOTE: This formula is derived for the CLASSIC Huber loss, not
            % pseudo-Huber. It is used here as a practical approximation
            % because the true pseudo-Huber leaf optimum has no closed form.
            % For theoretically correct leaf values, use ClassicHuberLoss.
            %
            %   gamma_k ≈ median(r) + mean(sign(r - median(r)) * min(|r - median(r)|, delta))
            %
            % This reduces to the median for pure L1 (delta -> 0) and the mean for L2.
            r       = Y_true(leafMask) - F_prev(leafMask);
            r_med   = median(r);
            diff    = r - r_med;
            d       = obj.Delta;
            gamma   = r_med + mean(sign(diff) .* min(abs(diff), d));
        end
    end
end
