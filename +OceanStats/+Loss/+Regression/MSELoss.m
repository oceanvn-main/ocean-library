classdef MSELoss < OceanStats.Loss.ITreeLoss
    % MSELOSS  Mean Squared Error loss for regression.
    %
    %   FORMULA:
    %     L(y, p)   = 0.5 * (y - p)^2        [per sample, factor 0.5 for clean gradient]
    %     dL/dp     = p - y                   [gradient]
    %     d²L/dp²   = 1                       [hessian, constant → Newton step = gradient]
    %
    %   WHEN TO USE:
    %     Standard regression when outliers are not a major concern.
    %     The 0.5 factor makes the gradient exactly (p - y), with no extra coefficient.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.MSELoss();
    %     L    = loss.forward(Y_true, Y_pred);   % scalar mean loss
    %     g    = loss.gradient(Y_true, Y_pred);  % N×1 residuals
    %     h    = loss.hessian(Y_true, Y_pred);   % N×1 ones
    %
    %   See also: OceanStats.Loss.Regression.PseudoHuberLoss
    %             OceanStats.Loss.Regression.ClassicHuberLoss
    %             OceanStats.Loss.Regression.MAELoss
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            r = Y_true - Y_pred;
            L = mean(0.5 * r .^ 2);
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            g = Y_pred - Y_true;          % N×1, positive when over-predicting
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            h = ones(size(Y_pred));       % constant Hessian = 1 everywhere
        end

        function n = name(~)
            n = "MSE";
        end

        function gamma = initPrediction(~, Y_true)
            % F_0 = mean(Y) — minimizes sum of squared errors
            gamma = mean(Y_true);
        end

        function gamma = computeLeafConstant(~, Y_true, F_prev, leafMask)
            % Leaf optimal = mean of residuals in this leaf.
            % For MSE the surrogate already gives this, but we compute
            % it explicitly for correctness with any F_prev.
            residuals = Y_true(leafMask) - F_prev(leafMask);
            gamma = mean(residuals);
        end
    end
end
