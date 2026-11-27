classdef MarginSquaredLoss < OceanStats.Loss.ILoss
    % MARGINSQUAREDLOSS  Squared Error for binary classification (margin-based).
    %
    %   CONVENTION:
    %     Y_true ∈ {-1, +1}  — signed class labels
    %     Y_pred             — raw decision score f(x)
    %
    %   FORMULA (per sample):
    %     margin  = y * f(x)
    %     L       = (1 - margin)^2
    %     dL/df   = -2 * y * (1 - margin)
    %     d²L/df² = 2 * y^2 = 2           (since y ∈ {-1, 1})
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Classification.MarginSquaredLoss();
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %     h    = loss.hessian(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Classification.HingeLoss
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            margin = Y_true .* Y_pred;
            L = mean((1 - margin).^2);
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            margin = Y_true .* Y_pred;
            g = -2 * Y_true .* (1 - margin);
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            h = 2 * ones(size(Y_pred));
        end

        function n = name(~)
            n = "Margin Squared";
        end
    end

    methods (Static, Access = private)
        function checkLabels_(Y_true)
            if ~all(Y_true == 1 | Y_true == -1)
                error('OceanStats:Loss:MarginSquaredLoss:InvalidLabels', ...
                    'MarginSquaredLoss requires Y_true in {-1, +1}. Got other values.');
            end
        end
    end
end
