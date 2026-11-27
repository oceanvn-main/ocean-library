classdef HingeLoss < OceanStats.Loss.ILoss
    % HINGELOSS  Hinge loss for binary classification (SVM-style).
    %
    %   CONVENTION:
    %     Y_true ∈ {-1, +1}  — signed class labels (same as ExponentialLoss)
    %     Y_pred             — raw decision score
    %
    %   FORMULA (per sample):
    %     margin  = y * p
    %     L       = max(0, 1 - margin)          [hinge — zero if correctly classified
    %                                             by more than margin 1]
    %     dL/dp   = -y  if margin < 1, else 0   [subgradient]
    %     d²L/dp² = 0                           [hessian; undefined at margin=1]
    %
    %   CONNECTION TO SVM:
    %     The SVM objective minimizes sum of hinge losses + L2 regularization.
    %     The hinge becomes zero ("support" is irrelevant) for well-classified points
    %     with margin ≥ 1 — only the "support vectors" contribute.
    %
    %   LIMITATION:
    %     Hessian is zero almost everywhere — not suitable for Newton-step methods
    %     (XGBoost-style). Gradient descent works fine.
    %
    %   WHEN TO USE:
    %     Binary classification, especially when SVM-like margin maximization is desired.
    %     Y_true must be {-1, +1}.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Classification.HingeLoss();
    %     L    = loss.forward(Y_true, Y_pred);
    %     g    = loss.gradient(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Classification.CrossEntropyLoss
    %             OceanStats.Loss.Classification.ExponentialLoss
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            margin = Y_true .* Y_pred;
            L = mean(max(0, 1 - margin));
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            margin = Y_true .* Y_pred;
            % Subgradient: -y where margin < 1, else 0
            g = zeros(size(Y_pred));
            g(margin < 1) = -Y_true(margin < 1);
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            % Hessian is 0 everywhere (non-smooth at margin=1)
            h = zeros(size(Y_pred));
        end

        function n = name(~)
            n = "Hinge (SVM)";
        end
    end

    methods (Static, Access = private)
        function checkLabels_(Y_true)
            if ~all(Y_true == 1 | Y_true == -1)
                error('OceanStats:Loss:HingeLoss:InvalidLabels', ...
                    'HingeLoss requires Y_true in {-1, +1}. Got other values.');
            end
        end
    end
end
