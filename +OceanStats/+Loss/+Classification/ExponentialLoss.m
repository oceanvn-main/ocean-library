classdef ExponentialLoss < OceanStats.Loss.ITreeLoss
    % EXPONENTIALLOSS  Exponential loss for binary classification.
    %
    %   CONVENTION:
    %     Y_true ∈ {-1, +1}  — signed class labels (AdaBoost convention)
    %     Y_pred             — raw score (positive → class +1, negative → class -1)
    %
    %   FORMULA (per sample):
    %     L(y, p)   = exp(-y * p)            [per sample]
    %     dL/dp     = -y * exp(-y * p)       [gradient]
    %     d²L/dp²   =  y^2 * exp(-y * p)    [hessian; since y^2=1, equals exp(-y*p)]
    %
    %   CONNECTION TO ADABOOST:
    %     The exponential loss is the theoretical loss function that AdaBoost.M1
    %     implicitly minimizes via coordinate descent. Gradient Boosting with
    %     ExponentialLoss and decision-stump weak learners recovers AdaBoost exactly.
    %
    %   IMPORTANT — LABEL CONVENTION:
    %     This loss requires Y_true ∈ {-1, +1}. If your labels are {0, 1},
    %     use CrossEntropyLoss instead.
    %
    %   WHEN TO USE:
    %     Binary classification problems where AdaBoost-style exponential weighting
    %     is desired. Less robust to outliers than Cross-Entropy (exponential tail
    %     grows fast), but yields the elegant AdaBoost solution.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Classification.ExponentialLoss();
    %     L    = loss.forward(Y_true, Y_pred);   % Y_true ∈ {-1,+1}
    %     g    = loss.gradient(Y_true, Y_pred);
    %     h    = loss.hessian(Y_true, Y_pred);
    %
    %   See also: OceanStats.Loss.Classification.CrossEntropyLoss
    %             OceanStats.Boosting.Algorithms.AdaBoostM1
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            L = mean(exp(-Y_true .* Y_pred));
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            g = -Y_true .* exp(-Y_true .* Y_pred);
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            obj.checkLabels_(Y_true);
            % y^2 = 1 for y in {-1, +1}
            h = exp(-Y_true .* Y_pred);
        end

        function n = name(~)
            n = "Exponential (AdaBoost)";
        end

        function gamma = initPrediction(~, Y_true)
            % INITPREDICTION  Optimal constant starting prediction.
            %   Minimizes sum(exp(-y * gamma)).
            %   Solution is 1/2 * log(N_pos / N_neg).
            N_pos = sum(Y_true == 1);
            N_neg = sum(Y_true == -1);
            if N_neg == 0
                gamma = 1.0; % Avoid Inf
            elseif N_pos == 0
                gamma = -1.0;
            else
                gamma = 0.5 * log(N_pos / N_neg);
            end
        end

        function gamma = computeLeafConstant(~, Y_true, F_prev, leafMask)
            % COMPUTELEAFCONSTANT  Optimal leaf output for Exponential Loss.
            %   Minimizes sum(w_i * exp(-y_i * gamma)) where w_i = exp(-y_i * F_i).
            %   Solution is 1/2 * log(W_pos / W_neg).
            y_leaf = Y_true(leafMask);
            f_leaf = F_prev(leafMask);
            
            w = exp(-y_leaf .* f_leaf);
            W_pos = sum(w(y_leaf == 1));
            W_neg = sum(w(y_leaf == -1));
            
            if W_neg == 0
                gamma = 1.0;
            elseif W_pos == 0
                gamma = -1.0;
            else
                gamma = 0.5 * log(W_pos / W_neg);
            end
        end
    end

    methods (Access = private)
        function checkLabels_(~, Y_true)
            % Enforce {-1, +1} label convention
            if ~all(Y_true == 1 | Y_true == -1)
                error('OceanStats:Loss:ExponentialLoss:InvalidLabels', ...
                    'ExponentialLoss requires Y_true in {-1, +1}. Got other values.');
            end
        end
    end
end
