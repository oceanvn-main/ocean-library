classdef BinaryBCELoss < OceanStats.Loss.ILoss
    % BINARYBCELOSS  Binary Cross-Entropy loss for binary classification.
    %
    %   Y_pred is the RAW LOGIT (score before sigmoid), consistent with the
    %   ILoss convention. Sigmoid is applied internally.
    %
    %   FORMULAS:
    %     p     = sigmoid(Y_pred) = 1 / (1 + exp(-Y_pred))
    %     L     = -mean( Y_true.*log(p) + (1-Y_true).*log(1-p) )
    %     dL/dp = p - Y_true                  [gradient w.r.t. logit]
    %     d²L   = p .* (1 - p)                [hessian w.r.t. logit]
    %
    %   Y_true must be in {0, 1}.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Classification.BinaryBCELoss();
    %     L    = loss.forward(Y_true, Y_pred);    % scalar
    %     g    = loss.gradient(Y_true, Y_pred);   % N x 1
    %     h    = loss.hessian(Y_true, Y_pred);    % N x 1
    %
    %   See also: OceanStats.Loss.ILoss
    %             OceanStats.Loss.Classification.CrossEntropyLoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            p   = 1 ./ (1 + exp(-Y_pred));
            eps = 1e-12;   % Clip to avoid log(0)
            p   = max(min(p, 1 - eps), eps);
            L   = -mean(Y_true .* log(p) + (1 - Y_true) .* log(1 - p));
        end

        function g = gradient(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            p = 1 ./ (1 + exp(-Y_pred));
            g = p - Y_true;   % N x 1
        end

        function h = hessian(obj, Y_true, Y_pred)
            obj.checkDims(Y_true, Y_pred);
            p = 1 ./ (1 + exp(-Y_pred));
            h = max(p .* (1 - p), 1e-8);   % N x 1, clipped for stability
        end

        function n = name(~)
            n = "BinaryCrossEntropy";
        end
    end
end
