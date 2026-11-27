classdef L2Regularizer < OceanStats.Regularization.IRegularizer
    % L2REGULARIZER  Ridge penalty: 0.5 * ||w||_2^2
    %
    %   Used to physically shrink the weights to prevent overfitting.
    %   The gradient is w, and the proximity operator scales w.
    %
    %   Author: Ocean Library | 2026-03-13

    methods
        function val = evaluate(~, w)
            val = 0.5 * sum(w.^2);
        end

        function grad = gradient(~, w)
            grad = w;
        end

        function w_prox = proximity(~, w, t, lambda)
            % PROXIMITY_L2  Analytic solution to:
            % argmin_x  0.5/t * ||x - w||^2 + lambda * 0.5 * ||x||^2
            %
            % => x - w + t * lambda * x = 0
            % => x * (1 + t * lambda) = w
            % => x = w / (1 + t * lambda)
            w_prox = w ./ (1 + t * lambda);
        end

        function name = getName(~)
            name = "L2 Ridge Penalty";
        end
    end
end
