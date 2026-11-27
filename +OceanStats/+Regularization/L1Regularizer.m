classdef L1Regularizer < OceanStats.Regularization.IRegularizer
    % L1REGULARIZER  Lasso penalty: ||w||_1
    %
    %   Used to physically shrink the weights, creating sparsity.
    %   The subgradient is sign(w), and the proximity operator is 
    %   Soft Thresholding (iterative shrinking).
    %
    %   Author: Ocean Library | 2026-03-13

    methods
        function val = evaluate(~, w)
            val = sum(abs(w));
        end

        function grad = gradient(~, w)
            % Returns a subgradient for the non-smooth L1 norm
            % (0 at w=0)
            grad = sign(w);
        end

        function w_prox = proximity(~, w, t, lambda)
            % SOFT_THRESHOLDING  Analytic solution to:
            % argmin_x  0.5/t * ||x - w||_2^2 + lambda * ||x||_1
            %
            % => Shrinks each component of w towards zero by threshold (t * lambda).
            % => If |w| < threshold, sets it exactly to zero.
            threshold = t * lambda;
            w_prox = sign(w) .* max(abs(w) - threshold, 0);
        end

        function name = getName(~)
            name = "L1 Lasso Penalty";
        end
    end
end
