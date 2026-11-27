classdef GiniIndex < OceanStats.Supervised.Tree.Impurity.IImpurityMeasure
    % GINIINDEX Gini impurity measure for classification trees.
    %
    % Formula:
    %   Gini = sum_{k=1}^{K} p_mk * (1 - p_mk)
    %        = 1 - sum_{k=1}^{K} p_mk^2
    %
    % Properties:
    %   - Ranges from 0 (pure) to 1 - 1/K (maximally mixed).
    %   - Differentiable — preferred for tree growing (gradient-based reasoning).
    %   - Equivalent to expected misclassification rate under random labeling.
    %
    % Reference: Breiman et al. (1984), CART.
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods
        function val = compute(~, P)
            % COMPUTE Gini = sum p_k * (1 - p_k) = 1 - sum p_k^2
            %
            % Args:
            %   P (N x K or 1 x K): Matrix of probabilities, or vector of counts
            %
            % Returns:
            %   val (N x 1 double): Gini impurity
            
            % If it's a single row and looks like raw counts (sum > 1.0001)
            if size(P, 1) == 1 && sum(P) > 1.0001
                N = sum(P);
                if N == 0
                    val = 0;
                    return;
                end
                P = P / N;
            end
            
            % Vectorized calculation
            val = 1 - sum(P.^2, 2);
        end

        function n = name(~)
            n = 'Gini Index';
        end
    end
end
