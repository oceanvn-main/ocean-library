classdef ExpectedLoss < OceanStats.Supervised.Tree.Impurity.IImpurityMeasure
    % EXPECTEDLOSS Expected misclassification cost impurity.
    %
    % Formula (Generalized Gini Index):
    %   I(t) = Sum over all pairs (True=j, Pred=i) where i is not j:
    %            L(i, j) * p(i|t) * p(j|t)
    %
    %   Where:
    %     j = True Class
    %     i = Predicted Class
    %     p(k|t) = Probability of class k in node t
    %              (i.e., count(k) / total_samples_in_node)
    %     L(i, j) = Cost of predicting i when true class is j
    %
    % This represents the expected cost of misclassification if we were to 
    % randomly assign labels based on the distribution p(k|t).
    %
    % Args:
    %   LossMatrix (K x K): Cost matrix.
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    properties
        LossMatrix
    end

    methods
        function obj = ExpectedLoss(lossMatrix)
            % EXPECTEDLOSS Constructor.
            obj.LossMatrix = lossMatrix;
        end

        function val = compute(obj, classCounts)
            % COMPUTE Expected Loss Impurity.
            
            N = sum(classCounts);
            if N == 0
                val = 0;
                return;
            end

            p = classCounts / N; % 1 x K
            
            % Formula: sum_{j} sum_{i != j} L(i, j) * p_i * p_j
            % Note: This is an "impurity" measure for splitting, distinct from the
            % "node risk" which uses min(Expected Loss).
            
            val = 0;
            K = length(p);
            
            % Double loop implementation for clarity (vectorized is p * L * p' if L symmetric-ish)
            % But let's be explicit:
            for j = 1:K % True class
                for i = 1:K % Predicted class (misinterpretation)
                    if i ~= j
                       val = val + obj.LossMatrix(i, j) * p(i) * p(j);
                    end
                end
            end
        end

        function n = name(~)
            n = 'Expected Loss (Gini-like)';
        end
    end
end
