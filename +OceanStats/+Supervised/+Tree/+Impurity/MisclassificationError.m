classdef MisclassificationError < OceanStats.Supervised.Tree.Impurity.IImpurityMeasure
    % MISCLASSIFICATIONERROR Misclassification error impurity measure.
    %
    % Formula:
    %   MiscErr = 1 - max_k(p_mk)
    %           = 1 - p_mk(m)    where k(m) is the majority class.
    %
    % Properties:
    %   - Ranges from 0 (pure) to 1 - 1/K (maximally mixed).
    %   - NOT differentiable at transitions — less suitable for tree growing.
    %   - Primarily used for pruning and final evaluation, not splitting.
    %
    % Reference: Hastie, Tibshirani, Friedman — ESL, Section 9.2.3.
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods
        function val = compute(~, P)
            % COMPUTE MisclassificationError = 1 - max(p_k)
            %
            % Args:
            %   P (N x K or 1 x K): Matrix of probabilities, or vector of counts
            %
            % Returns:
            %   val (N x 1 double): Misclassification rate
            
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
            val = 1 - max(P, [], 2);
        end

        function n = name(~)
            n = 'Misclassification Error';
        end
    end
end
