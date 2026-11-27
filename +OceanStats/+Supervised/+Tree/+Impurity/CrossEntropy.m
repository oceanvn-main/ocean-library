classdef CrossEntropy < OceanStats.Supervised.Tree.Impurity.IImpurityMeasure
    % CROSSENTROPY Cross-entropy (deviance) impurity measure.
    %
    % Formula:
    %   Deviance = -sum_{k=1}^{K} p_mk * log(p_mk)
    %
    % Properties:
    %   - Ranges from 0 (pure) to log(K) (maximally mixed).
    %   - More sensitive to changes in class probabilities near 0 and 1
    %     compared to Gini — preferred for tree growing in many textbooks.
    %   - Convention: 0 * log(0) = 0.
    %
    % Reference: Hastie, Tibshirani, Friedman — ESL, Section 9.2.3.
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods
        function val = compute(~, P)
            % COMPUTE Deviance = -sum p_k * log2(p_k), with 0*log2(0) = 0.
            %
            % Args:
            %   P (N x K or 1 x K): Matrix of probabilities, or vector of counts
            %
            % Returns:
            %   val (N x 1 double): Cross-entropy impurity
            
            % If it's a single row and looks like raw counts (sum > 1.0001)
            if size(P, 1) == 1 && sum(P) > 1.0001
                N = sum(P);
                if N == 0
                    val = 0;
                    return;
                end
                P = P / N;
            end
            
            % Apply 0*log2(0) = 0 convention safely without throwing NaN
            P_log = P;
            P_log(P == 0) = 1; % log2(1) = 0, avoiding log2(0)=NaN
            
            % Vectorized calculation
            val = -sum(P .* log2(P_log), 2);
        end

        function n = name(~)
            n = 'Cross-Entropy';
        end
    end
end
