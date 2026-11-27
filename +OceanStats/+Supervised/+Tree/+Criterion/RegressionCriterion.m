classdef RegressionCriterion < OceanStats.Supervised.Tree.Criterion.INodeCriterion
    % REGRESSIONCRITERION Node criterion for regression trees (RSS-based).
    %
    % Node value: mean(Y)
    % Node risk:  RSS = sum((Y - mean(Y))^2)
    % Split eval: Delegates to TreeSplitProblem (RSS minimization)
    % CV error:   sum((Y_true - Y_pred)^2)
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods
        function stats = computeNodeStats(~, Y)
            % COMPUTENODESTATS Returns mean and RSS for regression.
            N = length(Y);
            if N > 0
                val = mean(Y);
                risk = sum((Y - val).^2);
            else
                val = 0;
                risk = 0;
            end
            stats = struct('Value', val, 'Risk', risk);
        end

        function tf = isPure(~, Y)
            % ISPURE True if variance is negligible.
            tf = var(Y) < 1e-6;
        end

        function problem = createSplitProblem(~, X, Y, minLeaf, candidateGenerator)
            % CREATESPLITPROBLEM Delegates to existing TreeSplitProblem (RSS).
            arguments
                ~
                X
                Y
                minLeaf
                candidateGenerator = []
            end
            problem = OceanStats.Supervised.Tree.TreeSplitProblem(X, Y, minLeaf, [], candidateGenerator);
        end

        function err = cvError(~, Y_true, Y_pred)
            % CVERROR Residual Sum of Squares.
            err = sum((Y_true - Y_pred).^2);
        end
    end
end
