classdef HeuristicSelector < OceanStats.Supervised.PRIM.Selection.IBoxSelector
    % HEURISTICSELECTOR Simplistic greedy selection based entirely on training score.
    % FAST but prone to overfitting on tiny sample sizes.
    
    methods
        function bestIdx = select(~, boxesSeq, X, Y, objectiveFn, ~)
            % Evaluates objective monotonically avoiding expensive sub-calls 
            % (Greedy strategy prone to heavy overfitting). Ignore X, Y holdouts.
            scores = cellfun(@(b) b.Score, boxesSeq);
            [~, bestIdx] = max(scores);
        end
    end
end
