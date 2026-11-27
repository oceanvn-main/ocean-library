classdef HeuristicStrategy < OceanCS.Parallel.ISchedulingStrategy
    % HEURISTICSTRATEGY implementations the standard cost-based heuristic.
    
    methods
        function [shouldParallel, numWorkers] = shouldParallelize(obj, numTasks, taskCost, payloadBytes)
            shouldParallel = false;
            numWorkers = 0;
            
            % 1. Check if Parallel Toolbox is available and pool exists
            pool = gcp('nocreate');
            if isempty(pool)
                return;
            end
            numWorkers = pool.NumWorkers;
            
            if numTasks < numWorkers
                if numTasks < 2, return; end
            end
            
            if nargin < 4, payloadBytes = 0; end
            
            % 2. Heuristic Calculation
            totalWork = numTasks * taskCost;
            
            % Threshold: 1e6 "ops"
            minWorkThreshold = 1e6; 
            
            % Data Transfer penalty
            transferPenalty = payloadBytes * 1e-6; 
            
            effectiveWork = totalWork - transferPenalty;
            
            if effectiveWork > minWorkThreshold
                shouldParallel = true;
            end
        end
    end
end
