classdef SerialStrategy < OceanCS.Parallel.ISchedulingStrategy
    % SERIALSTRATEGY Always recommends serial execution.
    % Used as a control in benchmarks.
    
    methods
        function [shouldParallel, numWorkers] = shouldParallelize(obj, numTasks, taskCost, payloadBytes)
            shouldParallel = false;
            numWorkers = 0;
        end
    end
end
