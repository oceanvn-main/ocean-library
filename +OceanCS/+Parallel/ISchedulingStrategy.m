classdef (Abstract) ISchedulingStrategy < handle
    % ISCHEDULINGSTRATEGY Interface for parallel execution decision logic.
    %
    % Implementations define how to decide between serial and parallel execution.
    
    methods (Abstract)
        function [shouldParallel, numWorkers] = shouldParallelize(obj, numTasks, taskCost, payloadBytes)
            % SHOULDPARALLELIZE Decides if parallel execution is beneficial.
            %
            % Args:
            %   numTasks (double): Number of tasks.
            %   taskCost (double): Cost per task.
            %   payloadBytes (double): Data size per task.
            %
            % Returns:
            %   shouldParallel (logical)
            %   numWorkers (double)
        end
    end
end
