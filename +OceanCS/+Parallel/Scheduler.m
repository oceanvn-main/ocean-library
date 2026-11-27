classdef Scheduler
    % SCHEDULER Determines execution strategy (Serial vs. Parallel)
    %
    % Uses a cost model to estimate if parallelization overhead is justified.
    %
    % heuristic:
    %   Speedup = (CostPerTask * NumTasks) / (NumWorkers * Efficiency)
    %   if Speedup > (CostPerTask * NumTasks), then Parallelize? 
    %   Actually simpler: 
    %   ParallelTime = (TotalWork / NumWorkers) + Overhead
    %   SerialTime = TotalWork
    %   if ParallelTime < SerialTime -> Parallel
    %
    % Author: Ocean Library System
    % Date: 2026-02-18
    
    methods (Static)
        function setStrategy(strategy)
            % SETSTRATEGY Sets the current scheduling strategy.
            % Args:
            %   strategy (OceanCS.Parallel.ISchedulingStrategy)
            arguments
                strategy (1,1) OceanCS.Parallel.ISchedulingStrategy
            end
            OceanCS.Parallel.Scheduler.getSetStrategy(strategy);
        end
        
        function strategy = getStrategy()
            strategy = OceanCS.Parallel.Scheduler.getSetStrategy();
        end
        
        function tf = mockParallel(enable)
            % MOCKPARALLEL Helper to force parallel for testing.
            % Uses a simplified MockStrategy or internal override.
            % For simplicity, we just swap to a MockStrategy-like behavior or generic Override.
            % But to keep it strategy-compliant, let's just use a flag here that overrides 
            % BEFORE calling strategy, or we implement a MockStrategy class.
            
            persistent p_mock
            if nargin > 0
                p_mock = enable;
            end
            if isempty(p_mock)
                p_mock = false;
            end
            tf = p_mock;
        end

        function [shouldParallel, numWorkers] = shouldParallelize(numTasks, taskCost, payloadBytes)
            % SHOULDPARALLELIZE Decides between serial and parallel execution via Strategy.
            
            import OceanCS.Parallel.Scheduler
            
            % 1. Test Override
            if Scheduler.mockParallel()
                shouldParallel = true;
                numWorkers = 1;
                return;
            end
            
            % 2. Delegate to Strategy
            if nargin < 3, payloadBytes = 0; end
            
            strategy = Scheduler.getStrategy();
            [shouldParallel, numWorkers] = strategy.shouldParallelize(numTasks, taskCost, payloadBytes);
        end
    end
    
    methods (Static, Access = private)
        function out = getSetStrategy(newStrategy)
            persistent currentStrategy
            if isempty(currentStrategy)
                % Default
                currentStrategy = OceanCS.Parallel.HeuristicStrategy();
            end
            if nargin > 0
                currentStrategy = newStrategy;
            end
            out = currentStrategy;
        end
    end
end
