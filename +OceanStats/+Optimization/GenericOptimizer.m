classdef GenericOptimizer < handle
    % GENERICOPTIMIZER The Optimization Engine (The Chef).
    %
    % This class acts as a highly optimized execution loop. It possesses 
    % zero knowledge of the underlying mathematics. It strictly manages 
    % control flow, memory preallocation, convergence checking, and timing.
    %
    % Author: Ocean Library System
    % Date: 2026-02-11
    
    properties
        MaxIter (1,1) double = 100
        Verbose (1,1) logical = true
        
        % Memory Guardrails
        StoreParamHistory (1, 1) logical = false % Disabled by default
        KeepBestOnly (1, 1) logical = true       % Enabled by default
        
        History struct
        BestParams struct       % Strictly holds the single best state
        BestObjective (1, 1) double = Inf % Assuming minimization
    end
    
    methods
        function obj = GenericOptimizer(options)
            arguments
                options.MaxIter = 100;
                options.Verbose = true;
                options.StoreParamHistory = false; 
                options.KeepBestOnly = true;
            end
            obj.MaxIter = options.MaxIter;
            obj.Verbose = options.Verbose;
            obj.StoreParamHistory = options.StoreParamHistory;
            obj.KeepBestOnly = options.KeepBestOnly;
        end
        
        function [final_params, history] = run(obj, recipe, data)
            % RUN Executes the optimization loop via the Strategy Pattern.
            
            % 1. Architectural Guardrail
            if ~isa(recipe, 'OceanStats.Optimization.IOptimizationRecipe')
                error('Recipe must strictly implement IOptimizationRecipe.');
            end
            
            % -------------------------------------------------------------
            % OPTIMIZATION: Memory Preallocation
            % -------------------------------------------------------------
            empty_struct = struct('Iteration', 0, 'Objective', 0, 'Time', 0);
            
            % If the user explicitly accepts the memory risk, add the field
            if obj.StoreParamHistory
                empty_struct.Params = [];
            end
            
            obj.History = repmat(empty_struct, obj.MaxIter + 1, 1);
            
            startTime = tic;
            
            % 2. Evaluate and Record Initial State (Iteration 0)
            current_val = recipe.evaluate(data);
            current_params = recipe.getParams();
            
            % Safely initialize the Best trackers with Iteration 0
            obj.BestObjective = current_val; 
            obj.BestParams = current_params;
            
            obj.record(1, 0, current_params, current_val, 0); 
            
            if obj.Verbose
                fprintf('Starting Optimization: %s\n', recipe.getName());
                fprintf('Iter 0: Objective = %.6f\n', current_val);
            end
            
            % 3. The Execution Loop
            actual_iterations = 0; % Track true loop count for truncation
            
            for iter = 1:obj.MaxIter
                iterStart = tic;
                actual_iterations = iter;
                
                % --- THE MATH (Delegated strictly to the Recipe) ---
                recipe.step(data); % Scientist, let's do the heavy math job here
                
                % --- EVALUATION ---
                current_val = recipe.evaluate(data);
                current_params = recipe.getParams();
                
                % --- CACHING ---
                if obj.KeepBestOnly && (current_val < obj.BestObjective)
                    obj.BestObjective = current_val; % Fixed typo here
                    obj.BestParams = current_params; % Overwrite the single slot
                end
                
                duration = toc(iterStart);
                
                % --- RECORDING (O(1) memory insertion via index) ---
                obj.record(iter + 1, iter, current_params, current_val, duration);
                
                if obj.Verbose
                    fprintf('Iter %d: Objective = %.6f\n', iter, current_val);
                end
                
                % --- CONVERGENCE CHECK (Delegated to the Recipe) ---
                [converged, metric] = recipe.isConverged();
                if converged
                    if obj.Verbose
                        fprintf('Convergence Reached: %s\n', metric);
                    end
                    break;
                end
            end
            
            % 4. Finalization & Memory Truncation
            if obj.KeepBestOnly
                final_params = obj.BestParams;
            else
                final_params = current_params;
            end
            
            % Slice the preallocated array to drop unused iterations
            history = obj.History(1 : actual_iterations + 1); 
            obj.History = history; % Update internal state
            
            totalTime = toc(startTime);
            if obj.Verbose
                fprintf('Optimization Finished in %.4f seconds.\n', totalTime);
            end
        end
        
        function record(obj, memory_idx, iter, params, val, time)
            % RECORD Inserts state into the preallocated struct array.
            obj.History(memory_idx).Iteration = iter;
            obj.History(memory_idx).Objective = val;
            obj.History(memory_idx).Time = time;
            
            % Memory Guard: Only insert if the field physically exists
            if obj.StoreParamHistory
                obj.History(memory_idx).Params = params;
            end
        end
    end
end