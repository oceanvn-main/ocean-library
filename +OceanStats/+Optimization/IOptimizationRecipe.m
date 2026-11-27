classdef (Abstract) IOptimizationRecipe < handle
    % IOPTIMIZATIONRECIPE Interface for Optimization Strategies.
    %
    % A Recipe defines the mathematics of *how* to take a step in an
    % optimization landscape. It abstracts the "Science" away from the
    % "Engineering" of the optimization loop.
    %
    % Methods:
    %   step(data) -> params: Takes one optimization step.
    %   isConverged() -> bool: Checks if the latest step met criteria.
    %   evaluate() -> val: Returns the current objective function value.
    %   getParams() -> struct: Returns current parameters.
    %   getName() -> string: Name of the recipe (e.g., "EM for GMM").
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    methods (Abstract)
        % Take one step of optimization
        % Returns the updated parameters (optional return, state should be updated internal to recipe/model)
        [params] = step(obj, data);
        
        % Check for convergence (e.g., based on tolerance or max change)
        [converged, metric_val] = isConverged(obj);
        
        % Evaluate current objective function (e.g., Log Likelihood)
        [val] = evaluate(obj, data);
        
        % Get current parameters snapshot
        [params] = getParams(obj);
        
        % Get Recipe Name
        [name] = getName(obj);
    end
end
