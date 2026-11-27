classdef (Abstract) IMMRecipe < OceanStats.Optimization.IOptimizationRecipe
    % IMMRECIPE Abstract Base Class for Minorize-Maximize Recipes.
    %
    % This template enforces the MM structure:
    % 1. Construct a surrogate function (Minorizer) that touches the objective.
    %    - Method: minorize(data) -> aux_data
    % 2. Maximize this surrogate function.
    %    - Method: maximize(data, aux_data) -> params
    %
    % By inheriting from this, developers are forced to think in terms of
    % "Construction" (E-step) and "Maximization" (M-step).
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    methods
        function params = step(obj, data)
            % STEP Template Method for MM Algorithm.
            % Executes the standard Minorize-Maximize loop.
            
            % 1. Construction Step (Minorize)
            % Returns auxiliary data (e.g., Responsibilities, Q-function params)
            aux_data = obj.minorize(data);
            
            % 2. Maximization Step
            % Returns new parameters based on auxiliary data
            params = obj.maximize(data, aux_data);
            
            % 3. Update Model
            % Apply the new parameters to the underlying model
            obj.updateModel(params);
        end
    end
    
    methods (Abstract, Access = protected)
        % Create the surrogate function (e.g., E-Step)
        [aux_data] = minorize(obj, data);
        
        % Maximize the surrogate function (e.g., M-Step)
        [params] = maximize(obj, data, aux_data);
        
        % Update the specific model with generic params
        updateModel(obj, params);
    end
end
