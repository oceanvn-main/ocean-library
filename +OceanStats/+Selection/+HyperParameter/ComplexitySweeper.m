classdef ComplexitySweeper < handle
    % COMPLEXITYSWEEPER Grid Search for Single Hyperparameter.
    %
    % OVERVIEW:
    %   Iterates a parameter (usually complexity/degree) over a range,
    %   training and evaluating a strategy at each level.
    %   Computes AIC, AICc, BIC, and RSS.
    
    properties
        UseParallel (1, 1) logical = true
        BootstrapIters (1, 1) double = 0
    end
    
    methods
        function obj = ComplexitySweeper(useParallel, bootIters)
            if nargin > 0, obj.UseParallel = useParallel; end
            if nargin > 1, obj.BootstrapIters = bootIters; end
        end
        
        function results = sweep(obj, X, Y, strategy, param_name, range)
            % SWEEP Runs the grid search.
            % Inputs:
            %   strategy: IModelStrategy (must have a property 'param_name' or we inject it?)
            %   Ideally, the strategy should rely on a Factory or we clone it.
            %   
            %   BETTER ARCHITECTURE:
            %   The Strategy passed here is a FACTORY FUNCTION? 
            %   Or we assume the strategy has a method 'setParam(name, val)'?
            %   
            %   Legacy ComplexityAudit passed a function handle: TrainFcn(X,Y,level).
            %   To align with IModelStrategy, we should probably pass a 
            %   Strategy Factory: @(level) OceanStats.ModelSelection.PolyReg(level).
            %
            %   Or, if we pass an instance, we assume we can modify it.
            %   Let's use a Factory Function approach for maximum flexibility.
            %   
            %   New Signature: sweep(X, Y, strategyFactory, range)
            
            [N, ~] = size(X);
            num_levels = length(range);
            
            % Pre-allocate
            aics = zeros(num_levels, 1);
            bics = zeros(num_levels, 1);
            rss_vals = zeros(num_levels, 1);
            ks = zeros(num_levels, 1);
            models = cell(num_levels, 1);
            
            % Parallel or Serial
            % For simplicity in this implementation, let's stick to Serial first unless critical.
            % Actually, the legacy code used parfor. Let's try to support it.
            
            % We need to broadcast the factory.
            factory = strategy; 
            
            for i = 1:num_levels
                level = range(i);
                
                % 1. Instantiate Strategy for this level
                strat = factory(level); 
                
                % 2. Train
                mdl = strat.train(X, Y);
                models{i} = mdl;
                
                % 3. Evaluate
                y_pred = strat.predict(mdl, X);
                rss = strat.evaluate(Y, y_pred);
                rss_vals(i) = rss;
                
                k = strat.getComplexity(mdl);
                ks(i) = k;
                
                % 4. Compute Metrics
                % AICc (Corrected) is safer generally
                aics(i) = OceanStats.ModelSelection.aic(rss, k, 'N', N, 'Variant', 'AICc');
                bics(i) = OceanStats.ModelSelection.bic(rss, k, 'N', N, 'Type', 'RSS');
            end
            
            % Package Results
            results = table(range(:), rss_vals, aics, bics, ks, ...
                'VariableNames', {param_name, 'RSS', 'AIC', 'BIC', 'K'});
                
            % Identify Winners
            [~, idx_aic] = min(aics);
            [~, idx_bic] = min(bics);
            
            results.Properties.UserData = struct(...
                'BestAIC', range(idx_aic), ...
                'BestBIC', range(idx_bic), ...
                'Models', {models});
        end
    end
end
