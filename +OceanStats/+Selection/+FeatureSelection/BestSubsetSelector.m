classdef BestSubsetSelector < OceanStats.Base.IFeatureSelector
    % BESTSUBSETSELECTOR Exhaustive Search for Feature Selection.
    %
    % OVERVIEW:
    %   Evaluates all 2^P - 1 subsets of features to find the best model 
    %   for each size k (1 to P).
    %   Returns the global winner (usually based on an internal criteria or just the mask).
    %   Ideally, it returns the *Pareto Frontier* of models.
    %
    % WARNING:
    %   Computational Complexity is O(2^P). Do not use for P > 20.
    
    properties
        UseParallel (1, 1) logical = true
        ParallelThreshold (1, 1) double = 500 % Minimum candidates to trigger parfor
    end
    
    methods
        function obj = BestSubsetSelector(useParallel)
            if nargin > 0, obj.UseParallel = useParallel; end
        end
        
        function [mask, history] = select(obj, X, Y, strategy)
            % SELECT Runs exhaustive search. 
            % Returns the mask of the single best model (lowest training error? or AIC?).
            % Actually, standard Best Subset returns the frontier.
            % Adaptation: We return the mask of the model with minimum AIC if supported,
            % or minimum error if not (though that's usually the full model).
            % Better: Return the Best Model of a specific size K or the global best?
            % Let's return the mask of the "best" model according to BIC/AIC, 
            % defaulting to the full model if no penalized metric is available.

            history = obj.runExhaustiveSearch(X, Y, strategy);
            
            % Heuristic to pick "Best" mask from History
            % If strategy supports AIC, pick min AIC.
            % Else, return full model (or handle externally).
            
            best_idx = length(history); % Default to largest
            
            if strategy.SupportsAIC
                 % Calculate BIC for all candidate sizes
                 % Note: history(k+1) is model of size k.
                 N = size(X, 1);
                 bics = inf(length(history), 1);
                 for i = 1:length(history)
                     m = history(i);
                     if isempty(m.Model), continue; end
                     
                     k_param = strategy.getComplexity(m.Model);
                     % We assume m.Error is RSS or -2LogL
                     % Use standard BIC.
                     bics(i) = OceanStats.ModelSelection.bic(m.Error, k_param, 'N', N, 'Type', 'RSS');
                 end
                 [~, best_idx] = min(bics);
            end
            
            mask = history(best_idx).Mask;
        end
        
        function best_models = runExhaustiveSearch(obj, X, Y, strategy)
            % RUNEXHAUSTIVESEARCH The core engine extracted from FeatureSelectionAudit.
            
            [N, P] = size(X);
            
            % Pre-allocate structure array
            % Index k+1 stores model of size k
            best_models(P+1) = struct('Size', [], 'Error', [], 'Mask', [], 'Model', []);
            
            % 1. Null Model (k=0)
            best_models(1).Size = 0;
            best_models(1).Mask = false(1, P);
            best_models(1).Model = strategy.train(zeros(N,0), Y);
            y_null = strategy.predict(best_models(1).Model, zeros(N,0));
            best_models(1).Error = strategy.evaluate(Y, y_null);
            
            % 2. Combinatorial Loop (k=1 to P)
            % Set up DataQueue for progress if parallel
            if obj.UseParallel && P > 5
                D = parallel.pool.DataQueue;
                afterEach(D, @(~) fprintf('.')); 
                % Silent progress for now to avoid cluttering command window in library code
            end
            
            for k = 1:P
                combs = nchoosek(1:P, k);
                num_cands = size(combs, 1);
                err_cands = inf(num_cands, 1);
                
                if obj.UseParallel && num_cands > obj.ParallelThreshold
                    % Parallel Execution
                    % To use parfor, we must broadcast the strategy.
                    % Objects are broadcast automatically.
                    
                    parfor i = 1:num_cands
                        cols = combs(i, :);
                        X_sub = X(:, cols);
                        
                        % Train & Evaluate
                        mdl = strategy.train(X_sub, Y);
                        y_hat = strategy.predict(mdl, X_sub);
                        err_cands(i) = strategy.evaluate(Y, y_hat);
                        
                        % if mod(i, 500) == 0, send(D, 1); end
                    end
                else
                    % Serial Execution
                    for i = 1:num_cands
                        cols = combs(i, :);
                        X_sub = X(:, cols);
                        mdl = strategy.train(X_sub, Y);
                        y_hat = strategy.predict(mdl, X_sub);
                        err_cands(i) = strategy.evaluate(Y, y_hat);
                    end
                end
                
                % Find Winner for size k
                [min_err, idx] = min(err_cands);
                mask = false(1, P);
                mask(combs(idx, :)) = true;
                
                % Store result
                % We re-train to get the model object (save RAM in parallel loop)
                best_models(k+1).Size = k;
                best_models(k+1).Error = min_err;
                best_models(k+1).Mask = mask;
                best_models(k+1).Model = strategy.train(X(:, mask), Y);
            end
        end
    end
end
