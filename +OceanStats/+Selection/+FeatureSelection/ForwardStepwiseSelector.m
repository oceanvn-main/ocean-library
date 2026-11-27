classdef ForwardStepwiseSelector < OceanStats.Base.IFeatureSelector
    % FORWARDSTEPWISESELECTOR Scalable Feature Selection.
    %
    % ALGORITHM:
    %   1. Start with Null Model.
    %   2. For each step k:
    %       a. Try adding each unused feature.
    %       b. Select the one that improves the metric (RSS/AIC) the most.
    %   3. Stop when no improvement or k=P.
    %
    % COMPLEXITY: O(P^2) vs BestSubset's O(2^P).
    
    properties
        Criterion string = "RSS" % "RSS", "AIC", "BIC"
        Verbose logical = false
    end
    
    methods
        function obj = ForwardStepwiseSelector(criterion)
            if nargin > 0, obj.Criterion = criterion; end
        end
        
        function [mask, history] = select(obj, X, Y, strategy)
            [N, P] = size(X);
            
            % Current State
            current_mask = false(1, P);
            available_indices = 1:P;
            
            % History Storage
            history(P+1) = struct('Size', [], 'Error', [], 'Mask', [], 'Model', []);
            
            % Step 0: Null Model
            null_mdl = strategy.train(zeros(N,0), Y);
            y_null = strategy.predict(null_mdl, zeros(N,0));
            current_err = strategy.evaluate(Y, y_null);
            
            history(1).Size = 0;
            history(1).Error = current_err;
            history(1).Mask = current_mask;
            history(1).Model = null_mdl;
            
            if obj.Verbose, fprintf('Step 0: Null Error = %.4f\n', current_err); end
            
            % Forward Loop
            for k = 1:P
                best_j = -1;
                best_new_err = inf;
                
                % Try adding each available feature
                for j = available_indices
                    % Temporary mask
                    test_mask = current_mask;
                    test_mask(j) = true;
                    
                    X_sub = X(:, test_mask);
                    mdl = strategy.train(X_sub, Y);
                    y_hat = strategy.predict(mdl, X_sub);
                    err = strategy.evaluate(Y, y_hat);
                    
                    if err < best_new_err
                        best_new_err = err;
                        best_j = j;
                    end
                end
                
                % Update State
                current_mask(best_j) = true;
                available_indices(available_indices == best_j) = [];
                current_err = best_new_err;
                
                % Save History
                % We need to reconstruct the model to save it (or we could have saved it in loop)
                % But since we only run P iterations, retraining once is cheap.
                history(k+1).Size = k;
                history(k+1).Error = current_err;
                history(k+1).Mask = current_mask;
                history(k+1).Model = strategy.train(X(:, current_mask), Y);
                
                if obj.Verbose
                    fprintf('Step %d: Added Feat %d, Error = %.4f\n', k, best_j, current_err); 
                end
            end
            
            % Default Return: The mask of the model with min AIC/BIC if applicable, 
            % or let the user decide from history.
            % For 'select', we return the full path usually in stepwise.
            % But IFeatureSelector demands a single mask.
            % Let's return the mask of the best AIC/BIC model found.
            
            mask = obj.findBestMask(history, strategy, N);
        end
        
        function mask = findBestMask(obj, history, strategy, N)
             if strategy.SupportsAIC
                 bics = inf(length(history), 1);
                 for i = 1:length(history)
                     m = history(i);
                     % Avoid error if model failed
                     if isempty(m.Model), continue; end
                     k_param = strategy.getComplexity(m.Model);
                     bics(i) = OceanStats.ModelSelection.bic(m.Error, k_param, 'N', N, 'Type', 'RSS');
                 end
                 [~, best_idx] = min(bics);
                 mask = history(best_idx).Mask;
             else
                 % Fallback: Return the mask with lowest error (usually the last one)
                 % Or implement an "Elbow" heuristic?
                 % For now, return final (full) mask and let Auditor show the path.
                 mask = history(end).Mask;
             end
        end
    end
end
