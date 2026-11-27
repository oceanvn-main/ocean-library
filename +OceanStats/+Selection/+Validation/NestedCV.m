classdef NestedCV < OceanStats.Base.IValidator
    % NESTEDCV Validates feature selection stability via Nested Cross-Validation.
    %
    % LOGIC:
    %   For each fold k:
    %     1. Split Data -> Train/Test
    %     2. Run Feature Selection on Train -> Best Model(s)
    %     3. Evaluate Best Model(s) on Test -> Error
    
    properties
        Name string = "Nested Cross-Validation"
        CVStrategy OceanStats.Base.ICrossValidator
    end
    
    methods
        function obj = NestedCV(cv_strategy)
            if nargin < 1, cv_strategy = OceanStats.Selection.CrossValidation.KFold(5); end
            obj.CVStrategy = cv_strategy;
        end
        
        function result = validate(obj, selector, strategy, X, Y)
            K = obj.CVStrategy.NumFolds;
            [N, P] = size(X);
            
            fprintf('   > %s (K=%d)...\n', obj.Name, K);
            
            % Storage: [ModelComplexities x Folds]
            % We assume the selector returns a history of size P+1 (0..P features).
            % If selector returns something else, we need to adapt. 
            % For BestSubsetSelector, it returns struct array of size P+1.
            fold_errors = zeros(P+1, K);
            
            for k = 1:K
                [train_idx, test_idx] = obj.CVStrategy.getIndices(N, k);
                
                % 1. Train (Select Features)
                fold_history = selector.runExhaustiveSearch(X(train_idx,:), Y(train_idx), strategy);
                
                % 2. Test (Evaluate)
                X_test = X(test_idx, :); 
                Y_test = Y(test_idx);
                
                for p = 1:(P+1)
                    m = fold_history(p);
                    if m.Size == 0
                        X_sub = zeros(sum(test_idx), 0);
                    else
                        X_sub = X_test(:, m.Mask);
                    end
                    y_pred = strategy.predict(m.Model, X_sub);
                    
                    % MSE
                    fold_errors(p, k) = strategy.evaluate(Y_test, y_pred) / length(Y_test);
                end
            end
            
            result = struct();
            result.MeanError = mean(fold_errors, 2);
            result.SE = std(fold_errors, 0, 2) / sqrt(K);
            result.Raw = fold_errors;
        end
    end
end
