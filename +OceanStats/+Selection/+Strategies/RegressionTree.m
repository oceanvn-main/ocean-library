classdef RegressionTree < OceanStats.Base.IModelStrategy
    % DECISIONTREE Strategy for Regression Trees.
    %   Wraps MATLAB's 'fitrtree' for the Audit Engine.
    
    properties
        Name = "Decision Tree (CART)"
        SupportsAIC = false % Trees do not support standard AIC/BIC
    end
    
    methods
        function model = train(~, X, Y)
            % TRAIN Wraps fitrtree
            % We set MinLeafSize to prevent overfitting on small subsets
            if isempty(X)
                % Null model case (predict mean)
                model = mean(Y);
            else
                % 'StandardSE' helps with pruning, usually good for audits
                model = fitrtree(X, Y, 'MinLeafSize', 5, 'Prune', 'off');
            end
        end
        
        function preds = predict(~, model, X)
            % PREDICT Wraps predict
            if isnumeric(model) && isscalar(model)
                % Null model case (stored as scalar mean)
                preds = repmat(model, size(X,1), 1);
            else
                if isempty(X)
                    preds = [];
                else
                    preds = predict(model, X);
                end
            end
        end
        
        function err = evaluate(~, Y_true, Y_pred)
            % Metric: Residual Sum of Squares (RSS)
            err = sum((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(~, model)
            % Complexity Proxy: Number of Branch Nodes + 1
            % (Rough equivalent to degrees of freedom for trees)
            if isnumeric(model)
                k = 1; % Intercept only
            else
                k = sum(model.IsBranch) + 1;
            end
        end
    end
end