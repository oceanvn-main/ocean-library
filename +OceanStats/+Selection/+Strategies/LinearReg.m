classdef LinearReg < OceanStats.Base.IModelStrategy
    % LINEARREG Ordinary Least Squares Regression.
    %
    % MODEL:
    %   Min ||Y - Xw||^2
    %
    % USAGE:
    %   strat = OceanStats.Selection.Strategies.LinearReg();
    
    properties
        Name string = "Linear Regression"
        SupportsAIC logical = true
    end
    
    methods
        function obj = LinearReg()
        end
        
        function model = train(obj, X, Y, Weights)
            % Null Model check
            if size(X, 2) == 0
                model = struct('Beta', [], 'Intercept', mean(Y), 'df', 1);
                return;
            end
            
            % Handle optional weights
            if nargin < 4 || isempty(Weights)
                Weights = ones(size(X, 1), 1);
            end
            
            % Weighted Linear Regression: (X'WX)^-1 X'WY
            % Efficient implementation: sqrt(W)*X \ sqrt(W)*Y
            
            sqrtW = sqrt(Weights);
            X_design = [ones(size(X, 1), 1), X];
            
            % Apply weights
            X_w = sqrtW .* X_design;
            Y_w = sqrtW .* Y;
            
            beta_full = X_w \ Y_w;
            
            model = struct('Beta', beta_full(2:end), 'Intercept', beta_full(1), 'df', size(X,2)+1);
        end
        
        function y_pred = predict(obj, model, X)
             if isempty(model.Beta)
                y_pred = repmat(model.Intercept, size(X, 1), 1);
            else
                y_pred = X * model.Beta + model.Intercept;
            end
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            err = sum((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(obj, model)
            k = model.df;
        end
    end
end
