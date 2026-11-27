classdef NNLS < OceanStats.Base.IModelStrategy
    % NNLS Non-Negative Least Squares Strategy.
    %
    % OVERVIEW:
    %   Fits a linear model min ||Y - Xw||^2 subject to w >= 0.
    %   Dependent on MATLAB's lsqnonneg.
    %
    % COMPLIANCE:
    %   Implements IModelStrategy.
    %   SupportsAIC = false (Standard formula assumes unconstrained OLS).
    
    properties
        Name string = "Non-Negative Least Squares"
        SupportsAIC logical = false
    end
    
    methods
        function obj = NNLS()
            % Constructor
        end
        
        function model = train(~, X, Y)
            % TRAIN Fits NNLS.
            % X: [N x M] Design Matrix
            % Y: [N x 1] Target
            
            % lsqnonneg(C, d) solves min ||C*x - d||
            % So C=X, d=Y
            
            % Note: NNLS does not automatically add intercept unless X has column of ones.
            % For Stacking, X is usually predictions from base models, so we usually NOT want an intercept
            % (or we want weights to sum to 1). 
            % Since this is a generic strategy, we assume X is the full design matrix provided by caller.
            
            weights = lsqnonneg(X, Y);
            model.Coeffs = weights;
        end
        
        function preds = predict(~, model, X)
            preds = X * model.Coeffs;
        end
        
        function err = evaluate(~, Y_true, Y_pred)
             err = mean((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(~, model)
            k = sum(model.Coeffs > 0); % Number of active features
        end
    end
end
