classdef PLSReg < OceanStats.Base.IModelStrategy
    % PLSREG Partial Least Squares Regression.
    %
    % MODEL:
    %   Regresses Y on Latent Variables that explain variance in both X and Y.
    %
    % USAGE:
    %   strat = OceanStats.Selection.Strategies.PLSReg(3);
    
    properties
        Name string = "Partial Least Squares"
        SupportsAIC logical = false % Difficult to define K
        NumComponents (1, 1) double = 2
    end
    
    methods
        function obj = PLSReg(n_comp)
            if nargin > 0, obj.NumComponents = n_comp; end
        end
        
        function model = train(obj, X, Y)
            K = min(obj.NumComponents, size(X, 2));
            
            % plsregress(X, Y, ncomp)
            % Returns:
            %   XL, YL, XS, YS, BETA, PCTVAR, MSE, stats
            % BETA is (P+1)-by-M matrix of regression coefficients.
            % First row is intercept.
            [~, ~, ~, ~, BETA] = plsregress(X, Y, K);
            
            model = struct('Beta', BETA, 'K', K);
        end
        
        function y_pred = predict(obj, model, X)
            % plsregress returns beta with intercept at row 1.
            % Y_hat = [ones(N,1) X] * Beta
            
            y_pred = [ones(size(X, 1), 1), X] * model.Beta;
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            err = sum((Y_true - Y_pred).^2);
        end
        
        function k = getComplexity(obj, model)
            k = model.K; % Rough approximation
        end
    end
end
