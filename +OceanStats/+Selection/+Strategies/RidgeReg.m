classdef RidgeReg < OceanStats.Base.IModelStrategy
    % RIDGEREG Ridge Regression Strategy (L2 Penalty).
    %
    % MODEL:
    %   Min ||Y - Xw||^2 + Lambda * ||w||^2
    %
    % USAGE:
    %   strat = OceanStats.Selection.Strategies.RidgeReg(0.1);
    
    properties
        Name string = "Ridge Regression"
        SupportsAIC logical = true % Approximated via Effective Degrees of Freedom (df)
        Lambda (1, 1) double = 0.1 % Regularization Strength
    end
    
    methods
        function obj = RidgeReg(lambda)
            if nargin > 0, obj.Lambda = lambda; end
        end
        
        function model = train(obj, X, Y, Weights)
            % TRAIN Weighted Ridge Regression.
            % Model: Min || sqrt(W)(Y - Xw) ||^2 + Lambda * ||w||^2
                        
            % If X is empty (Null Model)
            if size(X, 2) == 0
                model = struct('Beta', [], 'Intercept', mean(Y), 'df', 1);
                return;
            end
            
            % Note: ridge returns b0 separately only if requested?
            % ridge(Y, X, k, 0) -> coefficients.
            % ridge centers and scales predictors to unit standard deviation.
            % Coeffs returned are for the scaled data, but 'ridge' documentation says
            % if '0' is passed as a flag it behaves differently?
            % Let's use closed form for transparency or robust built-in use.
            
            % Closed Form: w = (X'X + lambda*I)^-1 X'Y
            % Prerequisite: Center X and Y?
            % Professional libraries usually standardize X.
            
            % 1. Weighted Standardization
            sumW = sum(Weights);
            mu_X = sum(Weights .* X) / sumW;
            mu_Y = sum(Weights .* Y) / sumW;
            
            X_c = X - mu_X;
            Y_c = Y - mu_Y;
            
            % Weighted Variance: sum(w * (x-mu)^2) / sum(w)
            var_X = sum(Weights .* (X_c.^2)) / sumW;
            sigma_X = sqrt(var_X);
            sigma_X(sigma_X < 1e-9) = 1; 

            X_s = X_c ./ sigma_X;
            
            % 2. Solve Weighted Ridge
            % (X_s' W X_s + Lambda I) beta = X_s' W Y_c
            % Can be solved via data augmentation or direct normal equations.
            % Normal equations are usually fine for moderate dimensions.
            
            [~, P] = size(X_s);
            W = spdiags(Weights, 0, length(Weights), length(Weights));
            
            XTWX = X_s' * W * X_s;
            XTWY = X_s' * W * Y_c;
            
            I = eye(P);
            beta_s = (XTWX + obj.Lambda * I) \ XTWY;
            
            % 3. Effective Degrees of Freedom
            % df = tr( (X'WX + lI)^-1 X'WX )
            try
                % Efficient trace for small P
                H_part = (XTWX + obj.Lambda * I) \ XTWX;
                df = trace(H_part) + 1;
            catch
                 df = P + 1; % Fallback
            end
            
            % 4. Un-standardize Coefficients
            beta = beta_s ./ sigma_X';
            intercept = mu_Y - mu_X * beta;
            
            model = struct('Beta', beta, 'Intercept', intercept, 'df', df, 'Lambda', obj.Lambda);
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
