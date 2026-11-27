classdef PenalizedRegression < handle
    % PENALIZEDREGRESSION Generalized Ridge Regression with custom penalty
    %
    % Description:
    %   Solves the penalized linear regression problem:
    %   beta = argmin ||y - X*beta||^2 + lambda * beta^T * Omega * beta
    %
    %   Unlike standard Ridge Regression which assumes Omega = I (Identity),
    %   this class allows arbitrary positive semi-definite penalty matrices 
    %   Omega. This is required for models like Penalized Discriminant 
    %   Analysis (PDA) which enforce continuous spatial smoothness.
    %
    % Usage:
    %   Omega = OceanMath.LinearAlgebra.Optimization.generateSpatialPenalty(16, 16);
    %   model = OceanStats.Supervised.Regression.PenalizedRegression('Lambda', 0.1, 'PenaltyMatrix', Omega);
    %   model.fit(X, y);
    %   preds = model.predict(X_test);
    
    properties
        Lambda (1,1) double {mustBeNonnegative} = 1.0
        PenaltyMatrix double % p x p symmetric positive semi-definite matrix
        
        Beta % Fitted coefficients (p+1 x 1, first element is intercept)
    end
    
    methods
        function obj = PenalizedRegression(varargin)
            p = inputParser;
            addParameter(p, 'Lambda', 1.0);
            addParameter(p, 'PenaltyMatrix', []);
            parse(p, varargin{:});
            
            obj.Lambda = p.Results.Lambda;
            obj.PenaltyMatrix = p.Results.PenaltyMatrix;
        end
        
        function fit(obj, X, y)
            % FIT Trains the penalized regression model
            
            [N, p] = size(X);
            
            if isempty(obj.PenaltyMatrix)
                % Default to standard Ridge (L2 penalty)
                Omega = speye(p);
            else
                Omega = obj.PenaltyMatrix;
                if size(Omega, 1) ~= p || size(Omega, 2) ~= p
                    error('PenaltyMatrix must be of size p x p where p is the number of columns in X.');
                end
            end
            
            % Augment X with intercept
            X_aug = [ones(N, 1), double(X)];
            
            % Augment Omega to not penalize the intercept
            % Omega_aug = [0, 0; 0, Omega]
            if issparse(Omega)
                Omega_aug = blkdiag(sparse(0), Omega);
            else
                Omega_aug = blkdiag(0, Omega);
            end
            
            % Solve normal equations: (X^T X + lambda * Omega) * beta = X^T y
            % Using mldivide (\) for numerical stability over inv()
            LHS = (X_aug' * X_aug) + (obj.Lambda * Omega_aug);
            RHS = X_aug' * double(y);
            
            obj.Beta = LHS \ RHS;
        end
        
        function y_pred = predict(obj, X)
            % PREDICT Generates continuous predictions
            if isempty(obj.Beta)
                error('Model must be fitted before calling predict.');
            end
            
            N = size(X, 1);
            X_aug = [ones(N, 1), double(X)];
            y_pred = X_aug * obj.Beta;
        end
    end
end
