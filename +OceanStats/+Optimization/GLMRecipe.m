classdef GLMRecipe < OceanStats.Optimization.IOptimizationRecipe
    % GLMRECIPE A universal solver for Generalized Linear Models (GLMs).
    %
    % This is a general-purpose optimization strategy designed to solve 
    % Maximum Likelihood Estimation (MLE) for any distribution in the 
    % Exponential Family using Iteratively Reweighted Least Squares (IRLS).
    %
    % This recipe uses the Strategy Pattern via ILinkFunction to handle:
    % - K=1: Gaussian (Identity) or Bernoulli (Sigmoid) Experts.
    % - K>1: Multinomial (Softmax) Gating Networks.
    
    properties (Access = private)
        Theta          % [D x K] Parameters (Beta or Gamma)
        X              % [N x D] Design matrix
        Y_target       % [N x K] Target responses (labels or responsibilities)
        H_weight       % [N x 1] Observation weights (responsibilities from parent)
        Link           % OceanStats.Base.ILinkFunction
        InitialTheta   % Snapshot for convergence
        RecipeName string
    end
    
    methods
        function obj = GLMRecipe(X, Y, link, options)
            arguments
                X, Y, link
                options.InitialTheta = []
                options.Weights = []
                options.Name = "Unified GLM Solver"
            end
            obj.X = X;
            obj.Y_target = Y;
            obj.Link = link;
            obj.RecipeName = options.Name;
            
            [N, D] = size(X);
            K = size(Y, 2);
            
            if isempty(options.InitialTheta)
                obj.Theta = 0.01 * randn(D, K);
            else
                obj.Theta = options.InitialTheta;
            end
            obj.InitialTheta = obj.Theta;
            
            if isempty(options.Weights)
                obj.H_weight = ones(N, 1);
            else
                obj.H_weight = options.Weights;
            end
        end
        
        function params = step(obj, ~)
            % Iteratively Reweighted Least Squares (IRLS) Step
            obj.InitialTheta = obj.Theta;
            
            % 1. Linear Predictor (eta)
            eta = obj.X * obj.Theta;
            
            % 2. Map to space mu (p)
            mu = obj.Link.apply(eta);
            
            % 3. Calculate weights w (Fisher info)
            w = obj.Link.weights(mu);
            
            % 4. Solve the Stable Weighted Normal Equations
            % Theory: Instead of z = eta + (Y-mu)/W, where W can cause problem of division by zeros, we solve:
            % (X'WX) Beta = X'W*eta + X'(Y-mu)
            % This avoids catastrophic precision loss when W is very small.
            
            error_matrix = obj.Y_target - mu;
            [~, D] = size(obj.X);
            K = size(obj.Y_target, 2);
            
            % We initialize WLS with the design matrix dimensions
            % We'll use resolveStably to handle the math safely.
            N = size(obj.X, 1);
            for k = 1:K
                % Use a dummy z as we'll overwrite it with solveStably
                wLS = OceanMath.LinearAlgebra.IncrementalWLS(zeros(N, D), ones(N,1), zeros(N,1));
                wLS.resolveStably(obj.X, w, eta(:, k), error_matrix(:, k), obj.H_weight);
                obj.Theta(:, k) = wLS.solveForBeta();
            end
            
            params = obj.getParams();
        end
        
        function [converged, delta] = isConverged(obj)
            delta = norm(obj.Theta - obj.InitialTheta, 'fro') / (norm(obj.InitialTheta, 'fro') + 1e-10);
            converged = delta < 1e-5;
        end
        
        function val = evaluate(obj, ~)
            % Generalized Negative Log-Likelihood
            eta = obj.X * obj.Theta;
            mu = obj.Link.apply(eta);
            % Use broad cross-entropy as a generic objective
            val = -sum(sum(obj.H_weight .* (obj.Y_target .* log(max(mu, 1e-10)))));
        end
        
        function params = getParams(obj)
            params = struct('Theta', obj.Theta);
        end
        
        function name = getName(obj)
            name = obj.RecipeName;
        end
    end
end
