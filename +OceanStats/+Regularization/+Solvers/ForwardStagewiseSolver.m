classdef ForwardStagewiseSolver < OceanStats.Regularization.Solvers.IRegularizationSolver
    % FORWARDSTAGEWISESOLVER Infinitesimal Forward Stagewise Linear Regression.
    %
    %   OVERVIEW:
    %     Implements Algorithm 16.1 from ESL (Section 16.2.3).
    %     This traces the path of coefficients by making tiny updates
    %     along the feature most correlated with the current residual.
    %     This acts as a slow-to-overfit regularizer similar to LARS/Lasso.
    %
    %   See also: OceanStats.Supervised.Regression.RegularizedLinearModel
    
    methods
        function [beta_path, extra_info] = solve(~, X, Y, options)
            [~, p] = size(X);
            
            % Default options if not provided
            if isfield(options, 'NumIterations')
                M = options.NumIterations;
            else
                M = 5000;
            end
            
            if isfield(options, 'LearningRate')
                eps = options.LearningRate;
            else
                eps = 0.01;
            end
            
            beta = zeros(p, 1);
            beta_path = zeros(p, M);
            r = Y;
            Xt = X';
            
            for m = 1:M
                % 1. Find feature most correlated with residual
                c = Xt * r; 
                [~, best_j] = max(abs(c));
                
                % 2. Take a small step
                step_val = eps * sign(c(best_j));
                beta(best_j) = beta(best_j) + step_val;
                
                % 3. Update residual
                r = r - step_val * X(:, best_j);
                
                % Record path
                beta_path(:, m) = beta;
            end
            
            extra_info.NumIterations = M;
            extra_info.LearningRate = eps;
        end
    end
end
