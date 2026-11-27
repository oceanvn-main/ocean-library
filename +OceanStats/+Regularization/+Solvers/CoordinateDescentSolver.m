classdef CoordinateDescentSolver < OceanStats.Regularization.Solvers.IRegularizationSolver
    % COORDINATEDESCENTSOLVER Fast path solver via cyclical coordinate descent.
    %
    %   OVERVIEW:
    %     This solver internally delegates to MATLAB's highly optimized
    %     built-in lasso() function to compute the full path for Ridge,
    %     Lasso, or Elastic Net using warm starts.
    %
    %   See also: OceanStats.Supervised.Regression.RegularizedLinearModel
    
    methods
        function [beta_path, extra_info] = solve(~, X, Y, options)
            % Extract parameters
            alpha = options.Alpha;
            
            % Setup lasso arguments
            args = {'Alpha', alpha, 'Standardize', false};
            
            if isfield(options, 'Lambdas') && ~isempty(options.Lambdas)
                if isnumeric(options.Lambdas)
                    args = [args, {'Lambda', options.Lambdas}];
                end
                % If 'auto', we let lasso() pick the sequence (default)
            end
            
            % Execute MATLAB's optimized coordinate descent
            [beta_path, fitInfo] = lasso(X, Y, args{:});
            
            % If path comes out with largest lambda first (MATLAB default),
            % we reverse it so the path goes from highly regularized to unregularized
            % Actually, standard path plots (like ESL) often go from 
            % high lambda (all zero) to low lambda (OLS). We keep MATLAB's order
            % which is high lambda -> low lambda.
            
            extra_info.Lambdas = fitInfo.Lambda;
            extra_info.MSE = fitInfo.MSE;
            extra_info.DF = fitInfo.DF;
        end
    end
end
