classdef (Abstract) IRegularizationSolver < handle
    % IREGULARIZATIONSOLVER Interface for regularized linear solvers.
    %
    %   OVERVIEW:
    %     Defines the contract for back-end solvers used by RegularizedLinearModel.
    %     Solvers receive STANDARDIZED data (zero-mean, unit-variance X, zero-mean Y).
    %     Solvers return the full coefficient path. The parent model handles
    %     mapping the path back to the original scale.
    %
    %   METHODS:
    %     solve(obj, X, Y, options) -> [beta_path, extra_info]
    
    methods (Abstract)
        % SOLVE Compute the coefficient path.
        %   Inputs:
        %     X       - N x p standardized predictor matrix
        %     Y       - N x 1 centered response vector
        %     options - struct containing solver-specific options 
        %               (e.g., Alpha, Lambdas, NumIterations)
        %   Outputs:
        %     beta_path  - p x M matrix of coefficients across the path
        %     extra_info - struct with solver-specific info (e.g., lambda_sequence)
        [beta_path, extra_info] = solve(obj, X, Y, options)
    end
end
