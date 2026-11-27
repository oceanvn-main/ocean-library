classdef IPrimalSolver < handle
    % IPRIMALSOLVER  Abstract interface for primal unconstrained solvers.
    %
    %   Used to minimize the unconstrained composite objective 
    %   phi(w) = f(w) + lambda * r(w) directly. 
    %   Methods like SGD, Proximal Gradient, and L-BFGS fall here.
    %
    %   Contract:
    %   [w, b, status] = solve(obj, objFunc, w0, b0, X, Y, options)
    %
    %   Author: Ocean Library | 2026-03-13
    %   ELS: eq 12.43

    methods (Abstract)
        % SOLVE  Optimize the primal parameters w and bias b.
        %
        %   Inputs:
        %       objFunc : OceanStats.Optimization.CompositeObjective
        %       w0      : (p x 1) Initial weights
        %       b0      : (1,1) Initial bias
        %       X       : (N x p) Feature matrix
        %       Y       : (N x 1) Target labels or continuous values
        %       options : (Optional) struct with tolerances/hyperparameters
        %
        %   Outputs:
        %       w       : (p x 1) Optimized weights 
        %       b       : (1,1) Optimized bias
        %       status  : Struct containing convergence info
        [w, b, status] = solve(obj, objFunc, w0, b0, X, Y, options)

        % GETNAME  Return solver name.
        name = getName(obj)
    end
end
