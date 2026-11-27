classdef IKernelSolver < handle
    % IKERNELSOLVER  Abstract interface for kernel machine dual solvers.
    %
    %   All solvers must locate the Lagrange multipliers (alpha) and bias (b) 
    %   that minimize the chosen dual objective.
    %
    %   Contract:
    %   [alpha, b, status] = solve(obj, K, Y, loss, options)
    %
    %   Author: Ocean Library | 2026-03-13

    methods (Abstract)
        % SOLVE  Optimize the dual parameters alpha and bias b.
        %
        %   Inputs:
        %       K       : (N x N) Kernel Gram matrix
        %       Y       : (N x 1) Target labels or continuous values
        %       loss    : An instance of OceanStats.Loss.* (ILoss) Strategy
        %       options : (Optional) struct with tolerances/hyperparameters
        %
        %   Outputs:
        %       alpha   : (N x 1) Dual variables (Lagrange multipliers)
        %       b       : (1,1) Bias term
        %       status  : Struct containing convergence info (e.g., iterations, solved flag)
        [alpha, b, status] = solve(obj, K, Y, loss, options)

        % GETNAME  Return solver name.
        name = getName(obj)
    end
end
