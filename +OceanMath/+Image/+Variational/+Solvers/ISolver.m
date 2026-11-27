classdef ISolver < handle
    % ISOLVER  Abstract interface for image variational solvers.
    %
    %   All solvers take an ImageRegularizationEnergy and a noisy image g,
    %   and return the regularized estimate f.
    %
    %   Concrete implementations (OceanMath.Image.Variational.Solvers):
    %       IRLSSolver       — Iteratively Reweighted Least Squares
    %       PrimalDualSolver — Chambolle-Pock primal-dual splitting
    %       GNCSolver        — Blake-Zisserman Graduated Non-Convexity
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2

    methods (Abstract)
        % SOLVE  Run the solver and return the regularized image.
        %
        %   f = solver.solve(energy, g)
        %
        %   energy — ImageRegularizationEnergy (OceanMath.Image.Variational)
        %   g      — (rows x cols) double noisy input image
        %   f      — (rows x cols) double regularized output image
        f = solve(obj, energy, g)
    end

    methods (Access = public)
        function history = getHistory(obj)
            % GETHISTORY  Return convergence history from the last solve() call.
            %
            %   Returns a struct with fields:
            %     .Energy       — (1 x NumIter) total energy at each iteration
            %     .Residual     — (1 x NumIter) ||f_{k+1} - f_k|| / ||f_k||
            %     .NumIter      — actual iterations taken
            history = obj.History_;
        end
    end

    properties (Access = protected)
        % HISTORY_  Internal convergence record populated during solve().
        History_ = struct('Energy', [], 'Residual', [], 'NumIter', 0)
    end
end
