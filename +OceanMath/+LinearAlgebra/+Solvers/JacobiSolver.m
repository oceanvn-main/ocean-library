classdef JacobiSolver < OceanMath.LinearAlgebra.Solvers.ILinearSolver
    % JACOBISOLVER  Jacobi / Damped-Jacobi iterative solver for Ax = b.
    %
    % THEORY OVERVIEW:
    %   The Jacobi method solves A*x = b by updating ALL components of x
    %   simultaneously at each iteration, using only the values from the
    %   PREVIOUS sweep (unlike Gauss-Seidel, which uses freshly-updated
    %   values within a sweep):
    %
    %       x_i^(k+1) = ( b_i - sum_{j!=i} A_ij * x_j^(k) ) / A_ii
    %
    %   Damping (relaxation) parameter omega (w) gives Damped Jacobi:
    %
    %       x^(k+1) = (1 - w) * x^(k)  +  w * x_Jacobi^(k+1)
    %
    %   Special cases:
    %     w = 1   -> standard Jacobi
    %     0 < w < 1 -> damped (more stable, may converge where w=1 diverges)
    %
    %   KEY DIFFERENCE FROM GAUSS-SEIDEL:
    %     Jacobi stores a full copy x_old before starting each sweep, so
    %     updates within a sweep do NOT see each other. This makes Jacobi
    %     trivially parallelisable but usually slower than Gauss-Seidel in
    %     serial execution.
    %
    % CONVERGENCE GUARANTEE:
    %   Guaranteed when A is strictly diagonally dominant.
    %   For SPD matrices, Gauss-Seidel is preferred (it converges twice as
    %   fast asymptotically).
    %
    % USAGE:
    %   solver = JacobiSolver();
    %   solver = JacobiSolver('Omega', 0.8, 'MaxIter', 1000);
    %   [x, info] = solver.solve(A, b);
    %   [x, info] = solver.solve(A, b, x0);   % warm start
    %
    % AUTHOR: Ocean Library System
    % DATE:   2026-02-28

    properties
        Name = "Jacobi"
        MaxIter = 500
        Tol = 1e-8
        Verbose = false
        RecordHistory = false

        % OMEGA  Relaxation (damping) parameter.
        %   w = 1   : standard Jacobi
        %   0 < w < 1: damped Jacobi (can help when standard Jacobi diverges)
        Omega   (1,1) double  = 1.0
    end

    methods
        function obj = JacobiSolver(varargin)
            % CONSTRUCTOR  Optional name-value configuration.
            %
            % USAGE:
            %   JacobiSolver()
            %   JacobiSolver('Omega', 0.8, 'MaxIter', 1000, 'Tol', 1e-10)
            %   JacobiSolver('Verbose', true, 'RecordHistory', true)
            p = inputParser;
            addParameter(p, 'Omega',         1.0,   @(x) isscalar(x) && x > 0 && x <= 1);
            addParameter(p, 'MaxIter',       500,   @(x) isscalar(x) && x > 0);
            addParameter(p, 'Tol',           1e-8,  @(x) isscalar(x) && x > 0);
            addParameter(p, 'Verbose',       false, @islogical);
            addParameter(p, 'RecordHistory', false, @islogical);
            parse(p, varargin{:});

            obj.Omega         = p.Results.Omega;
            obj.MaxIter       = p.Results.MaxIter;
            obj.Tol           = p.Results.Tol;
            obj.Verbose       = p.Results.Verbose;
            obj.RecordHistory = p.Results.RecordHistory;

            if obj.Omega ~= 1
                obj.Name = sprintf("Jacobi Damped (omega=%.3g)", obj.Omega);
            end
        end

        function [x, info] = solve(obj, A, b, x0)
            % SOLVE  Find x such that A*x ≈ b via Jacobi sweeps.
            %
            % INPUTS:
            %   A   - (N x N) real square coefficient matrix
            %   b   - (N x 1) right-hand side
            %   x0  - (N x 1) optional initial guess [default: zeros(N,1)]
            %
            % OUTPUTS:
            %   x    - (N x 1) approximate solution
            %   info - struct {iters, residual, converged, history}

            N = size(A, 1);

            % --------------------------------------------------------
            % 0. Validate inputs
            % --------------------------------------------------------
            assert(size(A,1) == size(A,2), 'OceanMath:JacobiSolver:NonSquare', ...
                'A must be a square matrix.');
            assert(numel(b) == N, 'OceanMath:JacobiSolver:DimMismatch', ...
                'b must have %d elements to match A.', N);

            % --------------------------------------------------------
            % 1. Warn if not diagonally dominant
            % --------------------------------------------------------
            obj.checkDiagonalDominance(A);

            % --------------------------------------------------------
            % 2. Pre-extract diagonal (reused every sweep for efficiency)
            % --------------------------------------------------------
            d = diag(A);             % (N x 1) diagonal entries
            D_inv = 1 ./ d;          % element-wise inverse of diagonal

            % --------------------------------------------------------
            % 3. Initialise
            % --------------------------------------------------------
            if nargin < 4 || isempty(x0)
                x = zeros(N, 1);
            else
                x = x0(:);
            end

            b_norm  = norm(b);
            if b_norm == 0, b_norm = 1; end

            history   = [];
            iters     = 0;
            converged = false;
            w         = obj.Omega;
            rel_res   = Inf;

            % --------------------------------------------------------
            % 4. Jacobi sweep loop (all components updated from x_old)
            % --------------------------------------------------------
            for iter = 1:obj.MaxIter

                % Vectorised: x_j^(k+1) = D^-1 * (b - (A - D)*x^(k))
                %  A*x gives the full product; subtracting d.*x removes
                %  the diagonal contribution, leaving off-diagonal only.
                x_gs = D_inv .* (b - (A * x - d .* x));

                % Apply damping
                x_new = (1 - w) .* x + w .* x_gs;

                % ---- Relative residual ----
                rel_res = norm(b - A * x_new) / b_norm;

                if obj.RecordHistory
                    history(end+1, 1) = rel_res; %#ok<AGROW>
                end

                x = x_new;  % Commit update (all at once — Jacobi rule)

                if obj.Verbose
                    fprintf('  [%s] Iter %4d | rel_res = %.3e\n', obj.Name, iter, rel_res);
                end

                % ---- Convergence check ----
                if rel_res < obj.Tol
                    iters     = iter;
                    converged = true;
                    break;
                end
            end

            if ~converged
                iters = obj.MaxIter;
                warning('OceanMath:JacobiSolver:NoConvergence', ...
                    '%s did not converge in %d iterations. Final rel_res = %.3e', ...
                    obj.Name, obj.MaxIter, rel_res);
            end

            info = obj.buildInfo(iters, rel_res, converged, history);
        end
    end
end
