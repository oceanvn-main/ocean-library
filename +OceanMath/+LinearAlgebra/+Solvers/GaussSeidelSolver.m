classdef GaussSeidelSolver < OceanMath.LinearAlgebra.Solvers.ILinearSolver
    % GAUSSSEIDELSOLVERSOLVER  Gauss-Seidel / SOR iterative solver for Ax = b.
    %
    % THEORY OVERVIEW:
    %   The Gauss-Seidel method solves a linear system A*x = b by sweeping
    %   through each component of x one at a time, using the most-recently
    %   updated values of the other components (unlike Jacobi which batches
    %   all updates until the end of a sweep).
    %
    %   Standard update formula at step i:
    %
    %       x_i^(k+1) = ( b_i - sum_{j<i} A_ij * x_j^(k+1)
    %                            - sum_{j>i} A_ij * x_j^(k)  ) / A_ii
    %
    %   Successive Over-Relaxation (SOR) extends this with a relaxation
    %   parameter omega (w):
    %
    %       x_i^(k+1) = (1 - w) * x_i^(k)  +  w * x_i_GS^(k+1)
    %
    %   Special cases:
    %     w = 1   -> standard Gauss-Seidel
    %     0 < w < 1 -> Under-relaxation (stabilises difficult systems)
    %     1 < w < 2 -> Over-relaxation  (accelerates convergence for SPD)
    %
    % CONVERGENCE GUARANTEE:
    %   Guaranteed when A is:
    %     (a) Strictly diagonally dominant, OR
    %     (b) Symmetric positive definite (any w in (0,2))
    %
    % USAGE:
    %   solver = GaussSeidelSolver();                    % w = 1 (standard GS)
    %   solver = GaussSeidelSolver('Omega', 1.5);        % w = 1.5 (SOR)
    %   [x, info] = solver.solve(A, b);
    %   [x, info] = solver.solve(A, b, x0);             % warm start
    %
    % AUTHOR: Ocean Library System
    % DATE:   2026-02-28

    properties
        Name = "Gauss-Seidel (SOR)"
        MaxIter = 500
        Tol = 1e-8
        Verbose = false
        RecordHistory = false

        % OMEGA  Successive Over-Relaxation parameter.
        %   w = 1   : standard Gauss-Seidel
        %   0 < w < 1: under-relaxation (more stable on ill-conditioned systems)
        %   1 < w < 2: over-relaxation  (faster convergence on SPD systems)
        Omega   (1,1) double  = 1.0
    end

    methods
        function obj = GaussSeidelSolver(varargin)
            % CONSTRUCTOR  Optional name-value configuration.
            %
            % USAGE:
            %   GaussSeidelSolver()
            %   GaussSeidelSolver('Omega', 1.5, 'MaxIter', 1000, 'Tol', 1e-10)
            %   GaussSeidelSolver('Verbose', true, 'RecordHistory', true)
            p = inputParser;
            addParameter(p, 'Omega',         1.0,   @(x) isscalar(x) && x > 0 && x < 2);
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
                obj.Name = sprintf("Gauss-Seidel SOR (omega=%.3g)", obj.Omega);
            end
        end

        function [x, info] = solve(obj, A, b, x0)
            % SOLVE  Find x such that A*x ≈ b via Gauss-Seidel / SOR sweeps.
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
            assert(size(A,1) == size(A,2), 'OceanMath:GaussSeidelSolver:NonSquare', ...
                'A must be a square matrix.');
            assert(numel(b) == N, 'OceanMath:GaussSeidelSolver:DimMismatch', ...
                'b must have %d elements to match A.', N);

            % --------------------------------------------------------
            % 1. Warn if not diagonally dominant (convergence not guaranteed)
            % --------------------------------------------------------
            obj.checkDiagonalDominance(A);

            % --------------------------------------------------------
            % 2. Initialise
            % --------------------------------------------------------
            if nargin < 4 || isempty(x0)
                x = zeros(N, 1);
            else
                x = x0(:);
            end

            b_norm   = norm(b);
            if b_norm == 0, b_norm = 1; end  % avoid divide-by-zero

            history  = [];
            iters    = 0;
            converged = false;

            % --------------------------------------------------------
            % 3. SOR sweep loop
            % --------------------------------------------------------
            w = obj.Omega;

            for iter = 1:obj.MaxIter
                x_old = x;

                % Inner sweep: update each component in sequence
                for i = 1:N
                    % Compute Gauss-Seidel correction (uses already-updated x)
                    sigma = A(i, :) * x - A(i, i) * x(i);
                    x_gs  = (b(i) - sigma) / A(i, i);

                    % Apply relaxation
                    x(i)  = (1 - w) * x_old(i) + w * x_gs;
                end

                % ---- Relative residual ----
                rel_res = norm(b - A * x) / b_norm;

                if obj.RecordHistory
                    history(end+1, 1) = rel_res; %#ok<AGROW>
                end

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
                warning('OceanMath:GaussSeidelSolver:NoConvergence', ...
                    '%s did not converge in %d iterations. Final rel_res = %.3e', ...
                    obj.Name, obj.MaxIter, rel_res);
            end

            info = obj.buildInfo(iters, rel_res, converged, history);
        end
    end
end
