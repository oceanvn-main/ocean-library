classdef (Abstract) ILinearSolver < handle
    % ILINEARSOLVER  Abstract interface for iterative linear system solvers.
    %
    % THEORY OVERVIEW:
    %   Iterative solvers find a solution to the system Ax = b by starting
    %   from an initial guess x0 and repeatedly improving it until the
    %   residual r = b - Ax is sufficiently small.
    %
    % PURPOSE:
    %   This interface is the Strategy contract for all iterative solvers in
    %   OceanMath.LinearAlgebra.Solvers.  Concrete implementations include:
    %       - GaussSeidelSolver  (Gauss-Seidel / SOR)
    %       - JacobiSolver       (Jacobi / Damped Jacobi)
    %
    % CONTRACT:
    %   [x, info] = solve(obj, A, b)
    %   [x, info] = solve(obj, A, b, x0)
    %
    %   INPUTS:
    %     A   - (N x N) real square coefficient matrix
    %     b   - (N x 1) right-hand side vector
    %     x0  - (N x 1) optional initial guess  [default: zeros]
    %
    %   OUTPUTS:
    %     x    - (N x 1) approximate solution
    %     info - struct with fields:
    %              .iters       - number of iterations performed
    %              .residual    - final relative residual  ||b - Ax|| / ||b||
    %              .converged   - logical, true if Tol was reached
    %              .history     - (iters x 1) relative residual at each iteration
    %                             (empty if obj.RecordHistory == false)
    %
    % AUTHOR: Ocean Library System
    % DATE:   2026-02-28

    % ------------------------------------------------------------------
    % Abstract scalar configuration properties
    % ------------------------------------------------------------------
    properties (Abstract)
        Name           % Human-readable name, e.g. "Gauss-Seidel (SOR)"
        MaxIter        % Maximum number of sweep iterations
        Tol            % Convergence tolerance on relative residual
        Verbose        % Print residual every iteration when true
        RecordHistory  % Collect per-iteration residual history
    end

    % ------------------------------------------------------------------
    % Abstract solver method — every strategy must implement this
    % ------------------------------------------------------------------
    methods (Abstract)
        % SOLVE  Find x such that A*x ≈ b.
        [x, info] = solve(obj, A, b, x0)
    end

    % ------------------------------------------------------------------
    % Shared helpers available to all concrete strategies
    % ------------------------------------------------------------------
    methods (Access = protected)

        function checkDiagonalDominance(~, A)
            % CHECKDIAGONALDOMINANCE  Warn if A is not strictly diagonally dominant.
            %
            % Gauss-Seidel and Jacobi are guaranteed to converge only when
            %   |A(i,i)| > sum_{j!=i} |A(i,j)|  for all i.
            % If this fails the user is warned (not errored) since SPD matrices
            % also guarantee convergence even without diagonal dominance.
            N = size(A, 1);
            diag_abs = abs(diag(A));
            row_sums = sum(abs(A), 2) - diag_abs;
            if any(diag_abs <= row_sums)
                warning('OceanMath:LinearAlgebra:NotDiagonallyDominant', ...
                    ['Matrix is not strictly diagonally dominant. ' ...
                     'Convergence is only guaranteed for SPD matrices.']);
            end
        end

        function info = buildInfo(~, iters, rel_res, converged, history)
            % BUILDINFO  Construct the standard info return struct.
            info.iters     = iters;
            info.residual  = rel_res;
            info.converged = converged;
            info.history   = history;
        end

    end
end
