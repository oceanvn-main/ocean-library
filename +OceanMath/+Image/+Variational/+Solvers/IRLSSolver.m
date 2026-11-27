classdef IRLSSolver < OceanMath.Image.Variational.Solvers.ISolver
    % IRLSSOLVER  Iteratively Reweighted Least Squares for image regularization.
    %
    %   Minimises the regularised denoising energy:
    %       E(f) = (1/2)||f - g||^2  +  lambda * E_IR(f)
    %
    %   by iterating the following weighted least-squares sub-problem:
    %
    %       [I  +  lambda * D' * W_k * D] * f_{k+1}  =  g
    %
    %   where:
    %       D    — sparse first-difference operator (built by ImageRegularizationEnergy)
    %       W_k  — diagonal IRLS weight matrix: W_k = diag( w(D * f_k) )
    %       w(.) — penalty weight function: w(d) = rho'(d) / d
    %
    %   Algorithm properties:
    %     - With QuadraticPenalty (w=1): converges in exactly 1 iteration
    %       (the result equals the direct backslash solve — verified by test suite)
    %     - With L1Penalty / TV:    typically 20–50 iterations
    %     - With HuberPenalty:      typically 10–30 iterations
    %     - With PNormPenalty(p<1): may need 50–100 iterations + warm starts
    %     - With GNCPenalty:        used as inner solver by GNCSolver
    %
    %   Two linear backends:
    %     'backslash'  — MATLAB's direct sparse solver (reliable for all sizes)
    %     'pcg'        — Preconditioned Conjugate Gradient (faster for large images)
    %
    %   Package: OceanMath.Image.Variational.Solvers
    %
    %   Usage:
    %       import OceanMath.Regularization.*
    %       import OceanMath.Image.Variational.*
    %       import OceanMath.Image.Variational.Solvers.*
    %
    %       pen    = L1Penalty();
    %       energy = ImageRegularizationEnergy(pen, 'Lambda', 0.1);
    %       solver = IRLSSolver('MaxIter', 50, 'Tol', 1e-6);
    %       f      = solver.solve(energy, g_noisy);
    %       h      = solver.getHistory();
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2; Green (1984) IRLS origin

    % ── Properties ───────────────────────────────────────────────────────────
    properties
        % MAXITER  Maximum number of outer IRLS iterations.
        MaxIter (1,1) double {mustBePositive, mustBeInteger} = 50

        % TOL  Relative convergence tolerance: ||Δf|| / (||f|| + eps) < Tol.
        Tol (1,1) double {mustBePositive} = 1e-6

        % LINEARSOLVE  Backend for the inner linear system.
        %   'backslash' — MATLAB sparse direct solver (reliable for all sizes)
        %   'pcg'       — Preconditioned Conjugate Gradient (faster, large images)
        LinearSolve char {mustBeMember(LinearSolve, {'backslash','pcg'})} = 'backslash'

        % VERBOSE  Print convergence info each iteration.
        Verbose (1,1) logical = false
    end

    % ── Public API ────────────────────────────────────────────────────────────
    methods
        function obj = IRLSSolver(options)
            % IRLSSOLVER  Constructor.
            %
            %   solver = IRLSSolver()
            %   solver = IRLSSolver('MaxIter', 100, 'Tol', 1e-8, 'LinearSolve', 'pcg')
            arguments
                options.MaxIter     (1,1) double  = 50
                options.Tol         (1,1) double  = 1e-6
                options.LinearSolve char          = 'backslash'
                options.Verbose     (1,1) logical = false
            end
            obj.MaxIter     = options.MaxIter;
            obj.Tol         = options.Tol;
            obj.LinearSolve = options.LinearSolve;
            obj.Verbose     = options.Verbose;
        end

        function f = solve(obj, energy, g)
            % SOLVE  Run IRLS and return the regularised image.

            [rows, cols] = size(g);
            N = rows * cols;
            g_vec = g(:);

            lambda = energy.Lambda;
            D = energy.buildDifferenceMatrix(rows, cols);
            AtA = speye(N);
            Atg = g_vec;

            % Initialise: start from the noisy image (warm start)
            f_vec = g_vec;

            energy_hist   = zeros(1, obj.MaxIter);
            residual_hist = zeros(1, obj.MaxIter);

            for k = 1:obj.MaxIter
                % ── Step 1: IRLS weight matrix W_k ──────────────────────────
                dv   = D * f_vec;
                w    = energy.Penalty.weight(dv);
                W_k  = spdiags(w, 0, numel(dv), numel(dv));

                % ── Step 2: solve weighted LS system ─────────────────────────
                %   LHS = I + lambda * D' W D
                LHS = AtA + lambda .* (D' * W_k * D);
                RHS = Atg;

                if strcmp(obj.LinearSolve, 'backslash')
                    f_new = LHS \ RHS;
                else
                    L_pc = ichol(LHS, struct('type','ict','droptol',1e-3));
                    [f_new, ~] = pcg(LHS, RHS, 1e-10, 200, L_pc, L_pc');
                end

                % ── Step 3: convergence check ─────────────────────────────────
                delta = norm(f_new - f_vec) / (norm(f_vec) + eps);
                E_k   = energy.totalEnergy(reshape(f_new, rows, cols), g);

                energy_hist(k)   = E_k;
                residual_hist(k) = delta;

                if obj.Verbose
                    fprintf('IRLS iter %3d:  E = %.6f   ||Δf||/||f|| = %.2e\n', k, E_k, delta);
                end

                f_vec = f_new;

                if delta < obj.Tol
                    if obj.Verbose
                        fprintf('IRLS converged at iteration %d.\n', k);
                    end
                    energy_hist   = energy_hist(1:k);
                    residual_hist = residual_hist(1:k);
                    break
                end
            end

            f = reshape(f_vec, rows, cols);

            obj.History_ = struct( ...
                'Energy',   energy_hist, ...
                'Residual', residual_hist, ...
                'NumIter',  numel(energy_hist));
        end
    end
end
