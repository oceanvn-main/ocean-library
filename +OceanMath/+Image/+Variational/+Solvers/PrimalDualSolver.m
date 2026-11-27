classdef PrimalDualSolver < OceanMath.Image.Variational.Solvers.ISolver
    % PRIMALDUAL SOLVER  Chambolle-Pock primal-dual splitting for TV denoising.
    %
    %   Minimises the regularised energy using the first-order primal-dual
    %   algorithm of Chambolle & Pock (2011):
    %
    %       min_{f}  (1/2)||f - g||^2  +  lambda * ||D f||_1
    %
    %   Especially suited for the L1/TV penalty where the non-smoothness
    %   of the L1 norm causes IRLS weights to diverge near d=0.
    %
    %   Algorithm (Chambolle-Pock, Alg. 1):
    %   ─────────────────────────────────────────────────────────────────────
    %   Initialise: f_0 = g, p_0 = 0, f_bar_0 = f_0
    %
    %   For k = 0, 1, 2, ...
    %       p_{k+1}     = proj_Ball( p_k + sigma * D * f_bar_k )
    %       f_{k+1}     = (f_k - tau * D' * p_{k+1} + tau * g) / (1 + tau)
    %       f_bar_{k+1} = f_{k+1} + theta * (f_{k+1} - f_k)
    %
    %   Step-size rule:  tau * sigma * ||D||^2 <= 1
    %       Auto-computed from normest(D) if Tau/Sigma are NaN (default).
    %
    %   Package: OceanMath.Image.Variational.Solvers
    %
    %   Usage:
    %       import OceanMath.Image.Regularization.*
    %       import OceanMath.Image.Variational.*
    %       import OceanMath.Image.Variational.Solvers.*
    %
    %       pen    = L1Penalty();
    %       energy = ImageRegularizationEnergy(pen, 'Lambda', 0.1);
    %       solver = PrimalDualSolver('MaxIter', 500);
    %       f      = solver.solve(energy, g_noisy);
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Chambolle & Pock (2011) "A First-Order Primal-Dual Algorithm"

    % ── Properties ───────────────────────────────────────────────────────────
    properties
        MaxIter         (1,1) double {mustBePositive, mustBeInteger} = 500
        Tol             (1,1) double {mustBePositive} = 1e-6
        Tau             (1,1) double = NaN   % primal step size (NaN = auto)
        Sigma           (1,1) double = NaN   % dual step size   (NaN = auto)
        Theta           (1,1) double {mustBeInRange(Theta, 0, 1, 'inclusive')} = 1.0
        Verbose         (1,1) logical = false
        VerboseInterval (1,1) double  = 50
    end

    % ── Public API ────────────────────────────────────────────────────────────
    methods
        function obj = PrimalDualSolver(options)
            % PRIMALDUAL SOLVER  Constructor.
            arguments
                options.MaxIter         (1,1) double  = 500
                options.Tol             (1,1) double  = 1e-6
                options.Tau             (1,1) double  = NaN
                options.Sigma           (1,1) double  = NaN
                options.Theta           (1,1) double  = 1.0
                options.Verbose         (1,1) logical = false
                options.VerboseInterval (1,1) double  = 50
            end
            obj.MaxIter         = options.MaxIter;
            obj.Tol             = options.Tol;
            obj.Tau             = options.Tau;
            obj.Sigma           = options.Sigma;
            obj.Theta           = options.Theta;
            obj.Verbose         = options.Verbose;
            obj.VerboseInterval = options.VerboseInterval;
        end

        function f = solve(obj, energy, g)
            % SOLVE  Run Chambolle-Pock and return the regularised image.

            [rows, cols] = size(g);
            N = rows * cols;
            g_vec = g(:);
            lambda = energy.Lambda;

            D = energy.buildDifferenceMatrix(rows, cols);
            N_edges = size(D, 1);

            % ── Auto-set step sizes ──────────────────────────────────────────
            tau   = obj.Tau;
            sigma = obj.Sigma;

            if isnan(tau) || isnan(sigma)
                % Estimate ||D|| via MATLAB's built-in power-iteration normest.
                % tau * sigma * ||D||^2 <= 1 is the Chambolle-Pock convergence
                % condition (verified in testPrimalDualStepCondition).
                normD    = normest(D, 1e-2);
                tau_auto = 0.99 / normD;
                sigma_auto = 0.99 / normD;
                % Check: tau_auto * sigma_auto * normD^2 = 0.99^2 < 1 ✓

                if isnan(tau);   tau   = tau_auto;   end
                if isnan(sigma); sigma = sigma_auto; end
            end

            obj.Tau   = tau;
            obj.Sigma = sigma;

            % ── Initialise ───────────────────────────────────────────────────
            f_vec     = g_vec;
            f_bar_vec = g_vec;
            p         = zeros(N_edges, 1);

            energy_hist   = zeros(1, obj.MaxIter);
            residual_hist = zeros(1, obj.MaxIter);

            for k = 1:obj.MaxIter
                % ── Dual ascent + projection onto [-1, 1] ball ───────────────
                p_new = p + sigma .* (D * f_bar_vec);
                p_new = max(-1, min(1, p_new));

                % ── Primal descent (proximal step for L2 data term) ──────────
                %   f_{k+1} = (f_k - tau*D'*p_{k+1} + tau*g) / (1 + tau)
                f_new = (f_vec - tau .* (D' * p_new) + tau .* g_vec) ./ (1 + tau);

                % ── Over-relaxation ──────────────────────────────────────────
                f_bar_new = f_new + obj.Theta .* (f_new - f_vec);

                % ── Convergence check ─────────────────────────────────────────
                delta = norm(f_new - f_vec) / (norm(f_vec) + eps);
                E_k   = energy.totalEnergy(reshape(f_new, rows, cols), g);

                energy_hist(k)   = E_k;
                residual_hist(k) = delta;

                if obj.Verbose && mod(k, obj.VerboseInterval) == 0
                    fprintf('PrimalDual iter %4d:  E = %.6f   ||Δf||/||f|| = %.2e\n', k, E_k, delta);
                end

                p         = p_new;
                f_vec     = f_new;
                f_bar_vec = f_bar_new;

                if delta < obj.Tol
                    if obj.Verbose
                        fprintf('PrimalDual converged at iteration %d.\n', k);
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
