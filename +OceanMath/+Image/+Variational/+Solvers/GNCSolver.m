classdef GNCSolver < OceanMath.Image.Variational.Solvers.ISolver
    % GNCSOLVER  Blake-Zisserman Graduated Non-Convexity continuation solver.
    %
    %   Minimises the non-convex GNC energy by continuation annealing:
    %
    %       Start at large V  →  nearly convex (quadratic), easy to minimise
    %       Reduce V step-by-step  →  progressively more non-convex
    %       Each scale initialised from the previous (warm start)
    %
    %   Algorithm:
    %   ─────────────────────────────────────────────────────────────────────
    %   Require: GNCPenalty (or any non-convex IRobustPenalty with a 'V' property)
    %
    %   For scale s = 1, 2, ..., NumScales:
    %       V_s = VMax * (Vtarget/VMax)^((s-1)/(NumScales-1))   [log-spaced]
    %       Set energy.Penalty.V = V_s
    %       f_s = IRLSSolver.solve(energy, g)  with warm start f_{s-1}
    %
    %   Package: OceanMath.Image.Variational.Solvers
    %
    %   Usage:
    %       import OceanMath.Image.Regularization.*
    %       import OceanMath.Image.Variational.*
    %       import OceanMath.Image.Variational.Solvers.*
    %
    %       pen    = GNCPenalty('V', 0.1);
    %       energy = ImageRegularizationEnergy(pen, 'Lambda', 0.05);
    %       solver = GNCSolver('NumScales', 8, 'VMax', 100);
    %       f      = solver.solve(energy, g_noisy);
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Blake & Zisserman (1987); Szeliski (2021) §4.2

    % ── Properties ───────────────────────────────────────────────────────────
    properties
        NumScales    (1,1) double {mustBePositive, mustBeInteger} = 8
        VMax         (1,1) double = NaN   % NaN = auto: 100 * penalty.V
        InnerMaxIter (1,1) double = 30
        InnerTol     (1,1) double = 1e-6
        Verbose      (1,1) logical = false
    end

    % ── Public API ────────────────────────────────────────────────────────────
    methods
        function obj = GNCSolver(options)
            % GNCSOLVER  Constructor.
            arguments
                options.NumScales    (1,1) double  = 8
                options.VMax         (1,1) double  = NaN
                options.InnerMaxIter (1,1) double  = 30
                options.InnerTol     (1,1) double  = 1e-6
                options.Verbose      (1,1) logical = false
            end
            obj.NumScales    = options.NumScales;
            obj.VMax         = options.VMax;
            obj.InnerMaxIter = options.InnerMaxIter;
            obj.InnerTol     = options.InnerTol;
            obj.Verbose      = options.Verbose;
        end

        function f = solve(obj, energy, g)
            % SOLVE  Run GNC continuation and return the regularised image.

            import OceanMath.Image.Variational.Solvers.IRLSSolver

            % ── Validate: penalty must support the .V property ───────────────
            if ~isprop(energy.Penalty, 'V')
                error('GNCSolver:InvalidPenalty', ...
                    ['GNCSolver requires a penalty with a ''V'' property ' ...
                     '(e.g. GNCPenalty).  Got: %s.'], class(energy.Penalty));
            end

            if energy.Penalty.isConvex()
                warning('GNCSolver:ConvexPenalty', ...
                    'The penalty is already convex — GNC continuation is unnecessary.');
            end

            % ── Continuation schedule (log-spaced V values) ──────────────────
            V_target = energy.Penalty.V;
            V_max    = obj.VMax;
            if isnan(V_max)
                V_max = 100 * V_target;
            end

            if obj.NumScales == 1
                V_schedule = V_target;
            else
                V_schedule = exp(linspace(log(V_max), log(V_target), obj.NumScales));
            end

            % ── Inner IRLS solver ────────────────────────────────────────────
            inner = IRLSSolver( ...
                'MaxIter', obj.InnerMaxIter, ...
                'Tol',     obj.InnerTol);

            % ── Initialise from noisy image ──────────────────────────────────
            f = g;

            scale_energy = zeros(1, obj.NumScales);
            scale_V      = V_schedule;

            for s = 1:obj.NumScales
                V_s = V_schedule(s);
                energy.Penalty.V = V_s;

                f = obj.solveWithWarmStart_(energy, g, f, inner);

                E_s = energy.totalEnergy(f, g);
                scale_energy(s) = E_s;

                if obj.Verbose
                    fprintf('GNC scale %2d/%2d:  V = %.4g   E = %.6f\n', ...
                        s, obj.NumScales, V_s, E_s);
                end
            end

            % Restore original V
            energy.Penalty.V = V_target;

            obj.History_ = struct( ...
                'Energy',   scale_energy, ...
                'ScaleV',   scale_V, ...
                'NumIter',  obj.NumScales, ...
                'Residual', []);
        end
    end

    % ── Private helpers ───────────────────────────────────────────────────────
    methods (Access = private)
        function f = solveWithWarmStart_(obj, energy, g, f0, inner)
            % SOLVEWITHWARMSTART_  Run inner IRLS with warm-start from f0.

            [rows, cols] = size(g);
            N      = rows * cols;
            lambda = energy.Lambda;
            D      = energy.buildDifferenceMatrix(rows, cols);
            AtA    = speye(N);
            Atg    = g(:);

            f_vec = f0(:);

            for k = 1:inner.MaxIter
                dv  = D * f_vec;
                w   = energy.Penalty.weight(dv);
                W_k = spdiags(w, 0, numel(dv), numel(dv));

                LHS   = AtA + lambda .* (D' * W_k * D);
                f_new = LHS \ Atg;

                delta = norm(f_new - f_vec) / (norm(f_vec) + eps);
                f_vec = f_new;

                if delta < inner.Tol
                    break
                end
            end

            f = reshape(f_vec, rows, cols);
        end
    end
end
