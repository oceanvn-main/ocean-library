classdef ArchetypalAnalysis < handle
    % ARCHETYPALANALYSIS  Archetypal Analysis (Cutler & Breiman, 1994).
    %
    % ESL §14.6.1. Approximates data as:  X ≈ W * H,  H = B * X
    %
    % Unlike NMF, archetypes H are constrained to be convex combinations
    % of the DATA POINTS — forcing them to lie on the CONVEX HULL.
    % Each sample is also a convex combination of the r archetypes.
    %
    % CONSTRAINTS:
    %   W (N x r): w_ik >= 0,  sum_k w_ik = 1  (rows = simplex)
    %   B (r x N): b_ki >= 0,  sum_i b_ki = 1  (rows = simplex)
    %   H = B * X  (r x p): archetypes as convex combinations of data
    %
    % OBJECTIVE (ESL Eq. 14.77):
    %   min_{W, B}  ||X - W * B * X||_F^2
    %
    % ALGORITHM:
    %   W step  — fix H=B*Xc: solve N independent r-variable simplex QPs
    %             w_i = lsqnonneg(H', x_i') then normalize
    %
    %   B step  — fix W: update each row k of B via Frank-Wolfe (FW).
    %             FW gradient (1×N): ∂f/∂b_k = 2*(wk_sq*(b_k*Q) - c_k*Xc^T)
    %             where Q = Xc*Xc^T  (precomputed N×N Gram matrix)
    %                   c_k = (wk'*R_k)  (1×p partial-residual projection)
    %             FW vertex: e_{argmin(gradient)} — place all weight on one data point
    %             FW step:   b_k ← (1-γ)*b_k + γ*e_j*   with γ=2/(t+2)
    %
    %   Initialization — furthest-point sampling (much better than random):
    %     Archetype 1 = data point farthest from centroid (Xc is centered)
    %     Archetype k = data point farthest from all previous archetypes
    %     B initialized as one-hot rows (b_k = e_{i_k})
    %
    % USAGE:
    %   aa = OceanStats.Unsupervised.ArchetypalAnalysis();
    %   aa.fit(X, r, 'NumRestarts', 3, 'FWIter', 20, 'Verbose', true);
    %   H = aa.getArchetypes();      % (r x p) archetype images
    %   aa.printSummary();
    %
    % Author: Ocean Library | 2026-04-29

    properties (SetAccess = private)
        W             % (N x r) convex weight matrix (rows sum to 1)
        B             % (r x N) archetype coefficient matrix (rows sum to 1)
        Archetypes    % (r x p) = B * Xc, archetype images in centered space
        ReconError    % (1 x Iterations) reconstruction error history
        TotalEnergy   % scalar: ||Xc||_F^2
        ColumnMeans   % (1 x p) training column means
        ConvergedFlag % logical
        Iterations    % scalar
    end

    methods
        function fit(obj, X, r, opts)
            % FIT  Learn r archetypes via alternating minimization.
            %
            %   X            (N x p)  — data matrix (any range)
            %   r            scalar   — number of archetypes
            %   MaxIter      int      — max outer alternating iterations (default 200)
            %   FWIter       int      — Frank-Wolfe inner iterations per B update (default 20)
            %   Tol          scalar   — convergence: relative change in error (default 1e-4)
            %   NumRestarts  int      — keep best of N random restarts (default 3)
            %   Verbose      logical  — print progress (default false)

            arguments
                obj
                X                (:,:) double
                r                (1,1) double {mustBePositive, mustBeInteger}
                opts.MaxIter     (1,1) double {mustBePositive, mustBeInteger} = 200
                opts.FWIter      (1,1) double {mustBePositive, mustBeInteger} = 20
                opts.Tol         (1,1) double {mustBePositive} = 1e-4
                opts.NumRestarts (1,1) double {mustBePositive, mustBeInteger} = 3
                opts.Verbose     (1,1) logical = false
            end

            [N, ~] = size(X);

            % Center X
            obj.ColumnMeans = mean(X, 1);     % (1 x p)
            Xc = X - obj.ColumnMeans;         % (N x p)
            obj.TotalEnergy = norm(Xc,'fro')^2;

            % Precompute Gram matrix Q = Xc * Xc^T (N x N) — used in FW gradient.
            % O(N^2 * p) once. For N=658, p=256 → ~110M ops, stored as 658^2 doubles (~3MB).
            Q = Xc * Xc';   % (N x N)

            best_err  = Inf;
            best_W = []; best_B = []; best_hist = [];

            for restart = 1:opts.NumRestarts
                if opts.Verbose
                    fprintf('  [AA] Restart %d/%d\n', restart, opts.NumRestarts);
                end

                % -------------------------------------------------------
                % INITIALIZATION — furthest-point sampling
                % Sets b_k = e_{i_k} (archetype k starts as one data point)
                % -------------------------------------------------------
                B_cur = obj.initFarthest(Xc, r, N);   % (r x N), one-hot rows
                H_cur = B_cur * Xc;                    % (r x p)

                err_hist  = zeros(1, opts.MaxIter);
                converged = false;

                for iter = 1:opts.MaxIter

                    % ---------------------------------------------------
                    % W STEP — fix H, solve N simplex QPs (small: r vars)
                    %   min_{w_i>=0, sum=1}  ||x_i - w_i * H||^2
                    % ---------------------------------------------------
                    W_cur = obj.solveW(Xc, H_cur);   % (N x r)

                    % ---------------------------------------------------
                    % B STEP — Frank-Wolfe on each row k of B
                    %
                    % Objective for b_k (1×N) with H = B*Xc:
                    %   f(b_k) = ||R_k - wk * (b_k*Xc)||_F^2
                    % where R_k = Xc - WH + wk*H[k,:] (partial residual)
                    %
                    % FW gradient (1×N):
                    %   g = 2*(wk_sq*(b_k*Q) - c_k*Xc^T)
                    %   c_k = wk^T * R_k  (1×p)
                    %
                    % FW update:
                    %   j* = argmin_j g[j]    (cheapest: just min of a vector)
                    %   gamma = 2/(t+2)       (diminishing step)
                    %   b_k = (1-gamma)*b_k + gamma*e_{j*}
                    % ---------------------------------------------------
                    WH = W_cur * H_cur;   % (N x p) — shared across B loop

                    for k = 1:r
                        wk    = W_cur(:, k);       % (N x 1)
                        wk_sq = wk' * wk;           % scalar
                        if wk_sq < 1e-12, continue; end

                        old_hk = H_cur(k, :);                         % (1 x p)
                        R_k    = Xc - WH + wk * old_hk;               % (N x p) partial residual
                        c_k    = wk' * R_k;                           % (1 x p)

                        b_k = B_cur(k, :);   % (1 x N) current row

                        for t = 1:opts.FWIter
                            % Gradient: g = 2*(wk_sq * b_k*Q  -  c_k * Xc')
                            %   b_k*Q  : (1×N)*(N×N) = (1×N)
                            %   c_k*Xc': (1×p)*(p×N) = (1×N)
                            g = 2 * (wk_sq * (b_k * Q)  -  c_k * Xc');   % (1×N)

                            % FW vertex: unit simplex atom minimizing g
                            [~, j_star] = min(g);

                            % Diminishing step size (standard FW schedule)
                            gamma = 2 / (t + 2);

                            % Convex combination update (stays on simplex)
                            b_k          = (1 - gamma) * b_k;
                            b_k(j_star)  = b_k(j_star) + gamma;
                        end

                        B_cur(k, :) = b_k;
                        new_hk      = b_k * Xc;                        % (1 x p)
                        H_cur(k, :) = new_hk;

                        % Incremental WH update (avoids full recompute)
                        WH = WH + wk * (new_hk - old_hk);             % (N x p)
                    end

                    % Reconstruction error
                    err = norm(Xc - WH, 'fro')^2;
                    err_hist(iter) = err;

                    if opts.Verbose && mod(iter, 10) == 0
                        fprintf('  [AA] iter=%3d | err=%.2f (%.1f%% unexplained)\n', ...
                            iter, err, 100*err/obj.TotalEnergy);
                    end

                    % Convergence: relative change in error
                    if iter > 1
                        delta = abs(err_hist(iter-1) - err) / (err_hist(iter-1) + 1e-12);
                        if delta < opts.Tol
                            converged = true;
                            err_hist  = err_hist(1:iter);
                            break;
                        end
                    end
                end % iter

                final_err = err_hist(end);
                if final_err < best_err
                    best_err  = final_err;
                    best_W    = W_cur;
                    best_B    = B_cur;
                    best_hist = err_hist;
                    obj.ConvergedFlag = converged;
                    obj.Iterations    = length(err_hist);
                end
            end % restart

            obj.W          = best_W;
            obj.B          = best_B;
            obj.Archetypes = best_B * Xc;
            obj.ReconError = best_hist;
        end

        function W_new = transform(obj, X_new)
            % TRANSFORM  Find convex weights W_new for new data (M x p).
            if isempty(obj.Archetypes)
                error('OceanStats:AA:NotFitted', 'Call fit() first.');
            end
            Xc_new = X_new - obj.ColumnMeans;
            W_new  = obj.solveW(Xc_new, obj.Archetypes);
        end

        function X_hat = reconstruct(obj, W_in)
            % RECONSTRUCT  Return W*H + ColumnMeans.
            if isempty(obj.W)
                error('OceanStats:AA:NotFitted', 'Call fit() first.');
            end
            if nargin < 2, W_in = obj.W; end
            X_hat = W_in * obj.Archetypes + obj.ColumnMeans;
        end

        function H = getArchetypes(obj)
            % GETARCHETYPES  Return H = B*Xc (r x p).
            if isempty(obj.Archetypes)
                error('OceanStats:AA:NotFitted', 'Call fit() first.');
            end
            H = obj.Archetypes;
        end

        function printSummary(obj)
            if isempty(obj.W)
                fprintf('[AA] Not fitted.\n'); return;
            end
            [N, r] = size(obj.W);
            [~, p] = size(obj.Archetypes);
            fprintf('\n=== Archetypal Analysis Summary ===\n');
            fprintf('  Samples (N)         : %d\n', N);
            fprintf('  Features (p)        : %d\n', p);
            fprintf('  Archetypes (r)      : %d\n', r);
            fprintf('  Total energy        : %.4f\n', obj.TotalEnergy);
            fprintf('  Final recon error   : %.4f (%.1f%% unexplained)\n', ...
                obj.ReconError(end), 100*obj.ReconError(end)/obj.TotalEnergy);
            fprintf('  Iterations          : %d\n', obj.Iterations);
            fprintf('  Converged           : %s\n', mat2str(obj.ConvergedFlag));
            w_row_sums = sum(obj.W, 2);
            fprintf('  W row-sum range     : [%.4f, %.4f] (should be ~1)\n', ...
                min(w_row_sums), max(w_row_sums));
            fprintf('====================================\n');
        end
    end

    methods (Access = private)

        function W = solveW(~, Xc, H)
            % SOLVEW  N independent r-variable simplex QPs for W.
            % min_{w_i>=0, sum=1} ||x_i - w_i*H||^2  →  lsqnonneg then normalize.
            % Fast: r is small (2–8), p observations, N calls total.
            [N, ~] = size(Xc);
            r  = size(H, 1);
            W  = zeros(N, r);
            Ht = H';   % (p x r) — transpose once
            for i = 1:N
                w = lsqnonneg(Ht, Xc(i,:)');   % (r x 1)
                s = sum(w);
                if s > 1e-12
                    W(i,:) = (w / s)';
                else
                    W(i,:) = ones(1, r) / r;
                end
            end
        end

        function B = initFarthest(~, Xc, r, N)
            % INITFARTHEST  Furthest-point initialization for B.
            % Each archetype starts as a single data point (one-hot b_k).
            % Selects data points that are maximally spread across the data.
            %
            % Step 1: pick point farthest from centroid (Xc already centered → max norm)
            % Step 2: pick point farthest from all previous selections (greedy)
            norms = sum(Xc .^ 2, 2);       % (N x 1) squared distances from centroid
            idx = zeros(1, r);
            [~, idx(1)] = max(norms);

            min_dists = sum((Xc - Xc(idx(1),:)).^2, 2);   % dist to first archetype
            for k = 2:r
                [~, idx(k)]  = max(min_dists);
                d_new        = sum((Xc - Xc(idx(k),:)).^2, 2);
                min_dists    = min(min_dists, d_new);
            end

            % One-hot B: b_k = e_{idx(k)}
            B = zeros(r, N);
            for k = 1:r
                B(k, idx(k)) = 1;
            end
        end
    end
end
