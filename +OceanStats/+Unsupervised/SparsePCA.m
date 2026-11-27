classdef SparsePCA < handle
    % SPARSEPCA  Sparse PCA (Zou et al., 2006) — ESL §14.5.5 textbook approach.
    %
    % FORMULATION (ESL Eq. 14.70 — single component):
    %   min_{theta, v}  ||X - (X*v)*theta^T||_F^2 + lambda*||v||^2 + lambda1*||v||_1
    %   subject to:  ||theta||_2 = 1
    %
    % Both theta and v are (p x 1) feature-space vectors.
    % The reconstruction of X is: (X*v)*theta^T — a rank-1 (N x p) matrix.
    %
    % ALGORITHM (alternating minimization, ESL §14.5.5):
    %
    %   STEP A — fix v, solve for theta (Procrustes):
    %     Minimize ||X - (Xv)*theta^T||_F^2 s.t. ||theta||=1
    %     => theta = X^T*(X*v) / ||X^T*(X*v)||        (p x 1, unit norm)
    %
    %   STEP B — fix theta, solve for v (Elastic-Net):
    %     Minimize ||X - (Xv)*theta^T||_F^2 + lambda*||v||^2 + lambda1*||v||_1
    %     Normal equations (L2 part):  (X^T X + lambda*I)*v = X^T*(X*theta)
    %     => v_ridge = (X^T X + lambda*I)^{-1} * (X^T X * theta)
    %     => v = SoftThreshold(v_ridge, lambda1/2)     (L1 part)
    %     => v = v / ||v||                              (normalize for stability)
    %
    % MULTI-COMPONENT (ESL Eq. 14.71 — deflation):
    %   After converging on component k, subtract its rank-1 reconstruction:
    %     X_res <- X_res - (X_res*v_k)*theta_k^T
    %   Fit component k+1 on the residual X_res.
    %
    % OPTIMIZATION (vs naive implementation):
    %   X^T*X is computed ONCE per component (not per iteration).
    %   Its LU factorization is pre-computed ONCE per component.
    %   The inner loop then does O(p^2) back-substitution only (not O(p^3) solve).
    %
    % USAGE:
    %   spca = OceanStats.Unsupervised.SparsePCA();
    %   spca.fit(X, K, 'Lambda', 1e-4, 'Lambda1', 0.1, 'MaxIter', 500);
    %   Z_new = spca.transform(X_test);
    %   spca.printSummary();
    %
    % ASSUMPTIONS:
    %   - X is (N x p). Centered internally (zero-mean columns).
    %   - Lambda > 0 ensures (X^T X + lambda*I) is invertible when p > N.
    %   - Lambda1 = 0 recovers standard PCA (no sparsity).
    %
    % DEPENDENCIES:
    %   OceanStats.Regularization.L1Regularizer (soft-threshold proximity)
    %
    % See Also: OceanStats.Pipeline.PCA, OceanStats.Regularization.L1Regularizer
    %
    % Author: Ocean Library | 2026-04-29

    properties (SetAccess = private)
        Loadings          % (p x K) sparse loading matrix V
        Scores            % (N x K) scores = centered X * V
        DirectionMatrix   % (p x K) unit-norm direction matrix Theta
        ExplainedVariance % (1 x K) variance explained per component
        TotalVariance     % scalar: trace(cov(Xc)) — total variance
        ColumnMeans       % (1 x p) column means for centering new data
        ConvergedFlags    % (1 x K) logical — true if component k converged
        Iterations        % (1 x K) iterations used per component
    end

    methods
        function fit(obj, X, K, opts)
            % FIT  Compute K sparse PCs via alternating minimization + deflation.
            %
            % Arguments:
            %   X          (N x p)  — data matrix (centered internally)
            %   K          scalar   — number of sparse PCs
            %   Lambda     scalar   — L2 ridge on v (default 1e-4)
            %                         Ensures (X^T X + lambda*I) is invertible.
            %   Lambda1    scalar   — L1 on v, controls sparsity (default 0.1)
            %                         0 = dense (PCA); larger = sparser loadings.
            %   MaxIter    scalar   — max alternating iterations per component (default 500)
            %   Tol        scalar   — convergence on ||v_new - v_old||_2 (default 1e-6)
            %   Verbose    logical  — print convergence info per component (default false)

            arguments
                obj
                X          (:,:) double
                K          (1,1) double {mustBePositive, mustBeInteger}
                opts.Lambda  (1,1) double {mustBeNonnegative} = 1e-4
                opts.Lambda1 (1,1) double {mustBeNonnegative} = 0.1
                opts.MaxIter (1,1) double {mustBePositive, mustBeInteger} = 500
                opts.Tol     (1,1) double {mustBePositive} = 1e-6
                opts.Verbose (1,1) logical = false
            end

            [N, p] = size(X);

            % --- Center X (store means for transform) ---
            obj.ColumnMeans = mean(X, 1);                   % (1 x p)
            Xc = X - obj.ColumnMeans;                       % (N x p) centered

            % Total variance = trace(cov(Xc)): denominator for % explained variance
            obj.TotalVariance = sum(diag(cov(Xc)));

            % --- L1 regularizer (soft-threshold) ---
            l1 = OceanStats.Regularization.L1Regularizer();

            % --- Warm start: V initialized from standard PCA ---
            % Economy SVD of Xc (discards zero singular values).
            % Right singular vectors V_svd are the standard PC directions —
            % closest starting point for SPCA, accelerates convergence.
            [~, ~, V_svd] = svd(Xc, 'econ');               % (p x r), r = min(N,p)
            r_init = min(size(V_svd, 2), K);

            % --- Storage ---
            V_mat     = zeros(p, K);                        % sparse loadings
            Theta_mat = zeros(p, K);                        % unit-norm directions

            obj.ConvergedFlags = false(1, K);
            obj.Iterations     = zeros(1, K);

            % --- Deflation residual ---
            % Starts as full centered data. Each component's rank-1 reconstruction
            % is subtracted before fitting the next component.
            Xc_res = Xc;

            % === Loop over K components ===
            for comp = 1:K
                if opts.Verbose
                    fprintf('  [SparsePCA] Component %d / %d\n', comp, K);
                end

                % Warm-start v from corresponding standard PC
                if comp <= r_init
                    v = V_svd(:, comp);                     % (p x 1) unit vector
                else
                    v_rand = randn(p, 1);
                    v = v_rand / norm(v_rand);
                end

                % Pre-compute X^T*X once per component (uses deflated residual).
                % This matrix does NOT change during the inner iteration loop.
                XtX = Xc_res' * Xc_res;                    % (p x p)

                % Pre-factorize (X^T X + lambda*I) ONCE per component.
                % Inner loop then uses O(p^2) back-substitution, not O(p^3) solve.
                A = XtX + opts.Lambda * eye(p);             % (p x p) regularized gram
                [L_fac, U_fac, P_fac] = lu(A);             % LU factorization

                % === Inner alternating loop (Steps A and B) ===
                for iter = 1:opts.MaxIter
                    v_old = v;

                    % -------------------------------------------------
                    % STEP A — Procrustes (fix v, solve for theta)
                    %
                    % min_{||theta||=1} ||X - (Xv)*theta^T||_F^2
                    % Solution: theta = X^T*(X*v) / ||X^T*(X*v)||
                    %
                    % Derivation: differentiating and using unit-norm constraint.
                    % theta = X^T X v / ||X^T X v|| = XtX*v / ||XtX*v||
                    % (using precomputed XtX — no extra N*p pass needed)
                    % -------------------------------------------------
                    Mv  = XtX * v;                          % (p x 1) = X^T X v
                    nrm = norm(Mv);
                    if nrm < 1e-12
                        % v collapsed to null space — reinitialize
                        v_rand = randn(p, 1);
                        theta  = v_rand / norm(v_rand);
                    else
                        theta = Mv / nrm;                   % (p x 1) unit norm
                    end

                    % -------------------------------------------------
                    % STEP B — Elastic-Net (fix theta, solve for v)
                    %
                    % min_v ||X-(Xv)*theta^T||_F^2 + lambda*||v||^2 + lambda1*||v||_1. Equation 14.70 
                    %
                    % L2 normal equations: (X^T X + lambda*I)*v = X^T X * theta
                    % => v_ridge = A^{-1} * (XtX * theta). How to solve this without invert A
                    %    using pre-factorized LU (O(p^2) per iteration)
                    %
                    % L1 soft-threshold: v = sign(v_ridge).*max(|v_ridge|-lambda1/2, 0)
                    % PA = LU => PA v  = Pb -> LU v = Pb. Define y = Uv, So now we need to solve for Ly = Pb for y, and Uv = y for v
                    % P_frac * rhs . (Solve for Pb). Call this B_hat
                    % Then L_fac \  B_hat which derived from Ly = Pb = B_hat -> so y = L^-1 * B_hat or L \ B_hat (\ in Matlab means solve Ly = B_hat for y)
                    %  Now we have y. Solve for v by y = Uv => v = y / U or U \ y (\ in Matlab means solve Uv = y for v)
                    % -------------------------------------------------
                    rhs     = XtX * theta;                  % (p x 1) = X^T X theta. %Target is rhs
                    v_ridge = U_fac \ (L_fac \ (P_fac * rhs)); % (p x 1) O(p^2), Pure L2, fix L1

                    v = l1.proximity(v_ridge, 1, opts.Lambda1 / 2);  % soft-threshold

                    % Normalize v (scale-invariant objective; prevents drift)
                    v_nrm = norm(v);
                    if v_nrm > 1e-12
                        v = v / v_nrm;
                    else
                        % v zeroed out — Lambda1 is too large for this component.
                        % Storing v_old here (the PCA warm-start) would give
                        % artificially high explained variance, causing a V-shape
                        % in the sweep curve. Store zeros instead: ExplainedVar = 0.
                        if opts.Verbose
                            fprintf('  [SparsePCA] Warning: comp=%d zeroed at iter=%d. Lambda1 too large.\n', comp, iter);
                        end
                        v = zeros(p, 1);   % dead component — contributes 0 explained variance
                        break;
                    end

                    % --- Convergence check ---
                    delta = norm(v - v_old);
                    if delta < opts.Tol
                        obj.ConvergedFlags(comp) = true;
                        obj.Iterations(comp)     = iter;
                        if opts.Verbose
                            fprintf('  [SparsePCA] Component %d converged at iter %d (delta=%.2e)\n', comp, iter, delta);
                        end
                        break;
                    end
                end % inner iter loop

                if ~obj.ConvergedFlags(comp)
                    obj.Iterations(comp) = opts.MaxIter;
                    if opts.Verbose
                        fprintf('  [SparsePCA] Component %d did NOT converge in %d iters.\n', comp, opts.MaxIter);
                    end
                end

                % Store converged component
                V_mat(:, comp)     = v;
                Theta_mat(:, comp) = theta;

                % --- Deflation: subtract rank-1 reconstruction ---
                % Reconstruction = (Xc_res*v) * theta^T  (N x p)
                % This removes the explained variance before fitting the next component.
                score_k = Xc_res * v;                       % (N x 1). Projection of each sample onto this com loading vec
                Xc_res  = Xc_res - score_k * theta';        % (N x p). Get residuals

            end % component loop

            % --- Store results ---
            obj.Loadings        = V_mat;                    % (p x K)
            obj.DirectionMatrix = Theta_mat;                % (p x K)
            obj.Scores          = Xc * V_mat;               % (N x K)
            obj.ExplainedVariance = var(obj.Scores, 0, 1); % (1 x K)
        end

        function Z = transform(obj, X_new)
            % TRANSFORM  Project new data onto fitted sparse PCs.
            %   X_new (M x p) -> Z (M x K), centered using training ColumnMeans.
            if isempty(obj.Loadings)
                error('OceanStats:SparsePCA:NotFitted', 'Call fit() first.');
            end
            Z = (X_new - obj.ColumnMeans) * obj.Loadings;  % (M x K)
        end

        function r = sparsityRatio(obj)
            % SPARSITYRATIO  Fraction of exact-zero loadings per component. (1 x K)
            if isempty(obj.Loadings)
                error('OceanStats:SparsePCA:NotFitted', 'Call fit() first.');
            end
            r = sum(obj.Loadings == 0, 1) / size(obj.Loadings, 1);
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print formatted model summary to console.
            if isempty(obj.Loadings)
                fprintf('[SparsePCA] Not fitted.\n'); return;
            end
            [p, K] = size(obj.Loadings);
            fprintf('\n=== SparsePCA Summary (ESL §14.5.5 — Deflation) ===\n');
            fprintf('  Features (p)        : %d\n', p);
            fprintf('  Components (K)      : %d\n', K);
            fprintf('  Total variance      : %.4f\n', obj.TotalVariance);
            fprintf('  Convergence         :');
            for k = 1:K
                if obj.ConvergedFlags(k)
                    fprintf('  PC%d=YES(%d)', k, obj.Iterations(k));
                else
                    fprintf('  PC%d=NO(%d)', k, obj.Iterations(k));
                end
            end
            fprintf('\n  Explained variance  :');
            for k = 1:K
                fprintf('  PC%d=%.4f(%.1f%%)', k, obj.ExplainedVariance(k), ...
                    100*obj.ExplainedVariance(k)/obj.TotalVariance);
            end
            fprintf('\n  Sparsity (zero %%)   :');
            sparsity = obj.sparsityRatio();
            for k = 1:K
                fprintf('  PC%d=%.1f%%', k, 100*sparsity(k));
            end
            fprintf('\n====================================================\n');
        end
    end
end
