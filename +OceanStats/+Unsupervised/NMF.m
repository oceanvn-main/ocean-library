classdef NMF < handle
    % NMF  Non-negative Matrix Factorization (Lee & Seung, 1999).
    %
    % ESL §14.6. Approximates a non-negative data matrix as:
    %   X  ≈  W * H
    %
    % Where:
    %   X  (N x p) — input data matrix (must be non-negative)
    %   W  (N x r) — activation matrix (coefficients per sample)
    %   H  (r x p) — basis matrix (r basis images/components)
    %   r  ≤ min(N, p) — factorization rank
    %
    % All entries of W and H are constrained to be >= 0.
    % This non-negativity produces parts-based, interpretable components
    % unlike PCA which allows mixed-sign (cancelling) loadings.
    %
    % ALGORITHM — Lee & Seung (2001) multiplicative updates (Frobenius norm):
    %   Objective:  min_{W>=0, H>=0}  ||X - W*H||_F^2
    % ||X||_F2 = Trace (X'X) -> Let X = || X- W*H||. Then solve Trace (X'X) -> Take gradient with respect to H.
    % dL / dH = 2 (W'WH) - 2(W' X)
    % dL / dW = 2(WHH') - 2(XH')
    % Standard Gradient Descent: H <- H -eta * dL/dH. But we need H>=0. 
    % Solution: Split Gradient into positive and negative parts
    % dL / dH = W'WH - W'X . Choose per-element adaptive learning relative. Where W'X is negative part. and W'WH is positive.
    %  Choose eta_ij = H_ij / (W'WH)_ij. Substitute. This will cancels the negative parts exactly and guarantees non-negativity + self-normalizing property
    % Same argument applied to dL/dW with adaptive leanring rate choose to be n_ij = W_ij / (WHH')_ij
    %   H ← H .* (W^T * X)  ./  (W^T * W * H + eps)
    %   W ← W .* (X  * H^T) ./  (W  * H * H^T + eps)
    %
    %   Each update is element-wise non-negative and monotonically decreases
    %   the Frobenius reconstruction error.
    %   After each W update, columns are normalized to unit norm (scale absorbed into H).
    %
    % USAGE:
    %   nmf = OceanStats.Unsupervised.NMF();
    %   nmf.fit(X, r, 'MaxIter', 300, 'Verbose', true);
    %   H = nmf.getBasisImages();       % (r x p) basis images
    %   X_hat = nmf.reconstruct();      % training reconstruction
    %   W_new = nmf.transform(X_test);  % activate on new data
    %   nmf.printSummary();
    %
    % ASSUMPTIONS:
    %   - X must be non-negative (>= 0). Preprocess with:
    %       X_nn = (X - min(X(:))) / (max(X(:)) - min(X(:)))
    %     before calling fit(). The fit method validates this.
    %   - Larger r => better reconstruction, less sparsity in W/H
    %   - Small r (e.g. 25-49) gives interpretable part-based images
    %
    % See Also: OceanStats.Pipeline.PCA, OceanStats.Unsupervised.SparsePCA
    %
    % Author: Ocean Library | 2026-04-29

    properties (SetAccess = private)
        W             % (N x r) activation matrix (non-negative coefficients per sample)
        H             % (r x p) basis image matrix (non-negative basis vectors)
        ReconError    % (1 x Iterations) Frobenius reconstruction error per iteration
        TotalEnergy   % scalar: ||X||_F^2 — total energy of training data
        ConvergedFlag % logical — did the update loop converge within MaxIter?
        Iterations    % scalar — number of update iterations used
    end

    methods
        function fit(obj, X, r, opts)
            % FIT  Learn r non-negative basis images from data matrix X.
            %
            % Arguments:
            %   X        (N x p) double  — data matrix, must be non-negative (>= 0)
            %   r        scalar int      — factorization rank (number of basis images)
            %   MaxIter  scalar int      — max multiplicative update iterations (default 300)
            %   Tol      scalar > 0      — convergence on relative change in ||X-WH||_F (default 1e-4)
            %   Verbose  logical         — print progress every 10 iters (default false)

            arguments
                obj
                X            (:,:) double
                r            (1,1) double {mustBePositive, mustBeInteger}
                opts.MaxIter (1,1) double {mustBePositive, mustBeInteger} = 300
                opts.Tol     (1,1) double {mustBePositive} = 1e-4
                opts.Verbose (1,1) logical = false
            end

            % Validate non-negativity
            if min(X(:)) < -1e-10
                error('OceanStats:NMF:NegativeInput', ...
                    'NMF requires non-negative input. Shift X to [0,max] before calling fit().\n  min(X(:)) = %.4f', min(X(:)));
            end
            X = max(X, 0);   % clip tiny numerical negatives

            [N, p] = size(X);
            obj.TotalEnergy = norm(X, 'fro')^2;

            % --- Initialize W and H with non-negative random values ---
            % Small positive offset prevents immediate zero-locking
            W = abs(randn(N, r)) * 0.1 + 0.01;  % (N x r)
            H = abs(randn(r, p)) * 0.1 + 0.01;  % (r x p)

            % Normalize W columns to unit norm (initial scale in H). Solve problem of scaling ambiguity
            for k = 1:r
                cn = norm(W(:, k));
                if cn > 1e-12
                    H(k, :) = H(k, :) * cn;
                    W(:, k) = W(:, k) / cn;
                end
            end

            eps_val = 1e-10;   % prevent division by zero in multiplicative updates
            recon_err = zeros(1, opts.MaxIter);
            obj.ConvergedFlag = false;
            obj.Iterations    = opts.MaxIter;

            % --- Multiplicative update loop ---
            for iter = 1:opts.MaxIter

                % =========================================================
                % UPDATE H — fix W, solve for H
                %
                % min_{H>=0} ||X - W*H||_F^2
                % Multiplicative update: H ← H .* (W^T X) ./ (W^T W H + eps)
                %
                % Derivation: set gradient = -W^T X + W^T W H = 0,
                %   then take element-wise ratio form to ensure H >= 0.
                % =========================================================
                WtX  = W' * X;          % (r x p) numerator
                WtWH = (W' * W) * H;   % (r x p) denominator
                H    = H .* WtX ./ (WtWH + eps_val);
                H    = max(H, 0);       % clip floating-point negatives

                % =========================================================
                % UPDATE W — fix H, solve for W
                %
                % min_{W>=0} ||X - W*H||_F^2
                % Multiplicative update: W ← W .* (X H^T) ./ (W H H^T + eps)
                % =========================================================
                XHt  = X * H';          % (N x r) numerator
                WHHt = W * (H * H');   % (N x r) denominator
                W    = W .* XHt ./ (WHHt + eps_val);
                W    = max(W, 0);

                % Normalize W columns (transfer scale to H, prevents W blowup). Because update rules for W and H run independent. We need to explicit scale those c
                for k = 1:r
                    cn = norm(W(:, k));
                    if cn > 1e-12
                        W(:, k) = W(:, k) / cn;
                        H(k, :) = H(k, :) * cn;
                    end
                end

                % --- Reconstruction error ---
                err = norm(X - W * H, 'fro')^2;
                recon_err(iter) = err;

                if opts.Verbose && mod(iter, 10) == 0
                    fprintf('  [NMF] iter=%3d | ||X-WH||_F^2=%.4f (%.1f%% unexplained)\n', ...
                        iter, err, 100 * err / obj.TotalEnergy);
                end

                % --- Convergence check ---
                if iter > 1
                    delta = abs(recon_err(iter-1) - recon_err(iter)) / (recon_err(iter-1) + 1e-12);
                    if delta < opts.Tol
                        obj.ConvergedFlag = true;
                        obj.Iterations    = iter;
                        recon_err         = recon_err(1:iter);
                        if opts.Verbose
                            fprintf('  [NMF] Converged at iter %d (delta=%.2e)\n', iter, delta);
                        end
                        break;
                    end
                end
            end

            obj.W          = W;
            obj.H          = H;
            obj.ReconError = recon_err;
        end

        function W_new = transform(obj, X_new)
            % TRANSFORM  Find activations W_new for new data, holding H fixed.
            %   NDF learned both W and H, at inference, we have new data X_new and want to find its activations
            % This is also optimization problem. We fix the dictionary H (pattern), and optimize W_new
            %
            %   min_{W_new >= 0}  ||X_new - W_new * H||_F^2
            %
            %   Solved by running multiplicative updates on W_new only (H frozen).
            %   X_new must be non-negative (apply same preprocessing as training data).
            if isempty(obj.H)
                error('OceanStats:NMF:NotFitted', 'Call fit() first.');
            end
            X_new = max(X_new, 0);
            [M, ~] = size(X_new);
            r       = size(obj.H, 1);
            W_new   = abs(randn(M, r)) * 0.1 + 0.01;
            eps_val = 1e-10;
            HHt     = obj.H * obj.H';   % (r x r) precompute once
            XHt     = X_new * obj.H';   % (N x r)
            for iter = 1:100
                WHHt  = W_new * HHt; %Same as train. But does one operation instead of many 
                W_new = W_new .* XHt ./ (WHHt + eps_val);
                W_new = max(W_new, 0);
            end
        end

        function X_hat = reconstruct(obj, W_in)
            % RECONSTRUCT  Return W * H (training reconstruction if W_in omitted).
            if isempty(obj.W)
                error('OceanStats:NMF:NotFitted', 'Call fit() first.');
            end
            if nargin < 2
                W_in = obj.W;
            end
            X_hat = W_in * obj.H;
        end

        function basis = getBasisImages(obj)
            % GETBASISIMAGES  Return H (r x p) — each row is one basis image.
            if isempty(obj.H)
                error('OceanStats:NMF:NotFitted', 'Call fit() first.');
            end
            basis = obj.H;
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print formatted model summary.
            if isempty(obj.W)
                fprintf('[NMF] Not fitted.\n'); return;
            end
            [N, r] = size(obj.W);
            [~, p] = size(obj.H);
            fprintf('\n=== NMF Summary (Lee & Seung, 2001) ===\n');
            fprintf('  Samples (N)         : %d\n', N);
            fprintf('  Features (p)        : %d\n', p);
            fprintf('  Rank (r)            : %d\n', r);
            fprintf('  Total energy        : %.4f\n', obj.TotalEnergy);
            fprintf('  Final recon error   : %.4f (%.1f%% unexplained)\n', ...
                obj.ReconError(end), 100 * obj.ReconError(end) / obj.TotalEnergy);
            fprintf('  Iterations          : %d\n', obj.Iterations);
            fprintf('  Converged           : %s\n', mat2str(obj.ConvergedFlag));
            fprintf('========================================\n');
        end
    end
end
