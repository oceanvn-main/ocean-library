function [M, R, beta, X_aligned] = rigid_procrustes_average(X, tol, max_iter)
% RIGID_PROCRUSTES_AVERAGE  Iterative Generalized Procrustes Analysis (rigid + scale).
%
%   Solves:  min_{R_l in O(p), beta_l > 0, M}  sum_l ||beta_l * X_l * R_l - M||_F^2
%   via Alternating Least Squares (ALS). ESL Ex. 14.9, Eq. 14.115.
%
%   INPUT
%     X        (N x p x L) double  — landmark arrays, one per shape.
%                                    May be uncentered; centering is applied internally.
%     tol      scalar               — convergence threshold on ||M_new - M_old||_F
%     max_iter scalar               — maximum number of ALS iterations
%
%   OUTPUT
%     M         (N x p)      — Procrustes mean template (unit Frobenius norm)
%     R         (p x p x L)  — optimal rotation matrix per shape
%     beta      (L x 1)      — optimal scale factor per shape
%     X_aligned (N x p x L)  — each shape scaled and rotated to M
%
%   ASSUMPTIONS
%     - Centering is applied to each slice before any computation.
%     - R is solved as U*V' (Orthogonal Procrustes, may include reflections).
%       For proper rotations (det=+1) enforce the sign fix externally if needed.
%     - M is re-normalized after each mean update to prevent scale collapse.
%
%   REFERENCE
%     Hastie et al. (2009), ESL §14.5.3 and Exercise 14.9.

arguments
    X        (:,:,:) double
    tol      (1,1)   double  = 1e-8
    max_iter (1,1)   double  = 500
end

[N, p, L] = size(X);

% =========================================================================
% Pre-processing: center each shape at the origin (remove translation)
% =========================================================================
% Centering removes the mean landmark position from each slice.
% Broadcasting: bsxfun subtracts per-column mean from every row.
for l = 1:L
    mu_l    = mean(X(:,:,l), 1);         % (1 x p) centroid
    X(:,:,l) = X(:,:,l) - mu_l;          % zero-centred slice
end

% =========================================================================
% Initialization
% =========================================================================
% Initialize mean template to the first shape (unit Frobenius norm).
M = X(:,:,1);
M = M / norm(M, 'fro');                  % ||M||_F = 1  (prevents scale collapse)

% Preallocate outputs
R         = zeros(p, p, L);             % (p x p x L) rotation matrices
beta      = zeros(L, 1);               % (L x 1) scale factors
X_aligned = zeros(N, p, L);            % (N x p x L) aligned shapes

% =========================================================================
% ALS Iteration
% =========================================================================
for iter = 1:max_iter

    M_old = M;

    % ------------------------------------------------------------------
    % Step A: Align each shape to current M (Eq. 14.115 per-shape sub-problem)
    % ------------------------------------------------------------------
    for l = 1:L
        Xl = X(:,:,l);                   % (N x p) current shape

        % Cross-covariance matrix (ESL §14.5.3, step toward Eq. 14.57 family)
        C = Xl' * M;                     % (p x p)  = X_l^T * M

        % SVD Procrustes solution: R* = argmax_{R'R=I} tr(M^T X_l R)
        %   Equivalently: max_{R} tr(C^T R),  C = X_l^T M = U S V^T
        %   => R* = U * V^T                        (Orthogonal Procrustes)
        [U, S, V] = svd(C);              % C = U S V^T

        R(:,:,l) = U * V';              % (p x p) optimal rotation, ESL p.533

        % Optimal scale: beta* = tr(S) / ||X_l||_F^2
        %   Derived by: d/d(beta) ||beta X_l R - M||_F^2 = 0
        %   => beta = tr(M^T X_l R) / ||X_l||_F^2 = tr(S) / ||X_l||_F^2
        beta(l) = trace(S) / (norm(Xl, 'fro')^2);

        % Aligned shape: scaled + rotated version of X_l
        X_aligned(:,:,l) = beta(l) * Xl * R(:,:,l);   % (N x p)
    end

    % ------------------------------------------------------------------
    % Step B: Update mean template (arithmetic mean of aligned shapes)
    % ------------------------------------------------------------------
    M_raw = mean(X_aligned, 3);          % (N x p) — vectorized mean over L slices

    % Re-normalize to unit Frobenius norm (prevents degenerate scale collapse)
    M = M_raw / norm(M_raw, 'fro');

    % ------------------------------------------------------------------
    % Step C: Convergence check
    % ------------------------------------------------------------------
    delta = norm(M - M_old, 'fro');
    if delta < tol
        fprintf('[rigid_procrustes_average] Converged at iteration %d (delta=%.2e)\n', ...
            iter, delta);
        break;
    end

    if iter == max_iter
        warning('rigid_procrustes_average:NoConvergence', ...
            'Maximum iterations (%d) reached. delta=%.2e (tol=%.2e)', ...
            max_iter, delta, tol);
    end
end
end
