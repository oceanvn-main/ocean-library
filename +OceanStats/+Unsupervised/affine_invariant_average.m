function [M, X_aligned] = affine_invariant_average(X)
% AFFINE_INVARIANT_AVERAGE  Closed-form affine-invariant Procrustes mean.
%
%   Solves:  min_{A_l in GL(p), M: M^T M = I}  sum_l ||X_l * A_l - M||_F^2
%   via eigen-decomposition of the averaged projection matrix H_avg.
%   ESL Exercise 14.10, Eq. 14.60 (closed-form, no iteration required).
%
%   INPUT
%     X   (N x p x L) double  — landmark arrays, one per shape.
%                               May be uncentered; centering is applied internally.
%
%   OUTPUT
%     M         (N x p) double — affine-invariant mean template.
%                                Columns are orthonormal: M^T M = I_p.
%     X_aligned (N x p x L) double — (Optional) M projected onto each shape's column space.
%
%   ALGORITHM
%     1. Center each shape: X_l <- X_l - mean(X_l)
%     2. For each l: compute rank-p projection H_l = Q_l * Q_l^T
%                    where [Q_l, ~] = qr(X_l, 0)   (economy QR)
%                    This is numerically stable — avoids explicit matrix inversion.
%     3. H_avg = (1/L) * sum_l H_l
%     4. M = p largest eigenvectors of H_avg   (via eigs for efficiency)
%
%   NUMERICAL NOTES
%     - QR replaces X_l*(X_l^T X_l)^{-1}*X_l^T which is ill-conditioned.
%       Both give the same projection: onto col(X_l).      (ESL §14.5.3)
%     - eigs(H_avg, p, 'largestabs') is O(N^2 * p) vs eig(H_avg) O(N^3).
%     - real() discards negligible floating-point imaginary parts.
%
%   REFERENCE
%     Hastie et al. (2009), ESL §14.5.3 and Exercise 14.10.

arguments
    X (:,:,:) double
end

[N, p, L] = size(X);

% =========================================================================
% Pre-processing: center each shape at the origin (remove translation)
% =========================================================================
for l = 1:L
    mu_l     = mean(X(:,:,l), 1);        % (1 x p) centroid of slice l
    X(:,:,l) = X(:,:,l) - mu_l;          % zero-centred slice
end

% =========================================================================
% Compute and accumulate projection matrices  (ESL Eq. 14.61)
% =========================================================================
% H_l = X_l * (X_l^T X_l)^{-1} * X_l^T  is the orthogonal projector onto col(X_l).
% Numerically stable equivalent via economy QR:
%   X_l = Q_l * R_l  =>  H_l = Q_l * Q_l^T
%
H_avg = zeros(N, N);                     % (N x N) running sum of projectors
Q_cell = cell(L, 1);                     % Store Q_l for optional X_aligned

for l = 1:L
    Xl        = X(:,:,l);                % (N x p)
    [Q_l, ~]  = qr(Xl, 0);              % economy QR: Q_l is (N x p), orthonormal cols
    Q_cell{l} = Q_l;                     % save for later
    H_l       = Q_l * Q_l';             % (N x N) rank-p orthogonal projector
    H_avg     = H_avg + H_l;            % accumulate
end

% Average over L shapes
H_avg = H_avg / L;                      % (N x N) symmetric, PSD

% =========================================================================
% Eigen-decomposition: extract top-p eigenvectors  (ESL Eq. 14.60)
% =========================================================================
% H_avg is symmetric PSD with eigenvalues in [0, 1].
% The p largest eigenvectors span the "average column space" of the shapes.
% eigs is preferred over eig for large N: O(N^2 * p) vs O(N^3).

[M, ~] = eigs(H_avg, p, 'largestabs');  % M: (N x p), columns orthonormal

% Discard negligible imaginary parts from floating-point arithmetic
M = real(M);                             % (N x p), M^T M = I_p

% =========================================================================
% Optional: Project M back onto each shape's affine subspace
% =========================================================================
if nargout > 1
    X_aligned = zeros(N, p, L);
    for l = 1:L
        Q_l = Q_cell{l};
        X_aligned(:,:,l) = Q_l * (Q_l' * M);  % H_l * M
    end
end

end
