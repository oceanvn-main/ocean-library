function d = amariDist(A0, A_hat)
% AMARIDIST  Amari distance between true and estimated mixing matrices.
%
%   Measures how close A_hat is to A0 up to permutation and sign of columns.
%   d=0 means perfect recovery. Larger d = worse separation.
%
%   FORMULA (Bach & Jordan 2002, ESL Eq. 14.97):
%     R = A0 * inv(A_hat)
%     d(A0, A_hat) = (1/2p) * sum_i [sum_j |r_ij| / max_j |r_ij| - 1]
%                 + (1/2p) * sum_j [sum_i |r_ij| / max_i |r_ij| - 1]
%
%   where:
%     r_ij = (A0 * A_hat^{-1})_ij
%     Row term: measures that each row of R has one dominant entry (permutation)
%     Col term: measures that each col of R has one dominant entry (sign/scale)
%
%   INPUTS:
%     A0    (p x p) true mixing matrix
%     A_hat (p x p) estimated mixing matrix
%
%   OUTPUT:
%     d     scalar — Amari distance in [0, (p-1)/p * (p/(p-1) - 1)] approx [0, 1]
%
% Author: Ocean Library | 2026-04-30

p    = size(A0, 1);
R    = A0 / A_hat;          % A0 * inv(A_hat),  (p x p)
absR = abs(R);

% Row normalization: each row divided by its maximum element
row_max  = max(absR, [], 2);   % (p x 1)
row_norm = absR ./ row_max;    % (p x p)
term1    = sum(sum(row_norm, 2) - 1);   % sum_i [sum_j normalized - 1]

% Column normalization: each column divided by its maximum element
col_max  = max(absR, [], 1);   % (1 x p)
col_norm = absR ./ col_max;    % (p x p)
term2    = sum(sum(col_norm, 1) - 1);   % sum_j [sum_i normalized - 1]

d = (term1 + term2) / (2 * p);
end
