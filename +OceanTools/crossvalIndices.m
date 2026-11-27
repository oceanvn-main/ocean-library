function indices = crossval_indices(N, k)
% CROSSVAL_INDICES Generates random k-fold indices
%   indices = crossval_indices(N, k)
%
%   Inputs:
%       N - Total number of samples
%       k - Number of folds (e.g., 10)
%
%   Output:
%       indices - (N x 1) Vector where each element is a number 1..k

    % 1. Create a repeated sequence [1, 2, ..., k, 1, 2, ...]
    indices = repmat(1:k, 1, ceil(N/k));
    
    % 2. Trim it to the exact length of data N
    indices = indices(1:N)';
    
    % 3. Shuffle randomly
    indices = indices(randperm(N));
end