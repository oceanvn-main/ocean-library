function [x_star, y_star, w_star] = process_ties(x, y, w)
% PROCESS_TIES Preprocesses data for smoothing splines by handling ties in x.
%
%   [x_star, y_star, w_star] = process_ties(x, y, w)
%
%   Inputs:
%       x - Vector of independent variables (Nx1)
%       y - Vector of dependent variables (Nx1)
%       w - (Optional) Vector of weights (Nx1). Default is ones.
%
%   Outputs:
%       x_star - Sorted unique values of x (Mx1)
%       y_star - Weighted means of y corresponding to x_star (Mx1)
%       w_star - Sum of weights corresponding to x_star (Mx1)

    %Handle inputs
    %Ensure vectors are column vectors
    x = x(:);
    y = y(:);
    
    %If weighted are not provided, default to vector of ones
    if nargin < 3 || isempty(w)
        w = ones(size(x));
    else
        w = w(:);
    end
    
    %Identify unique X values
    [x_star, ~, idx] = unique(x);

    %Compute W* (Sum of weights for each unique x)
    w_star = accumarray(idx, w);
    
    %Compute Y* (Weighted mean)
    %First, compute the sum of (w*y) for each group
    y_weighted_sum = accumarray(idx, w.*y);
    % Then divide by the total weight of that group
    y_star = y_weighted_sum ./ w_star;
end
