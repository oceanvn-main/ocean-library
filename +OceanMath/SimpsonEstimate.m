function S = simpsonEstimate(g, a, b, n)
% simpsonEstimate: Approximates a definite integral using Simpson's 1/3 Rule.
%
%   S = simpsonEstimate(g, a, b, n) computes the numerical integral of
%   the function 'g' (a function handle) from a lower limit 'a' to an
%   upper limit 'b' using 'n' subintervals.
%
%   Assumptions:
%   - 'g' is a function handle that supports vectorized input 
%     (e.g., it uses '.*', './', '.^' instead of '*', '/', '^').
%   - 'n' must be a positive, even integer.
%
%   Inputs:
%     g - Function handle. The integrand (e.g., @(x) x.^2).
%     a - Scalar. The lower limit of integration.
%     b - Scalar. The upper limit of integration.
%     n - Integer. The number of subintervals (MUST be even).
%
%   Outputs:
%     S - Scalar. The approximate value of the definite integral.
%
%   Example:
%     % To find the integral of sin(x) from 0 to pi with 20 steps:
%     f = @(x) sin(x);
%     area = simpsonEstimate(f, 0, pi, 20);
%     % (The exact answer is 2. The result will be very close.)
%
%   See also: integral, integral2, integral3, quad, trapz

% --- 1. Input Validation ---
% Check if n is an even integer.
if mod(n, 2) ~= 0 || n <= 0
    error('Input ''n'' (number of intervals) must be a positive, even integer.');
end
if ~isa(g, 'function_handle')
    error('Input ''g'' must be a function handle (e.g., @(x) x.^2).');
end

% --- 2. Setup ---
% Calculate the width of each subinterval (Delta x).
dx = (b - a) / n;

% Generate a vector of all grid points from x0 to xn.
% This vector will have n+1 elements.
x = a:dx:b;

% --- 3. Evaluate Function ---
% Evaluate the function g at all grid points simultaneously.
% This vectorized approach is much faster than a for-loop.
try
    y = g(x);
catch ME
    fprintf('Error evaluating the function handle. Did you use vectorized operators (.*, .^, ./) ?\n');
    rethrow(ME);
end

% --- 4. Apply Simpson's Coefficients ---
% Create the coefficient vector [1, 4, 2, 4, ..., 2, 4, 1]

% Start with a vector of all ones.
coeffs = ones(1, n+1);

% Set the '4' coefficients for even-indexed elements (2, 4, ..., n)
% (These correspond to x1, x3, x5, ...)
coeffs(2:2:n) = 4;

% Set the '2' coefficients for inner odd-indexed elements (3, 5, ..., n-1)
% (These correspond to x2, x4, x6, ...)
coeffs(3:2:n-1) = 2;

% --- 5. Calculate Final Sum ---
% Apply the Simpson's 1/3 Rule formula: (dx/3) * sum(coeffs * y_values)
% We use '.*' for element-wise multiplication of the two vectors.
S = (dx / 3) * sum(coeffs .* y);

end % End of function
