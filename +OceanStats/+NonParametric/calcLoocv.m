function [cv_score, effective_dof, S_diagonals] = calcLoocv(X, Y, bandwidth, kernel_type, degree)
%CALC_LOOCV Calculates the exact LOOCV score for Local Polynomial Regression.
%   [cv, dof] = CALC_LOOCV(X, Y, h) computes the Leave-One-Out Cross-Validation
%   score for a local linear regression model with bandwidth h.
%
%   [cv, dof] = CALC_LOOCV(X, Y, h, kernel, degree) allows specification of the
%   smoothing kernel and polynomial degree.
%
%   Inputs:
%       X           - (Nx1) Predictor vector (numeric).
%       Y           - (Nx1) Response vector (numeric).
%       bandwidth   - (Scalar) Smoothing parameter (>0).
%       kernel_type - (String) 'gaussian' (default), 'epanechnikov', 'tricube'.
%       degree      - (Int) Polynomial degree: 0 (constant), 1 (linear), 2 (quadratic).
%
%   Outputs:
%       cv_score    - Total Residual Sum of Squares (LOOCV).
%       effective_dof - Trace of the smoother matrix (Model Complexity).
%
%   Example:
%       [score, df] = OceanStats.NonParametric.calcLoocv(x, y, 0.5, 'tricube', 2);
%
%   See also LOCAL_REG, VARYING_COEF.

    %Input Santiziation
    X = X(:); Y = Y(:);
    N = length(X);
    
    if nargin < 4, kernel_type ='gaussian'; end
    if nargin < 5, degree = 1; end
    S_diagonals = zeros(N, 1);
    Y_fitted = zeros(N, 1);

    %Loop: Compute Local Hat Values

    for i = 1:N

        x0 = X(i);

        %1. Kernel Weights
        dists = abs(X - x0) / bandwidth;
        switch lower(kernel_type)
            case 'gaussian', weights = exp(-(dists.^2) / 2);
            case 'epanechnikov', weights = 0.75 * (1 - dists.^2); weights(dists>1)=0;
            case 'tricube', weights = (1 - dists.^3).^3; weights(dists>1)=0;
            case 'uniform', weights = ones(N, 1); weights(dists>1)=0;
            otherwise, error('Unknown kernel type.');

        end
        W = diag(weights);

        %2. Design Matrix (Polynomial)
        Z = zeros(N, degree + 1);

        for p = 0:degree, Z(:, p+1) = (X - x0).^p; end

        %3. Local Projection (Hat Value)
        %We add a tiny ridge to prevent singular matrix
        try
            InvGram = inv(Z' * W * Z + eye(degree+1)* 1e-10);
            Projector = InvGram * Z' * W;
            S_row = Projector(1, :); %First row (Intercept at x0)
            Y_fitted(i) = S_row * Y;
            S_diagonals(i) = S_row(i); %The "Self-Influence"
        catch 
            Y_fitted(i) = NaN;
            S_diagonals = NaN;
        end

        %Ouput 1: Effective Degrees of Freedom
        %Definition: The sum of the diagonal elements (Trace of S)
        %Represents the equivalent number of parameters used by the
        %smoother.
        valid_idx = ~isnan(S_diagonals);
        effective_dof = sum(S_diagonals(valid_idx));

        %Output 2: Shortcut LOOCV Score
        residuals = Y - Y_fitted;

        %Filter unstable points (Leverage ~ 1 implies interpolation)
        stable_idx = valid_idx & (S_diagonals < 0.999);
        if sum(~stable_idx) > 0
            warning('Stability Warning: %d points have leverage ~1. Ignored in CV.', sum(~stable_idx));
        end

        %The Magic Formula (Exercise 6.7) 
        cv_score = sum( (residuals(stable_idx) ./ (1 - S_diagonals(stable_idx))).^2);
        
    end

end