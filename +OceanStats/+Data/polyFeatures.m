function out = polyFeatures(X, degree)
% POLYFEATURES Maps data to a higher-dimensional polynomial feature space.
%
%   out = OceanStats.Data.polyFeatures(X, degree)
%
%   OVERVIEW:
%       This function performs a Basis Expansion, projecting the input data X
%       from a low-dimensional linear space into a higher-dimensional 
%       Hilbert space. This allows linear models (like Logistic Regression 
%       or OLS) to learn non-linear decision boundaries.
%
%   THEORETICAL BACKGROUND:
%       Linear models compute predictions using a hyperplane: $y = w^T x$.
%       By expanding the features 'x' into polynomials $\phi(x)$, the linear
%       hyperplane in the high-dimensional space corresponds to a curved
%       surface in the original space.
%
%       Mapping $\phi(x_1, x_2)$ for Degree 2:
%       [x_1, x_2] -> [x_1, x_2, x_1^2, x_1*x_2, x_2^2]
%
%       Note: We exclude the bias term (intercept) here because standard 
%       MATLAB fitting functions (fitlm, mnrfit) add their own intercept.
%
%   INPUTS:
%       X      : [N x D] matrix of input features.
%       degree : Integer scalar defining the maximum polynomial power.
%
%   OUTPUTS:
%       out    : [N x M] matrix of expanded features.
%
%   ALGORITHM DETAILS:
%       - For D=2 (e.g., PCA projection or 2 selected features):
%         Generates full interaction terms (X1 * X2). This enables diagonal
%         rotation and "squeezing" of decision boundaries.
%       - For D~=2:
%         Generates simple powers (X.^d). This enables "bending" of the 
%         hyperplane but assumes feature independence (no interactions) to 
%         prevent combinatorial explosion (Curse of Dimensionality).

    % 1. Validation
    arguments
        X double
        degree (1,1) double {mustBeInteger, mustBePositive}
    end

    [N, D] = size(X);
    out = [];

    % 2. Case A: The "Wiggle" Engine (D=2)
    % Optimized for 2D visualizations (PCA or manual feature selection).
    % Includes interaction terms to allow boundaries to rotate.
    if D == 2
        X1 = X(:, 1);
        X2 = X(:, 2);
        
        % Iterate through polynomial orders
        for i = 1:degree
            % Iterate through combinations for the current order 'i'
            % Logic: X1^(i-j) * X2^j
            % i=2, j=0 => X1^2 * X2^0 = X1^2
            % i=2, j=1 => X1^1 * X2^1 = Interaction
            % i=2, j=2 => X1^0 * X2^2 = X2^2
            for j = 0:i
                term = (X1 .^ (i-j)) .* (X2 .^ j);
                out = [out, term]; %#ok<AGROW> 
            end
        end

    % 3. Case B: Simple Power Expansion (D != 2)
    % Fallback to prevent creating millions of features if D is large.
    else
        for d = 1:degree
            out = [out, X.^d]; %#ok<AGROW>
        end
    end
end