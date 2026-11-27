function N = getNSBasis(x, knots)
% GET_NS_BASIS Generates the Natural Cubic Spline Basis Matrix
%
%   N = GET_NS_BASIS(x, knots) computes the basis functions for a Natural
%   Cubic Spline given the input vector x and a specified set of knots.
%
%   INPUTS:
%       x     - (Nx1 Vector) The raw input variable (e.g., age, sbp).
%       knots - (Kx1 Vector) The sorted locations of the knots \xi.
%               Must include boundary knots (min/max) and interior knots.
%
%   OUTPUT:
%       N     - (Nx(K-1) Matrix) The design matrix.
%               Columns correspond to the basis functions N_2, ..., N_K.
%               Note: The intercept N_1 = 1 is EXCLUDED to prevent
%               collinearity with the global model intercept.
%
%   THEORY REFERENCES:
%       Based on "The Elements of Statistical Learning", Chapter 5.
%       - Enforces linearity beyond boundary knots (Natural Constraint).
%       - Implements Equations (5.4) and (5.5).
%
%  

    % Ensure inputs are column vectors for consistent broadcasting
    x = x(:);
    knots = knots(:);
    
    % K is the total number of knots (including boundaries)
    K = length(knots);
    
    % ---------------------------------------------------------------------
    % HELPER FUNCTION: d_k(X)
    % Reference: Equation (5.5) in the text
    %
    % d_k(X) = [ (X - xi_k)_+^3 - (X - xi_K)_+^3 ] / (xi_K - xi_k)
    %
    % This function constructs a curve that starts at knot k, but becomes
    % a quadratic parabola (constant curvature) after the last knot K.
    % The division normalizes the curvature to be exactly 6.
    % ---------------------------------------------------------------------
    d = @(vals, k) ( pos_part(vals - knots(k)).^3 - ...
                     pos_part(vals - knots(K)).^3 ) ./ ...
                     (knots(K) - knots(k));

    %Pre-allocate the Basis Matrix N
    %We need K-1 columns because:
    %Total DF = K.
    %We remove N_1 (Intercept) to avoid multicollinearity -> Remaining K-1
    N = zeros(length(x), K-1);
    % ---------------------------------------------------------------------
    % BASIS 1: The Linear Term
    % Reference: N_2(X) = X  (from Equation 5.4)
    %
    % Since we removed the intercept (N_1 = 1), the first column of our
    % matrix is the linear term X. This allows the model to default to
    % a straight line if all non-linear coefficients are zero.
    % ---------------------------------------------------------------------
    N(:, 1) = x;
    % BASIS 2: The Curvature Terms
    % Reference: N_{k+2}(X) = d_k(X) - d_{K-1}(X) (Eq 5.4)
    %
    % We loop from k = 1 to K-2.
    % By subtracting d_{K-1} from d_k, we ensure that the quadratic tails
    % cancel each other out.
    % Result: The second derivative becomes 0 past the last knot.
    %         This satisfies the Natural Boundary Condition (Linearity).
    % ---------------------------------------------------------------------
    for k = 1:(K-2)
        %Map loop index k to matrix column k+1
        N(:, k+1) = d(x, k) - d(x, K-1);
    end

end

function val = pos_part(val)
%POS_PART implements the Truncated Power operator ()_+
%Return vals if val > 0, else 0
    val = max(0, val);
end

