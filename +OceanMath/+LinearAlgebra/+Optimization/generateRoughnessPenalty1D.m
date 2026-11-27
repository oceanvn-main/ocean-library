function Omega = generateRoughnessPenalty1D(n)
    % GENERATEROUGHNESSPENALTY1D Constructs a 1D discrete 2nd-derivative penalty matrix
    %
    % Usage:
    %   Omega = OceanMath.LinearAlgebra.Optimization.generateRoughnessPenalty1D(21);
    %
    % Description:
    %   Used for Penalized Discriminant Analysis (PDA) and smoothing splines
    %   to enforce smoothness on 1D signal predictors (e.g., waveforms, spectra). 
    %   
    %   It computes the squared 2nd-order difference operator matrix \Omega
    %   such that x^T * \Omega * x = \sum_i (x_{i+1} - 2x_i + x_{i-1})^2.
    %   This is the discrete approximation to \int [f''(t)]^2 dt.
    %
    % Input:
    %   n - Length of the 1D signal sequence.
    %
    % Output:
    %   Omega - (n x n) sparse, symmetric positive semi-definite penalty matrix.
    
    arguments
        n (1,1) double {mustBePositive, mustBeInteger}
    end
    
    if n < 3
        Omega = sparse(n, n);
        return;
    end
    
    % ── Derivation of D ─────────────────────────────────────────────────────
    % The discrete 2nd derivative (second-difference) at position i is:
    %   (D*x)[i] = x[i] - 2*x[i+1] + x[i+2] / delta x^2. But for optimization, we drop the constant delta x^2
    %            = [ 1, -2, 1 ] dot x[i:i+2]
    %
    % Applied to all n-2 interior positions, D is an (n-2 x n) matrix.
    % Each row slides the stencil [1, -2, 1] one position to the right.
    %
    % For n=5 (offsets 0,1,2 place stencil columns on super-diagonals):
    %   row 1 → [  1  -2   1   0   0 ]   ← stencil at positions 1,2,3
    %   row 2 → [  0   1  -2   1   0 ]   ← stencil at positions 2,3,4
    %   row 3 → [  0   0   1  -2   1 ]   ← stencil at positions 3,4,5
    %
    % spdiags([e, -2*e, e], 0:2, n-2, n) places the three columns onto
    % diagonal offsets 0 (main), 1 (1st super), 2 (2nd super) of the matrix.
    
    e = ones(n, 1);
    D = spdiags([e, -2*e, e], 0:2, n-2, n);
    
    % ── Derivation of Omega = D'*D ───────────────────────────────────────────
    % We want a penalty P(x) = sum_i (x[i+1] - 2x[i] + x[i-1])^2
    %                        = ||D*x||^2                 (squared 2-norm of D*x)
    %                        = (D*x)' * (D*x)
    %                        = x' * D' * D * x
    %                        = x' * Omega * x            (quadratic form)
    %
    % Therefore:  Omega = D' * D
    %
    % Two key properties fall out automatically:
    %   1. Symmetric:  (D'D)' = D'(D')' = D'D            ✓
    %   2. Positive semi-definite: x'*D'D*x = ||Dx||^2 >= 0 for all x  ✓
    %
    % This makes Omega valid to add to any least-squares objective as a
    % regularization term without making the problem ill-posed.
    
    Omega = D' * D;
end
