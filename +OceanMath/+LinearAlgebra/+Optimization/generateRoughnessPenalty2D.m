function Omega = generateRoughnessPenalty2D(nrows, ncols)
    % GENERATEROUGHNESSPENALTY2D Constructs a 2D discrete roughness penalty matrix
    %
    % Usage:
    %   Omega = OceanMath.LinearAlgebra.Optimization.generateRoughnessPenalty2D(16, 16);
    %
    % Description:
    %   Used for Penalized Discriminant Analysis (PDA) on image or 2D-signal
    %   predictors (e.g., 16x16 handwritten digit pixel values).
    %
    %   Penalizes roughness in BOTH row and column directions simultaneously,
    %   via the Kronecker sum of two 1D roughness penalties:
    %
    %     Omega_2D = kron(I_nrows, Omega_col) + kron(Omega_row, I_ncols)
    %
    %   where Omega_col penalizes column-direction roughness (neighboring pixels
    %   in the same column) and Omega_row penalizes row-direction roughness
    %   (neighboring pixels in the same row).
    %
    %   The pixel vector is assumed to be in ROW-MAJOR order:
    %     pixel (i, j) -> index (i-1)*ncols + j
    %
    %   So penalizing column-direction means penalizing positions separated by
    %   ncols in the flat vector, which is kron(Omega_row, I_ncols).
    %
    % ── Derivation ─────────────────────────────────────────────────────────
    % Let x be the (nrows*ncols x 1) flattened image vector.
    %
    % 1D roughness along COLUMNS (vertical neighbors, stride = ncols apart):
    %   D_row = generateRoughnessPenalty1D(nrows)    (nrows x nrows)
    %   Omega_vertical = kron(D_row, I_ncols)         (nrows*ncols x nrows*ncols)
    %   => x' * Omega_vertical * x = sum over columns of 1D roughness per column
    %
    % 1D roughness along ROWS (horizontal neighbors, stride = 1 apart):
    %   D_col = generateRoughnessPenalty1D(ncols)     (ncols x ncols)
    %   Omega_horizontal = kron(I_nrows, D_col)        (nrows*ncols x nrows*ncols)
    %   => x' * Omega_horizontal * x = sum over rows of 1D roughness per row
    %
    % Total 2D roughness:
    %   Omega_2D = Omega_vertical + Omega_horizontal
    %   => x' * Omega_2D * x approximates the discrete integral of [d2f/dx1^2 + d2f/dx2^2]
    %
    % Properties:
    %   - Symmetric:           (A+B)' = A+B   ✓ (since each term is D'D)
    %   - Positive semi-def:   x'(A+B)x = x'Ax + x'Bx >= 0  ✓
    %   - Sparse:              built from sparse blocks
    %
    % Input:
    %   nrows - Number of pixel rows (e.g., 16)
    %   ncols - Number of pixel cols (e.g., 16)
    %
    % Output:
    %   Omega - (nrows*ncols x nrows*ncols) sparse symmetric PSD penalty matrix.

    arguments
        nrows (1,1) double {mustBePositive, mustBeInteger}
        ncols (1,1) double {mustBePositive, mustBeInteger}
    end

    % 1D penalty along each dimension
    Omega_row = OceanMath.LinearAlgebra.Optimization.generateRoughnessPenalty1D(nrows);
    Omega_col = OceanMath.LinearAlgebra.Optimization.generateRoughnessPenalty1D(ncols);

    % Kronecker sum: penalizes roughness along both axes
    % kron(Omega_row, I_ncols): applies row smoothing across each column stripe
    % kron(I_nrows, Omega_col): applies col smoothing within each row stripe
    Omega = kron(Omega_row, speye(ncols)) + kron(speye(nrows), Omega_col);
end
