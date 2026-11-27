function Omega = generateSpatialPenalty(rows, cols)
    % GENERATESPATIALPENALTY Constructs a 2D discrete spatial Laplacian matrix
    %
    % Usage:
    %   Omega = OceanMath.LinearAlgebra.Optimization.generateSpatialPenalty(16, 16);
    %
    % Description:
    %   Used for Penalized Discriminant Analysis (PDA) to enforce spatial 
    %   smoothness on image predictors. The penalty x^T * Omega * x measures 
    %   the sum of squared differences between vertically and horizontally 
    %   adjacent pixels in a flattened (rows * cols) x 1 vector.
    % 
    %   It computes the graph Laplacian of a 2D grid graph using the 
    %   Kronecker sum of 1D path Laplacians:
    %   Omega = (D_rows \otimes I_cols) + (I_rows \otimes D_cols)
    %
    % Input:
    %   rows - Number of pixel rows in the image (e.g., 16)
    %   cols - Number of pixel columns in the image (e.g., 16)
    %
    % Output:
    %   Omega - (rows*cols) x (rows*cols) sparse, symmetric positive 
    %           semi-definite penalty matrix.
    
    arguments
        rows (1,1) double {mustBePositive, mustBeInteger}
        cols (1,1) double {mustBePositive, mustBeInteger}
    end
    
    % 1D Laplacian matrices for paths (difference operator D^T D)
    D_r = create1DLaplacian(rows);
    D_c = create1DLaplacian(cols);
    
    % Identity matrices for dimensions
    I_r = speye(rows);
    I_c = speye(cols);
    
    % Kronecker sum for 2D grid graph Laplacian
    Omega = kron(D_r, I_c) + kron(I_r, D_c);
end

function L = create1DLaplacian(n)
    % Creates a 1D graph Laplacian for a path graph of length n.
    % E.g., for n=3:  [ 1 -1  0;
    %                  -1  2 -1;
    %                   0 -1  1 ]
    if n == 1
        L = sparse(0);
        return;
    end
    
    % Main diagonal: 2 for interior nodes, 1 for boundary nodes
    main_diag = 2 * ones(n, 1);
    main_diag(1) = 1;
    main_diag(n) = 1;
    
    % Off-diagonals: -1 for adjacent nodes
    off_diag = -1 * ones(n, 1);
    
    L = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], n, n);
end
