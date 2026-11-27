classdef IOrthogonalizer < handle
    % IORTHOGONALIZER Interface for orthogonalization strategies (QR decomposition).
    %
    % Strategies implementing this interface provide numerical methods to 
    % decompose matrices into Orthogonal (Q) and Upper Triangular (R) components,
    % as well as update those components incrementally to support stepwise models.
    
    methods (Abstract)
        % DECOMPOSE Performs a full QR decomposition of matrix A.
        % OUTPUTS:
        %   Q - Orthonormal basis matrix
        %   R - Upper triangular matrix
        [Q, R] = decompose(obj, A);
        
        % ADDCOLUMNS Incrementally updates Q and R with new columns H.
        % OUTPUTS:
        %   Q_new - Updated orthonormal basis
        %   R_new - Updated upper triangular matrix
        [Q_new, R_new] = addColumns(obj, Q_old, R_old, H);
        
        % FASTPROJECT Quickly orthogonalizes candidate vector h against Q_old
        % and calculates the residual drop. Highly optimized for inner knot loops
        % where full beta coefficients (R inverse) are not yet required.
        % OUTPUTS:
        %   rss - The new Residual Sum of Squares if h is added
        %   q_new - The orthogonalized version of h
        [rss, q_new] = fastProject(obj, Q_old, h, res_old);
    end
end
