classdef IncrementalQR < handle
    % INCREMENTALQR Maintains the QR decomposition of a growing design matrix.
    %
    % By storing the orthogonal Q matrix and upper triangular R matrix, this
    % class allows stepwise algorithms to add new basis functions to the model
    % in O(NM) time rather than re-inverting the entire matrix in O(NM^2) time.
    
    properties (SetAccess = private)
        Q       % N x M Orthogonal basis
        R       % M x M Upper triangular matrix
        N       % Number of rows
        M       % Number of columns
        Orthogonalizer % Strategy instance (IOrthogonalizer)
    end
    
    methods
        function obj = IncrementalQR(B_initial, orthoStrategy)
            if nargin < 2 || isempty(orthoStrategy)
                % Default to highest stability strategy
                obj.Orthogonalizer = OceanMath.LinearAlgebra.Orthogonalization.HouseholderReflection();
            else
                obj.Orthogonalizer = orthoStrategy;
            end
            
            % Initializes the QR state with the first basis function(s).
            [obj.N, obj.M] = size(B_initial);
            
            % Initial Strategy decomposition
            [obj.Q, obj.R] = obj.Orthogonalizer.decompose(B_initial); 
        end
        
        function [beta, rss, res] = addColumns(obj, H, Y)
            % ADDCOLUMNS Incrementally updates Q and R with new columns H.
            % Uses Modified Gram-Schmidt to orthogonalize H against Q.
            %
            % INPUTS:
            %   H - (N x K) Matrix of new basis functions to add
            %   Y - (N x 1) Target vector to recalculate coefficients
            
            % Delegate internal matrix updating to the configured strategy
            [obj.Q, obj.R] = obj.Orthogonalizer.addColumns(obj.Q, obj.R, H);
            obj.M = size(obj.Q, 2);
            
            % 5. Fast recalculation of coefficients (beta) and RSS
            % Because Q is orthogonal, beta = R \ (Q' * Y)
            
            if nargout > 0
                qty = obj.Q' * Y;
                
                warning('off', 'MATLAB:rankDeficientMatrix');
                beta = obj.R \ qty;
                warning('on', 'MATLAB:rankDeficientMatrix');
                
                % Fast RSS calculation
                if nargout > 1
                    res = Y - obj.Q * qty;
                    rss = sum(res.^2, 'all');
                end
            end
        end
        
        function [beta, rss, res] = recalculate(obj, Y)
            % RECALCULATE Fast coefficient update without changing the matrix.
            qty = obj.Q' * Y;
            beta = obj.R \ qty;
            res = Y - obj.Q * qty;
            rss = sum(res.^2, 'all');
        end
        
        function [rss, q_new] = fastResidualUpdate(obj, h, res_old)
            % FASTRESIDUALUPDATE Calculates the new RSS after adding candidate h
            % temporarily via the generic strategy.
            [rss, q_new] = obj.Orthogonalizer.fastProject(obj.Q, h, res_old);
        end
    end
end
