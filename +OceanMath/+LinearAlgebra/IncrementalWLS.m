classdef IncrementalWLS < handle
    % INCREMENTALWLS Maintains the Weighted Least Squares normal equations.
    %
    % Used by PolyMARSForwardSelector to maintain (B'WB) and (B'Ws)
    % incrementally as basis functions are added to the model. This allows
    % O(NM) updates instead of a full O(NM^2) re-solve each step.
    %
    % The weight matrix W is diagonal (N x 1 weight vector) and changes
    % at every forward step (driven by current softmax probabilities).
    %
    % Notation:
    %   B  - (N x M) current basis matrix
    %   W  - (N x 1) diagonal weights (Fisher info per sample = p_i*(1-p_i))
    %   s  - (N x K) score matrix  (Y_ind - P)
    %   BtWB - (M x M) = B' * diag(W) * B   (Gram matrix under W)
    %   BtWs - (M x K) = B' * diag(W) * s   (cross-term)

    properties (SetAccess = private, GetAccess = public)
        B       % (N x M) current design matrix
        
        %%% MATH %%%
        % This stores the (Negative) HESSIAN matrix: H = B^T * W * B
        BtWB    
        
        %%% MATH %%%
        % This stores the GRADIENT vector (Score): g = B^T * W * s
        % Because s = W^-1 * (Y - P), the W's cancel out leaving B^T * (Y - P)
        BtWs    
        
        N       % number of samples
        M       % number of current basis functions
    end

    methods
        function obj = IncrementalWLS(B0, W, s)
            % Constructor.
            % INPUTS:
            %   B0 - (N x M0) initial design matrix (at least the intercept column)
            %   W  - (N x 1)  initial diagonal weight vector
            %   s  - (N x K)  initial score matrix (Y_ind - P)
            [obj.N, obj.M] = size(B0);
            obj.B    = B0;
            
            % Element-wise scale avoids building a massive NxN diagonal matrix
            Bw       = B0 .* W;        
            
            %%% MATH: HESSIAN %%%
            % Curvature Matrix: H = B0^T * W * B0
            obj.BtWB = Bw' * B0;       
            
            %%% MATH: GRADIENT %%%
            % Score Vector: g = B0^T * W * s
            obj.BtWs = Bw' * s;        
        end

        function refresh(obj, B, W, s)
            % REFRESH Recomputes BtWB and BtWs from scratch.
            obj.B    = B;
            [obj.N, obj.M] = size(B);
            
            %%% MATH: HESSIAN %%%
            obj.BtWB = Bw' * B; 
            
            %%% MATH: GRADIENT %%%
            obj.BtWs = Bw' * s; 
        end

        function resolveStably(obj, B, W, eta, error, H)
            % RESOLVESTABLY Implements the division-free GLM update.
            % Theory: Avoids RHS = X'W(eta + (y-p)/W) which has a numerical
            % division by W. Instead solves (X'WX) beta = X'W*eta + X'H*(y-p).
            % This is the numerically stable form of IRLS when probabilities 
            % are near 0 or 1.
            obj.B = B;
            [obj.N, obj.M] = size(B);
            
            combined_weights = H .* W; % combined_weights = responsibilities * link_weights
            Bw = B .* combined_weights;
            
            %%% MATH: HESSIAN (CURVATURE) %%%
            % H = X' * W_combined * X
            obj.BtWB = Bw' * B;
            
            %%% MATH: GRADIENT (STABLE SCORE) %%%
            % Stable RHS: X'W*eta + X'H*(y-p)
            % This is exactly B^T * W * s but expanded to avoid W division.
            obj.BtWs = B' * (combined_weights .* eta + H .* error);
        end

        function addColumns(obj, H, W, s)
            % ADDCOLUMNS Incrementally extends BtWB and BtWs with new columns H.
            % INPUTS:
            %   H  - (N x K_new) new basis column(s) being committed
            %   W  - (N x 1)     current diagonal weights
            %   s  - (N x K)     current score matrix
            
            %%% MATH: ISOLATING THE NEW FEATURE (H) %%%
            % Instead of recalculating the old features against themselves, 
            % we only calculate the calculus for the new feature H.
            
            Hw  = H .* W;           
            HtWH = Hw' * H;         % Hessian block: H^T * W * H (New vs New)
            HtWB = Hw' * obj.B;     % Hessian block: H^T * W * B (New vs Old)
            HtWs = Hw' * s;         % Gradient block: H^T * W * s (New Gradient)
            BtWH = HtWB';           % Hessian block: B^T * W * H (Old vs New)

            %%% MATH: EXPANDING THE HESSIAN %%%
            % We physically build the block matrix for the updated Hessian:
            % [ B^T * W * B   |   B^T * W * H ]
            % [ H^T * W * B   |   H^T * W * H ]
            obj.BtWB = [obj.BtWB, BtWH;
                        HtWB,     HtWH];

            %%% MATH: EXPANDING THE GRADIENT %%%
            % We stack the new gradient below the old gradient:
            % [ B^T * (Y - P) ]
            % [ H^T * (Y - P) ]
            obj.BtWs = [obj.BtWs; HtWs]; %Gradient 

            % Extend design matrix
            obj.B = [obj.B, H];
            obj.M = obj.M + size(H, 2);
        end

        function beta = solveForBeta(obj)
            % SOLVEFORBETA Returns the current WLS coefficient estimate.
            
            %%% MATH: NEWTON-RAPHSON UPDATE %%%
            % The fundamental linear algebra equation is (X^T * W * X) * beta = X^T * W * s
            % By doing obj.BtWB \ obj.BtWs, we are mathematically computing:
            % beta = (Hessian)^-1 * (Gradient)
            
            warning('off', 'MATLAB:singularMatrix');
            warning('off', 'MATLAB:nearlySingularMatrix');
            warning('off', 'MATLAB:rankDeficientMatrix');
            
            % Add robust Ridge (Tikhonov) regularization.
            % We use a slightly larger epsilon (1e-7) to stabilize the "Divide" phase.
            lambda = 1e-8 * max(diag(obj.BtWB)) + 1e-7;
            H_stable = obj.BtWB + eye(size(obj.BtWB)) * lambda;
            
            beta = H_stable \ obj.BtWs;
            
            warning('on',  'MATLAB:singularMatrix');
            warning('on',  'MATLAB:nearlySingularMatrix');
            warning('on',  'MATLAB:rankDeficientMatrix');
        end

    end
end