classdef LeaveOneOut < OceanStats.Base.ICrossValidator
    % LEAVEONEOUT Validation Strategy for Tiny Data.
    %
    % OVERVIEW:
    %   A special case of K-Fold where K = N.
    %   - Training Set: N-1 samples.
    %   - Test Set: 1 sample.
    %   - Repeats N times.
    %
    % USAGE:
    %   cv = LeaveOneOut(N);
    
    properties
        NumFolds double
    end
    
    methods
        function obj = LeaveOneOut(N_known)
            if nargin > 0
                obj.NumFolds = N_known;
            else
                error('LeaveOneOut requires N to be known at construction. Usage: LeaveOneOut(N).');
            end
        end
        
        function [train_idx, test_idx] = getIndices(~, N, k)
            % GETINDICES Returns indices for the k-th sample held out.
            % k goes from 1 to N.
            if k > N, error('Index k exceeds sample size N'); end
            test_idx = k;
            train_idx = setdiff(1:N, test_idx)';
        end
    end
end
