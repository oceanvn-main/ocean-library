classdef BootstrapSplitter < OceanStats.Base.ICrossValidator
    % BOOTSTRAPSPLITTER Generates Training/Test splits via Bootstrap Resampling.
    %
    % OVERVIEW:
    %   Used when traditional K-Fold CV is unstable (e.g., small N).
    %   - Training Set: A random sample of size N with replacement (approx 63.2% unique).
    %   - Test Set: The "Out-Of-Bag" (OOB) samples (approx 36.8%).
    %
    % USAGE:
    %   cv = BootstrapSplitter(50); % 50 Bootstrap Repetitions
    
    properties
        NumFolds double % Number of Resamples (Repetitions)
        StoredIndices cell % Cache for reproducibility if needed
    end
    
    methods
        function obj = BootstrapSplitter(num_resamples)
            if nargin > 0, obj.NumFolds = num_resamples; end
        end
        
        function [train_idx, test_idx] = getIndices(obj, N, k)
            % GETINDICES Returns bootstrap sample and OOB for iteration k.
            % Note: We generate fresh samples unless we caching.
            % For simplicity, we generate on the fly using rng state.
            
            % 1. Resample (Training)
            train_idx = randsample(N, N, true); 
            
            % 2. OOB (Testing)
            all_idx = 1:N;
            test_idx = setdiff(all_idx, unique(train_idx))';
            
            % Edge Case: If by tiny chance resampling picks perfectly unique set (impossible for large N),
            % test_idx is empty.
            if isempty(test_idx)
                 warning('BootstrapSplitter:EmptyOOB', ...
                     'Bootstrap sample contains all unique indices. OOB is empty.');
            end
        end
    end
end
