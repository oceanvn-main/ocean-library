classdef (Abstract) ICrossValidator < handle
    % ICROSSVALIDATOR Interface for Cross-Validation strategies.
    %
    % METHODS:
    %   getIndices(N) -> [trainIdx, testIdx] 
    %   iterator(N)   -> Returns an iterator or count of folds.
    
    properties (Abstract)
        NumFolds double
    end
    
    methods (Abstract)
        % GETINDICES Returns the train/test logical indices for fold k.
        % Inputs: 
        %   N (int): Total number of samples.
        %   k (int): The fold index (1 to NumFolds).
        % Outputs:
        %   trainIdx (logical): [N x 1] mask for training.
        %   testIdx  (logical): [N x 1] mask for testing.
        [trainIdx, testIdx] = getIndices(obj, N, k)
    end
end
