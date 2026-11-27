classdef (Abstract) ISplitStrategy < handle
    % ISPLITSTRATEGY Interface for tree splitting algorithms.
    %
    % Implementations define how to find the best split for a dataset.
    %
    % Supported Algorithms:
    %   - CART (Binary splits on purity/MSE)
    %   - C5.0 (Multi-way splits on Gain Ratio)
    %   - Random (Random feature/threshold)
    %   - CHAID (Chi-square stats)
    %
    % Author: Ocean Library System
    % Date: 2026-02-17
    
    methods (Abstract)
        % FINDSPLIT Find the best split for the given data node.
        %
        % Inputs:
        %   X (N x P): Feature matrix
        %   Y (N x 1): Target vector
        %   Params (struct): Optional hyperparameters (MinLeaf, Criterion, etc.)
        %
        % Returns:
        %   result (SplitResult): Object describing the split (or Terminal).
        result = findSplit(obj, X, Y, params)
    end
end
