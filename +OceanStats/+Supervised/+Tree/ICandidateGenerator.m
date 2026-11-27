classdef (Abstract) ICandidateGenerator < handle
    % ICANDIDATEGENERATOR Interface for generating split candidates.
    %
    % Implementations define how to search for potential split points 
    % in a dataset (e.g., exhaustive midpoints, random sampling, quantiles).
    %
    % Author: Ocean Library System
    % Date: 2026-02-17

    methods (Abstract)
        % GENERATE Creates a list of split candidates.
        %
        % Args:
        %   X (N x P): Feature matrix.
        %   Y (N x 1): Target vector.
        %   minLeaf (double): Minimum samples required in each child.
        %
        % Returns:
        %   candidates: An array or cell array of candidate objects.
        candidates = generate(obj, X, Y, minLeaf)
    end
end
