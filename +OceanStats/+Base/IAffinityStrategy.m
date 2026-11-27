classdef (Abstract) IAffinityStrategy < handle
    % IAFFINITYSTRATEGY Abstract interface for Graph Affinity Construction.
    %
    % Transforms a raw data matrix X into an N x N affinity/similarity graph W.
    % This represents the "Metric/Neighbor" layer, entirely decoupling the 
    % distance logic from the matrix math engine.
    
    methods (Abstract)
        % COMPUTEAFFINITY Generate the N x N adjacency/similarity matrix.
        %
        % Inputs:
        %   X - (N x p) feature matrix
        %
        % Outputs:
        %   W - (N x N) similarity matrix (sparse or dense)
        W = computeAffinity(obj, X)
    end
end
