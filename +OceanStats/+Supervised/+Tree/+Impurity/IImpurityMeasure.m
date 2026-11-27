classdef (Abstract) IImpurityMeasure < handle
    % IIMPURITYMEASURE Abstract interface for node impurity criteria.
    %
    % Impurity measures quantify the "disorder" of class labels in a tree node.
    % Each implementation takes a vector of class counts and returns a scalar.
    %
    % Implementations:
    %   - GiniIndex
    %   - CrossEntropy (Deviance)
    %   - MisclassificationError
    %
    % Reference: Hastie, Tibshirani, Friedman — The Elements of Statistical
    %            Learning, 2nd Ed., Section 9.2.3.
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods (Abstract)
        % COMPUTE Calculate the impurity of a node or set of nodes.
        %
        % Args:
        %   P (N x K double): Matrix of class probabilities. If a 1xK vector
        %                     is provided, returns a scalar. If an NxK matrix
        %                     is provided, returns an Nx1 column vector.
        %                     Note: Can also accept raw class counts, which
        %                     it will internally normalize to probabilities.
        %
        % Returns:
        %   val (N x 1 double): Impurity values. 0 = pure node.
        val = compute(obj, P);
    end

    methods
        function n = name(obj)
            % NAME Returns a display-friendly name for this criterion.
            n = class(obj);
        end
    end
end
