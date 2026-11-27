classdef (Abstract) INodeCriterion < handle
    % INODECRITERION Abstract strategy for task-specific tree behavior.
    %
    % Encapsulates the differences between regression and classification trees:
    %   - How to compute node statistics (value, risk)
    %   - How to create a split problem for the optimizer
    %   - How to measure cross-validation error
    %
    % Implementations:
    %   - RegressionCriterion  (mean/RSS)
    %   - ClassificationCriterion (majority class / impurity)
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods (Abstract)
        % COMPUTENODESTATS Compute task-specific statistics for a node.
        %
        % Args:
        %   Y (N x 1): Target values in this node.
        %
        % Returns:
        %   stats (struct): Must contain at minimum:
        %     .Value  — prediction value (mean for regression, majority class for classification)
        %     .Risk   — node risk for pruning (RSS for regression, weighted misclass for classification)
        %     May also contain: .ClassProbs, .ClassCounts (for classification)
        stats = computeNodeStats(obj, Y);

        % ISPURE Check if a node's targets are sufficiently pure to stop splitting.
        %
        % Args:
        %   Y (N x 1): Target values in this node.
        %
        % Returns:
        %   tf (logical): true if no further splitting is beneficial.
        tf = isPure(obj, Y);

        %   candidateGenerator (ICandidateGenerator): Optional.
        %
        % Returns:
        %   problem (handle): Object with generateCandidates() and evaluate() methods.
        problem = createSplitProblem(obj, X, Y, minLeaf, candidateGenerator);

        % CVERROR Compute cross-validation error for a fold.
        %
        % Args:
        %   Y_true (N x 1): True target values.
        %   Y_pred (N x 1): Predicted values.
        %
        % Returns:
        %   err (double): Scalar error (RSS for regression, misclass count for classification).
        err = cvError(obj, Y_true, Y_pred);
    end
end
