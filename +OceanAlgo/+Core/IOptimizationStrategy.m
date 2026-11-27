classdef (Abstract) IOptimizationStrategy < handle
    % IOPTIMIZATIONSTRATEGY Interface for general optimization strategies.
    %
    % This interface decouples the "Search Algorithm" from the "Problem" being solved.
    % It allows models (like Decision Trees) to delegate the search for the best
    % solution (like a split point) to an interchangeable optimizer.
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    methods (Abstract)
        % OPTIMIZE Selects the best candidate solution.
        %
        % Args:
        %   problem (handle): An object describing the problem. It must usually
        %                     provide methods to generate candidates and evaluate them.
        %                     The exact interface depends on the specific Strategy implementation,
        %                     but typically involves a `generateCandidates()` and `evaluate()` method.
        %
        % Returns:
        %   bestSolution: The optimal solution found.
        %   bestScore: The objective score of that solution.
        [bestSolution, bestScore] = optimize(obj, problem);
    end
end
