classdef GreedySearch < OceanAlgo.Core.IOptimizationStrategy
    % GREEDYSEARCH A generic implementation of a Greedy Algorithm.
    %
    % "Greedy" here means:
    % 1. Generate all *immediate* possible moves/candidates (local search).
    % 2. Evaluate each candidate using an objective function.
    % 3. Pick the single best candidate that improves the objective the most.
    %
    % This is "General" because it doesn't know about Decision Trees.
    % It only knows about a "Problem" that can `generateCandidates()` and `evaluate()`.
    %
    % algorithm:
    %   best = null
    %   for candidate in problem.generateCandidates():
    %       score = problem.evaluate(candidate)
    %       if score is better than best:
    %           best = candidate
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    properties
        Verbose (1,1) logical = false
    end

    methods
        function obj = GreedySearch(options)
            arguments
                options.Verbose = false;
            end
            obj.Verbose = options.Verbose;
        end

        function [bestSolution, bestScore] = optimize(obj, problem)
            % OPTIMIZE Runs the greedy search over candidates provided by the problem.
            
            % 1. Get Iterator/List of candidates
            % The problem must typically return an array or cell array of candidates.
            % Or a generator (if Matlab supported yield, but here we assume list/array)
            candidates = problem.generateCandidates();
            
            bestSolution = [];
            bestScore = Inf; % Assuming Minimization (RSS, Error, etc). 
            % TODO: Make minimization/maximization configurable if needed. 
            % For Regression Trees (RSS), we minimize.
            
            iter = 0;
            if obj.Verbose
                fprintf('GreedySearch: Evaluating %d candidates...\n', length(candidates));
            end
            
            % 2. Iterate and Evaluate
            % If candidates is a generic object array or cell array
            isCell = iscell(candidates);
            n = length(candidates);
            
            for i = 1:n
                if isCell
                    cand = candidates{i};
                else
                    cand = candidates(i);
                end
                
                % Delegate evaluation (Calculation of RSS/Entropy) back to the problem
                % because the problem owns the data.
                score = problem.evaluate(cand);
                
                if score < bestScore
                    bestScore = score;
                    bestSolution = cand;
                end
                iter = iter + 1;
            end
            
            if obj.Verbose && ~isempty(bestSolution)
                fprintf('GreedySearch: Found best score %.4f\n', bestScore);
            end
        end
    end
end
