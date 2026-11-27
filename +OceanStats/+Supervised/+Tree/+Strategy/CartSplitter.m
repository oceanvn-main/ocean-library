classdef CartSplitter < OceanStats.Supervised.Tree.Strategy.ISplitStrategy
    % CARTSPLITTER Implements binary splitting strategy (CART).
    %
    % Uses a greedy search optimizer to find the best binary split (X <= t).
    % Supports pluggable Criterion (Gini, Entropy, RSS) via params.
    %
    % Author: Ocean Library System
    % Date: 2026-02-17
    
    properties
        Optimizer % OceanAlgo.Greedy.GreedySearch or similar
    end
    
    methods
        function obj = CartSplitter(optimizer)
            if nargin < 1 || isempty(optimizer)
                obj.Optimizer = []; % Placeholder
            else
                obj.Optimizer = optimizer;
            end
        end
        
        function result = findSplit(obj, X, Y, params)
            % FINDSPLIT  Find the best split for the given data node.
            
            import OceanStats.Supervised.Tree.Strategy.SplitResult
            
            % 1. Build the split problem (criterion owns data + evaluateFn)
            if isfield(params, 'CandidateGenerator')
                cg = params.CandidateGenerator;
            else
                cg = [];
            end
            problem = params.Criterion.createSplitProblem(X, Y, params.MinLeafSize, cg);
            
            % 2. Find the best split.
            %    FAST PATH: TreeSplitProblem.findBestSplit() sorts each feature
            %    ONCE and uses rssIncremental — O(N*p*logN + k*p) per node.
            %    FALLBACK:  GreedySearch.optimize() — O(N*k*p) per node.
            if ismethod(problem, 'findBestSplit')
                [bestSplit, bestScore] = problem.findBestSplit();
            else
                if isempty(obj.Optimizer)
                    error('CartSplitter:MissingOptimizer', ...
                        'An Optimizer (IOptimizationStrategy) is required.');
                end
                [bestSplit, bestScore] = obj.Optimizer.optimize(problem);
            end
            
            % 3. Package result
            result = SplitResult();
            
            if isempty(bestSplit) || isinf(bestScore)
                result.IsTerminal    = true;
                result.Description   = "Leaf (No Split)";
                return;
            end
            
            result.IsTerminal    = false;
            result.FeatureIndex  = bestSplit.FeatureIndex;
            result.Threshold     = bestSplit.Threshold;
            result.Gain          = -bestScore;
            result.Description   = sprintf("Feat %d <= %.4f", result.FeatureIndex, result.Threshold);
            result.LeftChildName  = "True";
            result.RightChildName = "False";
            
            % NO SURROGATE LOGIC HERE.
            % Use SurrogateSplitStrategy Decorator for surrogates.
        end
    end
end
