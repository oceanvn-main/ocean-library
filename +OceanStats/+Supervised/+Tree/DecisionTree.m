classdef DecisionTree < handle
    % DECISIONTREE A unified Decision Tree for regression and classification.
    %
    % Design:
    %   - Task-agnostic tree logic (growing, traversal, pruning, CV).
    %   - All task-specific behavior is delegated to an INodeCriterion strategy:
    %       * RegressionCriterion     (mean/RSS)
    %       * ClassificationCriterion (majority class / impurity)
    %   - Uses OceanAlgo.Core.IOptimizationStrategy to find best splits.
    %   - Supports Surrogate Splits via Decorator Pattern (SurrogateSplitStrategy).
    %
    % Usage:
    %   % Regression (default)
    %   tree = OceanStats.Supervised.Tree.DecisionTree('Criterion', 'regression');
    %   tree.fit(X, Y);
    %   Ypred = tree.predict(X);
    %
    %   % Classification with Surrogates
    %   tree = OceanStats.Supervised.Tree.DecisionTree('Criterion', 'gini', 'UseSurrogates', true);
    %   tree.fit(X, Y);
    %
    % Author: Ocean Library System
    % Date: 2026-02-17
    properties
        MaxDepth    (1,1) double = 3
        MinLeafSize (1,1) double = 5
        Optimizer   (1,1) OceanAlgo.Core.IOptimizationStrategy = OceanAlgo.Greedy.GreedySearch()
        CandidateGenerator (1,1) OceanStats.Supervised.Tree.ICandidateGenerator = ...
            OceanStats.Supervised.Tree.ExhaustiveCandidateGenerator()
        
        % SplitStrategy: Defines how splits are found (CART, C5.0, etc.)
        % Can be decorated with SurrogateSplitStrategy.
        SplitStrategy
        
        UseSurrogates (1,1) logical = false
        Criterion   % INodeCriterion — set in constructor via parseCriterion()
        
        % Model State
        RootNode struct = struct([])
        
        % Cost-Sensitive Learning
        LossMatrix  % K x K matrix of misclassification costs
    end

    properties (Access = private)
        % Scratch_  Shared output buffer for predictBatch_.
        % Declared as a handle property so all recursive frames see the
        % same memory — zero COW copies regardless of tree depth or N.
        Scratch_ double
    end
    
    % =====================================================================
    %  PUBLIC API
    % =====================================================================
    methods
        function obj = DecisionTree(options)
            % DECISIONTREE Constructor.
            %
            % Name-Value Args:
            %   MaxDepth (double): Maximum tree depth. Default: 3.
            %   MinLeafSize (double): Minimum samples per leaf. Default: 5.
            %   Criterion (string|INodeCriterion): Task criterion.
            %       Strings: 'regression' (default), 'gini', 'entropy',
            %                'misclassification'.
            %   Optimizer (IOptimizationStrategy): Search strategy. Default: GreedySearch.
            %   SplitStrategy (ISplitStrategy): Optional custom strategy.
            %   UseSurrogates (logical): specific for handling missing data.
            %   LossMatrix (double): Optional K x K cost matrix.
            
            arguments
                options.MaxDepth = 3;
                options.MinLeafSize = 5;
                options.Criterion = 'regression';
                options.Optimizer = OceanAlgo.Greedy.GreedySearch();
                options.CandidateGenerator = []; 
                options.SplitStrategy = [];
                options.UseSurrogates = false;
                options.LossMatrix = [];
            end
            
            obj.MaxDepth = options.MaxDepth;
            obj.MinLeafSize = options.MinLeafSize;
            obj.UseSurrogates = options.UseSurrogates;
            obj.Optimizer = options.Optimizer;
            obj.LossMatrix = options.LossMatrix;
            
            if isempty(options.CandidateGenerator)
                import OceanStats.Supervised.Tree.ExhaustiveCandidateGenerator
                obj.CandidateGenerator = ExhaustiveCandidateGenerator();
            else
                obj.CandidateGenerator = options.CandidateGenerator;
            end
            
            obj.Criterion = obj.parseCriterion(options.Criterion);
            
            % 1. Initialize Base Strategy
            if ~isempty(options.SplitStrategy)
                obj.SplitStrategy = options.SplitStrategy;
            elseif strcmpi(options.Criterion, 'c50')
                import OceanStats.Supervised.Tree.Strategy.C50Splitter
                obj.SplitStrategy = C50Splitter(obj.Optimizer);
            else
                % Default to CART
                import OceanStats.Supervised.Tree.Strategy.CartSplitter
                obj.SplitStrategy = CartSplitter(obj.Optimizer);
            end
            
            % 2. Apply Surrogate Decorator if requested
            if obj.UseSurrogates
                import OceanStats.Supervised.Tree.Surrogate.SurrogateSplitStrategy
                obj.SplitStrategy = SurrogateSplitStrategy(obj.SplitStrategy);
            end
        end
        
        function fit(obj, X, Y)
            % FIT Train the tree on data (X, Y).
            obj.RootNode = obj.growTree(X, Y, 0);
        end
        
        function Y_pred = predict(obj, X)
            % PREDICT  Predicts target values — vectorized, zero-copy.
            %
            %   Uses obj.Scratch_ (handle property) as shared output buffer.
            %   predictBatch_ writes Y_pred(idx)=value directly to Scratch_
            %   without passing the array through the recursion stack,
            %   eliminating all COW copies.

            error(nargoutchk(0, 1, nargout));
            N           = size(X, 1);
            obj.Scratch_ = zeros(N, 1);          % allocate shared buffer
            obj.predictBatch_(obj.RootNode, X, (1:N)');
            Y_pred = obj.Scratch_;               % single read-out, no copy
        end
        
        function Probs = predictProb(obj, X)
            % PREDICTPROB Returns class probabilities for each sample.
            % Output: N x K matrix where K is number of classes.
            
            if isa(obj.Criterion, 'OceanStats.Supervised.Tree.Criterion.RegressionCriterion')
                error('DecisionTree:PredictProbForRegression', 'predictProb is not supported for Regression trees.');
            end
            
            if isempty(obj.RootNode)
                 error('DecisionTree:NotFitted', 'Tree must be fitted before prediction.');
            end
            
            N = size(X, 1);
            % Use classes from Criterion or RootNode
            % Criterion.Classes is populated during fit.
            classes = obj.Criterion.Classes;
            K = length(classes);
            
            Probs = zeros(N, K);
            
            for i = 1:N
                leaf = obj.traverseToLeaf(obj.RootNode, X(i, :));
                
                % Leaf.ClassProbs is Kx1 or 1xK. Assign to row.
                % Ensure orientation.
                p = leaf.ClassProbs;
                Probs(i, :) = p(:)';
            end
        end
        
        function probs = predictProba(obj, X)
            % PREDICTPROBA Returns class probability vectors (classification only).
            [N, ~] = size(X);
            node0 = obj.traverseToLeaf(obj.RootNode, X(1, :));
            if ~isfield(node0, 'ClassProbs')
                error('DecisionTree:NotClassification', ...
                    'predictProba is only available for classification trees.');
            end
            K = length(node0.ClassProbs);
            probs = zeros(N, K);
            probs(1, :) = node0.ClassProbs;
            for i = 2:N
                node = obj.traverseToLeaf(obj.RootNode, X(i, :));
                probs(i, :) = node.ClassProbs;
            end
        end
        
        function tree = prune(obj, alpha)
            % PRUNE Returns a new DecisionTree pruned with cost-complexity alpha.
            newObj = obj.clone();
            newObj.RootNode = obj.pruneNode(obj.RootNode, alpha);
            tree = newObj;
        end
        
        function alphas = weakestLinkSequence(obj)
            % WEAKESTLINKSEQUENCE Finds candidate alphas using Breiman's algorithm.
            tempTree = obj.clone();
            tempTree.RootNode = obj.RootNode;
            alphas = 0;
            while ~tempTree.RootNode.IsLeaf
                tempTree.RootNode = tempTree.calculateSubtreeStats(tempTree.RootNode);
                minG = tempTree.solveWeakestLink(tempTree.RootNode);
                if isinf(minG) || isnan(minG)
                    break;
                end
                if minG > alphas(end)
                    alphas(end+1) = minG; %#ok<AGROW>
                end
                tempTree = tempTree.prune(minG + 1e-10);
            end
            alphas = unique([0, alphas]);
        end
        
        function bestTree = fitCV(obj, X, Y, validator)
            % FITCV Automatically tunes alpha using Cross-Validation.
            
            % 1. Train full tree to get pruning sequence
            obj.fit(X, Y);
            alphas = obj.weakestLinkSequence();
            nAlphas = length(alphas);
            cvErrors = zeros(nAlphas, 1);
            
            % 2. CV Loop
            [N, ~] = size(X);
            numIterations = validator.NumFolds; 
            
            for k = 1:numIterations
                [trainIdx, testIdx] = validator.getIndices(N, k);
                if ~any(testIdx), continue; end

                X_train = X(trainIdx, :); Y_train = Y(trainIdx);
                X_test  = X(testIdx, :);  Y_test  = Y(testIdx);
                
                tempTree = obj.clone();
                tempTree.fit(X_train, Y_train);
                
                for i = 1:nAlphas
                    prunedT = tempTree.prune(alphas(i));
                    Y_pred = prunedT.predict(X_test);
                    foldError = obj.Criterion.cvError(Y_test, Y_pred);
                    cvErrors(i) = cvErrors(i) + foldError;
                end
            end
            
            % 3. Pick best alpha
            [~, bestIdx] = min(cvErrors);
            bestAlpha = alphas(bestIdx);
            fprintf('Best Alpha found: %.4f\n', bestAlpha);
            
            % 4. Return final pruned tree on all data
            obj.fit(X, Y);
            bestTree = obj.prune(bestAlpha);
        end
        function n = countNodes(obj, node)
            % COUNTNODES Recursive count of all nodes.
            if nargin < 2
                node = obj.RootNode;
            end
            if isempty(node), n = 0; return; end
            
            n = 1; % Self
            if ~node.IsLeaf
                for i = 1:length(node.Children)
                    n = n + obj.countNodes(node.Children{i});
                end
            end
        end
        
        function n = countTerminalNodes(obj, node)
            % COUNTTERMINALNODES Recursive count of leaf nodes (|T|).
            if nargin < 2
                node = obj.RootNode;
            end
            if isempty(node), n = 0; return; end
            
            if node.IsLeaf
                n = 1;
            else
                n = 0;
                for i = 1:length(node.Children)
                    n = n + obj.countTerminalNodes(node.Children{i});
                end
            end
        end
        
        function d = depth(obj, node)
            % DEPTH Logic to find max depth of tree.
            if nargin < 2
                node = obj.RootNode;
            end
            if isempty(node), d = 0; return; end
            
            if node.IsLeaf
                d = 0;
            else
                childDepths = zeros(1, length(node.Children));
                for i = 1:length(node.Children)
                    childDepths(i) = obj.depth(node.Children{i});
                end
                d = 1 + max(childDepths);
            end
        end

        function [leafIds, nLeaves] = getLeafIds(obj, X)
            % GETLEAFIDS  Route training samples to leaves and assign integer IDs.
            %
            %   [leafIds, nLeaves] = tree.getLeafIds(X)
            %
            %   Used by GradientBoostingMachine after fitting the surrogate tree
            %   to identify which samples belong to each leaf region.
            %
            %   Output:
            %     leafIds  — N×1 integer array (1..nLeaves, one per sample)
            %     nLeaves  — total number of leaf nodes

            % Number all leaves depth-first
            cnt = 0;
            [obj.RootNode, cnt] = obj.numberLeaves_(obj.RootNode, cnt);
            nLeaves = cnt;

            N       = size(X, 1);
            leafIds = zeros(N, 1);
            leafIds = obj.getLeafIdsBatch_(obj.RootNode, X, (1:N)', leafIds);
        end

        function refitLeafValues(obj, gamma)
            % REFITLEAFVALUES  Override leaf Value fields with optimized constants.
            %
            %   tree.refitLeafValues(gamma)
            %
            %   gamma(k) = new value for leaf with LeafId_ = k.
            %   Called by GradientBoostingMachine for true-loss leaf optimization.
            obj.RootNode = obj.setLeafValues_(obj.RootNode, gamma);
        end
        
    end
    
    % =====================================================================
    %  PRIVATE — Tree Growing, Traversal, Pruning
    % =====================================================================
    methods (Access = private)
        function node = growTree(obj, X, Y, depth)
            % GROWTREE Recursive tree builder.
            nSamples = length(Y);
            
            % 0. Compute node statistics
            stats = obj.Criterion.computeNodeStats(Y);
            
            % 1. Stopping conditions
            if depth >= obj.MaxDepth || nSamples < (2 * obj.MinLeafSize) || obj.Criterion.isPure(Y)
                node = obj.makeLeaf(stats, nSamples);
                return;
            end
            
            % 2. Find best split via STRATEGY
            % DELEGATION: The SplitStrategy (possibly decorated) handles logic.
            params = struct();
            params.MinLeafSize = obj.MinLeafSize;
            params.Criterion = obj.Criterion;
            params.CandidateGenerator = obj.CandidateGenerator;
            
            result = obj.SplitStrategy.findSplit(X, Y, params);
            
            % 3. No valid split
            if result.IsTerminal
                node = obj.makeLeaf(stats, nSamples);
                return;
            end
            
            % 4. Routing and Recursion
            % DELEGATION: Use the SplitResult's routing logic to partition data.
            
            [N, ~] = size(X);
            childAssignments = zeros(N, 1);
            
            % Vectorized assignment if possible or row-by-row
            for i = 1:N
                childAssignments(i) = result.route(X(i, :));
            end
            
            % Handle NaN assignments (missing data with no surrogate)
            % Common policy: send to majority child.
            nanMask = isnan(childAssignments) | childAssignments == 0;
            if any(nanMask)
                 % Predict directions for nanMask based on primary split if possible
                 % or just majority. Let's do majority of NON-NAN routed rows.
                 validAssignments = childAssignments(~nanMask);
                 if isempty(validAssignments)
                     % All rows missing? Fallback to leaf.
                     node = obj.makeLeaf(stats, nSamples); return;
                 end
                 majChild = mode(validAssignments);
                 childAssignments(nanMask) = majChild;
            end
            
            % Recurse
            % Determine unique children to create
            % Note: For categorical splits, we might have K > 2.
            % For binary splits, childIdx will be 1 or 2.
            uniqueChildren = unique(childAssignments);
            uniqueChildren(uniqueChildren == 0) = []; % Safety
            K = length(uniqueChildren);
            
            childNodes = cell(1, max(uniqueChildren));
            for k = uniqueChildren'
                mask = (childAssignments == k);
                childNodes{k} = obj.growTree(X(mask, :), Y(mask), depth + 1);
            end
            
            % 5. Create Internal Node
            node = stats;
            node.Split = result;
            node.Children = childNodes; % Use dynamic Children array
            node.IsLeaf = false;
            node.NSamples = nSamples;
            
            % Attach Surrogates if present (for Traversal)
            node.Surrogates = result.Surrogates;
        end
        
        function node = makeLeaf(~, stats, nSamples)
            node = stats;
            node.IsLeaf = true;
            node.NSamples = nSamples;
        end
        
        function node = traverseToLeaf(~, node, x)
            % TRAVERSETOLEAF Walk tree using SplitResult logic.
            
            while ~node.IsLeaf
                % Use the route method of the SplitResult
                childIdx = node.Split.route(x);
                
                if isnan(childIdx) || childIdx == 0
                    % Missing value with no surrogate and no majority logic in route
                    % Fallback: Find child with most samples
                    childCounts = arrayfun(@(c) c{1}.NSamples, node.Children);
                    [~, childIdx] = max(childCounts);
                end
                
                node = node.Children{childIdx};
            end
        end
        
        function [node, isLeaf] = pruneNode(obj, node, alpha)
            if node.IsLeaf
                isLeaf = true;
                return;
            end
            
            for i = 1:length(node.Children)
                [node.Children{i}, ~] = obj.pruneNode(node.Children{i}, alpha);
            end
            
            [R_T, nLeaves] = obj.calcSubtreeRisk(node);
            R_t = node.Risk;
            
            if (R_t + alpha) <= (R_T + alpha * nLeaves)
                 % Prune this node
                 node = obj.makeLeaf(node, node.NSamples); % Reuse prediction stats
                 isLeaf = true;
            else
                isLeaf = false;
            end
        end
        
        function [risk, nLeaves] = calcSubtreeRisk(obj, node)
            if node.IsLeaf
                risk = node.Risk;
                nLeaves = 1;
            else
                risk = 0; nLeaves = 0;
                for i = 1:length(node.Children)
                    [r, n] = obj.calcSubtreeRisk(node.Children{i});
                    risk = risk + r;
                    nLeaves = nLeaves + n;
                end
            end
        end
        
        function minG = solveWeakestLink(obj, node)
            if node.IsLeaf
                minG = Inf;
                return;
            end
            R_T = node.SubtreeRisk;
            nLeaves = node.SubtreeLeaves;
            R_t = node.Risk;
            if nLeaves > 1
                g_t = (R_t - R_T) / (nLeaves - 1);
            else
                g_t = Inf;
            end
            
            g_child = Inf;
            for i = 1:length(node.Children)
                g_child = min(g_child, obj.solveWeakestLink(node.Children{i}));
            end
            minG = min(g_t, g_child);
        end
        
        function node = calculateSubtreeStats(obj, node)
            if node.IsLeaf
                node.SubtreeRisk = node.Risk;
                node.SubtreeLeaves = 1;
            else
                node.SubtreeRisk = 0;
                node.SubtreeLeaves = 0;
                for i = 1:length(node.Children)
                    node.Children{i} = obj.calculateSubtreeStats(node.Children{i});
                    node.SubtreeRisk = node.SubtreeRisk + node.Children{i}.SubtreeRisk;
                    node.SubtreeLeaves = node.SubtreeLeaves + node.Children{i}.SubtreeLeaves;
                end
            end
        end
        
        function newObj = clone(obj)
            newObj = OceanStats.Supervised.Tree.DecisionTree(...
                'MaxDepth', obj.MaxDepth, ...
                'MinLeafSize', obj.MinLeafSize, ...
                'Criterion', obj.Criterion, ...
                'Optimizer', obj.Optimizer, ...
                'UseSurrogates', obj.UseSurrogates);
        end
        
        function criterion = parseCriterion(obj, input)
             if isa(input, 'OceanStats.Supervised.Tree.Criterion.INodeCriterion')
                 criterion = input; return;
             end
             switch lower(input)
                 case {'regression', 'rss', 'mse'}
                     criterion = OceanStats.Supervised.Tree.Criterion.RegressionCriterion();
                 case 'gini'
                     criterion = OceanStats.Supervised.Tree.Criterion.ClassificationCriterion(...
                         OceanStats.Supervised.Tree.Impurity.GiniIndex(), 'LossMatrix', obj.LossMatrix);
                 case {'entropy', 'crossentropy', 'c50'}
                     criterion = OceanStats.Supervised.Tree.Criterion.ClassificationCriterion(...
                         OceanStats.Supervised.Tree.Impurity.CrossEntropy(), 'LossMatrix', obj.LossMatrix);
                 case {'misclassification'}
                     criterion = OceanStats.Supervised.Tree.Criterion.ClassificationCriterion(...
                         OceanStats.Supervised.Tree.Impurity.MisclassificationError(), 'LossMatrix', obj.LossMatrix);
                 otherwise
                     error('DecisionTree:BadCriterion', 'Unknown criterion "%s"', input);
             end
        end

        function [node, cnt] = numberLeaves_(obj, node, cnt)
            % NUMBERLEAVES_  Depth-first leaf numbering for getLeafIds.
            % Assigns node.LeafId_ = sequential integer.
            if node.IsLeaf
                cnt          = cnt + 1;
                node.LeafId_ = cnt;
            else
                for k = 1:length(node.Children)
                    [node.Children{k}, cnt] = obj.numberLeaves_(node.Children{k}, cnt);
                end
            end
        end


        function predictBatch_(obj, node, X, idx)
            % PREDICTBATCH_  Vectorized partition traversal — zero-copy.
            %
            %   Writes results to obj.Scratch_ (shared handle property).
            %   No Y_pred argument — no array is passed through the stack.
            %   COW is never triggered because we never modify a passed arg.

            if node.IsLeaf
                obj.Scratch_(idx) = node.Value;  % direct write to shared buffer
                return;
            end
            % Which feature this node to split the data
            j = node.Split.FeatureIndex;
            t = node.Split.Threshold;

            % NaN <= t is false in MATLAB, so NaN features route right.
            leftMask = X(idx, j) <= t; % leftMask is a logical array, true if the sample is less than or equal to the threshold

            if any(leftMask)
                obj.predictBatch_(node.Children{1}, X, idx(leftMask));  %Recursive on the child but only in the subset of indices that satisfy the condition
            end
            if any(~leftMask)
                obj.predictBatch_(node.Children{2}, X, idx(~leftMask));
            end
        end
        
        %Grouped samples into "buckets" 
        function leafIds = getLeafIdsBatch_(obj, node, X, idx, leafIds)
            % GETLEAFIDSBATCH_  Vectorized partition traversal for getLeafIds.
            %   Same logic as predictBatch_ but assigns leaf IDs instead of values.

            if node.IsLeaf
                leafIds(idx) = node.LeafId_;
                return;
            end

            j = node.Split.FeatureIndex;
            t = node.Split.Threshold;
            leftMask = X(idx, j) <= t;

            if any(leftMask)
                leafIds = obj.getLeafIdsBatch_(node.Children{1}, X, idx(leftMask),  leafIds);
            end
            if any(~leftMask)
                leafIds = obj.getLeafIdsBatch_(node.Children{2}, X, idx(~leftMask), leafIds);
            end
        end

        function node = setLeafValues_(obj, node, gamma)
            % SETLEAFVALUES_  Recursive leaf Value update for refitLeafValues.
            if node.IsLeaf
                node.Value = gamma(node.LeafId_);
            else
                for k = 1:length(node.Children)
                    node.Children{k} = obj.setLeafValues_(node.Children{k}, gamma);
                end
            end
        end
    end
end