classdef DecisionTree < OceanStats.Base.IModelStrategy
    % DECISIONTREE Adapter for OceanStats.Supervised.Tree.DecisionTree.
    %
    % OVERVIEW:
    %   Connects the audit/selection pipeline to our custom DecisionTree implementation.
    %   - Supports Classification (Gini) and Regression (MSE).
    %
    % PROPERTIES:
    %   MinLeafSize (default 5)
    %   MaxDepth    (default 5)
    %   Criterion   (default 'gini' for classification, 'regression' for regression)
    
    properties
        Name string = "Decision Tree (Custom)"
        SupportsAIC logical = false 
        
        % Hyperparameters
        MinLeafSize double = 5
        MaxDepth double = 5
        Criterion string = "gini" % 'gini', 'entropy', or 'regression'
        
        % Pruning
        PruneAlpha double = 0
        
        % Cost-Sensitive
        LossMatrix = []
    end
    
    methods
        function obj = DecisionTree(min_leaf, max_depth, criterion)
            if nargin > 0, obj.MinLeafSize = min_leaf; end
            if nargin > 1, obj.MaxDepth = max_depth; end
            if nargin > 2, obj.Criterion = criterion; end
        end
        
        function model = train(obj, X, Y)
            % TRAIN fits the custom DecisionTree.
            % Wraps it in a struct or returns the object directly if compatible.
            
            % 1. Create Tree Instance
            tree = OceanStats.Supervised.Tree.DecisionTree(...
                'MinLeafSize', obj.MinLeafSize, ...
                'MaxDepth', obj.MaxDepth, ...
                'Criterion', obj.Criterion, ...
                'LossMatrix', obj.LossMatrix);
            
            % 2. Fit
            tree.fit(X, Y);
            
            % 3. Prune (if requested)
            if obj.PruneAlpha > 0
                tree = tree.prune(obj.PruneAlpha);
            end
            
            % 4. Return as model artifact
            model = tree;
        end
        
        function preds = predict(obj, model, X)
            % PREDICT Returns predictions.
            %
            % If Classification (Gini/Entropy): Returns probability of class 2 (if binary).
            % If Regression: Returns values.
            
            tree = model; % Model is the DecisionTree object
            
            if strcmpi(obj.Criterion, 'regression')
                % Regression -> Values
                preds = tree.predict(X);
            else
                % Classification -> Probabilities for positive class (Class 2)
                % We assume binary classification for FeatureSelectionAudit usually.
                try
                    probs = tree.predictProb(X);
                    if size(probs, 2) >= 2
                        preds = probs(:, 2);
                    else
                        preds = probs(:, 1); % Fallback
                    end
                catch
                    % Fallback if predictProba fails (e.g. regression tree used on discrete data)
                    preds = tree.predict(X); 
                end
            end
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            % EVALUATE generic error metric
            if strcmpi(obj.Criterion, 'regression')
                % RSS
                err = sum((Y_true - Y_pred).^2);
            else
                % Misclassification Rate or Cost
                if isempty(obj.LossMatrix)
                    Y_hard = Y_pred >= 0.5;
                    if iscategorical(Y_true), Y_true = double(Y_true) - 1; end
                    err = mean(Y_true ~= Y_hard);
                else
                    % Use Criterion.cvError for consistent cost-sensitive eval
                    classes = unique(Y_true);
                    Y_hard = Y_pred >= 0.5;
                    L = obj.LossMatrix;
                    err = 0;
                    for i = 1:length(Y_true)
                         pIdx = double(Y_hard(i)) + 1; % 0->1, 1->2
                         tIdx = double(Y_true(i) == classes(2)) + 1;
                         err = err + L(pIdx, tIdx);
                    end
                    err = err / length(Y_true);
                end
            end
        end
        
        function k = getComplexity(~, model)
            % Complexity = Number of Terminal Nodes
            k = model.countTerminalNodes(model.RootNode);
        end
    end
end

function n = countLeaves(node)
    if strcmp(node.Type, 'Leaf')
        n = 1;
    else
        n = countLeaves(node.Left) + countLeaves(node.Right);
    end
end
