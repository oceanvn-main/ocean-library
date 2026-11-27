classdef ImportanceCalculator
    % IMPORTANCECALCULATOR Feature importance strategies for Random Forests.
    %
    % OVERVIEW:
    %   Provides two static methods implementing the two canonical
    %   importance measures from Breiman (2001) / ESL §15.3.2:
    %
    %   MDA — Mean Decrease Accuracy (permutation importance).
    %     For each tree b and each feature j: permute X_oob(:,j), re-predict,
    %     measure accuracy drop.  More reliable for correlated features. Sabotaging the Frozen Model. 
    %
    %   MDI — Mean Decrease Impurity (Gini / RSS importance).
    %     Sum over all nodes in all trees the impurity decrease weighted by
    %     the fraction of samples reaching each node, for feature j.
    %     Faster, but can over-rank high-cardinality features.
    %
    % USAGE:
    %   scores = OceanStats.Ensemble.ImportanceCalculator.mda( ...
    %       trees, oobIdx, X, Y, 'classification');
    %
    %   scores = OceanStats.Ensemble.ImportanceCalculator.mdi(trees, p);
    %
    % REFERENCE:
    %   Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
    %   ESL §15.3.2 (Hastie, Tibshirani, Friedman).
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    methods (Static)

        function scores = mda(trees, oobIdx, X, Y, task, classes)
            % MDA Mean Decrease Accuracy via OOB permutation.
            %
            % Algorithm (per tree b):
            %   1. Predict OOB samples → baseline accuracy A_b.
            %   2. For each feature j:
            %      a. Permute column j of X_oob.
            %      b. Predict with permuted data → accuracy A_bj.
            %      c. delta_bj = A_b - A_bj.
            %   3. scores(j) = mean over b of delta_bj  (positive = important).
            %
            % Args:
            %   trees   (cell 1xB):  Fitted DecisionTree objects.
            %   oobIdx  (cell 1xB):  OOB index vectors per tree.
            %   X       (N x p):     Full training features.
            %   Y       (N x 1):     True labels / responses.
            %   task    (string):    'classification' | 'regression'.
            %   classes (vector):    Class labels (classification only).
            %
            % Returns:
            %   scores (1 x p double): MDA importance for each feature.

            if nargin < 6, classes = []; end

            B = numel(trees);
            p = size(X, 2);
            delta = zeros(B, p);

            for b = 1:B
                idx_oob = oobIdx{b};
                if isempty(idx_oob), continue; end

                X_oob = X(idx_oob, :);
                Y_oob = Y(idx_oob);

                % Baseline accuracy / MSE
                base_err = ImportanceCalculator.treeError_( ...
                    trees{b}, X_oob, Y_oob, task, classes);

                % Permute each feature
                for j = 1:p
                    X_perm       = X_oob;
                    X_perm(:, j) = X_oob(randperm(numel(idx_oob)), j);

                    perm_err     = ImportanceCalculator.treeError_( ...
                        trees{b}, X_perm, Y_oob, task, classes);

                    % Positive = permuting this feature hurts performance
                    delta(b, j) = perm_err - base_err;
                end
            end

            scores = mean(delta, 1);   % 1 x p
        end

        function scores = mdi(trees, p)
            % MDI Mean Decrease Impurity (Gini / RSS reduction).
            %
            % Traverses the tree struct (node.Split.ImpurityDecrease,
            % node.NSamples) recursively, accumulates weighted impurity
            % reduction for each feature, then averages over trees.
            % Arcross all trees in the forest, how much total impuritation did this feature cause?
            %
            % Args:
            %   trees (cell 1xB):  Fitted DecisionTree objects.
            %   p     (double):    Total number of features.
            %
            % Returns:
            %   scores (1 x p double): MDI importance (normalised to sum=1).

            B = numel(trees);
            acc = zeros(1, p);

            for b = 1:B
                acc = acc + ImportanceCalculator.nodeImportance_( ...
                    trees{b}.RootNode, p, trees{b}.RootNode.NSamples);
            end

            scores = acc / B;

            % Normalise
            total = sum(scores);
            if total > 0
                scores = scores / total;
            end
        end
    end

    % =====================================================================
    %  Private helpers
    % =====================================================================
    methods (Static, Access = private)

        function err = treeError_(tree, X, Y, task, classes)
            % TREEERROR_ Compute error for one tree on a given (X, Y).
            if strcmpi(task, 'classification')
                P     = tree.predictProb(X);
                [~, idx] = max(P, [], 2);
                if ~isempty(classes)
                    Y_hat = classes(idx);
                else
                    Y_hat = idx;
                end
                err = mean(Y ~= Y_hat);
            else
                Y_hat = tree.predict(X);
                err   = mean((Y - Y_hat).^2);
            end
        end

        function imp = nodeImportance_(node, p, N_root)
            % NODEIMPORTANCE_ Recursively accumulate weighted impurity
            % reduction for each feature.  Requires that each internal node
            % has Split.ImpurityDecrease and Split.FeatureIndex fields.

            imp = zeros(1, p);

            if node.IsLeaf || ~isfield(node, 'Split') || isempty(node.Split)
                return;
            end

            % Weighted impurity reduction at this node
            j = node.Split.FeatureIndex;
            if isfield(node.Split, 'ImpurityDecrease')
                w          = node.NSamples / N_root; %Fraction of training samples reached this node
                imp(j)     = imp(j) + w * node.Split.ImpurityDecrease;
            end

            % Recurse into children
            for c = 1:numel(node.Children)
                if ~isempty(node.Children{c})
                    imp = imp + ImportanceCalculator.nodeImportance_( ...
                        node.Children{c}, p, N_root);
                end
            end
        end
    end
end
