classdef OOBTracker < handle
    % OOBTRACKER Out-of-Bag vote/prediction accumulator for Random Forests.
    %
    % OVERVIEW:
    %   Each tree is trained on a bootstrap sample.  The ~36.8% of training
    %   samples not selected (OOB samples) form an unbiased test set for
    %   that tree.  OOBTracker accumulates votes (classification) or sums
    %   (regression) over all trees and computes the final OOB error.
    %
    % USAGE:
    %   tracker = OceanStats.Ensemble.OOBTracker(N, K);   % K=1 → regression
    %   tracker.accumulate(tree, oobIdx, X, task);
    %   err = tracker.oobError(Y, task);
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties (Access = private)
        VoteAcc_   double   % N x K  (classification) | N x 1 (regression sum)
        Count_     double   % N x 1  — how many trees voted per sample
        N_         double
        K_         double
    end

    methods
        function obj = OOBTracker(N, K)
            % OOBTRACKER Constructor.
            %
            % Args:
            %   N (double): Total number of training samples.
            %   K (double): Number of classes (1 for regression).
            arguments
                N (1,1) double
                K (1,1) double = 1
            end
            obj.N_       = N;
            obj.K_       = K;
            obj.VoteAcc_ = zeros(N, K);
            obj.Count_   = zeros(N, 1);
        end

        function accumulate(obj, tree, oobIdx, X, task)
            % ACCUMULATE Add one tree's OOB predictions to the accumulator.
            % After training ONE tree. We have its OOB, and we want to use this as one vote for those OOB samples.
            %
            % Args:
            %   tree   (DecisionTree): Fitted tree object.
            %   oobIdx (vector):       Row indices of OOB samples.
            %   X      (N x p double): Full training feature matrix.
            %   task   (string):       'classification' or 'regression'.

            if isempty(oobIdx)
                return;
            end

            X_oob = X(oobIdx, :);

            if strcmpi(task, 'classification')
                % Soft-vote: accumulate class probabilities
                P_oob = tree.predictProb(X_oob);    % numel(oob) x K
                obj.VoteAcc_(oobIdx, :) = obj.VoteAcc_(oobIdx, :) + P_oob;
            else
                % Regression: accumulate raw predictions
                y_oob = tree.predict(X_oob);         % numel(oob) x 1
                obj.VoteAcc_(oobIdx, 1) = obj.VoteAcc_(oobIdx, 1) + y_oob;
            end

            obj.Count_(oobIdx) = obj.Count_(oobIdx) + 1;
        end

        function err = oobError(obj, Y, task, classes)
            % OOBEDERROR Compute OOB error across all samples that were OOB
            % for at least one tree. Evaluation step
            % Now we bring the true label Y to evaluate the Forest itself
            %
            % Args:
            %   Y       (N x 1): True training labels.
            %   task    (string): 'classification' or 'regression'.
            %   classes (vector, optional): Class labels for classification.
            %
            % Returns:
            %   err (double): Misclassification rate or MSE.

            voted = obj.Count_ > 0;   % samples covered by at least 1 tree
            if ~any(voted)
                err = NaN;
                return;
            end

            preds = obj.oobPredictions(task, classes);

            Y_v   = Y(voted);
            P_v   = preds(voted);

            if strcmpi(task, 'classification')
                err = mean(Y_v ~= P_v);
            else
                err = mean((Y_v - P_v).^2);
            end
        end

        function preds = oobPredictions(obj, task, classes)
            % OOBPREDICTIONS Final aggregated OOB prediction per sample.
            % Only trees that are never train on this OOB set are allowed to predict 
            % Returns NaN for samples that were never OOB (inBag every time).
            %After all B trees have finished, and we have collected votes for each sample (Many of them will have OOB votes overlap, we aggregate those)
            % 
            preds = nan(obj.N_, 1);
            voted = obj.Count_ > 0;

            if strcmpi(task, 'classification')
                % Average soft votes → argmax class
                avg_proba = obj.VoteAcc_(voted, :) ./ obj.Count_(voted);
                [~, idx]  = max(avg_proba, [], 2);
                if nargin >= 3 && ~isempty(classes)
                    preds(voted) = classes(idx);
                else
                    preds(voted) = idx;
                end
            else
                preds(voted) = obj.VoteAcc_(voted, 1) ./ obj.Count_(voted);
            end
        end

        function n = numVoted(obj)
            % NUMVOTED How many samples were OOB for at least one tree.
            n = sum(obj.Count_ > 0);
        end
    end
end
