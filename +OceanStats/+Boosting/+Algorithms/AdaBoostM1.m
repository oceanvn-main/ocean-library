classdef AdaBoostM1 < OceanStats.Boosting.Base.IBoostingStrategy
    % ADABOOSTM1  AdaBoost.M1 — multi-class discrete AdaBoost.
    %
    %   OVERVIEW:
    %     Implements the Freund & Schapire (1997) boosting algorithm for
    %     multi-class classification. At each round t:
    %       1. Train a weak classifier h_t on the re-weighted dataset.
    %       2. Compute weighted error: err_t = sum(W .* (h_t(X) ~= Y)).
    %       3. Compute model weight: alpha_t = log((1 - err_t) / err_t).
    %       4. Update sample weights: upweight misclassified examples.
    %       5. Normalize weights so they sum to 1.
    %     Final prediction: argmax over classes of sum_t(alpha_t * [h_t = k]).
    %
    %   EARLY STOPPING:
    %     If err_t >= 0.5, the weak learner is no worse than random chance.
    %     Boosting terminates early; subsequent rounds would be unstable.
    %     If err_t == 0, the data is perfectly classified in one round.
    %
    %   USAGE (via BoostingEnsemble):
    %     strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 50);
    %     wl = OceanStats.Selection.Strategies.DecisionTree('MaxDepth', 1);
    %     ens = OceanStats.Boosting.BoostingEnsemble(strategy, wl);
    %     fit = ens.train(X, Y);
    %     Yhat = ens.predict(fit, Xnew);
    %
    %   REFERENCE:
    %     Freund, Y. & Schapire, R.E. (1997). A decision-theoretic generalization
    %     of on-line learning and an application to boosting. Journal of Computer
    %     and System Sciences, 55(1), 119–139.
    
    properties
        Name          string = "AdaBoost.M1"
        NumIterations double = 50
        % EarlyStopThreshold: stop if weighted error >= this value (default 0.5)
        EarlyStopThreshold double = 0.5
    end
    
    methods
        function obj = AdaBoostM1(options)
            % ADABOOSTM1  Constructor with named options.
            %
            %   obj = AdaBoostM1('NumIterations', 50, 'EarlyStopThreshold', 0.5)
            arguments
                options.NumIterations       double {mustBePositive, mustBeInteger} = 50
                options.EarlyStopThreshold  double {mustBeInRange(options.EarlyStopThreshold, 0, 1)} = 0.5
            end
            obj.NumIterations      = options.NumIterations;
            obj.EarlyStopThreshold = options.EarlyStopThreshold;
        end
        
        function [alpha, W_new, fit] = boostingStep(obj, baseModel, X, Y, W, ~)
            % BOOSTINGSTEP  Execute one AdaBoost.M1 round.
            %
            %   Inputs:
            %     baseModel — IModelStrategy weak learner
            %     X         — N x p feature matrix
            %     Y         — N x 1 numeric class labels
            %     W         — N x 1 sample weights (must sum to 1)
            %     ~         — iteration index (unused by this algorithm)
            %
            %   Outputs:
            %     alpha  — scalar model weight (NaN triggers early stop)
            %     W_new  — N x 1 updated normalized sample weights
            %     fit    — trained model struct/object from baseModel.train()
            
            N = size(X, 1);
            
            % --- Step 1: Train weak learner on weighted data ---
            fit = baseModel.train(X, Y, W);
            
            % --- Step 2: Compute weighted error ---
            Y_hat   = baseModel.predict(fit, X);
            wrong   = (Y_hat ~= Y);                      % N x 1 logical
            err     = sum(W .* wrong) / sum(W);          % scalar in [0,1]
            
            % --- Early stop: degenerate classifier ---
            if err >= obj.EarlyStopThreshold
                alpha = NaN;
                W_new = W;
                return
            end
            
            % --- Step 3: Compute model weight alpha ---
            if err < eps
                % Perfect classifier — give it a large but finite weight
                alpha = log((1 - eps) / eps);
            else
                alpha = log((1 - err) / err);
            end
            
            % --- Step 4: Update sample weights (upweight misclassified) ---
            W_new = W .* exp(alpha .* wrong);            % N x 1
            
            % --- Step 5: Normalize ---
            W_new = W_new / sum(W_new);
            
            % Guard: clamp any numerical noise below zero
            W_new = max(W_new, 0);
            W_new = W_new / sum(W_new);
        end
        
        function Y_pred = aggregate(~, alphas, predictions, classes)
            % AGGREGATE  Weighted majority vote over all rounds.
            %
            %   For each sample, accumulate alpha-weighted votes per class,
            %   then pick the class with the highest total score.
            %
            %   Inputs:
            %     alphas      — 1 x T vector of model weights
            %     predictions — N x T matrix of class label predictions
            %     classes     — 1 x K sorted vector of unique class labels
            %
            %   Output:
            %     Y_pred — N x 1 final predicted class labels
            
            [N, ~] = size(predictions);
            K      = numel(classes);
            T      = numel(alphas);
            
            % Accumulate weighted votes: votes(n, k) = sum of alphas where
            % the prediction for sample n equals class k
            votes = zeros(N, K);
            for t = 1:T
                for k = 1:K
                    mask          = (predictions(:, t) == classes(k));
                    votes(:, k)   = votes(:, k) + alphas(t) * mask;
                end
            end
            
            % Pick the class with the highest accumulated vote per sample
            [~, idx] = max(votes, [], 2);
            Y_pred   = classes(idx)';          % N x 1, same type as classes
        end
    end
end
