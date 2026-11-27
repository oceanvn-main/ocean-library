classdef AdaBoostM2 < OceanStats.Boosting.Base.IBoostingStrategy
    % ADABOOSTM2 (SAMME) Multi-class extension of AdaBoost.
    %
    %   OVERVIEW:
    %     Implements the SAMME algorithm (Zhu, Zou, Rosset, Hastie 2005) which
    %     extends AdaBoost.M1 to multi-class scenarios naturally.
    %
    %     AdaBoost.M1 requires the weak learner's weighted accuracy to be > 50%.
    %     For K > 2 classes, this is often impossible (random guessing is 1/K).
    %     SAMME fixes this by adding a log(K-1) term to the alpha update.
    %     Now the weak learner only needs to be better than random guessing.
    %
    %   USAGE:
    %     strategy = OceanStats.Boosting.Algorithms.AdaBoostM2('NumIterations', 100);
    %     tree = OceanStats.Selection.Strategies.DecisionTree('MaxDepth', 3);
    %     ens = OceanStats.Boosting.BoostingEnsemble(strategy, tree);
    %     ens.train(X, Y);
    %
    %   See also: OceanStats.Boosting.Algorithms.AdaBoostM1
    
    properties
        Name          string = "AdaBoost.M2 (SAMME)"
        NumIterations double = 100
        LearningRate  double = 1.0  % Shrinkage nu
    end
    
    properties (Access = private)
        NumClasses_ double
    end
    
    methods
        function obj = AdaBoostM2(options)
            arguments
                options.NumIterations double = 100
                options.LearningRate  double = 1.0
            end
            obj.NumIterations = options.NumIterations;
            obj.LearningRate  = options.LearningRate;
        end
        
        function [alpha, W_new, fit] = boostingStep(obj, baseModel, X, Y, W, ~)
            % 1. Infer K if not set
            if isempty(obj.NumClasses_)
                obj.NumClasses_ = numel(unique(Y));
            end
            K = obj.NumClasses_;
            
            % 2. Train weak learner on weighted data
            fit = baseModel.train(X, Y, W);
            
            % 3. Get predictions and compute weighted error
            Y_hat = baseModel.predict(fit, X);
            
            misclassified = (Y_hat ~= Y);
            err = sum(W .* misclassified) / sum(W);
            
            % 4. Early stopping conditions
            % Must be better than random guessing: 1 - 1/K
            if err >= (1 - 1/K) || err == 0
                alpha = NaN;
                W_new = W;
                return;
            end
            
            % 5. Compute alpha (SAMME modification: + log(K-1))
            alpha = obj.LearningRate * (log((1 - err) / err) + log(K - 1));
            
            % 6. Update weights
            W_new = W .* exp(alpha * misclassified);
            
            % 7. Normalize
            W_new = W_new / sum(W_new);
        end
        
        function Y_pred = aggregate(~, alphas, predictions, classes)
            % AGGREGATE SAMME hard voting.
            %   Y_pred = argmax_k sum_{t} alpha_t * I(pred_t == k)
            N = size(predictions, 1);
            K = numel(classes);
            
            % Accumulate votes for each class
            votes = zeros(N, K);
            for k_idx = 1:K
                c = classes(k_idx);
                % For each sample, sum the alphas where the tree predicted class c
                is_c = (predictions == c);
                votes(:, k_idx) = is_c * alphas';
            end
            
            % Return the class with the highest vote sum
            [~, best_idx] = max(votes, [], 2);
            Y_pred = classes(best_idx);
        end
    end
end
