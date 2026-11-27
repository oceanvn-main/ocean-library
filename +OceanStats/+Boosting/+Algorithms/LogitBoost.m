classdef LogitBoost < OceanStats.Boosting.Base.IBoostingStrategy
    % LOGITBOOST Additive logistic regression via Newton steps.
    %
    %   OVERVIEW:
    %     Implements the LogitBoost algorithm (Friedman, Hastie, Tibshirani 2000).
    %     It casts AdaBoost as fitting an additive logistic regression model 
    %     using Newton-like weight updates.
    %
    %     IMPORTANT: Because LogitBoost fits a working continuous response (z),
    %     the base learner MUST be a REGRESSION model (e.g., a Regression Tree),
    %     even though the overall task is Binary Classification.
    %
    %   USAGE:
    %     strategy = OceanStats.Boosting.Algorithms.LogitBoost('NumIterations', 50);
    %     % Note: We use a regression tree as the weak learner
    %     tree = OceanStats.Selection.Strategies.DecisionTree('Criterion', 'mse', 'MaxDepth', 3);
    %     ens = OceanStats.Boosting.BoostingEnsemble(strategy, tree);
    %     ens.train(X, Y); % Y must have exactly 2 classes
    %
    %   See also: OceanStats.Boosting.Algorithms.GradientBoostingMachine
    
    properties
        Name          string = "LogitBoost"
        NumIterations double = 100
        LearningRate  double = 1.0  % Shrinkage nu
        ZMax          double = 4.0  % Bound on working response for numerical stability
    end
    
    properties (Access = private)
        F_          % N x 1 internal state F(x)
        Y_star_     % N x 1 internal mapped targets {0, 1}
    end
    
    methods
        function obj = LogitBoost(options)
            arguments
                options.NumIterations double = 100
                options.LearningRate  double = 1.0
                options.ZMax          double = 4.0
            end
            obj.NumIterations = options.NumIterations;
            obj.LearningRate  = options.LearningRate;
            obj.ZMax          = options.ZMax;
        end
        
        function [alpha, W_new, fit] = boostingStep(obj, baseModel, X, Y, W, t)
            N = size(X, 1);
            
            % 1. Initialization on the first step
            if t == 1
                classes = unique(Y);
                if numel(classes) ~= 2
                    error('LogitBoost:BinaryOnly', 'LogitBoost requires exactly 2 classes.');
                end
                % Map to 0 (first class) and 1 (second class)
                obj.Y_star_ = double(Y == classes(2));
                obj.F_ = zeros(N, 1);
            end
            
            % 2. Compute probabilities
            % p(x) = 1 / (1 + exp(-2 * F(x)))
            % Numerically stable sigmoid:
            P = 1 ./ (1 + exp(-2 * obj.F_));
            
            % Bound probabilities away from 0 and 1
            eps_p = 1e-5;
            P = max(eps_p, min(1 - eps_p, P));
            
            % 3. Compute working response (z) and weights (w)
            % z = (Y* - P) / (P * (1 - P))
            Z = (obj.Y_star_ - P) ./ (P .* (1 - P));
            
            % Bound Z for numerical stability (crucial in LogitBoost)
            Z = max(-obj.ZMax, min(obj.ZMax, Z));
            
            % Newton weights
            W_new = P .* (1 - P);
            % Normalize weights so they sum to 1 (expected by base learners)
            W_new = W_new / sum(W_new);
            
            % 4. Fit the regression weak learner to Z
            % NOTE: The ensemble passes Y into boostingStep, but we must
            % trick the base learner into fitting Z instead.
            fit = baseModel.train(X, Z, W_new);
            
            % 5. Update F(x)
            f_x = baseModel.predict(fit, X);
            obj.F_ = obj.F_ + 0.5 * obj.LearningRate * f_x;
            
            % Alpha is structurally 0.5 (or 0.5 * nu) in LogitBoost, 
            % but since we handle the 0.5 internally and at aggregation,
            % we just return nu here to track shrinkage.
            alpha = obj.LearningRate;
        end
        
        function Y_pred = aggregate(~, alphas, predictions, classes)
            % AGGREGATE Combine continuous tree predictions into a class.
            %   F(x) = sum( 0.5 * nu * f_t(x) )
            %   Y_pred = class 2 if F(x) > 0, else class 1.
            
            % predictions is N x T
            % alphas is 1 x T (contains the LearningRate)
            
            % Sum all predictions, scaled by 0.5 * alpha
            F = sum(predictions .* (0.5 * alphas), 2);
            
            % Threshold at 0
            is_class_2 = (F > 0);
            
            % Map back to original class labels
            Y_pred = repmat(classes(1), size(F, 1), 1);
            Y_pred(is_class_2) = classes(2);
        end
    end
end
