classdef (Abstract) IBoostingStrategy < handle
    % IBOOSTINGSTRATEGY  Strategy interface for sequential boosting algorithms.
    %
    %   OVERVIEW:
    %     Defines the contract for one full boosting loop. Concrete subclasses
    %     (AdaBoostM1, GradientBoosting, LogitBoost, ...) implement the step
    %     logic and the aggregation rule independently of the base learner.
    %
    %   USAGE (via BoostingEnsemble):
    %     strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 50);
    %     ensemble = OceanStats.Boosting.BoostingEnsemble(strategy, baseModel);
    %     fit = ensemble.train(X, Y);
    %     Yhat = ensemble.predict(fit, Xnew);
    %
    %   EXTENSION:
    %     1. Subclass IBoostingStrategy in +OceanStats/+Boosting/+Algorithms/
    %     2. Implement boostingStep() and aggregate()
    %     3. Inject into BoostingEnsemble — no other code changes needed.
    
    properties (Abstract)
        Name          string      % Human-readable name, e.g. "AdaBoost.M1"
        NumIterations double      % Maximum number of boosting rounds
    end
    
    methods (Abstract)
        % BOOSTINGSTEP  Execute one boosting round.
        %
        %   [alpha, W_new, fit] = boostingStep(obj, baseModel, X, Y, W, t)
        %
        %   Inputs:
        %     baseModel — any OceanStats.Base.IModelStrategy instance
        %     X         — N x p feature matrix
        %     Y         — N x 1 class labels (numeric, e.g. 1..K)
        %     W         — N x 1 sample weights (sum to 1)
        %     t         — current iteration index (1-based)
        %
        %   Outputs:
        %     alpha  — scalar weight for this weak learner (NaN → stop early)
        %     W_new  — N x 1 updated, normalized sample weights
        %     fit    — trained model object returned by baseModel.train()
        [alpha, W_new, fit] = boostingStep(obj, baseModel, X, Y, W, t)
        
        % AGGREGATE  Combine weak predictions into a final class label.
        %
        %   Y_pred = aggregate(obj, alphas, predictions, classes)
        %
        %   Inputs:
        %     alphas      — 1 x T vector of model weights
        %     predictions — N x T matrix of per-round class predictions
        %     classes     — 1 x K vector of unique class labels
        %
        %   Output:
        %     Y_pred — N x 1 final predicted class labels
        Y_pred = aggregate(obj, alphas, predictions, classes)
    end
end
