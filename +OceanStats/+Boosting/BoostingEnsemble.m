classdef BoostingEnsemble < OceanStats.Base.IModelStrategy
    % BOOSTINGENSEMBLE  Context / Facade for sequential boosting algorithms.
    %
    %   OVERVIEW:
    %     Implements IModelStrategy so any boosting ensemble is plug-and-play
    %     with CrossValidator, ComplexitySweeper, ModelFactory, and all audit
    %     infrastructure — the rest of the library sees a single black-box model.
    %
    %     The *algorithm logic* lives entirely in the injected IBoostingStrategy.
    %     BoostingEnsemble only orchestrates the loop and stores state.
    %
    %   STRATEGY PATTERN ROLES:
    %     Context   ← BoostingEnsemble (this class)
    %     Strategy  ← IBoostingStrategy (abstract interface)
    %     Concrete  ← AdaBoostM1, GradientBoosting, LogitBoost, ...
    %
    %   USAGE:
    %     strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 50);
    %     wl       = OceanStats.Selection.Strategies.DecisionTree('MaxDepth', 1);
    %     ens      = OceanStats.Boosting.BoostingEnsemble(strategy, wl);
    %
    %     fit  = ens.train(X, Y);           % standard IModelStrategy API
    %     Yhat = ens.predict(fit, Xnew);
    %     err  = ens.evaluate(Y, Yhat);
    
    % ------------------------------------------------------------------ %
    %  Public properties (required by IModelStrategy)
    % ------------------------------------------------------------------ %
    properties
        Name        string  = "Boosting Ensemble"
        SupportsAIC logical = false
    end
    
    % ------------------------------------------------------------------ %
    %  Configuration (set at construction, read-only after)
    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Strategy    % IBoostingStrategy — the injected algorithm
        BaseModel   % IModelStrategy    — the weak learner prototype
    end
    
    % ------------------------------------------------------------------ %
    %  Private training state (cached for diagnostics / plotting)
    % ------------------------------------------------------------------ %
    properties (Access = private)
        TrainingErrors_   double   % 1 x T weighted training errors per round
    end
    
    % ================================================================== %
    methods
        function obj = BoostingEnsemble(strategy, baseModel)
            % BOOSTINGENSEMBLE  Constructor.
            %
            %   ens = BoostingEnsemble(strategy, baseModel)
            %
            %   Inputs:
            %     strategy  — IBoostingStrategy subclass (e.g. AdaBoostM1)
            %     baseModel — IModelStrategy weak learner (e.g. DecisionTree)
            arguments
                strategy  OceanStats.Boosting.Base.IBoostingStrategy
                baseModel OceanStats.Base.IModelStrategy
            end
            obj.Strategy  = strategy;
            obj.BaseModel = baseModel;
            obj.Name      = strategy.Name + " (" + baseModel.Name + ")";
        end
        
        % -------------------------------------------------------------- %
        function model = train(obj, X, Y, W)
            % TRAIN  Run the full boosting loop.
            %
            %   model = train(X, Y)
            %   model = train(X, Y, W)    % W = initial sample weights
            %
            %   model fields:
            %     .Alphas       — 1 x T_actual model weights
            %     .WeakFits     — 1 x T_actual trained weak model objects
            %     .Classes      — 1 x K sorted unique class labels
            %     .TrainingErrors — 1 x T_actual per-round weighted errors
            %     .NumRounds    — T_actual (may be < NumIterations if early stop)
            arguments
                obj
                X   double
                Y   (:,1)
                W   (:,1) double = []
            end
            
            N = size(X, 1);
            T = obj.Strategy.NumIterations;
            
            % Uniform initialization if no weights provided
            if isempty(W)
                W = ones(N, 1) / N;
            else
                W = W / sum(W);    % Ensure normalized
            end
            
            classes = unique(Y);   % 1 x K sorted unique labels
            
            % Pre-allocate
            alphas       = zeros(1, T);
            weak_fits    = cell(1, T);
            train_errors = zeros(1, T);
            
            t_actual = 0;
            for t = 1:T
                [alpha, W_new, fit] = obj.Strategy.boostingStep( ...
                    obj.BaseModel, X, Y, W, t);
                
                % Early stop signal from strategy
                if isnan(alpha)
                    break
                end
                
                t_actual         = t_actual + 1;
                alphas(t_actual)     = alpha;
                weak_fits{t_actual}  = fit;
                
                % Record training error (for diagnostics)
                Y_hat = obj.BaseModel.predict(fit, X);
                train_errors(t_actual) = sum(W .* (Y_hat ~= Y)) / sum(W);
                
                W = W_new;
            end
            
            % Trim to actual rounds used
            model.Alphas        = alphas(1:t_actual);
            model.WeakFits      = weak_fits(1:t_actual);
            model.Classes       = classes;
            model.TrainingErrors = train_errors(1:t_actual);
            model.NumRounds     = t_actual;
            
            % Cache for external diagnostics
            obj.TrainingErrors_ = model.TrainingErrors;
        end
        
        % -------------------------------------------------------------- %
        function Y_pred = predict(obj, model, X)
            % PREDICT  Aggregate weak learner predictions into final labels.
            %
            %   Y_pred = predict(model, X)
            %
            %   Output: N x 1 class label vector
            
            T = model.NumRounds;
            N = size(X, 1);
            
            % Collect N x T predictions matrix
            predictions = zeros(N, T);
            for t = 1:T
                predictions(:, t) = obj.BaseModel.predict(model.WeakFits{t}, X);
            end
            
            Y_pred = obj.Strategy.aggregate(model.Alphas, predictions, model.Classes);
        end
        
        % -------------------------------------------------------------- %
        function err = evaluate(~, Y_true, Y_pred)
            % EVALUATE  Misclassification rate.
            %
            %   err = evaluate(Y_true, Y_pred)
            %   Returns a scalar in [0, 1].
            err = mean(Y_true ~= Y_pred);
        end
        
        % -------------------------------------------------------------- %
        function k = getComplexity(~, model)
            % GETCOMPLEXITY  Number of boosting rounds used.
            %   Used as the complexity proxy by ComplexitySweeper / audit tools.
            k = model.NumRounds;
        end
        
        % -------------------------------------------------------------- %
        function errors = getTrainingErrors(obj)
            % GETTRAININGERRORS  Return per-round training errors from last fit.
            %   Useful for convergence plots via OceanViz.
            errors = obj.TrainingErrors_;
        end
    end
end
