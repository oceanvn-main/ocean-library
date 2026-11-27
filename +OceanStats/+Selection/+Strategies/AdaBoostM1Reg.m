classdef AdaBoostM1Reg < OceanStats.Base.IModelStrategy
    % ADABOOSTM1REG  Thin IModelStrategy wrapper for BoostingEnsemble + AdaBoostM1.
    %
    %   OVERVIEW:
    %     Follows the exact same adapter pattern as +Selection/+Strategies/DecisionTree.m.
    %     Pre-configures a BoostingEnsemble with AdaBoostM1 strategy and a
    %     Decision Stump (DecisionTree, MaxDepth=1) weak learner, then
    %     exposes it as a standard IModelStrategy for CrossValidator,
    %     ComplexitySweeper, ModelFactory, and all audit infrastructure.
    %
    %   USAGE:
    %     m = OceanStats.Selection.Strategies.AdaBoostM1Reg();
    %     m = OceanStats.Selection.Strategies.AdaBoostM1Reg( ...
    %             'NumIterations', 100, 'MaxDepth', 2);
    %
    %     % Use in cross-validation:
    %     cv  = OceanStats.Selection.CrossValidation.CrossValidator(m, KFold(5));
    %     err = cv.run(X, Y);
    
    properties
        Name        string  = "AdaBoost.M1"
        SupportsAIC logical = false
        
        % Boosting hyperparameters
        NumIterations      double = 50
        EarlyStopThreshold double = 0.5
        
        % Weak learner hyperparameters (Decision Stump defaults)
        MaxDepth    double = 1
        MinLeafSize double = 1
    end
    
    properties (Access = private)
        Ensemble_   % BoostingEnsemble — built lazily on first train() call
    end
    
    methods
        function obj = AdaBoostM1Reg(options)
            % ADABOOSTM1REG  Constructor with named options.
            %
            %   obj = AdaBoostM1Reg('NumIterations', 50, 'MaxDepth', 1)
            arguments
                options.NumIterations      double {mustBePositive, mustBeInteger} = 50
                options.EarlyStopThreshold double = 0.5
                options.MaxDepth           double {mustBePositive, mustBeInteger} = 1
                options.MinLeafSize        double {mustBePositive, mustBeInteger} = 1
            end
            obj.NumIterations      = options.NumIterations;
            obj.EarlyStopThreshold = options.EarlyStopThreshold;
            obj.MaxDepth           = options.MaxDepth;
            obj.MinLeafSize        = options.MinLeafSize;
            obj.Name               = sprintf("AdaBoost.M1 (T=%d, D=%d)", ...
                                        obj.NumIterations, obj.MaxDepth);
        end
        
        function model = train(obj, X, Y, W)
            % TRAIN  Delegate to BoostingEnsemble.train().
            arguments
                obj
                X double
                Y (:,1)
                W (:,1) double = []
            end
            ens   = obj.buildEnsemble_();
            model = ens.train(X, Y, W);
        end
        
        function preds = predict(obj, model, X)
            % PREDICT  Delegate to BoostingEnsemble.predict().
            ens   = obj.buildEnsemble_();
            preds = ens.predict(model, X);
        end
        
        function err = evaluate(~, Y_true, Y_pred)
            % EVALUATE  Misclassification rate.
            err = mean(Y_true ~= Y_pred);
        end
        
        function k = getComplexity(~, model)
            % GETCOMPLEXITY  Number of boosting rounds used.
            k = model.NumRounds;
        end
    end
    
    methods (Access = private)
        function ens = buildEnsemble_(obj)
            % BUILDENSEMBLE_  Lazily constructs the BoostingEnsemble.
            %   Re-uses cached instance if hyperparameters haven't changed.
            if isempty(obj.Ensemble_)
                strategy  = OceanStats.Boosting.Algorithms.AdaBoostM1( ...
                    'NumIterations',      obj.NumIterations, ...
                    'EarlyStopThreshold', obj.EarlyStopThreshold);
                
                % Decision Stump: MaxDepth=1 is the canonical weak learner
                stump = OceanStats.Selection.Strategies.DecisionTree( ...
                    obj.MinLeafSize, obj.MaxDepth, 'gini');
                
                obj.Ensemble_ = OceanStats.Boosting.BoostingEnsemble(strategy, stump);
            end
            ens = obj.Ensemble_;
        end
    end
end
