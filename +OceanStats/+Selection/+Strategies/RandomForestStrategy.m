classdef RandomForestStrategy < OceanStats.Base.IModelStrategy
    % RANDOMFORESTSTRATEGY IModelStrategy adapter for RandomForest.
    %
    % OVERVIEW:
    %   Thin wrapper that exposes OceanStats.Ensemble.RandomForest through
    %   the IModelStrategy contract, making it usable with:
    %     - OceanStats.Selection.CrossValidation.CrossValidator
    %     - OceanStats.Selection.ComplexitySweeper
    %     - OceanStats.Selection.Auditors.BiasVarianceAuditor
    %     - OceanStats.Selection.Auditors.FeatureImportanceAuditor
    %
    % USAGE:
    %   strat = OceanStats.Selection.Strategies.RandomForestStrategy( ...
    %       'NumTrees', 100, 'Criterion', 'gini');
    %
    %   model = strat.train(X_train, Y_train);
    %   Y_pred = strat.predict(model, X_test);
    %   err    = strat.evaluate(Y_test, Y_pred);
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties
        Name       string  = "Random Forest"
        SupportsAIC logical = false

        %--- Hyperparameters forwarded to RandomForest ---
        NumTrees    double = 100
        Mtry        double = 0
        MaxDepth    double = Inf
        MinLeafSize double = 1
        Criterion   string = 'gini'
        Seed        double = 0
        ComputeOOB        logical = false  % off by default in strategy mode (speed)
        ComputeImportance logical = false
    end

    methods
        function obj = RandomForestStrategy(options)
            % RANDOMFORESTSTRATEGY Constructor (name-value).
            arguments
                options.NumTrees    double  = 100
                options.Mtry        double  = 0
                options.MaxDepth    double  = Inf
                options.MinLeafSize double  = 1
                options.Criterion   string  = 'gini'
                options.Seed        double  = 0
                options.ComputeOOB        logical = false
                options.ComputeImportance logical = false
            end
            obj.NumTrees          = options.NumTrees;
            obj.Mtry              = options.Mtry;
            obj.MaxDepth          = options.MaxDepth;
            obj.MinLeafSize       = options.MinLeafSize;
            obj.Criterion         = options.Criterion;
            obj.Seed              = options.Seed;
            obj.ComputeOOB        = options.ComputeOOB;
            obj.ComputeImportance = options.ComputeImportance;

            obj.Name = sprintf("Random Forest (B=%d, mtry=%s, %s)", ...
                obj.NumTrees, num2str(obj.Mtry), obj.Criterion);
        end

        function model = train(obj, X, Y, ~)
            % TRAIN Fits a RandomForest and returns it as the model artifact.
            rf = OceanStats.Ensemble.RandomForest( ...
                'NumTrees',          obj.NumTrees, ...
                'Mtry',              obj.Mtry, ...
                'MaxDepth',          obj.MaxDepth, ...
                'MinLeafSize',       obj.MinLeafSize, ...
                'Criterion',         obj.Criterion, ...
                'Seed',              obj.Seed, ...
                'ComputeOOB',        obj.ComputeOOB, ...
                'ComputeImportance', obj.ComputeImportance);
            rf.fit(X, Y);
            model = rf;   % RF handle is the artifact
        end

        function preds = predict(obj, model, X)
            % PREDICT Returns hard labels (classification) or values (regression).
            task = model.Task_;
            if strcmpi(task, 'classification')
                % Return soft probability of the positive class for binary,
                % or hard labels for multi-class.
                K = numel(model.Classes_);
                if K == 2
                    P     = model.predictProba(X);
                    preds = P(:, 2);   % probability of second class (AUC-friendly)
                else
                    preds = model.predict(X);
                end
            else
                preds = model.predict(X);
            end
        end

        function err = evaluate(obj, Y_true, Y_pred)
            % EVALUATE Misclassification rate (classif) or MSE (regression).
            task = obj.resolveTask_();
            if strcmpi(task, 'classification')
                Y_hard = round(Y_pred);   % threshold 0.5 for binary soft preds
                err    = mean(Y_true ~= Y_hard);
            else
                err = mean((Y_true - Y_pred).^2);
            end
        end

        function k = getComplexity(~, model)
            % GETCOMPLEXITY Returns NumTrees as the complexity proxy.
            k = model.NumTrees;
        end
    end

    methods (Access = private)
        function task = resolveTask_(obj)
            classifCriteria = {'gini','entropy','misclassification','crossentropy'};
            if any(strcmpi(obj.Criterion, classifCriteria))
                task = 'classification';
            else
                task = 'regression';
            end
        end
    end
end
