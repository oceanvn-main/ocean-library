classdef ExtraTreesStrategy < OceanStats.Base.IModelStrategy
    % EXTRATREESSTRATEGY IModelStrategy adapter for ExtraTreesForest.
    %
    % OVERVIEW:
    %   Thin wrapper exposing ExtraTreesForest through the IModelStrategy
    %   contract for use with CrossValidator, BiasVarianceAuditor, etc.
    %   Accessible via ModelFactory.create('extratrees').
    %
    % USAGE:
    %   strat = OceanStats.Selection.Strategies.ExtraTreesStrategy( ...
    %       'NumTrees', 100, 'Criterion', 'gini');
    %   model = strat.train(X_train, Y_train);
    %   Y_hat = strat.predict(model, X_test);
    %   err   = strat.evaluate(Y_test, Y_hat);
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties
        Name        string  = "Extra-Trees Forest"
        SupportsAIC logical = false

        NumTrees        double = 100
        Mtry            double = 0
        NumCandidates   double = 1
        MaxDepth        double = Inf
        MinLeafSize     double = 2
        Criterion       string = 'gini'
        UseBootstrap    logical = false
        Seed            double = 0
        ComputeImportance logical = false
    end

    methods
        function obj = ExtraTreesStrategy(options)
            arguments
                options.NumTrees        double  = 100
                options.Mtry            double  = 0
                options.NumCandidates   double  = 1
                options.MaxDepth        double  = Inf
                options.MinLeafSize     double  = 2
                options.Criterion       string  = 'gini'
                options.UseBootstrap    logical = false
                options.Seed            double  = 0
                options.ComputeImportance logical = false
            end
            obj.NumTrees        = options.NumTrees;
            obj.Mtry            = options.Mtry;
            obj.NumCandidates   = options.NumCandidates;
            obj.MaxDepth        = options.MaxDepth;
            obj.MinLeafSize     = options.MinLeafSize;
            obj.Criterion       = options.Criterion;
            obj.UseBootstrap    = options.UseBootstrap;
            obj.Seed            = options.Seed;
            obj.ComputeImportance = options.ComputeImportance;

            obj.Name = sprintf("Extra-Trees (B=%d, mtry=%s, K=%d, %s)", ...
                obj.NumTrees, num2str(obj.Mtry), obj.NumCandidates, obj.Criterion);
        end

        function model = train(obj, X, Y, ~)
            et = OceanStats.Ensemble.ExtraTreesForest( ...
                'NumTrees',        obj.NumTrees, ...
                'Mtry',            obj.Mtry, ...
                'NumCandidates',   obj.NumCandidates, ...
                'MaxDepth',        obj.MaxDepth, ...
                'MinLeafSize',     obj.MinLeafSize, ...
                'Criterion',       obj.Criterion, ...
                'UseBootstrap',    obj.UseBootstrap, ...
                'Seed',            obj.Seed, ...
                'ComputeOOB',      false, ...
                'ComputeImportance', obj.ComputeImportance);
            et.fit(X, Y);
            model = et;
        end

        function preds = predict(obj, model, X)
            task = model.Task_;
            if strcmpi(task, 'classification')
                K = numel(model.Classes_);
                if K == 2
                    P     = model.predictProba(X);
                    preds = P(:, 2);
                else
                    preds = model.predict(X);
                end
            else
                preds = model.predict(X);
            end
        end

        function err = evaluate(obj, Y_true, Y_pred)
            task = obj.resolveTask_();
            if strcmpi(task, 'classification')
                Y_hard = round(Y_pred);
                err    = mean(Y_true ~= Y_hard);
            else
                err = mean((Y_true - Y_pred).^2);
            end
        end

        function k = getComplexity(~, model)
            k = model.NumTrees;
        end
    end

    methods (Access = private)
        function task = resolveTask_(obj)
            if any(strcmpi(obj.Criterion, {'gini','entropy','misclassification','crossentropy'}))
                task = 'classification';
            else
                task = 'regression';
            end
        end
    end
end
